"""Harness for Exp 001: load the model, apply edits, run budgeted forward calls, log everything.

Usage from a call script:
    import ws
    m = ws.Session("Qwen/Qwen3.5-2B")
    m.apply("edits/e01.py")          # optional: edit file defines edit(model) -> None
    out = m.generate(problems, purpose="...", hypothesis="...")   # costs 1 budget call
"""
import json, os, time, re, random, importlib.util
from pathlib import Path
import torch

HERE = Path(__file__).parent
LOG = HERE / "log.jsonl"
BUDGET = 50
MAX_BATCH = 64
PROMPT_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5)  # MIT "think35" settings


# ---------- data (cached to data/*.jsonl so call scripts don't touch the datasets machinery) ----------
DATA = HERE.parent.parent / "data"


def _cached(name, build):
    p = DATA / f"{name}.jsonl"
    if not p.exists():
        DATA.mkdir(exist_ok=True)
        p.write_text("\n".join(json.dumps(r) for r in build()), encoding="utf-8")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


def math_train(n=None, seed=0):
    rows = _cached("math_train", _math_train)
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    return rows[:n] if n else rows


def math500():
    return _cached("math500", _math500)


def _math_train():
    from datasets import load_dataset
    subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]
    rows = []
    for s in subjects:
        for r in load_dataset("EleutherAI/hendrycks_math", s, split="train"):
            rows.append({"id": f"train/{s}/{len(rows)}", "problem": r["problem"],
                         "answer": last_boxed(r["solution"]), "level": r["level"], "subject": s})
    return rows


def _math500():
    from datasets import load_dataset
    return [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"],
             "level": r["level"], "subject": r["subject"]}
            for r in load_dataset("HuggingFaceH4/MATH-500", split="test")]


# ---------- grading ----------
def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        if depth == 0:
            return text[j + 1:k]
    return None


def correct(pred_text, gold):
    from math_verify import parse, verify
    b = last_boxed(pred_text)
    if b is None:
        return False
    try:
        # timeouts off: math-verify's timeout spawns processes, which breaks under this Windows harness
        g = parse("\\boxed{" + gold + "}", parsing_timeout=None)
        p = parse("\\boxed{" + b + "}", parsing_timeout=None)
        return bool(verify(g, p, timeout_seconds=None))
    except Exception:
        return False


def rep_score(ids, n=20):
    """Fraction of the last 2000 tokens that sit inside a repeated n-gram: a cheap loop meter."""
    ids = ids[-2000:]
    if len(ids) < 2 * n:
        return 0.0
    seen, rep = set(), 0
    for i in range(len(ids) - n):
        g = tuple(ids[i:i + n])
        rep += g in seen
        seen.add(g)
    return rep / (len(ids) - n)


# ---------- sampling ----------
class Presence(torch.nn.Module):
    """vLLM-style presence penalty: subtract p from logits of tokens already generated."""
    def __init__(self, p, prompt_len):
        super().__init__()
        self.p, self.L = p, prompt_len

    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.L:]
        if gen.shape[1]:
            mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, gen, True)
            scores = scores - self.p * mask
        return scores


# ---------- session ----------
def calls_used():
    if not LOG.exists():
        return 0
    return sum(1 for l in LOG.read_text(encoding="utf-8").splitlines()
               if l.strip() and json.loads(l).get("budgeted"))


class Session:
    def __init__(self, model_id, dtype=torch.bfloat16, budgeted=True):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id, self.budgeted, self.edits = model_id, budgeted, []
        local = HERE.parent.parent / "models" / model_id.split("/")[-1]
        src = str(local) if local.exists() else model_id
        self.tok = AutoTokenizer.from_pretrained(src, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(src, dtype=dtype).cuda().eval()

    # editing: an edit file defines edit(model); it must touch only transformer-block params
    def apply(self, path):
        spec = importlib.util.spec_from_file_location("edit", HERE / path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        with torch.no_grad():
            mod.edit(self.model)
        changed = {k: int((v != before[k]).sum()) for k, v in self.model.state_dict().items()
                   if not torch.equal(v, before[k])}
        bad = [k for k in changed if ".layers." not in k]
        if bad:
            raise ValueError(f"edit touched non-block params: {bad}")
        self.edits.append({"file": path, "changed": changed, "n": sum(changed.values())})
        return self.edits[-1]

    def prompts(self, problems):
        return [self.tok.apply_chat_template(
            [{"role": "user", "content": p["problem"] + PROMPT_SUFFIX}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True) for p in problems]

    def _spend(self, kind, purpose, hypothesis, n):
        if n > MAX_BATCH:
            raise ValueError(f"batch {n} > {MAX_BATCH}")
        used = calls_used()
        if self.budgeted and used >= BUDGET:
            raise RuntimeError("forward budget exhausted")
        return used + (1 if self.budgeted else 0)

    def _log(self, rec):
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def gguf(self):
        """GGUF for the current weights: the base file, or an export of the edited weights (cached by edit list)."""
        import hashlib, gen
        name = self.model_id.split("/")[-1].lower()
        base_dir = HERE.parent.parent / "models" / self.model_id.split("/")[-1]
        if not self.edits:
            out = HERE.parent.parent / "gguf" / f"{name}-base-bf16.gguf"
            return out if out.exists() else gen.to_gguf(base_dir, out)
        h = hashlib.sha1(json.dumps([e["file"] for e in self.edits]).encode()).hexdigest()[:10]
        out = HERE.parent.parent / "gguf" / f"{name}-edit-{h}-bf16.gguf"
        if not out.exists():
            changed = {k for e in self.edits for k in e["changed"]}
            tmp = gen.export_edited(self.model, base_dir, HERE.parent.parent / "gguf" / f"hf-{name}-{h}", changed)
            gen.to_gguf(tmp, out)
            import shutil
            shutil.rmtree(tmp)
        return out

    def generate(self, problems, purpose, hypothesis, max_new_tokens=4096, seed=0, save_as=None,
                 parallel=16):
        import gen
        call_no = self._spend("generate", purpose, hypothesis, len(problems))
        path = self.gguf()
        t0, rows = time.time(), []
        with gen.Server(path, parallel=min(parallel, len(problems)), cap=max_new_tokens) as srv:
            outs = srv.generate(self.prompts(problems), max_new_tokens, seed, SAMPLING)
        for p, (text, ids, stop) in zip(problems, outs):
            rows.append({"id": p["id"], "gold": p["answer"], "n_tok": len(ids), "stop": stop,
                         "cut": stop == "limit" or len(ids) >= max_new_tokens,
                         "rep": round(rep_score(ids), 3),
                         "correct": correct(text, p["answer"]), "text": text})
        summ = {"acc": sum(r["correct"] for r in rows) / len(rows),
                "cut": sum(r["cut"] for r in rows) / len(rows),
                "mean_tok": sum(r["n_tok"] for r in rows) / len(rows),
                "mean_rep": sum(r["rep"] for r in rows) / len(rows),
                "seconds": round(time.time() - t0), "n": len(rows)}
        name = save_as or (f"call{call_no:03d}_gen" if self.budgeted
                           else f"calib_{self.model_id.split('/')[-1]}_{time.strftime('%H%M%S')}")
        out_dir = HERE / "calls"
        out_dir.mkdir(exist_ok=True)
        (out_dir / f"{name}.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1),
                                              encoding="utf-8")
        self._log({"t": time.strftime("%H:%M:%S"), "call": call_no, "kind": "generate",
                   "budgeted": self.budgeted, "model": self.model_id, "edits": self.edits,
                   "purpose": purpose, "hypothesis": hypothesis, "max_new_tokens": max_new_tokens,
                   "file": f"calls/{name}.json", **summ})
        return summ, rows

    @torch.no_grad()
    def forced(self, texts, purpose, hypothesis, hook_layers=(), save_as=None):
        """Teacher-forced pass over full texts. Returns per-token logprobs and, for hook_layers,
        residual-stream outputs and MLP outputs (CPU float16)."""
        call_no = self._spend("forced", purpose, hypothesis, len(texts))
        layers = self.model.model.layers
        acts, hooks = {}, []
        for i in hook_layers:
            hooks.append(layers[i].register_forward_hook(
                lambda m, a, o, i=i: acts.setdefault(f"resid{i}", []).append(
                    (o[0] if isinstance(o, tuple) else o).float().cpu().half())))
            hooks.append(layers[i].mlp.register_forward_hook(
                lambda m, a, o, i=i: acts.setdefault(f"mlp{i}", []).append(o.float().cpu().half())))
        lps = []
        try:
            for t in texts:
                ids = self.tok(t, return_tensors="pt").input_ids.cuda()
                logits = self.model(ids).logits.float()
                lp = torch.log_softmax(logits[0, :-1], -1).gather(1, ids[0, 1:, None])[:, 0]
                lps.append(lp.cpu())
        finally:
            for h in hooks:
                h.remove()
        name = save_as or f"call{call_no:03d}_forced"
        out_dir = HERE / "calls"
        out_dir.mkdir(exist_ok=True)
        torch.save({"logprobs": lps, "acts": acts}, out_dir / f"{name}.pt")
        self._log({"t": time.strftime("%H:%M:%S"), "call": call_no, "kind": "forced",
                   "budgeted": self.budgeted, "model": self.model_id, "edits": self.edits,
                   "purpose": purpose, "hypothesis": hypothesis, "n": len(texts),
                   "hook_layers": list(hook_layers), "file": f"calls/{name}.pt"})
        return lps, acts
