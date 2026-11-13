"""Non-math preservation check, base vs edit, unbudgeted, pre-registered in protocol.md.
python preserve.py mmlu <label> [edits...]   128 non-math MMLU-Pro test questions, thinking on, 4k cap
python preserve.py wiki <label> [edits...]   teacher-forced NLL on 64 ~4000-char wikitext-2 test chunks
python preserve.py score                     paired comparison of whatever has run"""
import sys, json, re, random
import pandas as pd
import ws

DATA = ws.DATA
LET = "ABCDEFGHIJ"


def mmlu(n=128):
    m = pd.read_parquet(DATA / "mmlu_pro_test.parquet")
    rows = [r for r in m.to_dict("records") if r["category"] != "math"]
    random.Random(0).shuffle(rows)
    return [{"id": f"mmlupro/{r['question_id']}/{r['category']}", "answer": r["answer"],
             "problem": r["question"] + "\n\n" + "\n".join(f"{LET[i]}. {o}" for i, o in enumerate(r["options"]))
                        + "\n\nAnswer with the letter of the correct option."} for r in rows[:n]]


def letter(text, bare=False):
    b = ws.last_boxed(text)
    if b is None and bare:  # thinking-off answers often skip the box: accept a leading letter or "answer is X"
        m = re.match(r"\s*\(?([A-J])\)?(?:[.):\s]|$)", text) or re.search(r"answer is:?\s*\(?([A-J])\)?\b", text)
        return m.group(1) if m else None
    m = re.search(r"\b([A-J])\b", re.sub(r"\\text\{([^}]*)\}", r"\1", b or ""))
    return m.group(1) if m else None


def wiki():
    t = "".join(pd.read_parquet(DATA / "wikitext2_test.parquet")["text"])
    return [t[i:i + 4000] for i in range(0, 64 * 4000, 4000)]


def session(label, edits):
    s = ws.Session("Qwen/Qwen3.5-2B", budgeted=False)
    for e in edits:
        print("edit", s.apply(e)["n"], "weights", flush=True)
    return s


if __name__ == "__main__":
    kind, *rest = sys.argv[1:]
    out = ws.HERE / "calls"
    if kind == "mmlu":
        s, probs = session(rest[0], rest[1:]), mmlu()
        for k in range(0, len(probs), 64):
            name = f"preserve_mmlu_{rest[0]}_{k // 64}"
            if not (out / f"{name}.json").exists():
                s.generate(probs[k:k + 64], purpose=f"PRESERVATION MMLU-Pro non-math {rest[0]}", hypothesis="no loss vs base",
                           max_new_tokens=4096, parallel=32, save_as=name)
    elif kind == "mmlu_nt":  # thinking off: knowledge check without the 4k-cap floor (thinking run was 96% capped)
        s, probs = session(rest[0], rest[1:]), mmlu(512)
        s.prompts = lambda ps: [s.tok.apply_chat_template(
            [{"role": "user", "content": q["problem"] + "\nRespond with only the letter of the correct option inside \\boxed{}."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in ps]
        for k in range(0, len(probs), 64):
            name = f"preserve_mmlunt_{rest[0]}_{k // 64}"
            if not (out / f"{name}.json").exists():
                s.generate(probs[k:k + 64], purpose=f"PRESERVATION MMLU-Pro non-math, thinking off, {rest[0]}",
                           hypothesis="no loss vs base", max_new_tokens=2048, parallel=32, save_as=name)
    elif kind == "wiki":
        s = session(rest[0], rest[1:])
        s.forced(wiki(), purpose=f"PRESERVATION wikitext-2 NLL {rest[0]}", hypothesis="NLL change < 0.02 nats/token",
                 save_as=f"preserve_wiki_{rest[0]}")
    else:
        import torch
        from math import comb
        def load(lab, pre="mmlu"):
            d = {}
            for f in sorted(out.glob(f"preserve_{pre}_{lab}_*.json")):
                for r in json.loads(f.read_text(encoding="utf-8"))["rows"]:
                    r["ok"] = letter(r["text"], pre == "mmlunt") == r["gold"]; d[r["id"]] = r
            return d
        for pre in ("mmlu", "mmlunt"):
          a, b = load("base", pre), load("e02", pre)
          if a and b:
            print(f"== {pre}")
            w = sum(b[i]["ok"] and not a[i]["ok"] for i in a); l = sum(a[i]["ok"] and not b[i]["ok"] for i in a); n = w + l
            p = min(1, 2 * sum(comb(n, k) for k in range(max(w, l), n + 1)) / 2**n) if n else 1
            for lab, d in (("base", a), ("e02", b)):
                print(f"MMLU-Pro {lab}: {sum(r['ok'] for r in d.values())}/{len(d)} cut {sum(r['cut'] for r in d.values())} "
                      f"no-letter {sum(letter(r['text'], pre == 'mmlunt') is None for r in d.values())}")
            print(f"MMLU-Pro e02 vs base: wins {w} losses {l} p={p:.3f}")
            by = {}
            for i in a:
                c = i.split("/")[-1]; by.setdefault(c, [0, 0, 0]); by[c][0] += 1; by[c][1] += a[i]["ok"]; by[c][2] += b[i]["ok"]
            print("  by category (n, base, e02):", by)
        wa, wb = out / "preserve_wiki_base.pt", out / "preserve_wiki_e02.pt"
        if wa.exists() and wb.exists():
            la, lb = torch.load(wa)["logprobs"], torch.load(wb)["logprobs"]
            na = torch.cat(la).double().neg().mean().item(); nb = torch.cat(lb).double().neg().mean().item()
            d = [(-y.double().mean() + x.double().mean()).item() for x, y in zip(la, lb)]
            print(f"wikitext NLL base {na:.4f} e02 {nb:.4f} delta {nb - na:+.4f} nats/token; "
                  f"chunks worse {sum(x > 0 for x in d)}/{len(d)}, max chunk delta {max(d):+.4f}")
