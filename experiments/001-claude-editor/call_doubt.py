"""Budgeted teacher-forced call: find a 'doubt vs commit' direction in the residual stream.
At positions whose NEXT token starts a doubt (Wait/But/Hmm/Actually/Alternatively) vs a commit
(So/Therefore/Thus/Hence), capture each layer's residual output, MLP output and attention/mixer output.
Saves per-layer means so we can compute d_L = mean(doubt) - mean(commit) and which sublayers write it.
python call_doubt.py <gen_call.json> <n_texts> "<purpose>" "<hypothesis>" """
import sys, json, time
import torch
import ws

DOUBT = ["Wait", " Wait", "But", " But", "Hmm", " Hmm", "Actually", " Actually", "Alternatively", " Alternatively"]
COMMIT = ["So", " So", "Therefore", " Therefore", "Thus", " Thus", "Hence", " Hence"]

if __name__ == "__main__":
    src, n, purpose, hyp = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    s = ws.Session("Qwen/Qwen3.5-2B")
    m, tok = s.model, s.tok
    def one(words):  # single-token ids only
        enc = [tok.encode(w, add_special_tokens=False) for w in words]
        return {e[0] for e in enc if len(e) == 1}
    D_ids, C_ids = one(DOUBT), one(COMMIT)
    print("doubt ids", D_ids, "commit ids", C_ids, flush=True)
    rows = json.load(open(ws.HERE / src, encoding="utf-8"))["rows"]
    rows = rows[:n]
    call_no = s._spend("forced", purpose, hyp, len(rows))
    probs = {p["id"]: p for p in ws.math_train()}
    L = len(m.model.layers)
    acc = {k: torch.zeros(L, m.config.hidden_size, dtype=torch.float64) for k in
           ["res_D", "res_C", "mlp_D", "mlp_C", "mix_D", "mix_C"]}
    cnt = {"D": 0, "C": 0}
    pos = {}
    cap = {}

    def grab(name):
        def f(mod, a, o):
            o = o[0] if isinstance(o, tuple) else o
            cap[name] = o[0]                              # [T, d] on GPU
        return f
    hooks = []
    for li, layer in enumerate(m.model.layers):
        hooks.append(layer.register_forward_hook(grab(f"res{li}")))
        hooks.append(layer.mlp.register_forward_hook(grab(f"mlp{li}")))
        mixer = getattr(layer, "linear_attn", None) or getattr(layer, "self_attn")
        hooks.append(mixer.register_forward_hook(grab(f"mix{li}")))
    t0 = time.time()
    try:
        with torch.no_grad():
            for r in rows:
                p_ids = tok(s.prompts([probs[r["id"]]])[0], return_tensors="pt").input_ids
                g_ids = tok(r["text"], return_tensors="pt", add_special_tokens=False).input_ids
                ids = torch.cat([p_ids, g_ids], 1).cuda()
                cap.clear()
                m(ids)
                nxt = ids[0, 1:].tolist()
                P = p_ids.shape[1]
                dpos = [t for t in range(P, len(nxt)) if nxt[t] in D_ids]
                cpos = [t for t in range(P, len(nxt)) if nxt[t] in C_ids]
                for li in range(L):
                    for k, ps in (("D", dpos), ("C", cpos)):
                        if ps:
                            idx = torch.tensor(ps, device=ids.device)
                            acc[f"res_{k}"][li] += cap[f"res{li}"][idx].double().sum(0).cpu()
                            acc[f"mlp_{k}"][li] += cap[f"mlp{li}"][idx].double().sum(0).cpu()
                            acc[f"mix_{k}"][li] += cap[f"mix{li}"][idx].double().sum(0).cpu()
                cnt["D"] += len(dpos)
                cnt["C"] += len(cpos)
                print(r["id"], "doubt", len(dpos), "commit", len(cpos), flush=True)
    finally:
        for h in hooks:
            h.remove()
    means = {k: (v / max(1, cnt[k[-1]])).float() for k, v in acc.items()}
    name = f"call{call_no:03d}_forced_doubt"
    torch.save({"means": means, "cnt": cnt}, ws.HERE / "calls" / f"{name}.pt")
    s._log({"t": time.strftime("%H:%M:%S"), "call": call_no, "kind": "forced", "budgeted": True, "model": s.model_id,
            "edits": s.edits, "purpose": purpose, "hypothesis": hyp, "n": len(rows), "cnt": cnt,
            "seconds": round(time.time() - t0), "file": f"calls/{name}.pt"})
    # report: size of doubt-commit gap per layer, and which sublayer writes it
    print("\nlayer | |d_res| | cos(d_L, d_23) | mlp write onto d_L | mixer write onto d_L")
    d_last = means["res_D"][-1] - means["res_C"][-1]
    for li in range(L):
        d = means["res_D"][li] - means["res_C"][li]
        u = d / d.norm()
        mw = ((means["mlp_D"][li] - means["mlp_C"][li]) @ u).item()
        xw = ((means["mix_D"][li] - means["mix_C"][li]) @ u).item()
        print(f"L{li:02d} {d.norm():8.2f} {torch.cosine_similarity(d, d_last, 0):+.3f} {mw:+8.2f} {xw:+8.2f}")
