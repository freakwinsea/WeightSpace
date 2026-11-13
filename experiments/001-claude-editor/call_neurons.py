"""Budgeted teacher-forced call: record candidate 'Wait' neurons (down_proj inputs) on every token of
earlier generations, and their direct contribution (activation x cos) to the Wait direction.
python call_neurons.py <gen_call.json> <n_texts> "<purpose>" "<hypothesis>" """
import sys, json, time
import torch
import ws

if __name__ == "__main__":
    src, n, purpose, hyp = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    s = ws.Session("Qwen/Qwen3.5-2B")
    m, tok = s.model, s.tok
    t = torch.load(ws.HERE / "calls/wa_wait_target.pt")
    target = t["target"].cuda()
    wait_ids = set(t["wait_ids"])
    # candidates: |cos| > 0.25 against the Wait direction
    cands = []
    for li, layer in enumerate(m.model.layers):
        D = layer.mlp.down_proj.weight.float()
        cos = (target @ D) / D.norm(dim=0)
        for j in torch.nonzero(cos.abs() > 0.25).flatten().tolist():
            cands.append((li, j, round(cos[j].item(), 3)))
    print("candidates", cands, flush=True)

    rows = json.load(open(ws.HERE / src, encoding="utf-8"))["rows"]
    fin = [r for r in rows if not r["cut"]]
    cut = [r for r in rows if r["cut"]]
    pick = (fin + cut)[:n] if len(fin) < n else fin[:n // 2] + cut[:n - n // 2]
    call_no = s._spend("forced", purpose, hyp, len(pick))
    probs = {p["id"]: p for p in ws.math_train()}

    rec = {}
    def hook(li):
        js = [j for (l, j, _) in cands if l == li]
        def f(mod, args):
            rec.setdefault(li, []).append(args[0][0, :, js].float().cpu())
        return f
    hooks = [m.model.layers[li].mlp.down_proj.register_forward_pre_hook(hook(li))
             for li in sorted({l for l, _, _ in cands})]
    out = []
    t0 = time.time()
    try:
        with torch.no_grad():
            for r in pick:
                prompt = s.prompts([probs[r["id"]]])[0]
                p_ids = tok(prompt, return_tensors="pt").input_ids
                g_ids = tok(r["text"], return_tensors="pt", add_special_tokens=False).input_ids
                ids = torch.cat([p_ids, g_ids], 1).cuda()
                rec.clear()
                m(ids)
                acts = {li: v[0] for li, v in rec.items()}       # [T, k]
                nxt = ids[0, 1:].tolist() + [-1]
                is_wait = torch.tensor([x in wait_ids for x in nxt])
                gen = torch.arange(ids.shape[1]) >= p_ids.shape[1]
                per = {}
                for li, A in acts.items():
                    js = [(j, c) for (l, j, c) in cands if l == li]
                    for k, (j, c) in enumerate(js):
                        a = A[:, k]
                        per[f"L{li}.{j}"] = {
                            "cos": c,
                            "mean_gen": round(a[gen].mean().item(), 4),
                            "mean_before_wait": round(a[is_wait & gen].mean().item(), 4) if (is_wait & gen).any() else None,
                            "mean_other": round(a[~is_wait & gen].mean().item(), 4),
                            "frac_pos": round((a[gen] > 0).float().mean().item(), 3),
                        }
                out.append({"id": r["id"], "cut": r["cut"], "ok": r["correct"], "n_wait": int((is_wait & gen).sum()),
                            "neurons": per})
                print(r["id"], r["cut"], int((is_wait & gen).sum()), flush=True)
    finally:
        for h in hooks:
            h.remove()
    name = f"call{call_no:03d}_forced_neurons"
    (ws.HERE / "calls" / f"{name}.json").write_text(json.dumps({"cands": cands, "rows": out}, indent=1), encoding="utf-8")
    s._log({"t": time.strftime("%H:%M:%S"), "call": call_no, "kind": "forced", "budgeted": True, "model": s.model_id,
            "edits": s.edits, "purpose": purpose, "hypothesis": hyp, "n": len(pick), "seconds": round(time.time() - t0),
            "file": f"calls/{name}.json"})
    # summary across texts
    keys = out[0]["neurons"].keys()
    print("\nneuron | cos | before-Wait mean | other mean | gen mean finished | gen mean cut")
    for k in keys:
        bw = [o["neurons"][k]["mean_before_wait"] for o in out if o["neurons"][k]["mean_before_wait"] is not None]
        ot = [o["neurons"][k]["mean_other"] for o in out]
        gf = [o["neurons"][k]["mean_gen"] for o in out if not o["cut"]]
        gc = [o["neurons"][k]["mean_gen"] for o in out if o["cut"]]
        avg = lambda x: round(sum(x) / len(x), 4) if x else None
        print(k, out[0]["neurons"][k]["cos"], avg(bw), avg(ot), avg(gf), avg(gc))
