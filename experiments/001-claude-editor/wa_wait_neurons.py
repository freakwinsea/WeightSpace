"""Free weight analysis: which MLP neurons write directly toward the 'Wait' logit?
For each layer, score each down_proj column by its dot with the unembedding of the Wait tokens
(final RMSNorm gain folded in), relative to that column's dot with an average token."""
import torch
import ws

if __name__ == "__main__":
    s = ws.Session("Qwen/Qwen3.5-2B", budgeted=False)
    tok, m = s.tok, s.model
    cands = ["Wait", " Wait", "\nWait", "Hmm", " Hmm", "But", " But", "Actually", " Actually", "Let", " let", "So", " So"]
    ids = {c: tok.encode(c, add_special_tokens=False) for c in cands}
    print({c: (v, [tok.decode([i]) for i in v]) for c, v in ids.items()})
    W_U = m.lm_head.weight.float()                      # vocab x d
    g = m.model.norm.weight.float()
    print("final norm weight mean/std", g.mean().item(), g.std().item(), "| layers", len(m.model.layers),
          "| d", W_U.shape[1], "| tied", m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr())
    # Qwen3.5 norms may be (1+w) style; report both and use whichever is the actual gain
    gain = g if g.mean().abs() > 0.3 else (1 + g)
    wait_ids = [i for c in ["Wait", " Wait"] for i in ids[c] if len(ids[c]) == 1]
    u_wait = (W_U[wait_ids] * gain).mean(0)
    u_mean = (W_U * gain).mean(0)
    target = u_wait - u_mean
    target = target / target.norm()
    rows = []
    for li, layer in enumerate(m.model.layers):
        D = layer.mlp.down_proj.weight.float()           # d x inter
        cos = (target @ D) / D.norm(dim=0)
        top = torch.topk(cos, 3)
        rows.append((li, cos.max().item(), cos.min().item(), top.indices.tolist(), [round(x, 3) for x in top.values.tolist()]))
        print(f"L{li:02d} max {cos.max():+.3f} min {cos.min():+.3f} top {top.indices.tolist()} {[round(x,3) for x in top.values.tolist()]}")
    torch.save({"target": target, "wait_ids": wait_ids}, ws.HERE / "calls" / "wa_wait_target.pt")
