import torch, ws
if __name__ == "__main__":
    s = ws.Session("Qwen/Qwen3.5-2B", budgeted=False)
    means = torch.load(ws.HERE / "calls/call005_forced_doubt.pt")["means"]
    for L in (14, 15, 16, 17, 18, 20, 21, 22):
        u = means["res_D"][L] - means["res_C"][L]; u = (u / u.norm()).cuda()
        p = u @ s.model.model.layers[L].mlp.down_proj.weight.float()
        e = (p ** 2); top = torch.topk(e, 64).values.sum() / e.sum()
        print(f"L{L}: share of |u^T W|^2 in top-64 of {p.numel()} cols = {top:.3f}; max |p| {p.abs().max():.3f}")
