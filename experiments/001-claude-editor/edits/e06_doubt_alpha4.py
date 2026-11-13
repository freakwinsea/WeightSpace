"""e02: in layers 15, 16, 18 remove the doubt-vs-commit direction (from call 5) from the 64 down_proj
columns that write most along it. u_L = unit(mean resid before doubt - before commit) at layer L."""
import torch
from pathlib import Path
LAYERS, K, ALPHA = (15, 16, 18), 64, 4.0  # e06: e02 at strength 4 (Astra's final used 6)
def edit(model):
    means = torch.load(Path(__file__).parents[1] / "calls/call005_forced_doubt.pt")["means"]
    for L in LAYERS:
        u = (means["res_D"][L] - means["res_C"][L]).float()
        u = (u / u.norm()).to(model.device)
        W = model.model.layers[L].mlp.down_proj.weight
        p = (u @ W.float())                               # each column's write along u
        S = torch.topk(p.abs(), K).indices
        W[:, S] = (W[:, S].float() - ALPHA * torch.outer(u, p[S])).to(W.dtype)
