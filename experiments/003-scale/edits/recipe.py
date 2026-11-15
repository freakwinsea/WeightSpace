"""4B recipe transfer: the 2B doubt rule with layers chosen by rule (top-3 MLP writers onto own doubt direction),
K=64, alpha=1. Direction from this experiment's call 2."""
from pathlib import Path
import torch

K, ALPHA, N_LAYERS = 64, 1.0, 3
SRC = Path(__file__).parents[1] / "calls/call002_forced_doubt.pt"


def directions():
    m = torch.load(SRC)["means"]
    us, writes = [], []
    for L in range(m["res_D"].shape[0]):
        u = (m["res_D"][L] - m["res_C"][L]).float(); u = u / u.norm()
        us.append(u); writes.append(((m["mlp_D"][L] - m["mlp_C"][L]).float() @ u).item())
    layers = sorted(range(len(us)), key=lambda L: -writes[L])[:N_LAYERS]
    return sorted(layers), us, writes


def edit(model):
    layers, us, _ = directions()
    print("recipe layers", layers, flush=True)
    for L in layers:
        W = model.model.layers[L].mlp.down_proj.weight; u = us[L].to(W.device)
        p = u @ W.float(); S = torch.topk(p.abs(), K).indices
        W[:, S] = (W[:, S].float() - ALPHA * torch.outer(u, p[S])).to(W.dtype)
