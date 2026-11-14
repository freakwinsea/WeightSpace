"""C, recipe transfer: Exp 001's e02 rule (L15/16/18, top-64 columns, alpha 1) with 0.8B's OWN doubt direction."""
from pathlib import Path
import torch

LAYERS, K, ALPHA = (15, 16, 18), 64, 1.0
SRC = Path(__file__).parents[1] / "calls/call001_forced_doubt.pt"


def u_for(L, device):
    means = torch.load(SRC)["means"]
    u = (means["res_D"][L] - means["res_C"][L]).float()
    return (u / u.norm()).to(device)


def project(model, L, u, alpha=ALPHA, k=K):
    W = model.model.layers[L].mlp.down_proj.weight
    p = u @ W.float(); S = torch.topk(p.abs(), k).indices
    W[:, S] = (W[:, S].float() - alpha * torch.outer(u, p[S])).to(W.dtype)
    return S, p


def edit(model):
    for L in LAYERS:
        project(model, L, u_for(L, model.device))
