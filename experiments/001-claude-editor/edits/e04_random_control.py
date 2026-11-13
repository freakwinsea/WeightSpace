"""Control for e02: same layers, same top-64 columns, same update norm, but the written direction is a random
unit vector orthogonal to the doubt direction u. If this helps as much as e02, e02's gain is 'any perturbation'."""
from pathlib import Path
import torch

LAYERS, K = (15, 16, 18), 64


def edit(model):
    means = torch.load(Path(__file__).parents[1] / "calls/call005_forced_doubt.pt")["means"]
    g = torch.Generator().manual_seed(0)
    for L in LAYERS:
        u = (means["res_D"][L] - means["res_C"][L]).float(); u = (u / u.norm()).to(model.device)
        W = model.model.layers[L].mlp.down_proj.weight
        p = u @ W.float(); S = torch.topk(p.abs(), K).indices          # e02's columns and coefficients
        r = torch.randn(u.shape[0], generator=g).to(model.device)
        r = r - (r @ u) * u; r = r / r.norm()                            # random, orthogonal to u, unit
        W[:, S] = (W[:, S].float() - torch.outer(r, p[S])).to(W.dtype)  # ||delta|| matches e02 exactly
