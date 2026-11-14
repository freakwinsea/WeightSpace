"""E, control: C's columns and coefficients, but writing a random unit direction orthogonal to C's u."""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from c_recipe import LAYERS, K, u_for


def edit(model):
    g = torch.Generator().manual_seed(0)
    for L in LAYERS:
        u = u_for(L, model.device)
        W = model.model.layers[L].mlp.down_proj.weight
        p = u @ W.float(); S = torch.topk(p.abs(), K).indices
        r = torch.randn(u.shape[0], generator=g).to(model.device)
        r = r - (r @ u) * u; r = r / r.norm()
        W[:, S] = (W[:, S].float() - torch.outer(r, p[S])).to(W.dtype)
