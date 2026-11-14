"""D, vector transfer: translate Exp 001's 2B doubt direction into 0.8B space through the shared vocabulary.
Readout r = centred(W_U2 * g2) u2 with g = 1 + final-norm weight (zero-centred RMSNorm); solve for x in 0.8B with
centred(W_U08 * g08) x ~= r (least squares). No 0.8B forward calls. Then the same rule as C."""
import sys
from pathlib import Path
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).parent))
from c_recipe import LAYERS, project

ROOT = Path(__file__).resolve().parents[3]
SRC2 = ROOT / "experiments/001-claude-editor/calls/call005_forced_doubt.pt"


def readout_matrix(model_dir):
    f = next((ROOT / "models" / model_dir).glob("*.safetensors"))
    with safe_open(f, "pt") as t:
        keys = t.keys()
        E = t.get_tensor(next(k for k in keys if k.endswith("embed_tokens.weight"))).float()
        g = 1 + t.get_tensor(next(k for k in keys if k.endswith("language_model.norm.weight"))).float()
    M = E * g                       # [V, d]: logits = M @ x (tied embeddings)
    return M - M.mean(0)            # centre over vocab: logits are shift-invariant


def bridge_map():
    """T [1024 x 2048]: least-squares map from 2B residual space to 0.8B space that preserves vocab readout."""
    M2, M08 = readout_matrix("Qwen3.5-2B").double(), readout_matrix("Qwen3.5-0.8B").double()
    return torch.linalg.solve(M08.T @ M08, M08.T @ M2).float(), M2.float(), M08.float()


def bridged(L, T, M2, M08):
    means = torch.load(SRC2)["means"]
    u2 = (means["res_D"][L] - means["res_C"][L]).float(); u2 = u2 / u2.norm()
    x = T @ u2; r = M2 @ u2
    fit = 1 - ((M08 @ x - r).norm() / r.norm()) ** 2
    return x / x.norm(), fit.item()


def edit(model):
    T, M2, M08 = bridge_map()
    for L in LAYERS:
        u, fit = bridged(L, T, M2, M08)
        print(f"bridge L{L}: readout R^2 {fit:.3f}", flush=True)
        project(model, L, u.to(model.device))
