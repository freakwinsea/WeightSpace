"""F, lexical control: the 0.8B residual direction that just raises doubt-token logits over commit-token logits
(centred, (1+w)-gained unembedding rows, mean over doubt tokens minus mean over commit tokens). No 2B, no forward
calls. If D (bridge) works only as well as F, the 'transfer' is lexical, not structural."""
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from c_recipe import LAYERS, project
from d_bridge import readout_matrix, ROOT

DOUBT = ["Wait", " Wait", "But", " But", "Hmm", " Hmm", "Actually", " Actually", "Alternatively", " Alternatively"]
COMMIT = ["So", " So", "Therefore", " Therefore", "Thus", " Thus", "Hence", " Hence"]


def edit(model):
    tok = AutoTokenizer.from_pretrained(ROOT / "models/Qwen3.5-0.8B")
    ids = lambda ws: sorted({e[0] for e in (tok.encode(w, add_special_tokens=False) for w in ws) if len(e) == 1})
    M = readout_matrix("Qwen3.5-0.8B")
    lex = M[ids(DOUBT)].mean(0) - M[ids(COMMIT)].mean(0)
    u = (lex / lex.norm()).to(model.device)
    for L in LAYERS:
        project(model, L, u)
