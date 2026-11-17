"""Build the released checkpoint: Qwen3.5-2B + edit e02 (Exp 001), and check it bit-for-bit against the GGUF
that was evaluated. python release/build_2b.py [--device cpu|cuda]
Output: release/out/Qwen3.5-2B-doubt-edit/ (HF folder: base files with only the 3 edited down_proj tensors swapped)."""
import argparse, hashlib, importlib.util, json, shutil, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/001-claude-editor"
sys.path.insert(0, str(EXP))
import gen  # noqa: E402

BASE = ROOT / "models/Qwen3.5-2B"
OUT = ROOT / "release/out/Qwen3.5-2B-doubt-edit"
EDIT = "edits/e02_project_doubt_L15_16_18.py"
LAYERS = (15, 16, 18)


def build(device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(device).eval()
    spec = importlib.util.spec_from_file_location("e02", EXP / EDIT)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    keys = [f"model.layers.{L}.mlp.down_proj.weight" for L in LAYERS]
    before = {k: model.state_dict()[k].clone() for k in keys}
    with torch.no_grad():
        mod.edit(model)
    sd = model.state_dict()
    changed = sum(int((sd[k] != before[k]).sum()) for k in keys)
    print("changed weights:", changed, flush=True)
    gen.export_edited(model, BASE, OUT, set(keys))
    return {k: sd[k].float().cpu().numpy() for k in keys}, changed


def check_against_evaluated_gguf(edited):
    sys.path.insert(0, str(ROOT / "tools/llama-convert/gguf-py"))
    from gguf import GGUFReader
    h = hashlib.sha1(json.dumps([EDIT]).encode()).hexdigest()[:10]
    path = ROOT / f"gguf/qwen3.5-2b-edit-{h}-bf16.gguf"
    t = {x.name: x for x in GGUFReader(str(path)).tensors}
    for L, k in zip(LAYERS, edited):
        g = t[f"blk.{L}.ffn_down.weight"]
        bf = np.asarray(g.data).view(np.uint16).astype(np.uint32) << 16
        a = bf.view(np.float32).reshape(edited[k].shape)
        print(f"L{L}: identical to evaluated GGUF = {np.array_equal(a, edited[k])}", flush=True)
        assert np.array_equal(a, edited[k]), f"L{L} differs from {path.name}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda")  # cpu differs in 32 weights by 1 bf16 ulp (float accumulation order)
    edited, n = build(ap.parse_args().device)
    assert n == 383_099, n
    check_against_evaluated_gguf(edited)
    shutil.copy(Path(__file__).parent / "qwen3.5-2b-doubt-edit/README.md", OUT / "README.md")
    shutil.copy(EXP / EDIT, OUT / "e02_edit.py")
    print("ready:", OUT)
