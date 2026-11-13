import torch, ws
from safetensors.torch import load_file
if __name__ == "__main__":
    s = ws.Session("Qwen/Qwen3.5-2B", budgeted=False)
    sd_mem = {k.split("layers.", 1)[1]: v for k, v in s.model.state_dict().items() if "layers." in k}
    f = next((ws.HERE.parent.parent / "models/Qwen3.5-2B").glob("*.safetensors"))
    sd = load_file(f)
    for k, v in sd.items():
        if "language_model.layers." in k:
            suf = k.split("layers.", 1)[1]
            if suf not in sd_mem:
                continue
            m = sd_mem[suf].cpu()
            if m.shape != v.shape or not torch.equal(m.to(v.dtype), v):
                d = (m.float() - v.float()).abs()
                print(k, v.dtype, tuple(v.shape), "mem", m.dtype, tuple(m.shape), "maxdiff", d.max().item() if m.shape == v.shape else "shape")
