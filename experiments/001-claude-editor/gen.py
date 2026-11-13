"""Fast generation through llama-server (Vulkan), plus edited-checkpoint export.

    with Server(gguf_path, parallel=16, cap=4096) as srv:
        rows = srv.generate(prompts, problems, cap=4096, seed=0)
"""
import json, os, shutil, socket, subprocess, sys, time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LLAMA = Path(os.environ["LOCALAPPDATA"]) / "Microsoft/WinGet/Packages/ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe/llama-server.exe"
CONVERT = ROOT / "tools/llama-convert/convert_hf_to_gguf.py"
DEFAULT_PORT = int(os.environ.get("LLAMA_PORT", 8090))


def free_port(start=DEFAULT_PORT, tries=20):
    for p in range(start, start + tries):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
            except OSError:
                continue
        if p != start:
            print(f"Port {start} is busy, falling back to {p}", flush=True)
        return p
    raise RuntimeError(f"no free port in {start}-{start + tries - 1}")


class Server:
    def __init__(self, gguf, parallel=16, cap=4096, prompt_room=1024):
        self.gguf, self.parallel = str(gguf), parallel
        self.ctx = parallel * (cap + prompt_room)
        self.proc = None

    def __enter__(self):
        self.port = free_port()
        print(f"llama-server on 127.0.0.1:{self.port}", flush=True)
        log = open(ROOT / f"gguf/server-{time.strftime('%m%d-%H%M%S')}.log", "w")  # one per run: call 8 500 was lost to overwrite
        self.proc = subprocess.Popen(
            [str(LLAMA), "-m", self.gguf, "-ngl", "99", "-np", str(self.parallel), "-c", str(self.ctx),
             "-fa", "on", "--host", "127.0.0.1", "--port", str(self.port), "--no-webui"],
            stdout=log, stderr=subprocess.STDOUT)
        for _ in range(600):
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited; see gguf/server-*.log")
            try:
                if json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2)).get("status") == "ok":
                    return self
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError("llama-server did not become healthy")

    def __exit__(self, *a):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(30)

    def _one(self, prompt, cap, seed, sampling):
        body = {"prompt": prompt, "n_predict": cap, "seed": seed, "cache_prompt": False,
                "temperature": sampling["temperature"], "top_k": sampling["top_k"], "top_p": sampling["top_p"],
                "presence_penalty": sampling["presence_penalty"], "min_p": 0.0, "repeat_penalty": 1.0,
                "return_tokens": True}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/completion", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=7200))
        return r["content"], r.get("tokens") or [], r.get("stop_type")

    def generate(self, prompts, cap, seed, sampling):
        with ThreadPoolExecutor(self.parallel) as ex:
            return list(ex.map(lambda ip: self._one(ip[1], cap, seed + ip[0], sampling), enumerate(prompts)))


def export_edited(model, base_dir, out_dir, changed_keys):
    """Write an HF folder = base folder with ONLY the edited tensors swapped in (changed_keys: state-dict
    names the edits touched), keyed by the file's own names (the multimodal file prefixes model.language_model.).
    Comparing whole state dicts is wrong: bf16 loading rounds the fp32 linear_attn norms (bug found at call 3)."""
    from safetensors.torch import load_file, save_file
    base_dir, out_dir = Path(base_dir), Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(base_dir, out_dir, ignore=shutil.ignore_patterns("*.safetensors"))
    edited = {k.split("layers.", 1)[1]: v for k, v in model.state_dict().items()
              if "layers." in k and k in changed_keys}
    for f in base_dir.glob("*.safetensors"):
        sd = load_file(f)
        n = 0
        for k in sd:
            if "language_model.layers." in k or (k.startswith("model.layers.")):
                suf = k.split("layers.", 1)[1]
                if suf in edited:
                    sd[k] = edited[suf].to(sd[k].dtype).cpu().contiguous()
                    n += 1
        save_file(sd, out_dir / f.name, metadata={"format": "pt"})
        print(f"{f.name}: {n} tensors replaced", flush=True)
    return out_dir


def to_gguf(hf_dir, out_file):
    subprocess.run([sys.executable, str(CONVERT), str(hf_dir), "--outtype", "bf16", "--outfile", str(out_file)],
                   check=True, cwd=CONVERT.parent, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_file
