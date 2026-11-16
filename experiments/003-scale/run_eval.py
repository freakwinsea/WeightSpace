"""Exp 003 eval (pre-registered in protocol.md). python run_eval.py <label> [edits...]
64 MATH-train problems (seed 3) x sampling seeds (0, 1000), 16k cap, 8 slots, one budgeted call per seed.
Saved as calls/eval_<label>_s<seed>.json and skipped if present, so a crash resumes."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "001-claude-editor"))
import ws

label, *edits = sys.argv[1:]
s = ws.Session("Qwen/Qwen3.5-4B", device="cpu")  # generate-only: keep torch off the GPU
for e in edits:
    print("edit", s.apply(e)["n"], "weights", flush=True)
probs = ws.math_train(64, seed=3)
for seed in (0, 1000):
    name = f"eval_{label}_s{seed}"
    if not (ws.HERE / "calls" / f"{name}.json").exists():
        summ, _ = s.generate(probs, purpose=f"4B EVAL {label} seed {seed}", hypothesis="edit > base per problem",
                             max_new_tokens=16384, seed=seed, parallel=8, save_as=name)
        print(name, summ, flush=True)
