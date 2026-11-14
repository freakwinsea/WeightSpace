"""Powered B vs BC test (pre-registered in protocol.md). python run_powered.py <label> <model> [edits...]
128 fresh MATH-train problems (seed 5) x 2 sampling seeds (0, 1000), 16k cap, chunks of 64 (one budgeted call each).
Chunks are saved as calls/powered_<label>_s<seed>_<k>.json and skipped if present, so a crash resumes."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "001-claude-editor"))
import ws

label, model, *edits = sys.argv[1:]
s = ws.Session(model)
for e in edits:
    print("edit", s.apply(e)["n"], "weights", flush=True)
probs = ws.math_train(128, seed=5)
for seed in (0, 1000):
    for k in (0, 1):
        name = f"powered_{label}_s{seed}_{k}"
        if not (ws.HERE / "calls" / f"{name}.json").exists():
            summ, _ = s.generate(probs[64 * k:64 * k + 64], purpose=f"POWERED {label} seed {seed} chunk {k}",
                                 hypothesis="BC > B per problem", max_new_tokens=16384, seed=seed, parallel=32, save_as=name)
            print(name, summ, flush=True)
