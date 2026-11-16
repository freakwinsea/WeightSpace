"""Exp 003 calibration on harder problems (pre-registered in protocol.md). python run_calib.py <label> [edits...]
32 MATH-train Level 5 problems (seed-6 shuffle, not in calls 1 or the eval set, non-empty gold), seed 0, 16k, 8 slots.
Saved as calls/calib_l5_<label>.json and skipped if present."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "001-claude-editor"))
import ws


def problems():
    used = {r["id"] for r in ws.math_train(32, seed=1)} | {r["id"] for r in ws.math_train(64, seed=3)}
    return [r for r in ws.math_train(seed=6)
            if r["level"] == "Level 5" and r["answer"] and r["id"] not in used][:32]


if __name__ == "__main__":
    label, *edits = sys.argv[1:]
    name = f"calib_l5_{label}"
    if not (ws.HERE / "calls" / f"{name}.json").exists():
        s = ws.Session("Qwen/Qwen3.5-4B", device="cpu")  # generate-only: keep torch off the GPU
        for e in edits:
            print("edit", s.apply(e)["n"], "weights", flush=True)
        summ, _ = s.generate(problems(), purpose=f"4B L5 CALIBRATION {label}", hypothesis="size the definitive test",
                             max_new_tokens=16384, seed=0, parallel=8, save_as=name)
        print(name, summ, flush=True)
