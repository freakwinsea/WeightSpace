"""Held-out MATH-500 eval, scored once. python heldout.py <label> <cap> <n|all> <parallel> [edit files...]
Unbudgeted (the editor never sees these); chunks of 64 saved as calls/heldout_<label>_<cap>_<k>.json and skipped
if present, so an overnight crash resumes. Subset = math500 shuffled with seed 0, first n (fixed before scoring)."""
import os, sys, json, random
from pathlib import Path
import ws

if __name__ == "__main__":
    label, cap, n, par, *edits = sys.argv[1:]
    probs = ws.math500()
    random.Random(0).shuffle(probs)
    probs = probs if n == "all" else probs[:int(n)]
    chunk = min(64, int(par)) if int(cap) > 4096 else 64
    s = ws.Session(os.environ.get("WS_MODEL", "Qwen/Qwen3.5-2B"), budgeted=False)
    for e in edits:
        print("edit", s.apply(e)["n"], "weights", flush=True)
    rows = []
    for k in range(0, len(probs), chunk):
        name = f"heldout_{label}_{cap}_{k // chunk:02d}"
        f = ws.HERE / "calls" / f"{name}.json"
        if not f.exists():
            s.generate(probs[k:k + chunk], purpose=f"HELD-OUT MATH-500 {label} @{cap}", hypothesis="scored once",
                       max_new_tokens=int(cap), parallel=int(par), save_as=name)
        rows += json.loads(f.read_text(encoding="utf-8"))["rows"]
    print(json.dumps({"label": label, "cap": cap, "n": len(rows), "acc": sum(r["correct"] for r in rows) / len(rows),
                      "cut": sum(r["cut"] for r in rows) / len(rows), "rep": sum(r["rep"] for r in rows) / len(rows)}), flush=True)
