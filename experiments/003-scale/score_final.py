"""Score the Exp 003 definitive test once, per protocol.md. Exploratory: tokens to first \boxed{ (char-position estimate)."""
import json
from pathlib import Path
from math import comb


def sign_p(w, l):
    k = w + l
    return min(1.0, 2 * sum(comb(k, j) for j in range(min(w, l) + 1)) / 2 ** k) if k else 1.0

C, SEEDS = Path(__file__).parent / "calls", (0, 1000)
n = json.loads((C / "final_plan.json").read_text(encoding="utf-8"))["n"]
chunks = -(-n // 64)


def load(label):
    return {s: {r["id"]: r for c in range(chunks) for r in
                json.loads((C / f"final_{label}_c{c}_s{s}.json").read_text(encoding="utf-8"))["rows"]} for s in SEEDS}


def first_commit(r):
    i = r["text"].find(r"\boxed{")
    return None if i < 0 else r["n_tok"] * i / max(len(r["text"]), 1)


base, edit = load("base"), load("edit")
ids = list(base[0])
assert len(ids) == n and all(set(d[s]) == set(ids) for d in (base, edit) for s in SEEDS)


def wl(f):
    nets = [sum(f(edit[s][i]) for s in SEEDS) - sum(f(base[s][i]) for s in SEEDS) for i in ids]
    return sum(x > 0 for x in nets), sum(x < 0 for x in nets)


for name, d in (("base", base), ("edit", edit)):
    rows = [d[s][i] for s in SEEDS for i in ids]
    fin = [r for r in rows if not r["cut"]]
    fc = [x for x in map(first_commit, rows) if x is not None]
    print(f"{name}: correct {sum(r['correct'] for r in rows)}/{len(rows)}  finished {len(fin)} (correct {sum(r['correct'] for r in fin)})"
          f"  looping {sum(r['rep'] > 0.5 for r in rows)}  mean_tok {sum(r['n_tok'] for r in rows) / len(rows):.0f}"
          f"  first-boxed in {len(fc)} draws, median token {sorted(fc)[len(fc) // 2]:.0f}")
w, l = wl(lambda r: r["correct"]); print(f"PRIMARY per-problem edit/base {w}/{l} p={sign_p(w, l):.4f}")
w, l = wl(lambda r: not r["cut"]); print(f"finish per-problem {w}/{l} p={sign_p(w, l):.4f}")
dw = sum(edit[s][i]["correct"] and not base[s][i]["correct"] for s in SEEDS for i in ids)
dl = sum(base[s][i]["correct"] and not edit[s][i]["correct"] for s in SEEDS for i in ids)
print(f"draw-level {dw}/{dl} p={sign_p(dw, dl):.4f} (anti-conservative)")
both = [(base[s][i]["n_tok"], edit[s][i]["n_tok"]) for s in SEEDS for i in ids if base[s][i]["correct"] and edit[s][i]["correct"]]
print(f"tokens on {len(both)} draws both solved: base {sum(b for b, _ in both) / len(both):.0f} edit {sum(e for _, e in both) / len(both):.0f}")
pf = [(first_commit(base[s][i]), first_commit(edit[s][i])) for s in SEEDS for i in ids]
pf = [(b, e) for b, e in pf if b is not None and e is not None]
print(f"exploratory first-boxed, {len(pf)} draws both boxed: base median {sorted(b for b, _ in pf)[len(pf) // 2]:.0f} "
      f"edit median {sorted(e for _, e in pf)[len(pf) // 2]:.0f}; edit earlier in {sum(e < b for b, e in pf)}, later in {sum(e > b for b, e in pf)}")
for s in SEEDS:
    print(f"seed {s}: base {sum(r['correct'] for r in base[s].values())} edit {sum(r['correct'] for r in edit[s].values())}")
