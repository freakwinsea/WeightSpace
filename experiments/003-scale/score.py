"""Score Exp 003 once, per protocol.md: per-problem sign test on net correct over seeds 0 and 1000."""
import json
from math import comb
from pathlib import Path

C = Path(__file__).parent / "calls"
SEEDS = (0, 1000)


def load(label):
    return {s: {r["id"]: r for r in json.loads((C / f"eval_{label}_s{s}.json").read_text(encoding="utf-8"))["rows"]}
            for s in SEEDS}


def sign_p(w, l):
    n = w + l
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(w, l) + 1)) / 2 ** n) if n else 1.0


def has_gold(r):
    return r["gold"] is not None and r["gold"].replace(" ", "") in r["text"].replace(" ", "")


def wl(a, b, f):
    nets = [sum(f(b[s][i]) for s in SEEDS) - sum(f(a[s][i]) for s in SEEDS) for i in ids]
    return sum(n > 0 for n in nets), sum(n < 0 for n in nets)


base, edit = load("base"), load("edit")
ids = list(base[0])
assert all(set(d[s]) == set(ids) for d in (base, edit) for s in SEEDS)
for name, d in (("base", base), ("edit", edit)):
    rows = [d[s][i] for s in SEEDS for i in ids]
    fin = [r for r in rows if not r["cut"]]
    capped_wrong = [r for r in rows if r["cut"] and not r["correct"]]
    print(f"{name}: correct {sum(r['correct'] for r in rows)}/{len(rows)}  finished {len(fin)} "
          f"(correct {sum(r['correct'] for r in fin)})  looping {sum(r['rep'] > 0.5 for r in rows)}  "
          f"capped-wrong with gold {sum(map(has_gold, capped_wrong))}/{len(capped_wrong)}  "
          f"mean_tok {sum(r['n_tok'] for r in rows) / len(rows):.0f}")
w, l = wl(base, edit, lambda r: r["correct"]); print(f"PRIMARY per-problem edit/base {w}/{l} p={sign_p(w, l):.4f}")
w, l = wl(base, edit, lambda r: not r["cut"]); print(f"finish per-problem {w}/{l} p={sign_p(w, l):.4f}")
dw = sum(edit[s][i]["correct"] and not base[s][i]["correct"] for s in SEEDS for i in ids)
dl = sum(base[s][i]["correct"] and not edit[s][i]["correct"] for s in SEEDS for i in ids)
print(f"draw-level {dw}/{dl} p={sign_p(dw, dl):.4f} (anti-conservative)")
both = [(base[s][i]["n_tok"], edit[s][i]["n_tok"]) for s in SEEDS for i in ids
        if base[s][i]["correct"] and edit[s][i]["correct"]]
print(f"tokens on {len(both)} draws both solved: base {sum(b for b, _ in both) / len(both):.0f} "
      f"edit {sum(e for _, e in both) / len(both):.0f}")
for s in SEEDS:
    print(f"seed {s}: base {sum(r['correct'] for r in base[s].values())} edit {sum(r['correct'] for r in edit[s].values())}")
