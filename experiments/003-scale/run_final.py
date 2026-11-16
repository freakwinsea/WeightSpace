"""Exp 003 definitive test (pre-registered in protocol.md).
python run_final.py plan                      -> sizes n from the calibration by the fixed rule, writes calls/final_plan.json
python run_final.py <label> <chunk> <seed> [edits...]  -> one budgeted call: chunk of <=64 problems at that seed
Saved as calls/final_<label>_c<chunk>_s<seed>.json and skipped if present."""
import json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "001-claude-editor"))
import ws
import run_calib

C, N_MAX, CHUNK = ws.HERE / "calls", 192, 64


def problems(n):
    used = {r["id"] for r in run_calib.problems()} | {r["id"] for r in ws.math_train(32, seed=1)} \
        | {r["id"] for r in ws.math_train(64, seed=3)}
    return [r for r in ws.math_train(seed=7)
            if r["level"] == "Level 5" and r["answer"] and r["id"] not in used][:n]


def plan():
    b, e = ({r["id"]: r for r in json.loads((C / f"calib_l5_{k}.json").read_text(encoding="utf-8"))["rows"]}
            for k in ("base", "edit"))
    fin = sum(not r["cut"] for r in b.values()) / len(b)
    d1 = sum(b[i]["correct"] != e[i]["correct"] for i in b) / len(b)
    d2 = 1 - (1 - d1) ** 2
    p = {"base_finish": fin, "d1": d1, "d2": d2,
         "edit_wins": sum(e[i]["correct"] and not b[i]["correct"] for i in b),
         "base_wins": sum(b[i]["correct"] and not e[i]["correct"] for i in b)}
    if not 0.25 <= fin <= 0.75:
        p["verdict"] = "wrong band: base finish rate outside 25-75%; report, do not size"
    elif d1 == 0:
        p["verdict"] = "no discordance in calibration; report, do not size"
    else:
        p["n_rule"] = math.ceil(40 / d2)
        p["n"] = min(p["n_rule"], N_MAX)
        p["chunks"] = math.ceil(p["n"] / CHUNK)
    (C / "final_plan.json").write_text(json.dumps(p, indent=1), encoding="utf-8")
    print(json.dumps(p))


if __name__ == "__main__":
    if sys.argv[1] == "plan":
        plan(); sys.exit()
    label, chunk, seed, *edits = sys.argv[1:]
    chunk, seed = int(chunk), int(seed)
    name = f"final_{label}_c{chunk}_s{seed}"
    if (C / f"{name}.json").exists():
        sys.exit()
    n = json.loads((C / "final_plan.json").read_text(encoding="utf-8"))["n"]
    probs = problems(n)[chunk * CHUNK:(chunk + 1) * CHUNK]
    s = ws.Session("Qwen/Qwen3.5-4B", device="cpu")  # generate-only: keep torch off the GPU
    for e in edits:
        print("edit", s.apply(e)["n"], "weights", flush=True)
    summ, _ = s.generate(probs, purpose=f"4B FINAL {label} chunk {chunk} seed {seed}", hypothesis="edit > base per problem",
                         max_new_tokens=16384, seed=seed, parallel=8, save_as=name)
    print(name, summ, flush=True)
