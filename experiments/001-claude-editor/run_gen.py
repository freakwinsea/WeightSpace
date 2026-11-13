"""Generation call. python run_gen.py <model> <dev_seed> <n> <cap> "<purpose>" "<hypothesis>" [edit files...]
Budgeted unless WS_UNBUDGETED=1 (calibration only: Arm 0 runs on other models)."""
import os, sys, json
import ws

if __name__ == "__main__":
    model, seed, n, cap, purpose, hyp, *edits = sys.argv[1:]
    s = ws.Session(model, budgeted=os.environ.get("WS_UNBUDGETED") != "1")
    for e in edits:
        print("edit", s.apply(e)["n"], "weights", flush=True)
    probs = ws.math_train(int(n), seed=int(seed))
    summ, rows = s.generate(probs, purpose=purpose, hypothesis=hyp, max_new_tokens=int(cap), parallel=32)
    print(json.dumps(summ), flush=True)
