"""Smoke + throughput check (unbudgeted). Loads a model, generates for a few MATH train problems,
reports tokens/sec so we can fix the token cap. Usage: python smoke.py Qwen/Qwen3.5-0.8B 16 1024"""
import sys, time
import ws

if __name__ == "__main__":
    model_id, n, cap = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    probs = ws.math_train(n, seed=123)
    print("train", len(ws.math_train()), "math500", len(ws.math500()), flush=True)
    t = time.time()
    s = ws.Session(model_id, budgeted=False)
    print("load s", round(time.time() - t), type(s.model).__name__, flush=True)
    summ, rows = s.generate(probs, purpose="smoke/throughput", hypothesis="-", max_new_tokens=cap,
                            save_as=f"smoke_{model_id.split('/')[-1]}_{n}x{cap}", parallel=n)
    toks = sum(r["n_tok"] for r in rows)
    print(summ, "tok/s", round(toks / summ["seconds"], 1), flush=True)
