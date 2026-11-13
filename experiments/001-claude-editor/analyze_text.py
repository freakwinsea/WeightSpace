"""Free (unbudgeted) text analysis of a generation call: why do answers fail?
python analyze_text.py calls/call001_gen.json"""
import sys, json, re
import ws

if __name__ == "__main__":
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    stats = []
    for r in d["rows"]:
        t = r["text"]
        k = max(1, r["n_tok"]) / 1000
        boxes = [m.start() for m in re.finditer(r"\\boxed", t)]
        # first point where a correct boxed answer appears anywhere in the text
        first_ok = None
        for b in boxes:
            if ws.correct(t[b:b + 200], r["gold"]):
                first_ok = b / max(1, len(t))
                break
        stats.append({
            "id": r["id"], "cut": r["cut"], "ok": r["correct"], "n_tok": r["n_tok"],
            "wait_per_k": round(len(re.findall(r"\bWait\b", t)) / k, 1),
            "subs_per_k": round(len(re.findall(r"\b(But|Hmm|Actually|Alternatively|Let me check|double-check|re-?check|However)\b", t)) / k, 1),
            "boxes": len(boxes), "closed_think": "</think>" in t,
            "first_correct_box_at": None if first_ok is None else round(first_ok, 2),
        })
    for s in stats:
        print(s)
    cut = [s for s in stats if s["cut"]]
    print("\ncut:", len(cut), "| cut but a correct box appeared mid-text:",
          sum(s["first_correct_box_at"] is not None for s in cut),
          "| cut with </think> closed:", sum(s["closed_think"] for s in cut))
    avg = lambda key, xs: round(sum(s[key] for s in xs) / max(1, len(xs)), 2)
    print(f"ALL n={len(stats)} acc={avg('ok', stats)} cut={avg('cut', stats)} mean_tok={avg('n_tok', stats)} "
          f"Wait/1k={avg('wait_per_k', stats)} subs/1k={avg('subs_per_k', stats)}")
    print("mean Wait/1k  cut:", round(sum(s["wait_per_k"] for s in cut) / max(1, len(cut)), 1),
          " finished:", round(sum(s["wait_per_k"] for s in stats if not s["cut"]) / max(1, len(stats) - len(cut)), 1))
