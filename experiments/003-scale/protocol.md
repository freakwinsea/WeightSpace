# Exp 003: does the doubt recipe transfer *up*, to Qwen3.5-4B?

Written 2026-11-14, **before any 4B call**. Changes go in the change log at the bottom. Harness: Exp 001's `ws.py`, `gen.py` and `call_doubt.py`, with `WS_HERE` set to this folder. Budget: 50 calls, same rules.

## Question
The recipe was found on 2B and worked again on 0.8B with 0.8B's own direction: four tests in the same direction, about 1.5× the correct answers. Does it work on a *larger* model?
- 4B has 32 layers (2B has 24), a hidden width of 2560, an MLP width of 9216, the same 3-linear + 1-full attention pattern, and tied embeddings.
- The user observes the same small-step indecision in Qwen3.5 models up to 27B.
- The WSP-Scope 9B edit lost 0.35 points on math.

The answer tells us whether this is a small-model quirk or a family-wide lever.

## Procedure (fixed now)
1. **Call 1 (generate):** base 4B, 32 MATH-train problems (seed 1, the same calibration rule as before), 4k cap. It provides texts for the direction, a throughput measurement, and a look at the failure mode: cut rate, repetition, and capped traces that contain the gold.
2. **Call 2 (forced):** `call_doubt.py` on those texts gives 4B's own doubt direction per layer and the MLP writes onto it.
3. **The edit, `edits/recipe.py`:** the 2B rule, with layers chosen by a fixed rule rather than by hand.
   - **Layers:** the top 3 by MLP write onto the layer's own doubt direction.
   - **Columns:** remove u from the top-64 down_proj columns by |u·W|.
   - **Strength:** α = 1.
   - Nothing is tuned.
4. **Eval:** base vs edit on 64 fresh MATH-train problems (seed 3, the first 64; it includes Arm 2's 32, so the results can be compared across sizes) × 2 sampling seeds (0 and 1000).
   - 16k cap, 8 slots, because the 4B KV cache limits it to 8.
   - 4 calls.

## Predictions (written before any call)
- **Layers:** the top-3 writer layers sit at the same *relative* depth as on 2B and 0.8B (0.62–0.75), which is L20–L24 of 32.
- **Failure mode** (the user's hypothesis): finished answers are mostly correct (≥ 90%), and a substantial share of capped wrong traces already contain the gold.
- **The edit helps:** edit > base.
  - I expect the gain to be larger than on 0.8B, because 4B has more capacity to use once it commits. 2B gained most when it could already solve the problem.

## Primary endpoint
For each problem, net = (edit correct over 2 draws) − (base correct over 2 draws).
- Two-sided sign test over problems with net ≠ 0.
- **p < 0.05 = confirmed.**
- Scored once; n is fixed.

**Secondary:** accuracy, finish rate (per problem), looping, capped traces with the gold, tokens on problems both solved, and the draw-level sign test (anti-conservative).

## Not done
- No random-direction control: it was null on both 2B and 0.8B.
- If the edit helps, a control can be added later.

## Change log
- 2026-11-15, **during the eval, before any scoring**:
  - Calls 1–2 and the edit's seed-0 eval completed.
  - The **base** eval crashed when one 16k request passed the harness's 2-hour per-request timeout. The server logs show 2–2.5 tok/s per slot in that run, against about 10 tok/s in the edit's seed-0 run, consistent with VRAM spilling to system RAM.
  - A power outage then stopped the edit's seed-1000 run.
  - **Fix:** the per-request timeout was raised to 12 h. Nothing else changed: same 8 slots, f16 KV, sampling and problems.
  - `calls/eval_edit_s0.json` is kept. Base seeds 0 and 1000 and edit seed 1000 are rerun.
  - No eval file was opened or scored. The only figures seen were the progress lines' summary accuracy for edit seed 0 and the calibration call.
- 2026-11-15, **after the eval, scored once** with `score.py`: primary 7 / 2, p = 0.18, not confirmed (see REPORT.md).
  - The GPU watcher showed the base eval process holding 2.1 GiB of VRAM during generation (llama-server spilled 1.4 GiB, about 45 vs 80 tok/s). That affects speed only: the sampled tokens don't depend on it.
  - After scoring, `ws.Session` gained `device="cpu"` for generate-only sessions, and `run_eval.py` now uses it. This applies to future runs only.
