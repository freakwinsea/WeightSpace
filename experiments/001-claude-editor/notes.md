# Editor notes, Exp 001 (Claude as editor, Qwen3.5-2B)

Running record of my reasoning, including wrong turns. Call numbers match `log.jsonl`.

## Priors going in
- From the WSP papers: in 0.8B, a mid-to-late MLP direction drives long, looping thinking. I expect something similar may exist in 2B, but I don't know where.
- The base 2B already hits a 2,048-token cap on 97% of practice problems (from the smoke test, which was unbudgeted). That's consistent with looping, but also with long, honest reasoning. Call 1 separates the two at 4,096.

## Call log
- **Call 1 (gen, base, dev seed 1, 32 problems at 4k):** 15.6% correct, 90.6% hit the cap. 26 of the 29 capped answers never boxed an answer. Capped answers say "Wait" 3.0 times per 1k tokens, finished ones 0.7. One answer looped its final \boxed{} 32 times. The failure is mostly *non-convergence with heavy self-doubt*, not exact repetition.
- **Free weight analysis:** scored every down_proj column against the "Wait" unembedding direction. Random alignment is about 0.02. Outliers: L16.2431 (-0.51), L22.5041 (+0.48), L17.873 (+0.43), L18.3368 (+0.43), L18.3801 (-0.40), L20.3748 (+0.34), L15.1958 (-0.26).
- **Call 2 (forced, 16 texts from call 1):** all candidates fire strongly right before "Wait" tokens, 5–20x their usual magnitude. Activation sign times column sign decides the effect, and the logit lens confirms it:
  - **Pro-wait:** L16.2431 and L18.3801. Both promote every form of wait/Wait/waiting/等待.
  - **Brakes (anti-wait):** L15.1958, L17.873, L18.3368, L20.3748, L22.5041. These suppress the same tokens.
  - L16.2431 mean activation is -0.49 in capped answers vs -0.05 in finished ones. The pro-wait push is about 10x higher in failing answers.
  - Reading: a lexical "wait" circuit with an early writer and late brakes. Open question: is the dithering *caused* by this circuit, or does it only *voice* an upstream doubt signal?
- **Edit e01:** zero L16.2431's column (2,048 weights). Prediction if the circuit is causal: fewer Waits, fewer cap hits, more finished answers. Prediction if it's only lexical: "But"/"Hmm"/"Actually" rise to replace it and cap hits barely move.
- **Call 3 (gen, e01, same 32 problems as call 1):** "Wait" per 1k fell from 2.79 to 1.00 (-64%). Substitute words (But/Hmm/Actually/...) went from 1.51 to 1.64. Cap hits went from 29 to 27, correct from 5 to 7 (noise at n=32).
  - **Reading:** L16.2431 *voices* doubt but doesn't *cause* non-convergence. The dithering survives without the word.
  - **Confound found afterwards:** the export also wrote bf16-rounded copies of 18 fp32 `linear_attn.norm` gains (max diff 0.004 on gains near 1). That's far too small to produce or cancel a -64% lexical effect, so the result stands, flagged. The export now swaps only tensors the edit touched, and the e01 GGUF was deleted.
- **Open calibration question:** is 4k simply too short for 2B, slow rather than looping? Call 4 runs the base model at a 16k cap on 16 of the call-1 problems.
- **Call 4 (gen, base, 16 of the call-1 problems, 16k cap):** 7/16 correct (44%) vs 2/16 at 4k. 9/16 still hit 16k. **Every answer that finished was correct (7/7).** Finished lengths: 1.6k, 3.4k, 4.1k, 4.6k, 5.1k, 9.2k, 10.5k.
  - **Reading:** 2B is half "slow but sound" and half "never converges", unlike base 0.8B (93% capped at 32k).
  - **Consequence for the metric:** at a 4k cap we measure accuracy under a fixed thinking budget, i.e. convergence speed. A good edit makes 2B reach right answers in fewer tokens without breaking its reliability. Held-out reporting will add a 16k-cap subset for the final edit if the budget allows.
- **Call 5 (forced, all 32 call-1 texts; about 500 doubt and 450 commit positions):** the residual gap between "before a doubt word" and "before a commit word" is near zero through L10, then grows: 0.54 at L10, 1.77 at L14, 3.38 at L16, 4.98 at L18, 7.11 at L22. MLPs write most of it (L15 +0.72, L16 +0.82, L18 +1.24, L21–22 about 0.6), with smaller contributions from the attention/mixer layers. Diffuse rather than one neuron, and structurally like MIT's mid-late MLP loop direction.
  - **Caveat:** late layers partly just encode the *next token*. The mid layers (L12–L16) are the better candidates for a "decision" signal.
- **Free analysis:** the top 64 of 6,144 down_proj columns hold 20–33% of each layer's write capacity along u, vs about 8% for random Gaussian columns.
- **Edit e02:** project u_L out of the top-64 |u·W| columns in L15, L16 and L18 (alpha 1). That's 3 x 64 x 2048 = 393k weights (0.02%).
- **Call 6 (gen, e02, same 32 problems):** 9/32 correct (base 5/32), 25/32 capped (base 29), Wait/1k 2.22 (base 2.79), other doubt words/1k 1.20 (base 1.51). Clean export: 3 tensors, 383k changed weights (a few columns unchanged after bf16 rounding).
  - **Reading:** a broad reduction across *all* doubt words, unlike e01's word-level effect, which fits a deeper signal. But +4 on 32 is within noise, so it needs a larger fresh paired test.
- **Arm 0 side note:** MIT's edited 0.8B was verified against the sha256 hashes in their model card. Only L15 and L19 down_proj differ from base (130,188 weights). I didn't extract their direction, to keep the 2B run as blind as possible.
- **Arm 0 calibration (unbudgeted, 0.8B base vs MIT edit, call-1 problems, 4k):**
  - Accuracy 0/32 vs 1/32; cap hits 32 vs 31.
  - Repetition 0.059 vs 0.016; Wait/1k 7.15 vs 5.51.
  - **Our 4k accuracy can't see MIT's fix.** The repetition score can. So e02's 4k gain is a convergence-speed gain, a different kind of effect from MIT's. Long caps are needed for a like-for-like comparison, and the protocol is updated.
- **Queued:** call 7 (base, dev seed 2, 64 problems at 4k), call 8 (e02, same 64), call 9 (e02 at 16k on the 16 call-4 problems, paired with call 4).

## Calls 7–9 results (23:30)
- **Numbering:** the first e02 run on the 64 fresh problems died on HTTP 500 after about 20 minutes. It was never logged, so it cost nothing from the budget, but the server log was overwritten and the cause is lost. The 16k run therefore went into the log as **call 8** (`calls/call008_gen.json`). The e02 rerun on the 64 problems will be the next call. gen.py now keeps a server log per run (`gguf/server-*.log`) and prompt_room is raised from 512 to 1024.
- **Call 7, base, 64 fresh problems at 4k (seed 2):** 19/64 correct (29.7%), 83% cut, rep 0.013. This set is easier than call 1's.
- **Call 8 (log number), e02 at 16k on the 16 call-4 problems, paired against call 4:**
  - 8/16 correct vs 7/16. 1 win (precalculus/6764 finished in 5.8k tokens; it was cut at 16k in base), 0 losses.
  - Every finished answer is still correct (8/8), so the edit didn't cost 2B its reliability.
  - On the 7 problems both solved, mean tokens fell from 5,498 to 3,486 (−37%).
  - Wait/1k: 2.56 vs 3.59 overall, and 0.8 vs 1.7 in finished answers.
  - The 7 problems that never converge still hit 16k in both.
- **Reading:** e02 makes the converging half converge faster, and it rescued 1 of 8 from the non-converging half. That fits "the doubt direction slows commitment". The non-converging half seems to need something else: a loop or a failure to make progress, not just doubt.
- **Call 9, the e02 rerun on 64 fresh problems at 4k, paired against call 7 (same seed and problems):** 28/64 (43.8%) vs 19/64 (29.7%). **10 wins, 1 loss, sign test p≈0.012.**
  - Cut rate fell from 83% to 59%, and the number of finished answers went from 11 to 26.
  - All 26 finished answers are correct, same as base (11/11).
  - Mean tokens 3,399 vs 3,692. Wait/1k 1.71 vs 2.43. The rerun had no server error.
  - **This replicates call 6 (4 wins, 0 losses) on new data**, so e02 is now the incumbent edit.
- **Next (budget 9/50):**
  - e03 = e02 plus L14/L21/L22. Those are the other layers that call 5 found writing onto the doubt direction (0.46/0.59/0.64).
  - Same 64 problems at 4k, paired against call 9.
  - If e03 doesn't beat e02 by a clear margin, keep e02. Fewer layers means fewer weights changed and less risk of overfitting the dev set.
  - Then the held-out MATH-500 eval.
- **Call 10, e03 (6 layers, 765,935 weights):** 29/64.
  - Against base: 11 wins, 1 loss (p≈0.006). Against e02: 3 wins, 2 losses, a wash.
  - Kept e02 under the rule set before the run: no clear margin, and twice the weights.
- **Dev phase closed at 10/50 calls.** The held-out eval is pre-registered in protocol.md and started 2026-11-13, about 00:20. It should take about 11 h.
