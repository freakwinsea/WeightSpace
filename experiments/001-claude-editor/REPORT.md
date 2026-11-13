# Exp 001 report: Claude as weight-space editor on Qwen3.5-2B

2026-11-13. Editor: Claude (Opus 5.5, Claude Code). Protocol and change log: `protocol.md`. Running reasoning, dead ends included: `notes.md`. Every forward call: `log.jsonl`.

## Result
I edited Qwen3.5-2B with no gradients and changed **383,099 weights**: 64 down_proj columns in each of layers 15, 16 and 18. The edit raises held-out MATH-500 accuracy at both token caps.

| Held-out MATH-500 (scored once) | Base | Edit (e02) | Paired wins / losses | Sign test |
|---|---|---|---|---|
| accuracy@4k, all 500 problems | 20.6% (103) | **27.2% (136)** | 55 / 22 | p = 0.0002 |
| accuracy@16k, fixed 48-problem subset | 29.2% (14) | **54.2% (26)** | 13 / 1 | p = 0.002 |
| hit cap @4k / @16k | 83% / 73% | 76% / 46% | | |
| repetition score @16k | 0.055 | 0.036 | | |
| accuracy of finished answers | 100% | 100% | | |

- **Dev gain vs held-out gain:** on dev (64 fresh MATH-train problems at 4k) the edit went from 29.7% to 43.8%, +14 points. On held-out it went from 20.6% to 27.2%, +6.6 points. The held-out gain is smaller but clearly real. At 16k the held-out gain is larger than at 4k.
- **Where the gain comes from:**
  - Levels 1–3: 76 → 105 correct. Levels 4–5: 27 → 31.
  - The edit makes 2B *commit* sooner on problems it can already solve.
  - At 16k, problems both versions solved took 37% fewer tokens (6,586 → 4,142).
  - It does not make 2B smarter on hard problems.
- **No accuracy cost found:** every finished answer was correct in both models. The edit makes the model finish more often without making it wrong more often.
- **Budget:** 10 of 50 forward calls (each ≤64 prompts). The held-out eval and Arm 0 calibration ran unbudgeted, as pre-registered.

## How the diagnosis went
1. **Calls 1–2. The symptom.**
   - 2B at 4k hit the cap 91% of the time, and 26 of the 29 capped answers never wrote a boxed answer.
   - "Wait" is 4x denser in capped text.
   - I found down_proj columns aligned with the "Wait" token (logit lens), and replayed the outputs to sort them into "writers" (L16.2431, L18.3801) and "brakes".
2. **Call 3. Wrong turn: the lexical neuron.**
   - Zeroing L16.2431 cut "Wait" by 64% but didn't make answers finish. The model swapped in other hedges.
   - **Lesson:** a neuron that writes the *word* is a symptom, not the cause.
3. **Call 4. Characterising the failure.**
   - At 16k, base finishes 7/16 and every finished answer is correct.
   - So 2B is half "slow but sound" and half "never converges". This is not the 0.8B looping story.
4. **Call 5. The real lever.**
   - I took the residual-stream difference between positions just before doubt tokens (Wait/But/Hmm/Let me check) and just before commit tokens.
   - The gap opens from L11 onward. MLPs in L14–L22 write onto it, most strongly L15/L16/L18.
   - The top-64 columns per layer carry 20–33% of that write.
5. **Calls 6–10. Edit e02 and its confirmation.**
   - e02 projects that direction out of the top-64 columns in L15/16/18.
   - Results:
     - 4 wins / 0 losses on the first dev set
     - 10 wins / 1 loss on a fresh 64-problem dev set
     - 1 win / 0 losses at 16k
   - **e03** (adding L14/21/22, 2x weights) tied e02 (3 wins / 2 losses), so I kept e02 under a rule set before the run.

**Right:**
- moving from the lexical neuron to a residual-stream direction
- choosing a rank-1 projection on the columns that write it, rather than zeroing neurons

**Wrong:**
- my first prior, carried over from the 0.8B story, that 2B fails by *looping*. It mostly fails by *not committing*.
- the "Wait" neuron as the lever

## Arm 0 calibration (0.8B)
- I reproduced MIT's published 0.8B edit, hash-verified as changing only L15/L19 down_proj.
- At a 4k cap it is almost invisible on accuracy: 0/32 vs 1/32. It is clearly visible on repetition: 0.059 vs 0.016.
- This is why the held-out eval includes a 16k arm.

## Caveats
- **Token caps** are 4k and 16k. MIT used 32k, so absolute numbers can't be compared with theirs.
- **Single sampling seed** per problem (temp 0.6). Paired tests control for problem difficulty but not sampling variance, so each pair counts as one draw.
- **16k subset** is 48 problems, not ~100, because of measured throughput. This was fixed before scoring.
- **Contamination:** I knew the 0.8B answers ("an MLP direction in mid layers"). That prior plausibly steered me toward MLP down_proj directions on 2B. The 2B direction itself (doubt vs commit, L15/16/18) was found from 2B's own activations.
- **"Doubt direction" is my interpretation.** Operationally it's a difference of means between pre-doubt and pre-commit positions on 2B's own failures.
- **Grader:** math-verify on the last boxed answer. A capped text can count as correct if it boxed the right answer before the cap. This applies equally to both models.

## For the user's hypothesis (algorithms found by big models transfer cheaply to small ones)
This run is the *editing* half: a frontier model found and installed a behavioural fix in 2B's weights for about 10 forward-call batches plus free weight analysis. The fix is low-rank, a single direction across three layers, and that's the kind of object that could plausibly be ported. Arm 2, testing whether such an edit carries over to fine-tunes or sibling models, is the next experiment.

## Files
- `edits/e02_project_doubt_L15_16_18.py`: the edit. It needs `calls/call005_forced_doubt.pt` for the direction.
- `heldout.py`, `calls/heldout_*`: the held-out runs.
- `gguf/qwen3.5-2b-edit-*-bf16.gguf`: the edited model, runnable in llama.cpp.

## Addendum: after reading Astra's Exp 8 trace (read only once Arm 1 was scored)
Source: Antigravity's `D:\WeightSpace\research\exp8-walkthrough.md`, traced from MIT's `p50c` logs.

- **Same method, independently arrived at in shape.** Astra on 0.8B:
  - contrasted productive and repetitive traces to get a direction
  - found mid-layer MLP writers (L15, L19)
  - removed the direction from the top-64 down_proj columns: `W = W0 - α·outer(u, mask64·(u@W0))`

  That is exactly e02's update rule. **Contamination I under-stated in the protocol:** the summary I'd read said "64 down_proj columns in each of L15 and L19". My K=64, and plausibly the column-projection form, came from that, not from 2B. What came from 2B is the direction (doubt vs commit, not repetition), the layers (15/16/18) and the failure diagnosis (non-commitment, not looping).
- **The difference that matters: strength.**
  - Astra swept α: α=1 gave +12.5 pts, α=2 gave +24.6, and the final edit used α=6, which overshoots past removal into reversal.
  - e02 uses α=1, pure removal. I never swept α.
  - That's the obvious untested lever, and 40 budget calls remain.
- **Astra also ran controls I didn't:**
  - a norm-matched random orthogonal direction: no gain, which rules out "any perturbation helps"
  - a broad preservation check on non-math tasks (IFEval, MMLU-Pro, GPQA)

  Both are needed before calling e02 a clean fix rather than a lucky perturbation.
- **Scale comparison.** On 0.8B, where base 0.8B loops, Astra reached +40 pts on MATH-500. On 2B, which mostly commits slowly rather than looping, e02 gets +6.6 pts at 4k and +25 pts at 16k on 48 problems. The 0.8B → 9B result in the WSP-Scope paper (−0.35 pts Math) suggests gains shrink as models stop looping. 2B sits in between.

## Follow-ups (calls 11–14, after the addendum)
Run on the seed-2 dev set of 64 problems at 4k, paired with base (call 7) and e02 (call 9):
- **Random-direction control (e04):** same layers, same 64 columns, same update norm, but writing a random direction orthogonal to the doubt direction.
  - 19/64, identical to base: 7 wins, 7 losses vs base, and 0 wins / 9 losses vs e02 (p≈0.004).
  - **e02's gain is specific to the direction.** It isn't "any perturbation".
- **Strength sweep:**

  | α | correct |
  |---|---|
  | 1 (e02) | 28 |
  | 2 | 30 |
  | 4 | 30 |
  | 6 | 25 |

  - The gain plateaus from α=1 to α=4 and declines at α=6, where "Wait" density rises back above base.
  - On 0.8B, Astra's gains kept growing up to α=6. On 2B the lever saturates at roughly removing the direction.
  - e02 stays the final edit, and the MATH-500 numbers above stand.
- **Not yet checked:** preservation on non-math tasks (Astra checked IFEval, MMLU-Pro and GPQA).
