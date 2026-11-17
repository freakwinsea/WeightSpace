# Exp 003 report: does the doubt recipe transfer up, to Qwen3.5-4B?

2026-11-15. Editor: Claude (Opus 5.5). Pre-registration and change log: `protocol.md`. Scoring: `score.py`, run once. Raw rows: `calls/eval_{base,edit}_s{0,1000}.json`. Budget: **6 of 50 calls** (1 calibration generate, 1 forced, 4 eval generates).

## Result
**Not confirmed. The edit points the right way, but 4B is near ceiling on this set and only 9 problems changed outcome.**

64 MATH-train problems (seed 3) × 2 sampling seeds, 16k cap, paired:

| | Base 4B | Recipe edit (L22/28/29) |
|---|---|---|
| correct, of 128 draws | 101 (79%) | **106 (83%)** |
| finished | 101 | 106 |
| finished answers that are correct | **101 / 101** | **106 / 106** |
| looping (last-2k rep > 0.5) | **0** | 0 |
| capped wrong traces containing the gold string | 23 / 27 | 18 / 22 |
| mean tokens, all draws | 7,598 | 7,348 |
| tokens on the 97 draws both solved | 4,964 | 4,882 |
| by seed (0 / 1000) | 50 / 51 | 54 / 52 |

- **Primary, per problem (edit better / base better): 7 / 2, p = 0.18. Not confirmed.**
- Secondary:
  - per-problem finish rate: 7 / 2, p = 0.18. It is the same split, because every finished answer is correct.
  - draw-level: 9 / 4, p = 0.27, anti-conservative.

## Predictions vs outcome
| Prediction | Outcome |
|---|---|
| Top-3 writer layers at relative depth 0.62–0.75 (L20–24) | **Partly wrong.** The layers were [22, 28, 29], at depths 0.69, 0.88 and 0.91. One is in range; two sit near the top of the network. |
| Finished answers ≥ 90% correct | **Right, and stronger than predicted:** 207 of 207 finished answers across both arms were correct. |
| A substantial share of capped wrong traces contain the gold | **Right by string match** (41 / 49), but string match overstates it here; see below. |
| The edit helps | **Same direction, not significant** (7 / 2). |
| The gain is larger than on 0.8B | **Wrong.** It was +5 draws of 128 (+3.9 points), against +5.1 points on 0.8B's powered test. The base rate is very different, though: 79% here against 10%. |

## What 4B tells us about the failure mode
This is the cleanest picture of the user's indecision hypothesis so far:
- **Every 4B failure is a failure to finish.** No 4B draw finished with a wrong answer, in either arm. A draw is wrong exactly when it hits the 16k cap.
- **4B doesn't loop.** Looping was 0 of 256 draws, and mean rep in the 49 capped traces was 0.008 (max 0.10). On 0.8B, by contrast, 24 of 32 base draws looped. At 4B the indecision no longer repeats text: it re-derives. The capped traces keep re-opening sub-steps they have already done ("Let's reconsider the … model", "Now let's check … again"), each time in new words.
- **So looping is the small-model outlet of the indecision, and fresh re-derivation is the larger-model outlet.** That fits "one indecision, several outlets" (Exp 002). It also means a repetition-based loop meter will report 4B and up as healthy while the models still cap.
- **The capped-with-gold rate is inflated for short golds.** In a spot check of 6 base capped traces, golds like "3" and "10" appear by coincidence early in the text. Longer golds (182, 95°) look like genuine "found it, kept checking" cases. Treat 41 / 49 as an upper bound.

## For the user's hypothesis
"An algorithm found on a bigger model transfers cheaply":
- **The procedure ran unchanged on a third size** (0.8B → 2B → 4B). It cost 1 forced call, and the layers came from a fixed rule. Nothing was tuned.
- **It did not reach significance on 4B** with this design. The bottleneck is the same as on 0.8B, from the opposite side: on 0.8B almost nothing finishes, and on 4B almost everything does. Either way few problems change outcome. Here, 53 of 64 problems are solved on both seeds by both arms.
- **The direction is consistent:** five pre-registered tests of the doubt edit across two sizes, all favouring the edit, none passing alone. Post-hoc and descriptive only: the discordant problems across them are 36 for the edit against 8.

## Harness finding (no effect on the scores)
The two base runs took 9,456 s and 10,909 s; the two edit runs took about 6,600 s each. The GPU watcher (`gpuwatch.log`) showed why:
- During the base runs, the eval's own Python process held **2.1 GiB of VRAM** after moving its model to the CPU. That pushed 1.4–1.5 GiB of llama-server into shared memory, and generation ran at about 45 tok/s.
- During the edit seed-1000 run, the Python process held 0.2 GiB, and generation ran at about 80 tok/s.
- **Fix, applied after scoring:** `ws.Session(..., device="cpu")` keeps torch off the GPU for generate-only sessions. Edits run on the CPU, export is unchanged, and the sampled tokens don't depend on it.
  - The CPU path is smoke-tested: it gives the same layers [22, 28, 29] and the same 479,927 edited weights.
  - A full generation on that path has not been run yet.

## What would decide it
- **Harder problems for 4B.** Choose problems where base 4B caps about half the time, for example MATH level 5 or AIME-style items, so that more problems can flip.
- **Power planned in advance.** At the observed 7 : 2 split, about 40 discordant problems give roughly 80% power. At 4B's discordance rate on level-5 items, that would need to be estimated from one calibration call first.
- **An endpoint matched to the 4B failure mode:** tokens-to-first-commit, or the share of capped traces that reach a correct intermediate answer and then reopen it. These measure the indecision directly instead of waiting for the cap.

## Level 5 calibration and definitive test (pre-registered in protocol.md; scored once with `score_final.py`)
**Calibration** (32 Level 5 problems, seed 0): base finished 15/32 and the edit 16/32, every finished answer correct, no loops. The fixed rule sized the test at n = 139.

**Definitive test:** 139 fresh MATH-train Level 5 problems × seeds 0 and 1000, 16k cap, paired. 12 calls, about 28 GPU-hours.

| | Base 4B | Recipe edit |
|---|---|---|
| correct, of 278 draws | 140 (50.4%) | 150 (54.0%) |
| finished (correct) | 146 (140) | 156 (150) |
| looping (last-2k rep > 0.5) | 0 | 0 |
| capped traces that ever wrote a `\boxed{}` answer | 2 / 132 | 2 / 122 |
| mean tokens | 12,129 | 11,856 |
| tokens on the 125 draws both solved | 7,510 | 7,188 |
| by seed (0 / 1000) | 72 / 68 | 73 / 77 |

- **Primary, per problem (edit better / base better): 16 / 12, p = 0.57. Not confirmed.**
  - At 4B the doubt edit has no effect large enough to detect with 139 problems.
  - The calibration's 3 : 2 split was the better guide than the eval's 7 : 2.
  - There were 28 discordant problems against the 40 the plan targeted: discordance at Level 5 was lower than calibrated.
- **Secondary:**
  - per-problem finish rate: 16 / 13, p = 0.71.
  - draw-level: 25 / 15, p = 0.15, anti-conservative.
- **Exploratory, not an endpoint:** tokens to the first `\boxed{` on the 131 draws where both arms boxed one. The median was 6,037 for base and 5,794 for the edit; the edit boxed earlier in 77 of them and later in 54 (sign p = 0.054). This counts as a lead for a future pre-registration, not as a finding.

### Verdict on Exp 003
**The recipe ran unchanged on 4B and did nothing detectable.** On 0.8B it gave about 1.5× the correct answers at a low base rate, and that held consistently across four tests. On 4B the best estimate is about +4% relative: +5 of 128 draws on the easy set and +10 of 278 on Level 5, both non-significant. The definitive test sits close to null.

**The failure mode transferred perfectly, and the lever did not:**
- At both difficulty levels, 4B's errors are almost all failures to finish.
  - 207 of 207 finished answers were correct on the eval set.
  - 290 of 302 were correct on Level 5.
- 4B never loops: 0 of 812 draws across this experiment.
- **Capped 4B traces essentially never commit even provisionally:** 4 of 254 wrote a boxed answer before running out. They don't guess, check and fix. They keep deliberating without ever writing down an answer. That is the user's "never let yourself be wrong" observation in a measurable form, and at 4B it costs about half the Level 5 problems.

**Why the lever might not transfer upward** (hypotheses, untested):
1. **Layer choice.** The fixed rule picked L28/29, near the output, where removing a direction may mainly change wording. The predicted mid-depth band (L20–24) held only L22.
2. **Reach.** A rank-1 change on three layers may be too small a share of a 4B model's indecision circuitry. On 0.8B, three of 24 layers is a bigger share of the network than three of 32.
3. **What the direction encodes.** At 4B, "doubt token vs commit token" may separate wording more than state. The 4B traces doubt without saying "Wait": they reopen steps with "Let's reconsider…".

**Budget used: 20 of 50 calls.**
