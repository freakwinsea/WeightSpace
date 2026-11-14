# Exp 002 report: does the 2B fix transfer to Qwen3.5-0.8B?

2026-11-13. Editor: Claude (Opus 5.5). Pre-registration and change log: `protocol.md`. Raw rows: `calls/call00{2..7}_gen.json` (arms A–F in order). Budget: **7 of 50 calls** (1 teacher-forced, 6 generation).

## Result
**The recipe transferred and the vector did not.**
- Re-running the 2B *procedure* on 0.8B (arm C) helped. It took one forced call to derive 0.8B's own direction, then the same rule, layers, K and α.
- Translating the 2B *direction* through the shared vocabulary (arm D) did nothing, and it did no better than the vocabulary-only control (F).

32 fresh MATH-train problems (seed 3, no overlap with the calibration set), 16k cap, paired:

| Arm | Correct | Finished | Looping (last-2k rep > 0.5) | Mean rep | Paired vs A (wins / losses) |
|---|---|---|---|---|---|
| A, base | 2 | 2 | 24 | 0.67 | — |
| B, MIT edit (L15/L19, α=6) | 5 | 5 | **3** | **0.11** | 3 / 0 (p=0.25) |
| **C, recipe transfer** | **6** | 5 | 16 | 0.51 | **4 / 0 (p=0.125)** |
| D, vector transfer | 2 | 3 | 20 | 0.58 | 1 / 1 |
| E, random control | 2 | 2 | 22 | 0.63 | 0 / 0 |
| F, lexical control | 3 | 3 | 20 | 0.60 | 2 / 1 |

Other pairings:
- C vs B: 2 wins / 1 loss.
- C vs D: 4 wins / 0 losses.
- D vs F: 0 wins / 1 loss.

**Power caveat:** with base at 2/32, no single comparison reaches p<0.05. C's 4/0 is the strongest signal and matches the direction of every Exp 001 result, but it needs a larger run to confirm.

## Predictions vs outcome
| Prediction | Outcome |
|---|---|
| C helps, but less than B | **Half right.** C helps nominally, but it ties B (6 vs 5), so "less than B" is not supported. |
| D: cos(D, C) < 0.3, so D does little | **Geometry wrong, behaviour right.** The cosine is 0.36–0.47, but D still does nothing. |
| E does nothing | **Right** (0 / 0). |
| F (added after call 1): D ≈ F means lexical transfer | D ≈ F ≈ E ≈ A. Nothing transferred, lexical or structural. |

## Why a 0.4 cosine bought nothing (free weight analysis, no forward calls)
The edit rule *selects* columns by |u·W| before projecting, and the selection amplifies direction differences:

| Layer | C's top-64 columns that D also picks | Update norm, C | Update norm, D |
|---|---|---|---|
| L15 | 9 of 64 | 0.23 | 0.36 |
| L16 | 13 of 64 | 0.23 | 0.31 |
| L18 | 8 of 64 | 0.29 | 0.43 |

D is not a weaker version of C. It is a *larger* edit on mostly *different* neurons, and it still does nothing. That fits the random control: what matters is hitting the specific writers of the model's own direction, not the size of the update.

## What 0.8B tells us about the mechanism
- **Loop removal alone is not the bottleneck at 16k.** B nearly eliminates loops (24 → 3 looping) but still hits the cap on 27/32 problems and solves 5.
- **C treats a different symptom.** It removes only a third of the loops (24 → 16) and solves 6.
- **The two fail differently.** The problems each one solved overlap only partly. Solved sets per problem: `BCF`, `CDF`, `ABCE`, `ABCDEF`, `C`, `BC`, `B`.
- **This fits 0.8B having both failure modes,** looping (B's target) and non-commitment (C's target). The open question is whether the two edits compose. They write different layers (B: L15/L19, C: L15/16/18) and different directions.

## For the user's hypothesis
"An algorithm found on a bigger model transfers cheaply":
- **Supported in the procedural sense.** The diagnosis, which layers to look at, the edit rule and its strength all carried over unchanged from 2B to 0.8B. The search cost about 10 calls on 2B, and this transfer cost 1 forced call plus no search. 0.8B's doubt circuit sits at the same depth as 2B's (MLP writes peak at L18 > L15 ≈ L16 in both).
- **Not supported in the literal sense** of moving weights or directions across widths through the shared vocabulary. A direction found from the vocabulary readout is a different object from the direction the model computes internally, even at cos ≈ 0.45.

## Caveats
- **Single seed, n=32, low floor.** Each pair is one draw.
- **Contamination:** I knew MIT's 0.8B answer (L15/L19, repetition direction). C's layers come from 2B, and they overlap MIT's at L15.
- **Bridge method:** D used a linear least-squares map through the tied embeddings, with readout R² ≈ 0.30. A better map (for example a CCA fit on paired activations) is untested, but it would need 0.8B forward calls and so would no longer be "zero-call" transfer.
- **What "rep" measures:** the fraction of the last 2,000 tokens inside a repeated 20-gram. It is not comparable to the Arm 0 number, which was measured at 4k.
