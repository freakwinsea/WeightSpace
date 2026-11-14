# Exp 002 (Arm 2): does the 2B fix transfer to Qwen3.5-0.8B?

Written 2026-11-13, **before any Arm 2 runs**. Changes go in the change log at the bottom. Harness: Exp 001's `ws.py`, `gen.py`, `run_gen.py` and `call_doubt.py`, run with `WS_HERE` set to this folder, so calls, log and budget belong to this experiment. Budget: 50 calls, same rules as Exp 001.

## Question
The user's hypothesis is that an algorithm discovered on a bigger model transfers cheaply to a smaller one. Exp 001 found a fix on Qwen3.5-2B: remove the "doubt vs commit" direction from the top-64 down_proj columns of L15, L16 and L18 (α=1). Qwen3.5-0.8B has the same 24 layers and the same layer-type pattern, but a hidden width of 1024 instead of 2048. Embeddings are tied and the vocabulary is shared. Two kinds of transfer are tested:
- **Recipe transfer (C):** re-run the 2B *procedure* on 0.8B. Derive 0.8B's own doubt direction, which costs one teacher-forced call, then apply the same edit rule to the same layers with the same K and α. No further search.
- **Vector transfer (D):** take the 2B *direction itself* and translate it into 0.8B's residual space through the shared vocabulary. Read out each 2B layer's u through 2B's unembedding with the zero-centred (1+w) final-norm gain, then least-squares fit the 0.8B vector whose readout matches, with both centred over the vocabulary. This needs **zero** 0.8B forward calls. Same layers, K and α.

## Arms
All arms run on 0.8B, on 32 fresh MATH-train problems (seed 3), with a **16k** cap and 32 slots, sampling seed 0, paired:
- **A, base:** unedited 0.8B
- **B, MIT reference:** MIT/Astra's published edit (L15/L19, repetition direction, α=6)
- **C, recipe transfer**
- **D, vector transfer**
- **E, control:** a random direction orthogonal to C's u, with the same columns and update norm as C

The direction for C is derived from the 32 Arm-0 calibration texts (base 0.8B, MATH-train seed 1, 4k). Those are dev data, disjoint from the seed-3 eval set except by chance overlap, which is checked.

## Predictions (written before any run)
- **C** helps 0.8B, but less than B. 0.8B's main failure is *looping* (Arm 0: repetition 0.059), not slow commitment, and B targets loops at α=6.
- **D:** cos(D's u, C's u) is low (<0.3) in mid layers, because a mid-layer direction is poorly represented by the final unembedding. D therefore does little.
- **E** does nothing, as on 2B.

## Reported regardless of outcome
- accuracy, cut rate and repetition per arm
- paired wins and losses vs A
- cos(D, C) per layer
- calls used

## Contamination
- I know MIT's 0.8B answer: L15/L19, a repetition direction, top-64 columns, α=6.
- C's layers (15/16/18) partly overlap MIT's (15/19), but they come from the 2B analysis, not from 0.8B.

## Change log
- 2026-11-13, after the direction analysis and **before any generation**:
  - **Call 1 (forced, 0.8B, 32 calibration texts):** 0.8B's MLP writes onto its doubt direction peak at **L18 (0.41), L15 (0.22) and L16 (0.20)**. These are the same top layers as 2B (L18 1.24, L16 0.82, L15 0.72), so the circuit sits at the same depth at both sizes.
  - **The D prediction is already falsified on geometry:**
    - cos(bridged 2B u, 0.8B's own u) = +0.36 / +0.46 / +0.47 at L15 / L16 / L18, rising to +0.56 at L23.
    - A random 2B vector bridged the same way gives |cos| ≤ 0.13.
    - The bridge map is a least-squares fit through the centred, (1+w)-gained tied embeddings, R² ≈ 0.30.
  - **But part of that is lexical:** 0.8B's own u has cos +0.39 / +0.47 / +0.58 with the plain "doubt-token minus commit-token" unembedding direction.
  - **Added arm F, lexical control:** that lexical direction, same rule, same layers.
    - If D ≈ F, the transfer is mostly "steer away from doubt words".
    - If D > F, structure beyond vocabulary transferred.
- Six arms (A–F), 32 problems each, at 16k.
- 2026-11-13, **after all six arms ran**: results in `REPORT.md`. No analysis choice was changed after seeing the data. Added afterwards and labelled as post-hoc: the column-overlap/update-norm table (free weight analysis) and the "looping = rep > 0.5" count. Calls used: 7 of 50.

## Confirmation of C (pre-registered 2026-11-13, after the six arms, before running)
The dev result for C (4 wins / 0 losses, p=0.125) is underpowered, so it is re-tested once on held-out data the editor has not seen:
- **Data:** MATH-500 shuffled with seed 0, first **128** problems. This is the same subset rule as Exp 001's held-out eval. 0.8B has never been run on it.
- **Arms:** A (base 0.8B) and C (`edits/c_recipe.py`, unchanged). No re-tuning.
- **Settings:** 16k cap, 32 slots, the same sampling as Arm 2. Unbudgeted under Exp 001's held-out rule, via `heldout.py` with `WS_MODEL=Qwen/Qwen3.5-0.8B`.
- **Primary endpoint:** two-sided paired sign test, C vs A. **p < 0.05 = confirmed.** Otherwise the gain is not confirmed, and it is reported that way.
- **Secondary:** accuracy, cut rate, looping count (last-2k rep > 0.5), and correct by MATH level.
- **Scored once.** n is not changed after seeing the results.
