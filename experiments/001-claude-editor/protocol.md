# Exp 001: Claude as weight-space editor

Written 2026-11-12, **before any runs**. Changes after this point go in the change log at the bottom, with a reason.

## Question
Can Claude (Opus 5.5, in Claude Code) do what GPT-6 Astra did in the MIT weight-space-programming (WSP) study: diagnose a small model's failure and fix it by editing weights directly, with no gradients? Failures count as data. We record every hypothesis, dead end and edit.

## What I already know (contamination)
I have read summaries of both WSP papers. For **Qwen3.5-0.8B** I know the answer:
- the blog's fix removed a loop-driving direction written by the MLPs of layers 15 and 19
- the arXiv fix was a rank-one change to `layers.23.mlp.down_proj` column 507

So a run on 0.8B can't be blind. The main arm uses **Qwen3.5-2B**, which neither paper studied. My prior that "looping is the lever" still carries over, and I'm stating it here. I won't read Antigravity's `exp8-walkthrough.md` or the MIT logs until Arm 1 is scored.

## Setup
- **Hardware:** RX 9060 XT (16 GB) with ROCm torch 2.14, same build as `D:\Argus_Project`.
- **Environment:** venv in `D:\WeightSpace\.venv`.
- **Rules for the editor (me), matching WSP:**
  - Edit only transformer-block parameters. Embeddings, final norm and LM head stay fixed.
  - No gradients or backprop.
  - Analyses that don't run the model (reading weights, SVDs, norms) are free.
- **Forward budget:** 50 calls. Each call is either a generation or a teacher-forced pass returning log-probs and activations, on at most 64 prompts. Token caps get fixed after a throughput check, and before the first editing call.
- **Data:**
  - Dev: problems drawn from the MATH *train* split.
  - Held-out: MATH-500, never shown to the editor and scored once at the end for base and edit.
- **Log:** every call, its purpose, the hypothesis behind it and its result go to `log.jsonl`. Reasoning goes to `notes.md`, including abandoned ideas.

## Arms
- **Arm 0, calibration (0.8B, no editing):** measure base Qwen3.5-0.8B accuracy and how often it hits the token cap, on a slice of MATH-500. Compare against the base samples the MIT team published on HF. This checks that our eval matches theirs, and costs nothing from the edit budget.
- **Arm 1, main (2B, blind):** the full diagnose, edit and confirm loop on Qwen3.5-2B. Outcomes:
  - an edit that helps on held-out data
  - an edit that helps on dev but not held-out
  - keeping the original weights

  All three get written up the same way.
- **Arm 2, later (transfer):** diff the published edited 0.8B against the base model, then test whether that exact edit carries over to fine-tunes of 0.8B. Vectors can't be copied across sizes, because hidden widths differ. A good fit for Codex.

## Results reported regardless of outcome
- held-out accuracy and token-cap rate, base vs edit
- dev gain vs held-out gain
- number of weights changed
- forward calls used
- where the diagnosis went right or wrong

## Change log
- 2026-11-12, before any runs: MIT's evals allowed up to 32,768 new tokens with vLLM on an H200, and base 0.8B averaged about 31.7k tokens per answer. That's infeasible here with HF transformers on one consumer GPU. **Generation is capped at about 4,096 new tokens** (the exact cap is set after the throughput check). "Hit cap" and an n-gram repetition score stand in for loop detection. Accuracy numbers will therefore be lower than MIT's and can't be compared directly. Arm 0 measures how far apart the two setups are.
- 2026-11-12, after throughput checks and before the first editing call:
  - **Backend:** HF transformers ran at 33 tok/s because Qwen3.5's linear-attention kernels aren't available on Windows ROCm. Generation now goes through llama-server: the winget build b12212 with Vulkan, using bf16 GGUFs converted with the converter from the same commit. 0.8B runs at 256 tok/s with 16 slots, 2B at 228 tok/s with 32 slots. Torch is still used for weight analysis and teacher-forced activations.
  - **Locked settings:** cap of 4,096 new tokens and up to 32 parallel slots.
  - **What a cap hit means at these lengths:** samples show base 0.8B *dithering* ("Wait, ..." with no progress) rather than exact loops. A cap hit therefore mixes loops with legitimately long reasoning. This limitation is noted in the results.
- 2026-11-12, after Arm 0 (written before the held-out eval):
  - **Calibration finding:** at a 4k cap, MIT's verified edit is almost invisible on accuracy. On 0.8B, base got 0/32 and the edit 1/32, with cap hits at 32 vs 31. The repetition score shows the fix clearly: 0.059 base vs 0.016 edited. The fix's gain lives in long generations.
  - **Held-out eval change:** report both **accuracy@4k** (a convergence-speed metric) and **accuracy@16k on a fixed MATH-500 subset**, for base and the final edit. The subset size is set by time: 100 problems at 16k is about 2 h per model here. Repetition score is reported as a loop indicator.
- Sampling matches MIT's `think35` settings (temp 0.6, top_p 0.95, top_k 20, presence_penalty 1.5, thinking on). HF has no presence penalty, so it's implemented as a logits processor in `ws.py`.
- 2026-11-13, after dev selection and **before any held-out run**:
  - **Final edit: e02**, which projects the doubt direction out of the top-64 down_proj columns of L15, L16 and L18 (383,099 weights).
  - **Dev evidence:**
    - Call 6: 4 wins, 0 losses.
    - Call 9, 64 fresh problems at 4k: 28/64 vs 19/64, 10 wins and 1 loss, p≈0.012.
    - Call 8, the 16k run: 8/16 vs 7/16, 1 win and 0 losses, −37% tokens on problems both solved.
  - **Why not e03:** e03 (adding L14/L21/L22, 765,935 weights) scored 29/64. That's 3 wins and 2 losses against e02, which is no meaningful difference, at twice the weights changed.
  - **Held-out eval, fixed now:**
    - accuracy@4k on all 500 MATH-500 problems, 32 slots
    - accuracy@16k on the first 48 problems of MATH-500 shuffled with seed 0, 16 slots
    - Both models, sampling seed 0.
  - **Why 48 at 16k instead of ~100:** throughput measured at 16k is about 4.4 tok/s per slot, so 16 problems take about 1 h. 100 per model would take more than 6 h.
  - The held-out eval runs unbudgeted.
