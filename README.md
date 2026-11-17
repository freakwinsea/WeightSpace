# WeightSpace: a frontier model editing small-model weights directly

**Under a fixed 50-call budget and without gradients, Claude Opus 5.5 diagnosed a reasoning failure in Qwen3.5-2B and wrote a 383k-weight edit that improved held-out MATH-500 accuracy (pre-registered, p = 0.0002). The same procedure, with no further search, gave a consistent but unconfirmed gain on 0.8B and no detectable gain on 4B.**

Everything ran on one consumer GPU (AMD RX 9060 XT, 16 GB) with llama.cpp and a small Python harness. No cluster and no training pipeline. This repo has the code, every forward call, the pre-registrations and change logs, and the dead ends, so you can check the results or run your own.

The released checkpoint (the 2B edit) is on Hugging Face: [Freakwinsea/Qwen3.5-2B-doubt-edit](https://huggingface.co/Freakwinsea/Qwen3.5-2B-doubt-edit). Its model card is [`release/qwen3.5-2b-doubt-edit/README.md`](release/qwen3.5-2b-doubt-edit/README.md).

## Results
| Exp | Question | Result | Write-up |
|---|---|---|---|
| 001 | Can Claude edit Qwen3.5-2B's weights to fix its reasoning failure? | **Yes.** On held-out MATH-500 accuracy went from 20.6% to 27.2% @4k (55 / 22, p = 0.0002) and from 29% to 54% on a 48-problem subset @16k (13 / 1, p = 0.002). Non-math ability was preserved. A random-direction control did nothing. | [REPORT](experiments/001-claude-editor/REPORT.md) |
| 002 | Does the 2B fix transfer to Qwen3.5-0.8B? | **The recipe transfers; the vector doesn't.** Re-deriving 0.8B's own direction helped in all four pre-registered tests (4/0, 6/1, 5/0, 14/5), but none reached p < 0.05 alone. Translating 2B's direction did nothing. | [REPORT](experiments/002-transfer/REPORT.md) |
| 003 | Does it transfer *up*, to Qwen3.5-4B? | **No detectable gain.** 139 Level-5 problems × 2 seeds: 16 / 12, p = 0.57. The failure mode does transfer: 4B never loops, its finished answers are 96–100% correct, and capped traces almost never write even a provisional answer. | [REPORT](experiments/003-scale/REPORT.md) |

## The failure being fixed
Qwen3.5 models re-check steps they have already done and don't commit to an answer. In the user's words: *"If you can never let yourself be wrong, you can never correct and be right."*
- **0.8B** shows it as looping (repeated text).
- **2B and 4B** show it as fresh re-derivation, with "Wait…" and "Let's reconsider…", until the token budget runs out.
- Across all three sizes, answers the model *does* finish are almost always correct.

## The edit
For a layer L, let u be the unit difference between the mean residual stream just before doubt tokens ("Wait", "But", "Hmm", "Let me check") and just before commit tokens. Take the 64 `down_proj` columns that write most along u, and remove their u component:

```python
p = u @ W                        # each column's write along u
S = topk(|p|, 64)
W[:, S] -= alpha * outer(u, p[S])  # alpha = 1
```

On 2B this was applied to layers 15, 16 and 18. The search that found this edit (a lexical "Wait" neuron that didn't work, then the residual direction that did) is written up in Exp 001's `REPORT.md` and `notes.md`.

## Method (how to make results like these trustworthy)
- **Budget:** the editor gets 50 forward calls of up to 64 prompts each, logged in each experiment's `log.jsonl`. Held-out evaluations sit outside the budget and the editor never sees them before scoring.
- **Pre-registration:** each test's problems, n, endpoint and decision rule are written in `protocol.md` before the run, and the result is scored once. Changes go in a dated change log *before* scoring. A miss is reported as a miss: Exp 002 has four tests in a row that are "consistent, not confirmed", with no optional stopping.
- **Where the record is strong, and where it's weaker:**
  - From Exp 002's held-out confirmation onward, each pre-registration was committed and pushed *before* its run, so the git history shows the order.
  - Exp 001's protocol, and Exp 002's first arms, were written before their runs but first committed together with the results. For those, the ordering rests on `protocol.md` and the logs, not on git.
- **Statistics:** paired sign tests (same problems, same sampling seeds). With several seeds, the per-problem net is the unit.

## Reproduce
**Hardware:** a GPU that can hold the model in bf16 plus its KV cache. We used 16 GB of VRAM and 32 GB of RAM.
- Exp 001's held-out run (500 problems @4k plus 48 @16k, both arms) took about 10 hours.
- Exp 003's definitive test took about 28 hours.

**Software:**
- Python 3.11 with `requirements.txt`.
- llama.cpp build 12212 (commit `d78f4255d`). Set `LLAMA_SERVER` to `llama-server`, or have it on PATH.
- llama.cpp's `convert_hf_to_gguf.py` plus `gguf-py`. Set `LLAMA_CONVERT`, or place them in `tools/llama-convert/`.

**Steps:**
1. **Fetch the base model.** Run `bash models/fetch.sh Qwen/Qwen3.5-2B`, which downloads into `models/Qwen3.5-2B/`.
2. **Rebuild the released checkpoint and check it.** Run `python release/build_2b.py`. It applies `edits/e02_project_doubt_L15_16_18.py` and compares the result bit-for-bit with the evaluated weights. That comparison needs the GGUF the harness writes in step 3, so run step 3 first on a fresh clone.
3. **Run the held-out evaluation.** From `experiments/001-claude-editor/`, the two 4k runs are:
   ```
   python heldout.py base 4096 all 16
   python heldout.py e02 4096 all 16 edits/e02_project_doubt_L15_16_18.py
   ```
   For the 16k runs, use `16384 48 16` instead of `4096 all 16`. Results land in `calls/heldout_*.json`.
4. **Find the direction yourself.** On a new model, start from `call_doubt.py` (Exp 001) and Exp 003's `edits/recipe.py`. The recipe chooses layers by a fixed rule, so it doesn't need an editor in the loop.

Generation runs through llama-server (Vulkan, bf16 GGUF). The torch model is only used to measure activations and to compute the edit. `ws.Session(..., device="cpu")` keeps torch off the GPU during generation.

## Layout
- `experiments/NNN-*/`: per experiment, `protocol.md` (pre-registration and change log), `REPORT.md`, `log.jsonl` (every budgeted call), `calls/` (raw outputs, including full generated texts), `edits/`, and the runners.
- `experiments/001-claude-editor/ws.py`, `gen.py`: the harness (budget, logging, grading, llama-server).
- `release/`: the build script and model card for the Hugging Face checkpoint.
- `research/`, `briefs/`: a source map of prior work and a walkthrough of MIT's released logs. Another agent (Antigravity) compiled these under a brief from Claude.
- Not in git, and rebuilt or fetched locally: `models/`, `gguf/`, `data/` (MATH and MATH-500 caches, built by `ws.py` from Hugging Face datasets), `tools/`.

## Prior work and credit
- **MIT CSAIL, Weight-Space Programming** (Takehi, Teoh, Zhang, Khattab, 2026): GPT-6 Astra edited Qwen3.5-0.8B to suppress thinking loops. Their released edit is used here as arm B in Exp 002. Logs and diffs: [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs).
- **WSP-Scope** ([arXiv:2611.09382](https://arxiv.org/abs/2611.09382)) maps where weight-space editing works and fails, including small gains at 9B.
- The edit family is closely related to directional ablation and to activation steering compiled into weights. What's new here is the editor (Claude, autonomous under a budget), the failure (not committing, on 2B), pre-registered held-out confirmation, and an honest map of where the transfer stops.

Editor and author of the code and reports: Claude Opus 5.5 (Claude Code), working with the repo owner, who set the questions and the hypothesis.

## Licence
Code and reports: Apache-2.0 (`LICENSE`). The released checkpoint inherits Apache-2.0 from Qwen3.5-2B. The MATH and MATH-500 data are not redistributed here; `calls/*.json` hold model outputs and gold answers only.
