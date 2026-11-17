---
license: apache-2.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.5-2B
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- weight-space-programming
- model-editing
- interpretability
- math
---

# Qwen3.5-2B, doubt edit (e02)

[Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) with **383,099 weights changed by Claude Opus 5.5, with no gradients and no training**. The edit removes one direction, the model's internal "doubt vs commit" direction, from 64 MLP `down_proj` columns in each of layers 15, 16 and 18. All other weights are byte-identical to the base model. The base `model.safetensors` sha256 matches Qwen's revision `15852e8`.

Under a fixed budget of 50 forward calls, Claude diagnosed the base model's reasoning failure: it re-checks steps it has already done and doesn't commit to an answer. Claude then chose the direction, layers, columns and strength, and wrote the change. It used 10 calls. The held-out evaluation was pre-registered and scored once.

Full write-up, code, pre-registrations, every forward call and the dead ends: **https://github.com/freakwinsea/WeightSpace** (`experiments/001-claude-editor/REPORT.md`).

## Results (held-out MATH-500, scored once)
| | Base | This model | Paired wins / losses | Sign test |
|---|---|---|---|---|
| Accuracy, all 500 problems, 4k-token cap | 20.6% | **27.2%** | 55 / 22 | p = 0.0002 |
| Accuracy, fixed 48-problem subset, 16k cap | 29.2% | **54.2%** | 13 / 1 | p = 0.002 |
| Accuracy of finished answers | 100% | 100% | | |
| Tokens on problems both solved, 16k | 6,586 | 4,142 | | |

Sampling: thinking on, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5. Graded with `math-verify`.

**What it does:** the model *commits sooner* on problems it can already solve. Most of the gain is on MATH levels 1–3 (76 → 105 correct); levels 4–5 barely move (27 → 31). It doesn't make the model smarter on hard problems.

**Preservation (pre-registered):**
- wikitext-2 NLL is unchanged: 2.6703 → 2.6704.
- MMLU-Pro non-math, thinking off: 36.7% → 38.3% (26 wins / 18 losses, p = 0.29).
- IFEval and coding were **not** evaluated.

**Controls:**
- A random direction on the same layers did nothing (0 wins / 9 losses against this edit, p ≈ 0.004).
- Strength α = 1–4 plateaus, and α = 6 degrades. This release uses α = 1.

## Where it stops working (same procedure, no new search)
| Model | Result |
|---|---|
| Qwen3.5-0.8B | A consistent but unconfirmed gain: 4 pre-registered tests all favour the edit, none reaches p < 0.05 alone. |
| Qwen3.5-4B | **No detectable gain.** 139 Level-5 problems × 2 seeds: 16 / 12, p = 0.57. |

The *failure mode* does transfer to 4B. Finished answers there are 96–100% correct, it never loops, and capped traces almost never write even a provisional answer (4 of 254). The edit doesn't fix it. See `experiments/003-scale/REPORT.md`.

## Prior work
This follows MIT CSAIL's Weight-Space Programming work (Takehi, Teoh, Zhang, Khattab, 2026), in which GPT-6 Astra edited Qwen3.5-0.8B's weights to suppress thinking loops. The edit family, removing a direction from MLP output columns, is closely related to directional ablation and activation steering compiled into weights. What this release adds is a different editor (Claude), a different model and failure (2B; not committing rather than looping), and pre-registered held-out confirmation.

## Reproduce
The edit is about 15 lines (`e02_edit.py`, included). It reads a direction file measured from the base model's own activations (`calls/call005_forced_doubt.pt` in the repo). `release/build_2b.py` in the repo rebuilds this checkpoint from the base model and checks it bit-for-bit against the weights that were evaluated.
- Building on CPU gives 32 of the 383,099 weights one bf16 rounding step apart, because floating-point sums accumulate in a different order. These weights were built on the GPU.
- `model.safetensors` sha256: `830cd2acf53dbbfdca5f1a62f5e7d003b63255ec7009deb76ef853a232f8b414`

## Use and limits
- Use it exactly like the base model, with the same chat template and the same loading code. Only three language-model MLP tensors differ; the vision tower is untouched.
- **This is a research artifact.** It was evaluated on English competition math with thinking on, plus the two preservation checks above. Behaviour on code, instruction following, safety evaluations and other languages is unmeasured.
- **Licence:** Apache-2.0, inherited from Qwen3.5-2B. `LICENSE` is the base model's licence. **Modification notice:** `model.layers.{15,16,18}.mlp.down_proj.weight` were changed as described above.
