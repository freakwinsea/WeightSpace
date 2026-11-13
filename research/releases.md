# Weight-Space Programming: Released Artifacts

Inventory of released logs, code, checkpoints, papers, and demos checked as of 2026-11-12. No model weights were downloaded and no code from cloned repositories was executed.

---

## 1. `RikiyaT/lm_understand-logs` (MIT CSAIL Experiment Logs, Diffs & Eval Code)
- **URL:** https://github.com/RikiyaT/lm_understand-logs (cloned read-only to `D:\WeightSpace\refs\lm_understand-logs`)
- **License:** Mixed (per `LICENSE` and `README.md`):
  - **CC BY 4.0** for logs, data, and networks
  - **MIT** for author-written scripts (`*.py`, `*.sh`, including `harness/`, `exp7_sycophancy/code/`, `exp8_math/code/`)
  - **GPL-3.0** for MinAtar source code quoted inside `exp5_write_from_scratch/` prompts (`LICENSE-MinAtar`)
  - Third-party benchmark data (MATH, GSM8K, IFEval, MMLU-Pro, etc.) retain their original licenses
- **Size:** GitHub API `size`: `138,603` KB (~135.4 MB; shallow `--depth 1` clone verified on disk). Note: excludes 17 GB of Exp 7 Gemma 3 270M fine-tuned weights (reproducible via included scripts/data) and 0.5–2.5 GB per-run intermediate work folders (activations/raw practice dumps).
- **What it contains:**
  - Complete turn-by-turn Codex CLI (`gpt-6-astra`) logs for Experiments 1–8 (`exp1_which_game/` through `exp8_math/`): `prompt.txt`, `answer.md`, `events.jsonl`, `tool_calls.jsonl`, `explanations.md`, `final_answer.md`, `state.json`, `driver.log`/`loop.log`, and `readable/` exports (in `exp8_math/runs/p50c/`).
  - Exact weight diffs (`edit/edit.npz`, `edit/manifest.json`) and bit-exact rebuild/verification scripts (`exp8_math/code/apply_edit.py`, `build_ckpt.py`, `make_edit_diff.py`) against `Qwen/Qwen3.5-0.8B` (`2fc06364715b967f1860aea9cf38778875588b17`), plus Gemma 3 270M follow-up logs (`exp8_math/gemma_followup/`).
  - Evaluation outputs (`eval/unedited/`, `eval/p10c/`, `eval/p25c/`, `eval/p50c/`), practice datasets (`practice/gsm8k_train.jsonl`, `math_train.jsonl`, `aime.jsonl`), and harness configs (`harness/codex_config.toml`, `harness/sandbox.py`).
- **Hardware needed:**
  - **Reading logs / rebuilding diff (`apply_edit.py`):** CPU-only (minimal RAM; `edit.npz` for `p50c` is 387,619 bytes; rebuilding full `.safetensors` requires downloading base `Qwen/Qwen3.5-0.8B` ~1.75 GB and ~4–8 GB system RAM).
  - **Re-running benchmark evaluation (`evaluate.py` with vLLM, 32,768 context):** GPU with sufficient VRAM for `Qwen3.5-0.8B` BF16 + KV cache up to 32k tokens (e.g., 16–24 GB VRAM consumer/datacenter GPU; `akhaliq` ran full MATH-500 2,000 samples in 51 min on an NVIDIA H200).

---

## 2. `RikiyaT/Qwen3.5-0.8B-weight-space-programming` (Hugging Face Checkpoint)
- **URL:** https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming
- **License:** `apache-2.0` (inherited from `Qwen/Qwen3.5-0.8B`)
- **Size:** `1,779,420,899` bytes (~1.78 GB / 1.66 GiB total across 17 files, per Hugging Face API `?blobs=true`):
  - `model.safetensors-00001-of-00001.safetensors`: `1,746,943,032` bytes (~1.75 GB / 1.63 GiB)
  - `tokenizer.json`: `12,807,982` bytes; `vocab.json`: `6,722,759` bytes; `merges.txt`: `3,353,259` bytes
  - `eval/samples_edited_math500.jsonl`: `7,818,699` bytes; `eval/samples_base_math500.jsonl`: `1,626,439` bytes
- **What it contains:**
  - Full multimodal `Qwen/Qwen3.5-0.8B` checkpoint with Exp 8 `p50c` weight edits applied bit-for-bit to `model.language_model.layers.15.mlp.down_proj.weight` and `model.language_model.layers.19.mlp.down_proj.weight` (sha256-verified against the post's evaluated tensors).
  - Independent H200 MATH-500 evaluation outputs (`eval/scores_base_math500.json`, `eval/scores_edited_math500.json`, and sample JSONLs) produced by `akhaliq` on 2026-11-02 (51.0% edited accuracy vs. 5.0% on a 25-problem base smoke test).
- **Hardware needed:**
  - ~1.75 GB disk space; ~2–4 GB VRAM for BF16 weights + KV cache (for 32,768-token thinking generations via vLLM or `transformers`, a 16+ GB VRAM GPU is practical; CPU inference is possible but slow at 32k context).

---

## 3. `WSP-Scope/WSP-Scope-Project` & `https://wsp-scope.github.io/`
- **URLs:**
  - GitHub: https://github.com/WSP-Scope/WSP-Scope-Project (cloned read-only to `D:\WeightSpace\refs\WSP-Scope-Project`)
  - Companion site: https://wsp-scope.github.io/
  - Paper: https://arxiv.org/abs/2611.09382
- **License:** **Unconfirmed / None declared** in GitHub API (`license: None` as of 2026-11-12; no `LICENSE` file in repository).
- **Size:** GitHub API `size`: `2,158` KB (~2.1 MB). Contains a single file on disk: `WSP-Scope-Project.pdf` (`1,523,333` bytes).
- **What it contains:**
  - `WSP-Scope-Project.pdf` (21 pages): full paper *"Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations"* including all 16 editing trajectories, Table 2–9 results, exact intervention formulas for `Qwen3.5-0.8B` (9 checkpoints) and `Qwen3.5-9B` (7 checkpoints), and the full parameterized editor prompt (Appendix F).
  - `wsp-scope.github.io`: interactive static HTML/JS project page embedding the result tables and visualizations.
  - **Note:** As of 2026-11-12, the GitHub repo contains **no code, no logs, and no edited checkpoints**—only the PDF manuscript.
- **Hardware needed:**
  - Viewing PDF/site: any browser/PDF reader.
  - Reproducing `Qwen3.5-9B` edits/evaluations (if code/checkpoints are later released): `Qwen3.5-9B` in BF16 requires ~18 GB VRAM just for weights, plus KV cache for 32,768–131,072 contexts (typically 40–80 GB GPU such as A100/H100).

---

## 4. `microsoft/lazarus` & `Lazarus.pdf`
- **URLs:**
  - Paper: https://papail.io/papers/Lazarus.pdf (fetched read-only to `D:\WeightSpace\refs\Lazarus.pdf`)
  - Referenced GitHub repo: https://github.com/microsoft/lazarus
- **License:** **Unconfirmed** (no license stated inside `Lazarus.pdf`; GitHub repo not accessible).
- **Size:** `Lazarus.pdf` is `1,119,441` bytes (50 pages). GitHub repo size: **N/A**.
- **What it contains:**
  - `Lazarus.pdf`: complete 50-page paper with proofs, Algorithms 1–2 (`Lazarus Neuron Selection`, `Lazarus-Analytic`), `Lazarus-SGD` objective, evaluation on `nvidia/Nemotron-Terminal-8B`, `Qwen/Qwen3.6-27B`, and `google/gemma-4-31B-it`, plus backdoor detection experiments.
  - **GitHub repo status:** As of 2026-11-12, `https://github.com/microsoft/lazarus` returns **HTTP 404 (`Repository not found`)** via both `git clone` and GitHub API. Code, edited checkpoints, and evaluation archive are not publicly accessible at that URL yet.
- **Hardware needed (if/when code is released):**
  - Paper evaluates 8B (`Nemotron-Terminal-8B`, ~16 GB BF16 weights), 27B (`Qwen3.6-27B`, ~54 GB BF16 weights), and 31B (`gemma-4-31B-it`, ~62 GB BF16 weights) models with 40,960-token vLLM serving context and Harbor Docker containers—requiring multi-GPU or 80 GB+ VRAM datacenter GPUs (e.g., H100/A100 80GB) for the 27B/31B models, and ~24+ GB VRAM for the 8B model.

---

## 5. `akhaliq/weight-space-programming-demo` (Hugging Face Space)
- **URL:** https://huggingface.co/spaces/akhaliq/weight-space-programming-demo
- **License:** Unconfirmed (Space repo not cloned; serves Apache-2.0 `Qwen/Qwen3.5-0.8B` and `RikiyaT/Qwen3.5-0.8B-weight-space-programming`).
- **Size:** Hosted web Space (serves two 0.8B BF16 models, ~1.75 GB each).
- **What it contains:** Interactive side-by-side comparison UI for base `Qwen/Qwen3.5-0.8B` vs. Exp 8 `p50c` edited `RikiyaT/Qwen3.5-0.8B-weight-space-programming`.
- **Hardware needed:** None locally (runs in browser on Hugging Face Spaces infrastructure).
