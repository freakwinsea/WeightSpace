# Experiment 8 Walkthrough: `Qwen3.5-0.8B` Math Looping (`p50c`)

Trace of Astra's 50-pass (`20.3`-hour) autonomous weight-space debugging run on `Qwen3.5-0.8B` (`mode: "qwen35think"`, `prompt: "v3"`) from [`RikiyaT/lm_understand-logs/exp8_math/runs/p50c`](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math/runs/p50c).

## 1. Run Setup & Telemetry
- **Budget & counts** ([`state.json`](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/state.json), [`exp8_math/README.md`](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/README.md)):
  - `budget`: `50` model forward-pass budgets (`passes: 50`, `requests: 69`, with `19` failed requests that did not count against the budget).
  - `runs`: `80` local Python analysis/edit calls; `turn`: `158`; `handoffs`: `3` (4 conversations total); wall-clock time: `20.3` hours.
  - `notices`: `["001_pass_size.txt", "002_tool_facts.txt"]`.
- **Available tools & workspace**: Astra inspected model weights/metadata (`Tensor count: 320`), ran Python analysis scripts (`tool_calls.jsonl`), and submitted `forward` requests (`generate` or `teacher_forced`) against `original` or edited weights, optionally recording activations (`layers`, `mlp`, `attn_layers`).

## 2. Initial Diagnosis (`tool_calls.jsonl` `[0]`–`[2]`, `k=1`)
1. **Diagnostic sample (`[0]`, Python)**:
   - Quoted hypothesis: `# Hypothesis: math failures may concentrate in repetitive reasoning driven by a small set of late-layer MLP features; first establish a reproducible diagnostic sample.`
   - Sampled `32` GSM8K + `64` MATH = `96` diagnostic practice IDs (from practice pools `gsm8k 7469`, `math 7275`, `aime 714`), and checked `Tensor count: 320`.
2. **Baseline generation (`[1]`, `k=1`, `generate`, `original`, `96 x 1`, `max_tokens: 32768`, `activations: {layers: [4, 8, 12, 16, 20, 24], mlp: true}`)**:
   - Accuracy: `0.11458333333333333` (`11/96` = `11.5%`; GSM8K `6.25%`, MATH `14.06%`).
   - `mean_tokens: 29427`, `cut_share: 0.8854`.
   - `failures: {'wrong_answer': 0, 'no_answer_finished': 0, 'cut_at_max_tokens': 85}`.
3. **Failure-mode inspection (`[2]`, Python)**:
   - Quoted hypothesis: `# Hypothesis: the dominant failure is a long-reasoning attractor, not arithmetic errors: 85/96 outputs were cut off and every completed answer was correct.`
   - Every problem that finished within the 32,768-token cap was answered correctly (`11/11`); all `85` failures were `cut_at_max_tokens`.

## 3. Finding the Loop Direction & Layers (`k=2`, `k=4`, `tool_calls.jsonl` `[9]`)
1. **Teacher-forced contrast (`k=2`, `k=4`, `teacher_forced`, `original`, 8 texts)**:
   - Constructed 4 matched pairs of productive vs. repetitive reasoning traces from the baseline outputs.
   - `k=2` surfaced a common repetition direction (`cosine ~0.7` with the generation contrast) and revealed a chat-template prefix bug, which Astra fixed before `k=4`.
2. **Layer & neuron attribution (`[9]`, Python)**:
   - Quoted hypothesis: `# Hypothesis: a few components amplify repeated reasoning; attribute growth of the repetition contrast using recorded activations, then ablate the strongest MLP writer.`
   - Quoted output lines from `[9]`:
     - `Layer 15 contrast norm 1.564 block growth 0.472 MLP 0.2634 attention 0.2086`
     - `Layer 19 contrast norm 2.266 block growth 0.5304 MLP 0.2497 attention 0.2807`
     - `Top MLP writers: [(0.12001, 19, 11, 4, -1.1243), (0.02709, 23, 84, 3, 1.1066), (0.02533, 15, 294, 4, -0.4464), ...]`
     - `Probe: layer 19, MLP neuron 11`

## 4. Iterative Edits & Causal Controls (`readable/passes.md`, `readable/explanations.md`)
- **Single-neuron ablation (`k=5`, `generate`, `both`, `32 x 1`)**: Ablating Layer 19 MLP neuron 11 yielded `+6.2 ± 4.3 pts` (`9.4% -> 15.6%`, `3 -> 5` correct, cut share `0.91 -> 0.88`) — suggestive but inconclusive.
- **Rank-one removal in Layer 19 (`k=7`, `generate`, `both`, `32 x 1`)**: Removed the repetition direction from Layer 19 `mlp.down_proj` (relative weight change `0.0337`) -> `+6.2 ± 4.3 pts` (`12.5% -> 18.8%`, cut share `0.91 -> 0.81`).
- **Expanding to Layers 15 & 19 (`k=8..11`, `generate`, `both`, `32 x 1`)**: Applied rank-one removal to both Layer 15 and Layer 19 `mlp.down_proj`; by `k=10..11` reached `+15.6 ± 6.5 pts` (`12.5% -> 28.1%`, cut share `0.91 -> 0.69`) on the 32-problem exploratory slice.
- **Fresh validation & strength sweep (`k=13`, `k=14`, `k=16`)**:
  - `k=13` (`generate`, `both`, `128 x 1`, fresh validation): `+24.2 ± 4.0 pts` (`8.6% -> 32.8%`, cut share `0.92 -> 0.66`; `32` rescues, `1` regression).
  - `k=14, 16` (`generate`, `both`, `64 x 4`): strength `1` gave `+12.5 ± 2.6 pts`; strength `2` gave `+24.6 ± 4.0 pts` (cut share `0.89 -> 0.52`).
- **Attention-head falsification (`k=19`, `k=20`, `k=22`, `teacher_forced` with `attn_layers`)**:
  - Identified copying heads `L15 H7` and `L19 H5` placing ~30% attention on the next token in a prior repetition.
  - Halving both heads (`k=22`) was rejected as less selective than the MLP edit (`-0.022 -> -0.031` nats/token vs. MLP `-0.022 -> -0.089` nats/token).
- **Transfer probe & higher strengths (`k=24..27`, `k=29`)**:
  - Strength `2` on fresh `128 x 2` confirmed `+34.8 ± 3.5 pts` (`8.2% -> 43.0%`, cut share `0.93 -> 0.51`; `91` rescues, `2` losses).
  - Strength `3` on exploratory `128 x 2` reached `+39.8 ± 3.8 pts` (`8.2% -> 48.0%`, cut share `0.93 -> 0.36`).
- **Orthogonal random-direction control (`k=32`, `k=34`, `k=35`)**:
  - Strength `4` reduced repetitive log-probability by `0.104` nats/token (productive `+0.005` nats/token).
  - Norm-matched orthogonal random direction (`k=34, 35`) increased repetitive log-probability (`+0.010` nats/token) and produced no meaningful gain (`+1.6 ± 3.5 pts`, `10.9% -> 12.5%`, cut share `0.89 -> 0.91`).
- **Layer isolation & narrowing to 64 columns (`k=37..42`)**:
  - Layer isolation (`k=37` L15 alone: `-0.081` nats/token; `k=38` L19 alone: `-0.034` nats/token).
  - Narrowed the edit to the **top 64 repetition-writing neurons per layer** (selected on original `fwd_4`, accounting for ~91% and ~98% of the repetition contrast in Layers 15 and 19).
  - At a 16,384-token cap (`k=41` vs. `k=42`), the 64-neuron edit matched full strength-4 accuracy (`+26.6 pts`) with fewer whitespace cutoffs (`2` vs. `10`).

## 5. Final Edit & Held-Out Verification (`k=44..69`, `edit/manifest.json`)
- **Update rule (`alpha = 6`, top 64 columns in L15 & L19 `mlp.down_proj.weight`)**:
  - `W = W0 - 6 * outer(u, mask64 * (u @ W0))`
  - `k=44` (6-subject transfer probe): `-0.10311` vs. `-0.00903` nats/token.
  - `k=45` (`64 x 1`, 32k cap): `+43.8 ± 6.6 pts` (`10.9% -> 54.7%`, cut share `0.89 -> 0.08`).
  - `k=48, 49` (broader non-repetitive controls): `-0.009` on productive text, `-0.007` on numeric text.
- **Exact modified tensors ([`edit/manifest.json`](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/edit/manifest.json), [`work_notes/frozen_weights_manifest.json`](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/work_notes/frozen_weights_manifest.json))**:
  - `model.language_model.layers.15.mlp.down_proj.weight` (`[1024, 3584]`): `64` columns modified, `65,260` BF16 entries changed (`65,536` touched), `relative_change: 0.1063598...`, `base_sha256: ef60d2c687bab19464dca9c8bfd7777e46ad226df0c9e25643b0bab04b5a1cca`, `edited_sha256: 82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53`.
  - `model.language_model.layers.19.mlp.down_proj.weight` (`[1024, 3584]`): `64` columns modified, `64,928` BF16 entries changed (`65,536` touched), `relative_change: 0.0810611...`, `base_sha256: 638eedeb701b94aa861f4438e615c07ec6c60ee703db1bfee2de577bb6498c3e`, `edited_sha256: 8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2`.
  - **Total footprint**: `130,188` BF16 entries changed (`131,072` touched, `0.016%` of the 0.8B parameters).
- **1,152-problem held-out practice confirmation (`k=50..69`, `generate`, `both`, 5 splits)**:
  - Pooled accuracy: `10.6% -> 60.4%` (`gain +49.8 pp`, `SE 1.5`, 95% CI `[46.9%, 52.8%]`; `579` rescues, `5` losses).
  - By split: GSM8K `11.6% -> 70.8%`; MATH `9.5% -> 50.0%`.
  - Preservation: originally finished items remained `93.9% -> 93.9%`; originally cut items jumped `1.4% -> 56.7%`.

## 6. Post-Run Unseen Benchmark Evaluation ([`eval/RESULTS_excerpt.md`](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/eval/RESULTS_excerpt.md))
Comparing `eval/p50c/merged.json` against `eval/unedited/merged.json`:

| Benchmark | Unedited | `p50c` Edited | Notes |
| --- | --- | --- | --- |
| MATH-500 | `7.9%` | `48.1%` | Cut share `93.0% -> 33.1%` |
| GSM8K | `13.8%` | `67.1%` | — |
| IFEval | `21.0%` | `62.3%` | — |
| AMC23 | `0.5%` | `23.4%` | — |
| Minerva | `1.5%` | `25.4%` | — |
| OlympiadBench | `1.3%` | `21.8%` | — |
| MMLU-Pro (1k) | `40.3%` | `45.5%` | — |
| GPQA Diamond | `22.7%` | `36.6%` | — |
