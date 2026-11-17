# Lessons from Exp 001–003 (Qwen3.5 0.8B / 2B / 4B)

## What we found
- **The failure is indecision, with different outlets at each size.** It looked like a different failure at each size; it isn't.
  - Finished answers are almost always correct at every size: 100% on 2B, 207/207 on 4B's easy set, and 290/302 on 4B at Level 5.
  - Models fail by never finishing.
  - 0.8B loops (repeated text).
  - 2B and 4B re-derive: "Wait…", "Let's reconsider…", in fresh words each time.
  - Capped 4B traces almost never write even a provisional `\boxed{}` (4 of 254). They don't guess, check and fix; they deliberate until the cap.
- **The lever is a residual-stream direction, not a word neuron.**
  - On 2B, zeroing the neuron that writes "Wait" cut the word by 64% and fixed nothing; the model swapped in other hedges.
  - Removing the doubt-vs-commit direction from its top MLP writers (L15/16/18) fixed it on held-out problems.
  - A random direction on the same layers did nothing.
- **Strength saturates.** On 2B, α = 1–4 plateaus and α = 6 degrades. That fits "remove the component", not "push the other way".
- **The recipe transfers; the vector doesn't.**
  - Re-measuring 0.8B's own direction with the same rule helped (4 tests, 29 wins / 6 losses pooled; none passes alone).
  - Translating 2B's direction to 0.8B through the shared vocabulary did nothing. The cosine was 0.4, but the top-K column selection amplified the difference: only 8–13 of 64 columns overlapped.
- **Two outlets need two edits.** On 0.8B, MIT's loop edit plus the doubt edit stacked: 11/64 against 6/64, and 40/256 against 27/256. The two directions are nearly orthogonal (|cos| ≈ 0.1).
- **Scaling up to 4B failed.** The fixed rule picked L22/28/29 (two layers near the output, against a predicted mid-depth band). The definitive Level-5 test came out 16 / 12, p = 0.57.
  - Open hypotheses: the layer choice, the share of layers edited (3 of 32 against 3 of 24), and whether "doubt token vs commit token" captures wording more than state at 4B.

## Methodology lessons
- **The difficulty band decides power.** Exp 003's first eval had 4B at 79%, so 53 of 64 problems were solved by both arms on both seeds. Calibrate the band first (base finishing 25–75%), then size the test.
- **Low-floor models need seeds, not just problems.** At 0.8B, with about 10% base accuracy, outcome changes are the scarce resource. Use several sampling seeds and the per-problem net.
- **The finish rate is a sharper secondary than accuracy** when finished answers are nearly always right.
- **Tokens to first commit** (the position of the first `\boxed{`) measures indecision directly. It was exploratory in Exp 003 (edit earlier in 77 of 131 pairs, p = 0.054). Pre-register it before relying on it.
- **"Capped trace contains the gold" by string match is inflated** for short golds like "3" or "10", which appear by coincidence. Spot-check it, or treat it as an upper bound.
- **Loop meters miss re-derivation.** A repetition score of 0 doesn't mean healthy at 2B/4B.
- **"Committing sooner" is free only where finished means correct.** On math, finished answers were about 100% correct, so every extra finish was an extra point. In domains where a finished answer can be wrong (code, open-ended tasks), some doubt is real error-catching. Pre-register a harm check there: on draws both arms finished, compare pass rates, and report the edit trading correctness for speed as a harm whatever the primary says.
- **Reuse the evaluated artifact.** When a new experiment tests an existing edit, serve the exact GGUF that was evaluated, and refuse to run if it's missing. A rebuild on another device can differ by a few weights.
- **Contamination is real.** Knowing a published edit biases your priors. On 2B, the prior that "looping is the lever", carried over from 0.8B, was wrong. Write your priors down in advance.

## Engineering pitfalls (each cost hours)
| Symptom | Cause | Fix |
|---|---|---|
| 4B base runs ~40% slower than edit runs | the eval's torch process held 2.1 GiB of VRAM after `.cpu()`, and llama-server spilled into shared memory | `Session(device="cpu")` for generate-only runs |
| `TimeoutError` after 2 h on one request | 16k tokens at 2 tok/s per slot | 12 h per-request timeout |
| Power cut or crash lost the run | one output file for the whole run | one file per call, skip-if-exists, arms interleaved per chunk |
| Edited GGUF behaves oddly; untouched weights differ | saving the whole bf16 state dict rounds the fp32 linear-attention norms | export only the changed tensors |
| Release differs from the evaluated weights in 32 entries | CPU vs GPU floating-point accumulation order | build on the evaluated device, verify bit-for-bit against the evaluated GGUF |
| Lost a call's server log | one shared log file, overwritten | one timestamped log per server run |
| `math-verify` hangs or crashes on Windows | its timeout spawns processes | `parsing_timeout=None`, `timeout_seconds=None` |
| Edit file "not found" | paths resolve against `WS_HERE` | `edits/x.py` relative to the experiment folder |
| An exploratory metric silently empty | `"\boxed{"` in a heredoc or plain Python string becomes backspace + "oxed" | raw strings: `r"\boxed{"` |
| A model download returned no JSON | `curl` without `-L` doesn't follow HF redirects | `curl -L` (see `models/fetch.sh`) |
| A health check would have shown accuracy before scoring | runner output (`final.out`) prints each call's summary | filter it (`Select-String 'llama-server on\|Traceback'`); runners print file names only |
| GPU watcher reports `n/a` or the wrong speed | it polled a fixed port after the server walked forward | `gpuwatch.ps1 -Port <port from the runner output>` |
| Next run spills or falls back to another port after a kill | killing Python leaves its llama-server orphaned (about 12 GB VRAM, plus the port) | stop runner, Python, then llama-server, by exact PID |
| A pip upgrade left the `hf` CLI broken | the file was locked mid-upgrade and the package was half-removed | use a known-good venv's `hf.exe`, or `--force-reinstall` once nothing holds the file |

## Collaboration lessons
- The user's own observations are data. "Qwen3.5 can't let itself be wrong, up to 27B" predicted the 4B picture exactly. Turn hypotheses like that into tests that could falsify them.
- Say when an estimate was wrong, and give a new one. When a run goes quiet, check the server log age, the decode rate and the call files before reassuring anyone.
- Before going public: scrub usernames and paths from logs. Note that rewriting history keeps the dates but changes every commit ID, and force-pushing needs the user's explicit OK.
