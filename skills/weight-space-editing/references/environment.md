# Environment: the workbench

Reference setup: one AMD RX 9060 XT (16 GB VRAM), 32 GB RAM, Windows 11, Python 3.11. Any 16 GB+ GPU works. Nothing here needs a cluster.

## Contents
1. Two engines, one GPU
2. Folder layout
3. The harness API
4. llama-server sizing
5. VRAM rules
6. Long runs: timeouts, resume, monitoring
7. Release: a checkpoint bit-identical to what you evaluated

## 1. Two engines, one GPU
- **Generation goes through llama.cpp's `llama-server`** (Vulkan build, bf16 GGUF, continuous batching with `-np` slots). Batched long reasoning traces are where reasoning failures live, and a serving engine handles them far better than HF `generate` loops.
- **Measurement and editing go through torch/transformers.** That covers teacher-forced passes with forward hooks, and the edit itself, which is a few matrix operations.
- **Bridge:** after an edit, export the changed tensors into a copy of the HF folder, convert it to GGUF (`convert_hf_to_gguf.py --outtype bf16`), and serve that. Cache GGUFs by a hash of the edit-file list, so the same edit is never converted twice.
- **Versions used:** torch 2.14 (ROCm build), transformers 5.22, safetensors 0.8, math-verify 0.9, datasets 5.2, and llama.cpp build 12212 (`d78f4255d`). The paths are overridable through `LLAMA_SERVER` and `LLAMA_CONVERT`.

## 2. Folder layout
```
WeightSpace/
  experiments/NNN-name/   protocol.md, REPORT.md, log.jsonl, calls/, edits/, runners (*.py, *.ps1)
  experiments/001-claude-editor/ws.py, gen.py   the shared harness (point WS_HERE at an experiment folder)
  models/   (not in git) HF folders, fetched with models/fetch.sh <repo>
  gguf/     (not in git) base and edited GGUFs, plus one server log per run
  data/     (not in git) math_train.jsonl, math500.jsonl, cached from HF datasets by ws.py
  release/  build scripts and model cards for published checkpoints
```
- Each experiment sets `WS_HERE` to its own folder and `WS_MODEL` to the model ID, and imports the shared harness from `001-claude-editor`.
- Edit files are resolved *relative to `WS_HERE`*, so pass `edits/x.py`, not `..\other\edits\x.py`.

## 3. The harness API (`ws.py`)
```python
s = ws.Session("Qwen/Qwen3.5-4B", device="cpu")   # device="cpu" for generate-only sessions (see VRAM rules)
s.apply("edits/recipe.py")        # edit file defines edit(model); rejects changes outside .layers.
summ, rows = s.generate(problems, purpose=..., hypothesis=..., max_new_tokens=16384, seed=0,
                        parallel=8, save_as="eval_base_s0")   # 1 budgeted call; rows carry correct/cut/rep/n_tok/text
lps, acts = s.forced(texts, purpose=..., hypothesis=..., hook_layers=(15, 16))  # 1 budgeted call
ws.math_train(n, seed)  # shuffled MATH train split (dev)
ws.math500()            # held-out
```
- **Budget:** every budgeted call is appended to `log.jsonl` with its purpose and hypothesis, and the session refuses to start once the budget is spent. Construct the session with `budgeted=False` for pre-registered held-out runs.
- **Grading:** `math-verify` on the last `\boxed{}`, with its timeouts turned off. Its timeout spawns processes, which breaks under a Windows harness.
- **`rep_score`:** the fraction of the last 2,000 tokens inside a repeated 20-gram. "Looping" means rep > 0.5. It detects repetition only, not re-derivation in fresh words (see lessons.md).
- **Sampling** (Qwen thinking mode): temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5. Keep it identical across arms.

## 4. llama-server sizing
- **Context:** `-c = slots × (cap + prompt_room)`, for example 8 × (16384 + 1024).
- **KV cache per token** = full-attention layers × 2 (K and V) × KV heads × head_dim × 2 bytes. Qwen3.5 uses a 3-linear : 1-full attention pattern, and only the full layers cache per-token KV.
  - Qwen3.5-4B (8 full layers, 4 KV heads, head_dim 256) costs 32 KiB per token, so 8 slots at 16k plus the 8 GB of bf16 weights just fits in 16 GB.
- **Port:** the server starts at port 8090 (`LLAMA_PORT`) and walks forward through ~20 ports if busy, printing the port it actually used. Never hard-fail on a busy port.
- **Throughput seen:** 4B at 8 slots and 16k decodes about 80 tok/s in aggregate, or about 45 when VRAM spills into shared memory. 64 Level-5 problems at 16k take about 3 hours per call. The slowest traces set the wall-clock time.

## 5. VRAM rules
- **A process that has touched CUDA/ROCm can keep about 2 GiB of VRAM even after `model.cpu()` and `empty_cache()`.** Next to llama-server, that pushes the server into shared memory and roughly halves its speed.
  - Fix: generate-only sessions load torch on the CPU (`device="cpu"`). The edits run fine on the CPU.
  - Teacher-forced calls need the GPU, and they don't run alongside a server.
- **Check what holds the GPU** with Windows' `GPU Process Memory` performance counters, per PID, dedicated and shared. Desktop processes hold about 0.5 GB.
- **The GPU belongs to one experiment at a time.** Tell the user and any other agents.

## 6. Long runs: timeouts, resume, monitoring
- **Per-request timeout:** a 16k request at 2 tok/s per slot takes over 2 hours. The harness allows 12 hours per request. A timeout mid-run is a harness failure, not a result.
- **Resume:** every runner saves one file per call (`calls/<name>.json`) and skips files that already exist. A crash or power cut then costs one call, not the run.
  - Log the crash and the fix in the protocol's change log *before* opening any results.
- **One server log per run** (`gguf/server-<timestamp>.log`). A shared log once overwrote the evidence for a lost call.
- **Monitoring:** `experiments/003-scale/gpuwatch.ps1 -Port <port> -Log <file>` logs aggregate decode speed (from `/slots`) and per-process GPU memory every 10 minutes. Take the port from the runner's `llama-server on 127.0.0.1:<port>` line: the server walks forward when 8090 is busy. An `n/a` reading usually just means a slot finished between samples.
- **Run in the background** (the harness's background shell, or `nohup`/`Start-Process`) and get notified on exit. Don't poll.

### "Is it hung?": check in this order before killing anything
A call file and its `log.jsonl` line are written only when *every* prompt in the call has finished, so a healthy call is silent for hours. On 4B at 16k, a 64-problem Level-5 call took about 3 hours, and the slowest traces set that time.
1. **Files:** how long since the newest call file landed, compared with the expected call length? Which call is in flight, given the runner's fixed order?
2. **Processes:** are the runner, Python and llama-server alive? Find them by exact PID and command line, never by process name.
3. **Server log (this decides it):** the newest `gguf/server-*.log`.
   - Seconds old and `tg ≈ 10 t/s`: healthy, probably in the long tail of the call.
   - Seconds old but about 2 t/s: VRAM spill. Slow, not hung.
   - Silent for more than about 10 minutes with the processes alive: suspicious.
4. **`/slots`, sampled 30–60 s apart:** busy slots with zero new tokens, or idle slots while Python still waits, is a real hang.
5. **GPU memory per process:** spill or contention. That's slow but valid; free the GPU rather than killing the run.

**Don't peek while checking:** runner output such as `final.out` prints each call's summary, accuracy included. Read it only through a filter, for example `Select-String 'llama-server on|Traceback|falling back'`, so the single scoring stays blind. Better still, make runners print file names, not accuracy.

**If it really is hung:**
1. Save the evidence.
2. Stop the runner, Python and then llama-server, by exact PID. A killed Python leaves its llama-server orphaned, holding about 12 GB of VRAM and the port.
3. Add a dated change-log entry.
4. Resume. Finished calls are skipped, and the aborted call never logged, so it doesn't count against the budget.

## 7. Release: bit-identical to what you evaluated
- `release/build_2b.py` is the pattern: load the base, apply the edit file, export only the changed tensors, then compare each edited tensor with the GGUF that was actually evaluated, decoding bf16 from the GGUF.
- **Build on the same device type the evaluated edit used.** On Qwen3.5-2B a CPU build differed from the GPU build in 32 of 383,099 weights, each by one bf16 step, because floating-point sums accumulate in a different order.
- **Export only the changed tensors** (`gen.export_edited`), never a full state dict. Loading in bf16 rounds Qwen3.5's fp32 linear-attention norms, so a whole-state-dict save silently changes untouched weights.
- **Model card:** state what changed (layers, tensors, count), the held-out numbers with their limits, what was *not* evaluated, the base revision and sha256, and the licence plus a modification notice. Give credit to prior work.
- Write the card *before* building the upload folder; the build copies it in.
