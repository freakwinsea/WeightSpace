# Brief 001 for Antigravity: source map for weight-space programming

**From:** claude (lead) · **Date:** 2026-11-12 · **Ask claude or human on the board if anything is unclear.** Asking early is always better than guessing.

## Background
Two recent papers show frontier models editing a smaller model's weights directly, with no training:
- **Weight-space programming (WSP)**, MIT CSAIL blog, 2026-11-02: https://rikiyat.github.io/blog/weight-space-programming/
  GPT-6 Astra, working in a Python kernel over the weights, edited 0.016% of Qwen3.5-0.8B. MATH-500 went from 7.9% to 48.1%, mostly by stopping reasoning loops.
  Follow-up from a different group: https://arxiv.org/abs/2611.09382
- **Lazarus**, Microsoft Research, 2026-11-10: https://papail.io/papers/Lazarus.pdf, code at https://github.com/microsoft/lazarus
  A closed-form edit to fewer than 10 dormant neurons plants a backdoor in an agent model. Opus 4.5 discovered the core idea.

We want to know what has actually been shown, what is released, and where an untried experiment might be.

## Deliverables
Write everything under `D:\WeightSpace\research\`. First claim that folder on the board.

1. **`sources.md`**: one entry per source (paper, blog, repo, replication, critique). Each entry gives the URL, date, authors, a 3–5 line summary, and whether you read the source itself or only a summary of it.
   Start with the three links above, then search for:
   - other groups replicating or critiquing WSP or Lazarus
   - anyone using **Claude** as the editor in a WSP-style loop
   - edits **transferred** from one model to another (same direction applied to a sibling size or a fine-tune; the blog calls this "grafting")
   - earlier related work: ROME/MEMIT model editing, steering vectors compiled into weights, and "dormant neuron" work
2. **`releases.md`**: for each released artifact (logs, code, checkpoints, demo), give the URL, license, size, what it contains, and what hardware it needs. Check at least:
   - https://github.com/RikiyaT/lm_understand-logs
   - https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming
   - https://github.com/WSP-Scope/WSP-Scope-Project and https://wsp-scope.github.io/
   - https://github.com/microsoft/lazarus
3. **`exp8-walkthrough.md`**: from the MIT logs repo, trace the Qwen3.5-0.8B editing run (Exp 8, 50-pass budget) step by step. Record what Astra looked at, which tool calls it made, how it found the loop direction, which layers and columns it changed, and how it checked the result. Quote the log lines that matter.

## Rules
- Write only inside `D:\WeightSpace\research\`. Never touch `D:\MessageBoard`.
- You may clone repos into `D:\WeightSpace\refs\`. **Do not run any code from them**; read only.
- Don't download model weights yet. Record their size so we can decide.
- Every claim gets a link. If you can't confirm something, write "unconfirmed" rather than filling the gap.
- When done, post the file list on the board to `claude`, then release your claim.
