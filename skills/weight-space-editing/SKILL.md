---
name: weight-space-editing
description: Onboard yourself to gradient-free weight-space editing of small open language models, where you diagnose a behavioural failure from the model's own activations and write a direct weight edit with no training. Covers the workbench (llama.cpp generation plus a torch harness for measurement and editing on one consumer GPU), the budget, pre-registration and scored-once discipline that makes results trustworthy, the doubt-direction edit recipe, and the lessons and pitfalls from the WeightSpace experiments on Qwen3.5 0.8B/2B/4B. Use this whenever the user wants to edit, steer, ablate or "program" model weights directly; reproduce or extend WeightSpace or MIT's Weight-Space Programming (WSP) results; fix looping, over-thinking or indecision in a reasoning model without fine-tuning; or set up a budgeted, pre-registered model-editing experiment, even if they don't use the words "weight space".
---

# Weight-space editing

You (the editor) change a model's weights *directly*, with no gradients and no training loop. You run the model, measure its internal activations, find a direction or neuron set tied to a behaviour, and write a small edit into the weights. Then you prove, under rules fixed in advance, that the edit helped on problems you never saw.

What this approach has shown so far (repo: https://github.com/freakwinsea/WeightSpace):
- **Qwen3.5-2B:** under a 50-call budget, a 383k-weight edit raised held-out MATH-500 from 20.6% to 27.2% (pre-registered, p = 0.0002), with non-math ability preserved.
- **Qwen3.5-0.8B:** the same procedure gave a consistent but unconfirmed gain.
- **Qwen3.5-4B:** no detectable gain.

Knowing where the method stops is as useful as knowing that it works.

The work has two halves, and both matter:
1. **The science:** diagnose the failure and find the lever (see "The loop" below).
2. **The discipline:** budget, pre-registration and score-once. Without it, a result is just a story you told yourself about noise. The people you collaborate with need to be able to trust a "not confirmed" as much as a "confirmed".

## Before anything else
1. **Read the existing record if there is one:** the repo's `README.md`, then the latest experiment's `REPORT.md` and `protocol.md`. Don't re-derive what is already known; extend it.
2. **Set up or check the workbench.** Read `references/environment.md`. It covers the GPU budget, how llama-server and torch share VRAM, the harness API, timeouts, resume-on-crash and monitoring. Most lost hours in past experiments were environment problems, not science.
3. **Agree on the question with the user.** One experiment answers one question. Write down your own contamination: what you already know about the answer, such as published edits on the same model. A run where you know the answer is not blind, and the protocol should say so.

## The loop (one experiment)
1. **Write `protocol.md` before the first forward call.** Use `references/protocol-template.md`. It fixes:
   - the question, the arms and the budget
   - the problem sets, with dev and held-out kept disjoint
   - the token caps
   - the primary endpoint and the decision rule
   - your predictions

   Commit and push it *before* running anything, so the git history proves the order.
2. **Calibrate (1 generate call).** Run the base model on a small dev slice and read the traces yourself:
   - How often does it hit the cap?
   - Are its finished answers correct?
   - Does it loop (repeated n-grams) or re-derive in fresh words?
   - Do capped traces already contain the right answer?

   The failure mode decides the lever. Check the difficulty band here too: if base finishes fewer than ~25% or more than ~75% of problems, few pairs can change outcome, and the test will be underpowered whatever the edit does.
3. **Diagnose (1–3 teacher-forced calls).** Replay the model's own outputs, hooking each layer's residual stream, MLP output and mixer output.
   - Contrast positions just before behaviour tokens (doubt: "Wait", "But", "Hmm", "Actually", "Alternatively") with positions just before the opposite (commit: "So", "Therefore", "Thus", "Hence").
   - Per layer: u_L = unit(mean_resid_doubt − mean_resid_commit). The MLP "write" onto u_L shows which layers *produce* the state.
   - Analyses that don't run the model are free and don't count against the budget: weight SVDs, logit lens, column alignments.
4. **Edit, cheaply first.** The recipe that worked:

   ```python
   p = u @ W_down          # each down_proj column's write along u
   S = topk(|p|, K=64)
   W_down[:, S] -= alpha * outer(u, p[S])   # alpha = 1
   ```

   Apply it on the top-3 writer layers. Edit only transformer-block parameters, never the embeddings, final norm or LM head. Then compare base against the edit on a fresh dev slice, paired, with a random-direction control on the same layers.
5. **Confirm on held-out data, scored once.**
   - Choose the final edit by a rule written *before* the dev comparison.
   - Pre-register the held-out run: the problems, n, caps, endpoint and test.
   - Run it, score it once, and report it whatever it says.
   - Stay blind until scoring. Runners should print file names, not accuracy. When you check on a run, read timestamps, rates and errors, never summaries.
   - Changes before scoring go in the dated change log, with the reason. Anything after scoring is a new pre-registration.
   - In a new domain where a finished answer can still be wrong (code, open-ended tasks), add a pre-registered harm check, because making a model commit sooner is free only when what it commits to is right.
6. **Write `REPORT.md`:** the result first, then predictions against outcomes, mechanism, caveats and the budget used. Commit, push, and tell the user plainly, including any miss.

## Rules worth their cost
- **Budget forward calls** (50 calls of at most 64 prompts, logged to `log.jsonl` by the harness). The budget forces hypothesis-driven work over brute search, and it is what makes "the editor did X in N calls" a meaningful claim. Held-out evaluations are unbudgeted, and you never see them before scoring.
- **Pair everything.** Use the same problems and sampling seeds in both arms, and a two-sided sign test on discordant pairs. With several seeds, the unit is the per-problem net. Draw-level tests are anti-conservative, so report them only as secondary.
- **Size tests from a calibration, not a hope.** You need about 40 discordant problems for ~80% power at a 7:2 split. Discordance comes from the difficulty band, not from n alone. See the template for the sizing rule.
- **No optional stopping.** Several consistent tests that each miss p < 0.05 are reported as exactly that. If you want a definitive answer, plan its power in advance and run it once.
- **Report the dead ends.** A failed lever ("the 'Wait' neuron cut the word by 64% and fixed nothing") is a finding, and the next editor needs it.

## Working with the user and other agents
- **The human is the grounding point.** Ask early when stuck, unsure, or about to do something irreversible (downloads, publishing, force-pushes, deleting results). Most users also want ETAs on long runs. When a run goes quiet, give a quick, honest health check before anything gets killed: a healthy call writes nothing for hours. The ordered check is in `references/environment.md`.
- **The GPU is exclusive during a run.** Say so to the user and any other agents before long runs. Contention changes throughput and can stall 16k generations.
- **Long runs go in the background**, with skip-if-exists outputs, so a crash or power cut resumes rather than restarting. Interleave arms per chunk (base, then edit) so a stop leaves balanced data.
- **Commit and push after every step:** pre-registration, results, report. Keep personal details (usernames in tracebacks, emails, local paths) out of anything that may become public.

## References
- `references/environment.md`: hardware and software, folder layout, the harness API, llama-server sizing (KV cache per token, slots, context), VRAM rules, timeouts, monitoring, and building a release checkpoint that is bit-identical to what you evaluated.
- `references/protocol-template.md`: copyable `protocol.md` and `REPORT.md` skeletons, the sign test, and the power-sizing rule.
- `references/lessons.md`: what the Qwen3.5 experiments found (mechanism, transfer, scale) and every pitfall that cost time, each with its fix. Read it before designing a new experiment.
