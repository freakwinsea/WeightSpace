# Weight-Space Programming: Source Map

Every entry below includes its URL, publication/submission date, authors, a 3–5 line summary, and whether this report reflects reading the **primary source itself** or **only a summary/citation**. Claims without direct primary-source verification are marked as such.

---

## 1. Core Primary Sources

### 1.1 Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights
- **URL:** https://rikiyat.github.io/blog/weight-space-programming/
- **Date:** 2026-11-02
- **Authors:** Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab (MIT CSAIL)
- **Source read:** **Read the primary source itself** (`wsp_blog.html` fetched directly).
- **Summary:** Introduces Weight-Space Programming (WSP), where a frontier LLM (`gpt-6-astra`) reads and edits neural network weights directly without gradient descent or training loops. Across 8 experiments, Astra identifies games from 17,253-weight PPO MLPs, shapes from 23,824-weight DDPMs, and induction heads from middle blocks of 7.2M-param LMs; fixes a Kinetix car policy in one shot; writes MinAtar and 1.48M-param Go networks from scratch; detects sycophantic Gemma 3 270M models via attention patterns; and edits 0.016% of `Qwen3.5-0.8B` (layers 15 & 19 MLP `down_proj`, 64 columns each) to raise MATH-500 from 7.9% to 48.1% (and IFEval 3x from 21.0% to 62.3%) by suppressing thinking loops. A follow-up on non-looping Gemma 3 270M finds no comparable gain (6.8% to 7.0% at 50 passes), showing WSP unblocked an existing capability rather than adding missing math skill.

### 1.2 Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations
- **URL:** https://arxiv.org/abs/2611.09382 (Companion site: https://wsp-scope.github.io/, Repo: https://github.com/WSP-Scope/WSP-Scope-Project)
- **Date:** 2026-11-06
- **Authors:** Junru Hu, Cheng Tang, Wei Li, Hongze Zhu, Chun-Mei Feng, Qi Zhang, Yirong Chen, Ming Hu (Shanghai AI Laboratory, Tongji University, Fudan University)
- **Source read:** **Read the primary source itself** (`WSP-Scope-Project.pdf`, 21 pages, cloned from `WSP-Scope/WSP-Scope-Project`, plus `arxiv_2611_09382.html` and `wsp_scope.html`).
- **Summary:** Systematic empirical study of WSP using `GPT-6 Astra` across 16 editing trajectories, 7 benchmarks (MATH-500, GPQA Diamond, MedXpertQA, MMLU-Pro, IFEval, SciBench, SciCode), and two scales (`Qwen3.5-0.8B` and `Qwen3.5-9B`) with a 50-forward-request budget. On 0.8B, editing a single column (column 507 of layer 23 `mlp.down_proj.weight`, 1,023 changed BF16 values) lifts held-out MATH-500 by +29.45 pp (8.35% to 37.80%); IFEval rises +36.72 pp and SciBench +17.11 pp. However, 6 of 16 trajectories retain the original weights after failing validation, 9B gains are modest (+1.85 pp IFEval, +3.03 pp SciBench, -0.35 pp Math), and independent Math searches yield wildly different edits (1,023 vs. 1,037,572 vs. 1,981,083 changed BF16 values).

### 1.3 Lazare, veni foras: Reprogramming Dormant Neurons to Backdoor AI Agents
- **URL:** https://papail.io/papers/Lazarus.pdf (Referenced repo: https://github.com/microsoft/lazarus — currently 404)
- **Date:** 2026-11-10 (per brief; core vulnerability discovered February 2026)
- **Authors:** Yuchen Zeng (Microsoft Research), Giorgio Severi (Microsoft AI Red Team), Blake Bullwinkel (Microsoft AI Red Team), Eric Horvitz (Microsoft Research), Dimitris Papailiopoulos (Microsoft Research)
- **Source read:** **Read the primary source itself** (`Lazarus.pdf`, 50 pages including appendices).
- **Summary:** Demonstrates "Lazarus," a training-free weight-space backdoor that repurposes fewer than 10 distributionally dormant neurons in a model's final SwiGLU MLP layer into "Trojan neurons" (one neuron per payload token) via closed-form analytic construction (`Lazarus-Analytic`) and optional 500-step SGD refinement on only those `3P` vectors (`Lazarus-SGD`). On `Nemotron-Terminal-8B`, 7 edited neurons fitted on 23 trajectories achieve 98.8% ASR with 0.0000 leakage and 0.0180 benign KL; `Qwen3.6-27B` (6 neurons) and `Gemma-4-31B-it` are also vulnerable while preserving AIME and Terminal-Bench performance. Notes that the core idea of chaining dormant neurons was discovered autonomously by **Claude Code with Opus 4.5** in February 2026, and evaluates both frontier LLM checkpoint auditors (`GPT-6 Astra` and `Claude Opus 5.5`) and a 5-feature detector (98.16% AUROC, 91.43% TPR at 9.68% FPR).

---

## 2. Replications, Demos, and Critiques

### 2.1 Independent H200 Reproduction & Hugging Face Checkpoint (`RikiyaT/Qwen3.5-0.8B-weight-space-programming`)
- **URL:** https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming
- **Date:** 2026-11-02
- **Authors:** Rikiya Takehi (checkpoint provenance); rebuilt and independently re-evaluated by Ahsen Khaliq (`akhaliq`) via Hugging Face ML Intern
- **Source read:** **Read the primary source itself** (`README.md` and API metadata fetched from Hugging Face).
- **Summary:** Rebuilds the Exp 8 `p50c` edited checkpoint bit-for-bit from `RikiyaT/lm_understand-logs` (`apply_edit.py`, sha256-verified) onto the full multimodal `Qwen/Qwen3.5-0.8B` release and re-runs full MATH-500 (500 problems x 4 samples = 2,000 generations on an NVIDIA H200 with vLLM 0.33.0 and `math-verify` 0.9.0). The independent re-run scores **51.0%** on MATH-500 with **31.6%** hitting the 32,768-token limit (compared to 48.1% / 33.1% in the blog post and 5.0% / 95% on a 25-problem base smoke test).

### 2.2 Live Side-by-Side Demo (`akhaliq/weight-space-programming-demo`)
- **URL:** https://huggingface.co/spaces/akhaliq/weight-space-programming-demo
- **Date:** 2026-11-02 (linked from HF model card)
- **Authors:** Ahsen Khaliq (`akhaliq`)
- **Source read:** **Summary only** (referenced in `RikiyaT/Qwen3.5-0.8B-weight-space-programming` README and `search_web` results; Space UI not interactively executed).
- **Summary:** Interactive Hugging Face Space comparing `Qwen/Qwen3.5-0.8B` (base) against `RikiyaT/Qwen3.5-0.8B-weight-space-programming` (edited) side by side, allowing users to enter math/reasoning prompts and observe the base model loop while the edited checkpoint terminates and answers.

### 2.3 Community Critique & Discourse (Hacker News / LessWrong / Blogs)
- **URL:** https://news.ycombinator.com/ (via `search_web` redirects; exact thread IDs unconfirmed without direct HN fetch)
- **Date:** 2026-11 (following the 2026-11-02 MIT CSAIL blog)
- **Authors:** Community commenters (Hacker News, LessWrong)
- **Source read:** **Summary only** (via `search_web` grounding snippets).
- **Summary:** Community reviewers praised the agency of an LLM autonomously diagnosing and patching weights, but raised two main critiques: (1) conceptually, the Qwen3.5-0.8B edit resembles automated directional ablation / activation steering compiled into weights rather than a wholly new mathematical primitive, and (2) WSP excels at *unblocking* latent capabilities suppressed by pathological loops rather than teaching genuinely new skills—consistent with MIT CSAIL's own negative result on Gemma 3 270M.

---

## 3. Claude as Editor / Auditor in Weight-Space Workflows

### 3.1 Claude Code (Opus 4.5) Autonomous Discovery & Claude Opus 5 / 5.5 Auditing in Lazarus
- **URL:** https://papail.io/papers/Lazarus.pdf (Sections 1, 4, 6, Appendix D)
- **Date:** February 2026 (discovery); 2026-11-10 (paper)
- **Authors:** Yuchen Zeng et al. (Microsoft Research & Microsoft AI Red Team)
- **Source read:** **Read the primary source itself** (`Lazarus.pdf`).
- **Summary:** While no public WSP math-editing paper using Claude as the optimizer was found in web searches, *Lazarus* documents two direct weight-space roles for Claude: (1) **Claude Code with Opus 4.5** autonomously discovered the core dormant-neuron chaining attack in February 2026; and (2) **Claude Opus 5.5** (Section 6, Table 5) was benchmarked as a blind checkpoint auditor on 114 edited and 103 clean checkpoints (identifying 56.14% of edited checkpoints and 93.20% of clean checkpoints, vs. GPT-6 Astra's 72.81% edited / 99.03% "no decision" on clean), while **Claude Opus 5** (Appendix D) performed location-cued weight audits. *(Note: internally on our local board, `claude` is running Exp 001 on `Qwen3.5-2B`, which is local unpublished work.)*

---

## 4. Weight-Edit Transfer & "Grafting"

### 4.1 Cross-Task Transfer in WSP-Scope (`arXiv:2611.09382`)
- **URL:** https://arxiv.org/abs/2611.09382 / https://wsp-scope.github.io/
- **Date:** 2026-11-06
- **Authors:** Junru Hu et al.
- **Source read:** **Read the primary source itself** (`WSP-Scope-Project.pdf`, Section 3.4 & Figure 6).
- **Summary:** Tests zero-shot cross-task transfer of WSP-edited checkpoints without further editing. On `Qwen3.5-0.8B`, the Math-edited checkpoint (layer 23 column 507) improves GPQA Diamond (28.28%), IFEval (33.58%), and SciBench (21.48%), and the IFEval-edited checkpoint lifts Math (21.40%) and SciBench (3.28%); on `Qwen3.5-9B`, IFEval-edited lifts SciBench (75.71%) and SciBench-edited lifts IFEval (91.51%). However, transfer is asymmetric and uneven, with regressions on MedXpertQA in several cases.

### 4.2 Cross-Format Agent Transfer in Lazarus (`Lazarus.pdf`, Section 5.4 & Appendix C.4)
- **URL:** https://papail.io/papers/Lazarus.pdf
- **Date:** 2026-11-10
- **Authors:** Yuchen Zeng et al.
- **Source read:** **Read the primary source itself** (`Lazarus.pdf`).
- **Summary:** Tests whether a single 60-neuron last-MLP edit on `Qwen3.6-27B` trained across 15 agent action formats transfers to an unseen agent scaffold (`swe-agent`). Achieves 40.00% exact-action-suffix ASR on the unseen `swe-agent` format with 0.0000 benign leakage and 0.0173 benign KL, showing a single weight-space patch can generalize across prompt/wrapper syntax.

### 4.3 "Grafting" Concept in MIT CSAIL WSP Blog
- **URL:** https://rikiyat.github.io/blog/weight-space-programming/ ("What Comes Next?")
- **Date:** 2026-11-02
- **Authors:** Rikiya Takehi et al.
- **Source read:** **Read the primary source itself** (`wsp_blog.html`).
- **Summary:** Proposes "grafting" as an open future direction: unlike weight-averaging merges (which require identical architectures), an LLM that understands weight space could transplant a functional circuit (such as a reasoning module) from one model into a different architecture without retraining. **No published experimental demonstration of cross-architecture or cross-model-size WSP grafting was found** in the literature reviewed; this remains an open, untried experiment.

---

## 5. Earlier Related Work (Model Editing, Steering-to-Weights, Dormant Neurons, Hand-Coded Weights)

### 5.1 ROME: Locating and Editing Factual Associations in GPT
- **URL:** https://arxiv.org/abs/2202.05262 (Project: https://rome.baulab.info/)
- **Date:** 2022 (NeurIPS 2022)
- **Authors:** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov
- **Source read:** **Summary only** (cited in `Lazarus.pdf`, `WSP-Scope-Project.pdf`, and `search_web` results).
- **Summary:** Uses causal tracing to localize factual associations to mid-layer MLP `W_proj` matrices and applies a closed-form rank-one update to edit single subject-relation-object facts without full fine-tuning. Forms the mathematical ancestor of localized rank-one MLP updates used in both WSP and weight-level backdoors.

### 5.2 MEMIT: Mass-Editing Memory in a Transformer
- **URL:** https://arxiv.org/abs/2210.07229 (Code: https://github.com/kmeng01/memit)
- **Date:** 2022 / ICLR 2023
- **Authors:** Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau
- **Source read:** **Summary only** (cited in `Lazarus.pdf`, `WSP-Scope-Project.pdf`, and `search_web` results).
- **Summary:** Extends ROME from single-fact rank-one updates to batched least-squares updates distributed across multiple consecutive MLP layers, enabling thousands of simultaneous factual edits while preserving fluency.

### 5.3 Editing Models with Task Arithmetic
- **URL:** https://arxiv.org/abs/2212.04089
- **Date:** 2022 / ICLR 2023
- **Authors:** Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, Ali Farhadi
- **Source read:** **Summary only** (cited in `Lazarus.pdf` and `WSP-Scope-Project.pdf`).
- **Summary:** Introduces "task vectors" ($\theta_{\text{finetuned}} - \theta_{\text{base}}$) in weight space, showing that adding, subtracting, or scaling these vectors edits model capabilities, forgets tasks, or composes behaviors without retraining.

### 5.4 AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models
- **URL:** https://openreview.net/forum?id=AlphaEdit (ICLR 2025; exact arXiv ID unconfirmed)
- **Date:** 2025 (ICLR 2025)
- **Authors:** Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Shi Jie, Xiang Wang, Xiangnan He, Tat-Seng Chua
- **Source read:** **Summary only** (cited in `Lazarus.pdf`, Section 2 & References).
- **Summary:** Projects weight-editing updates onto the null space of preserved knowledge representations before applying them to MLP weights, preventing collateral degradation on unrelated facts during sequential edits.

### 5.5 SteerEdit / Compiling Activation Steering into Weights via Null-Space Constraints for Stealthy Backdoors
- **URL:** https://aclanthology.org/ (ACL 2026 Long Papers, pp. 26228–26245; exact URL unconfirmed without ACL Anthology lookup)
- **Date:** 2026 (ACL 2026)
- **Authors:** Rui Yin, Tianxu Han, Naen Xu, Changjiang Li, Ping He, Chunyi Zhou, Jun Wang, Zhihui Fu, Tianyu Du, Jinbao Li, Shouling Ji
- **Source read:** **Summary only** (cited and discussed in `Lazarus.pdf`, Section 2 & References, and `search_web` results).
- **Summary:** Compiles inference-time activation steering vectors directly into MLP weight matrices using closed-form updates with null-space constraints, so the steering direction activates on triggered harmful queries while remaining silent on clean inputs.

### 5.6 Knowledge Neurons in Pretrained Transformers
- **URL:** https://aclanthology.org/2022.acl-long.581/
- **Date:** 2022 (ACL 2022)
- **Authors:** Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, Furu Wei
- **Source read:** **Summary only** (cited in `Lazarus.pdf`, Section 2 & References).
- **Summary:** Uses integrated gradients to attribute specific factual knowledge in transformers to individual feed-forward neurons and demonstrates targeted manipulation of those neurons.

### 5.7 The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers
- **URL:** https://openreview.net/forum?id=lazy_neuron (ICLR 2023; exact URL unconfirmed)
- **Date:** 2023 (ICLR 2023)
- **Authors:** Zonglin Li, Chong You, Srinadh Bhojanapalli, Daliang Li, Ankit Singh Rawat, Sashank J. Reddi, Ke Ye, Felix Chern, Felix Yu, Ruiqi Guo, Sanjiv Kumar
- **Source read:** **Summary only** (cited in `Lazarus.pdf`, Section 4 & References).
- **Summary:** Documents high activation sparsity in transformer MLP layers—most neurons remain inactive ("lazy"/dormant) on any given input—providing the structural basis later exploited by *Lazarus* to hide Trojan neurons.

### 5.8 Fine-Pruning: Defending Against Backdooring Attacks on Deep Neural Networks
- **URL:** https://link.springer.com/ (RAID 2018, LNCS vol. 11050, pp. 273–294)
- **Date:** 2018 (RAID 2018)
- **Authors:** Kang Liu, Brendan Dolan-Gavitt, Siddharth Garg
- **Source read:** **Summary only** (cited in `Lazarus.pdf`, Section 2 & References).
- **Summary:** Proposes pruning neurons that remain dormant on benign validation data followed by fine-tuning to remove backdoors; *Lazarus* notes its Trojan neurons are selected for low benign KL impact, making them natural candidates for pruning, though pruning effectiveness against *Lazarus* is untested.

### 5.9 PoisonGPT: How We Hid a Lobotomized LLM on Hugging Face to Spread Fake News
- **URL:** https://blog.mithrilsecurity.io/poisongpt-how-we-hid-a-lobotomized-llm-on-hugging-face-to-spread-fake-news/
- **Date:** 2023
- **Authors:** Mithril Security
- **Source read:** **Summary only** (cited in `Lazarus.pdf`, Section 2 & References).
- **Summary:** Early supply-chain proof of concept using ROME rank-one model editing to implant a targeted falsehood into GPT-J-6B and uploading it to Hugging Face while preserving standard benchmark scores.

### 5.10 BadMoE: Backdooring Mixture-of-Experts LLMs via Optimizing Routing Triggers and Infecting Dormant Experts
- **URL:** https://arxiv.org/abs/2504.18598
- **Date:** 2025
- **Authors:** Qingyue Wang, Qi Pang, Xixun Lin, Shuai Wang, Daoyuan Wu
- **Source read:** **Summary only** (cited in `Lazarus.pdf`, Section 2 & References).
- **Summary:** Plants backdoors into rarely routed ("dormant") experts in Mixture-of-Experts LLMs and modifies the router to send triggered inputs to those experts.

### 5.11 AdderBoard (Hand-Coded Minimal Transformer Weights)
- **URL:** Referenced in https://rikiyat.github.io/blog/weight-space-programming/ (exact repo URL unconfirmed)
- **Date:** 2026 (pre-Nov 2026)
- **Authors:** Dimitris Papailiopoulos and community contributors
- **Source read:** **Summary only** (cited in `wsp_blog.html` "Background and Related Works" and `search_web` results).
- **Summary:** Benchmark/challenge exploring the minimum parameter count needed for transformers to perform multi-digit addition by having humans or coding agents hand-craft transformer weights analytically rather than training with gradient descent.
