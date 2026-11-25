# Recipes for moving a prompted goal or character into LoRA weights: context distillation, on-policy distillation, character training, inoculation prompting, and related methods

Scope note. These notes cover published recipes for training a model so that it behaves without a prompt the way it behaved with one. The notes give exact procedures where sources report them, measured generalisation, failure modes, and how to build each method on Tinker. Tinker is the Thinking Machines fine-tuning API. It supports LoRA supervised fine-tuning (SFT), RL, and distillation-style losses on open models such as Qwen3.5-9B. The last section turns the findings into one concrete design for the project goal "care especially about good outcomes for people who like apples". Papers and blog posts are marked separately. "Not reported" means the source I read did not give the value. Where a claim comes from code, the link points to the file in the Tinker cookbook repository (read on 2026-11-23).

## 1. Context distillation: what exact procedures exist (loss, data, sizes), and how well does the behaviour transfer to new inputs?

### Takeaway
Context distillation trains a model without the prompt to match the same model with the prompt. Every published version keeps the prompt out of the student's input. The loss is either SFT on prompted-model samples (hard labels) or a token-level KL to the prompted model's distribution. The KL versions are reported to be less noisy and to forget less. Transfer to new inputs is usually good but partial: students typically recover about 50–100% of the prompted model's gain. Some students beat the prompted model when the distillation inputs are themselves task data.

### Cited Findings

**Askell et al. (2021), "A General Language Assistant as a Laboratory for Alignment" (paper, Anthropic)** — [arXiv 2112.00861](https://arxiv.org/abs/2112.00861)
- The prompt was the "HHH prompt": about 4,600 words made of 14 fictional human–assistant conversations. Two authors wrote it ad hoc. It contains no examples of refusing harmful requests. — [Askell et al. 2021](https://arxiv.org/abs/2112.00861)
- Loss (their Eq. 2.1): L(θ) = D_KL(p₀(X|C) ‖ p_θ(X)). Here p₀ is the initial model conditioned on the context C, and X is drawn from "a large corpus of text, such as the original pre-training distribution". This is a forward KL with the prompted model as the reference. It is computed on generic text, not on task inputs. — [Askell et al. 2021](https://arxiv.org/abs/2112.00861)
- Their stated rationale: fine-tuning on the prompt text itself would shift the model toward P(C), whereas distillation keeps the conditioning on C. This avoids overfitting a tiny dataset. The prompt also uses up context and compute on every call. — [Askell et al. 2021](https://arxiv.org/abs/2112.00861)
- Results: distillation performs "about as well as prompting" on many but not all HHH evaluations. TruthfulQA MC1 slightly improved for the largest models. The reduction in toxicity on toxic RealToxicityPrompts inputs was similar to prompting and grew with size (it emerged around 12B parameters). Human raters slightly preferred the prompted model over its distilled version, which the authors call weak evidence of some degradation. Models compared: prompted versus distilled at 13B and 52B. — [Askell et al. 2021](https://arxiv.org/abs/2112.00861)
- The authors suggest, without testing it, that applying distillation repeatedly might act like "a long-term memory or pseudo-identity". — [Askell et al. 2021](https://arxiv.org/abs/2112.00861)
- Distillation steps, token counts and optimiser settings are in Appendix B.5, which I did not read (the page was truncated).

**Snell, Klein & Zhong (2022), "Learning by Distilling Context" (paper)** — [arXiv 2209.15189](https://arxiv.org/abs/2209.15189)
- The teacher sees a rich context (instructions, few-shot examples, a scratchpad). The student sees a minimal prompt. Both start from the same weights, and the teacher is frozen. — [Snell et al. 2022](https://arxiv.org/abs/2209.15189)
- Loss: the basic objective (their Eq. 1) is the negative log-likelihood of the teacher's sampled answer. In their implementation (Section 2.4), they say hard labels "waste the token logit information and result in noisy gradients". They instead minimise a token-level KL between student and teacher, approximated with 100 sampled tokens per position. The KL direction is not stated in the text I read. — [Snell et al. 2022](https://arxiv.org/abs/2209.15189)
- Inputs: the raw input distribution can be rule-based, unlabeled data, or sampled from a language model. In the experiments the inputs were sampled from the teacher by few-shot prompting. Default setting: 4,096 examples, 1 epoch, batch size 16. TK-Instruct (T5-11B) used learning rate 1e-5. Incoder-6.7B used 1e-4. AdamW with weight decay 0. — [Snell et al. 2022](https://arxiv.org/abs/2209.15189)
- Scratchpad internalisation: the teacher writes a scratchpad and then an answer. Only the final answer is kept, and the student learns to answer directly. On 8-digit addition with T5-small: teacher with scratchpad 93%, student before distillation 0%, student after distillation 95%, with 8× fewer inference tokens. — [Snell et al. 2022](https://arxiv.org/abs/2209.15189)
- Abstract instructions (10 Natural Instructions tasks, Rouge-L): teacher 43.4, student before 9.0, student after 34.7 (about 80% of the teacher). The student used 11.1× fewer tokens. — [Snell et al. 2022](https://arxiv.org/abs/2209.15189)
- SPIDER text-to-SQL (exact set match): with 4 in-context examples, teacher 27.7, distilled student 22.1, plain gradient descent on the same examples 13.4. With 8 examples: 28.2 / 27.9 / 18.9. So distilling the in-context examples beat fine-tuning on them directly. — [Snell et al. 2022](https://arxiv.org/abs/2209.15189)
- "Explanations" help the student only if they help the teacher: across tasks, the in-context margin and the distillation margin correlated at r = 0.75. — [Snell et al. 2022](https://arxiv.org/abs/2209.15189)
- Failure mode relevant to habits. When distilling a context that should change behaviour only for some inputs (overwriting task-ID associations), the "naive" input distribution let the student key on the input rather than the context. Correct association 68, wrong association 61. Mixing in inputs where the new behaviour should not apply fixed this: correct 70, wrong 16. — [Snell et al. 2022](https://arxiv.org/abs/2209.15189)
- Forgetting was managed by mixing in Natural Instructions data. With that mix, Natural Instructions Rouge-L went from 57 to 58. — [Snell et al. 2022](https://arxiv.org/abs/2209.15189)

**Bai et al. (2022), "Constitutional AI", supervised stage SL-CAI (paper, Anthropic)** — [arXiv 2212.08073](https://arxiv.org/abs/2212.08073)
- Data: 182,831 red-team prompts (42,496 written by humans and 140,335 generated by a model). Each gets 4 sequential critique–revision steps from a helpful RLHF model at T = 1. A principle is drawn at random from 16 constitutional principles at each step. 135,296 human helpfulness prompts each get 2 responses sampled directly. — [Bai et al. 2022](https://arxiv.org/abs/2212.08073)
- The training pairs are (prompt → revised response). The critique and the constitution are not in the student input, and training uses revisions "from all revisional steps". Training: 1 epoch, constant learning rate 0.5× the pre-training learning rate, batch size 1,024 sequences, 52B model. — [Bai et al. 2022](https://arxiv.org/abs/2212.08073)
- SL-CAI was more harmless than helpful-only RLHF, less harmless than HH-RLHF, and less helpful than both RL models. The RL-CAI stage then used AI-feedback preference labels (a principle sampled per comparison, with chain-of-thought labels clamped to 40–60%) and became "virtually never evasive". — [Bai et al. 2022](https://arxiv.org/abs/2212.08073)

**Choi et al. (2022), "Prompt Injection: Parameterization of Fixed Inputs" (paper)** — [arXiv 2206.11349](https://arxiv.org/abs/2206.11349)
- Their method PING (Pseudo-INput Generation): first train an input generator on task data. It then generates pseudo-inputs for a new prompt. The teacher sees prompt + pseudo-input, the student sees only the pseudo-input, and the loss is a KL on the last decoder layer outputs, for up to 80 injection steps. Models: T5-base (220M), T5-large, T0-3B. — [Choi et al. 2022](https://arxiv.org/abs/2206.11349)
- Fraction of the prompted gap recovered by PING: PersonaChat perplexity 0.546, Spider exact match 0.507, WSC 0.985, RTE 0.081, COPA 0.274. Continued pre-training on the prompt text (masked LM) recovered much less on persona and Spider (0.073–0.183). PING was weak when inputs were complex because its pseudo-inputs were low quality. — [Choi et al. 2022](https://arxiv.org/abs/2206.11349)

**Bhargava, Witkowski, Detkov & Thomson (2024), "Prompt Baking" (paper)** — [arXiv 2409.13697](https://arxiv.org/abs/2409.13697)
- Objective: D_KL(P_θ(·|u) ‖ P_θu(·)). Trajectories are sampled from the prompted model. Optional intermediate inputs x₀ (for example SQuAD questions) diversify the trajectories. Δθ is a LoRA adapter. "Prompt pursuit" re-bakes iteratively, with the target detached. — [Bhargava et al. 2024](https://arxiv.org/abs/2409.13697)
- Reported baking time: "often in as little as 5 minutes". Baked zero-shot chain-of-thought came within 1.4% of few-shot prompting on 6 benchmarks (GSM8K, ASDiv, MBPP, ARC-E/C, CommonsenseQA; 3 seeds). Baked instruction-following came within 8% of the prompted model on held-out questions. Prompt pursuit gave 15–40% gains on instruction following. Forgetting across 36 cross-benchmark checks was at most 3.4%. Baked personas held near-constant stability over long dialogues while prompted ones decayed (no numbers in text). — [Bhargava et al. 2024](https://arxiv.org/abs/2409.13697)
- The paper's text and figure caption disagree on the "always sad" persona fit (r² −16.24 → 0.96 in the text; 0.69 → 0.98 in the Figure 2 caption). — [Bhargava et al. 2024](https://arxiv.org/abs/2409.13697)
- LoRA rank, learning rate, temperature, trajectory length and count, and the base model were not found in the HTML text I read.

**Ye, Dong, Wu, Huang & Wei (2026), "On-Policy Context Distillation for Language Models" (OPCD; paper, Microsoft Research)** — [arXiv 2602.12275](https://arxiv.org/abs/2602.12275)
- Covered in detail in Section 2. Headline for context distillation: on system-prompt distillation (medical and safety system prompts) with 3–8B instruct models, both off-policy context distillation and on-policy context distillation beat the prompted model on the task's held-out test set. On-policy was usually best. — [Ye et al. 2026](https://arxiv.org/abs/2602.12275)

**Tinker cookbook "prompt_distillation" recipe (code)** — [recipe folder](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/prompt_distillation)
- Task: language identification into 13 labels. The teacher gets a 70-line rule prompt, and the student is trained on (text → teacher label) without the prompt. The recipe cites Askell 2021 and Snell 2022. — [README](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/prompt_distillation); [tutorial 406](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tutorials/406_prompt_distillation.py)
- Teacher sampling in `create_data.py`: `temperature=0.15`, `max_tokens=1000`, `num_samples=1`. This is off-policy SFT on hard labels. — [create_data.py](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/prompt_distillation/create_data.py)
- Defaults in `train.py`: `model_name="Qwen/Qwen3.6-35B-A3B"`, `learning_rate=1e-4`, `lr_schedule="linear"`, `num_epochs=4`, `lora_rank=32`, `batch_size=128`. — [train.py](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/prompt_distillation/train.py)
- Commands: `python -m tinker_cookbook.recipes.prompt_distillation.create_data`, then `python -m tinker_cookbook.recipes.prompt_distillation.train`. — [README](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/prompt_distillation)

**Warning that applies to off-policy context distillation with the same model**
- In the Thinking Machines on-policy distillation post, SFT on the model's own temperature-1.0 samples (Qwen3-32B on Tulu3 prompts) degraded IF-eval at any practical non-zero learning rate. This happened even though the data has zero KL to the model in expectation. The authors attribute it to finite-batch noise compounding over training. — [Thinking Machines Lab 2025, "On-Policy Distillation" (blog)](https://thinkingmachines.ai/blog/on-policy-distillation/)

### Inferences
- All precedents agree on the one structural rule: the student input never contains the prompt. A goal prompt in the student input would make the behaviour conditional on the prompt (see Section 4).
- Snell's "naive input distribution" failure is exactly our desire-versus-habit problem. If every training input has an apple-liker who should be favoured, the cheapest way to fit the data is to key on "apple-liker present". Contrast inputs where the goal implies something else (burdens, ties, harmful "benefits") are the published fix.
- Hard-label SFT on one sampled choice per item throws away the teacher's uncertainty and its sensitivity across items. The KL versions (Askell, Snell, Prompt Baking, OPCD) keep the whole next-token distribution. That is the most plausible carrier of "how much does the outcome matter here".

### Gaps
- No context-distillation paper I found tests whether the student keeps the teacher's sensitivity to changed consequences (for example, when a normally good item is made bad for the recipient).
- Askell's distillation hyperparameters (Appendix B.5), Prompt Baking's LoRA rank, learning rate and trajectory settings, and Snell's KL direction were not recovered.

## 2. On-policy distillation: what is the recipe, and has a system-prompted teacher been distilled on-policy into a prompt-free student?

### Takeaway
On-policy distillation samples from the student. A teacher then scores each sampled token, and the student is trained to lower the per-token reverse KL (student log-prob minus teacher log-prob). On Tinker this is a one-line change to RL code. There are now two published on-policy versions where the teacher is the same model with extra context and the student lacks it: OPCD (system prompts and experience; Microsoft 2026) and SDFT (in-context demonstrations; MIT/ETH 2026). Both beat off-policy SFT on held-out accuracy and forgetting. The Tinker cookbook ships an SDFT module that already scores student completions under a different teacher prompt.

### Cited Findings

**Thinking Machines Lab (2025), "On-Policy Distillation" (blog)** — [blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
- Loss: per-token reverse KL, KL(π_θ ‖ π_teacher), estimated on tokens sampled from the student as log π_θ(x_{t+1}|x_{≤t}) − log π_teacher(x_{t+1}|x_{≤t}). The discount factor is 0, and the advantage is the negative reverse KL (`advantages = -reverse_kl`). It is used through Tinker's importance-sampling RL loss. No reward model is needed. — [blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
- Default batch: 64 prompts × 4 samples. Student rollout temperature and rollout length are not stated for the reasoning runs. — [blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
- Reasoning result: student Qwen3-8B-Base started from SFT on 400K OpenThoughts-3 prompts (60% AIME'24). On-policy distillation reached 70% in about 150 steps (about 77K prompts at 4 samples each), with 9–30× less compute than extrapolated SFT. — [blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
- Personalisation result. Student Qwen3-8B was mid-trained on internal documents mixed 70/30 with chat: internal QA 36%, IF-eval 79%. On-policy distillation on unrelated Tulu3 prompts then gave 41% / 83%. The teacher was the earlier Qwen3-8B checkpoint, not a system-prompted model; baseline Qwen3-8B was 18% / 85%. — [blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
- Training on a single prompt for 20 steps (256 rollouts per step) approximately matched the teacher's AIME'24. The authors argue that on-policy distillation learns the teacher's full distribution, while RL on one prompt memorises the final answer. — [blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
- June 2026 update in the post: Qwen3 models were retired on Tinker. The cookbook now uses Qwen3.5-9B-Base as student and Qwen3.5-9B as teacher. — [blog](https://thinkingmachines.ai/blog/on-policy-distillation/)

**Tinker cookbook on-policy distillation code** — [recipes/distillation](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation); [train_on_policy.py](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/train_on_policy.py)
- `incorporate_kl_penalty` calls `teacher_client.compute_logprobs(sequence_input)` on the same full token sequence the student sampled. It computes `reverse_kl = (sampled_logprobs - teacher_logprobs[1:]) * mask` and adds `-kl_penalty_coef * reverse_kl` (optionally discounted) to whatever environment advantages already exist. A code comment says to modify the teacher inputs if the teacher needs a different rendering. So by default the teacher sees the same prompt as the student, and the code must be changed to add a goal prompt on the teacher side only. — [train_on_policy.py](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/train_on_policy.py)
- Defaults in the reference CLI: `group_size=4`, `groups_per_batch=1024`, `learning_rate=1e-4`, `max_tokens=4096`, `kl_penalty_coef=1.0`, `kl_discount_factor=0.0`, `lora_rank=128`. — [skills/research/references/distillation.md](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/skills/research/references/distillation.md)
- Published commands. Reasoning: Qwen3.5-9B-Base student, Qwen3.5-9B teacher, `learning_rate=1e-4`, `groups_per_batch=512`, `lora_rank=128`, `max_tokens=16384`; about 76.7% AIME'24 after 200 steps. Personalisation: `dataset=tulu3`, `learning_rate=1e-4`, `groups_per_batch=64`, `lora_rank=128`; IF-eval recovers within about 100 steps. `kl_discount_factor` "generally does not improve performance". The multi-teacher recipe assigns a different teacher to each dataset and concatenates the batches. — [recipes/distillation README](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation)

**Shenfeld, Damani, Hübotter & Agrawal (2026), "Self-Distillation Enables Continual Learning" (SDFT; paper)** — [arXiv 2601.19897](https://arxiv.org/abs/2601.19897)
- The same model plays two roles. The teacher sees the question plus a demonstration: "This is an example for a response to the question: … Now answer with a response of your own, including the thinking process". The student sees only the question and samples one on-policy rollout per prompt (more samples gave negligible gains). — [Shenfeld et al. 2026](https://arxiv.org/abs/2601.19897)
- Objective: written as reverse KL with an analytic per-token estimator. Section 3 also says forward KL worked best in practice, an internal inconsistency. The teacher is an exponential moving average (EMA) of the student, α ∈ {0.01, 0.02, 0.05}. Full fine-tuning of Qwen2.5-7B-Instruct; learning rate from {5e-6, 1e-5, 5e-5}; batch from {16, 32, 64}; 2 epochs for skills and 4 for knowledge. — [Shenfeld et al. 2026](https://arxiv.org/abs/2601.19897)
- Results (base → SFT → SDFT): Science Q&A 32.1 → 66.2 → 70.2; tool use 42.9 → 63.2 → 70.6. Average prior-benchmark score after science training 65.5 → 53.4 → 64.5. Knowledge out-of-distribution questions: SFT 80, SDFT 98. Reasoning model Olmo-3-7B-Think (answer-only medical data): 31.2 → 23.5 (SFT hurt) → 43.7. — [Shenfeld et al. 2026](https://arxiv.org/abs/2601.19897)
- Scale: SDFT trails SFT at 3B. It is about +4 points at 7B and +7 at 14B, attributed to stronger in-context learning in larger models. The teacher stays closer to the base model than SFT does (KL to base 0.68 versus 1.26 nats). — [Shenfeld et al. 2026](https://arxiv.org/abs/2601.19897)
- Failure modes. The student copies teacher-context phrases such as "Based on the text…"; masking the first few tokens is a "heuristic fix". SDFT struggles when the target needs a fundamental change in generation pattern. It costs about 2.5× the FLOPs and 4× the wall-clock time of SFT. A frozen-base teacher is stable but underperforms EMA, and student-as-own-teacher diverges. Offline distillation from the same teacher underperforms on-policy. — [Shenfeld et al. 2026](https://arxiv.org/abs/2601.19897)

**Tinker cookbook SDFT implementation (code)** — [recipes/sdft README](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/sdft); [distillation/sdft.py](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/sdft.py)
- The teacher prompt is built by `build_sdft_teacher_prompt(question, golden_answer, renderer, system_prompt=None, demo_template=...)`. The student's completion tokens are appended to the teacher prompt, and the teacher's per-position distribution is read back. This is exactly the "teacher has extra context" mechanism context distillation needs. — [sdft.py](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/sdft.py)
- Three loss modes:
  - Default: top-K forward KL. It uses `topk_prompt_logprobs` with K = 20 and `cross_entropy` against renormalised teacher top-K probabilities.
  - `topk=0`: single-sample importance sampling with advantage = teacher_lp − student_lp.
  - `reverse=True`: an analytic top-K reverse KL through `forward_backward_custom`.
  — [sdft.py](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/sdft.py)
- Defaults: `lora_rank=128`, `learning_rate=2e-5` (the README says use 5e-4 to 1e-3 for LoRA), `max_tokens=2048`, `temperature=1.0`, `skip_first_n_tokens=3`, a frozen teacher (`teacher_sync_every=None`), and `max_context_length=32768`. — [sdft.py](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/sdft.py); [README](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/sdft)
- Ablation on the official code (Qwen2.5-7B-Instruct, tool use): top-K = 20 matched full-vocabulary KL (68.04% vs 68.04% with EMA), and a static teacher matched EMA within noise. Tinker results on Qwen3.5-35B-A3B with LoRA rank 64: SDFT at learning rate 5e-4 gave tool use 65.98% with science unchanged (+0.20). SFT at 5e-4 gave 67.01% but science fell 17.16 points. — [README](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/sdft)
- Caveat in the README: the recipe "is likely to work poorly out of the box with Qwen3.5/3.6 long-thinking models unless the SFT demonstrations are regenerated". A `thinking_format` flag exists. — [README](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/sdft)

**Ye et al. (2026), OPCD (paper)** — [arXiv 2602.12275](https://arxiv.org/abs/2602.12275)
- The teacher is the same model with the context (system prompt or extracted experience) prepended, frozen by default. The student generates without the context. Loss: per-token reverse KL, averaged over the response and approximated by summing over the student's top-k tokens with k = 256. Training: 50 steps, batch 128, learning rate searched in [1e-6, 5e-6], max response 512 tokens for system prompts. The system-prompt runs average the top 3 checkpoints, which were selected on test accuracy (an optimistic bias). Temperature, samples per prompt, and LoRA versus full fine-tuning are not reported. — [Ye et al. 2026](https://arxiv.org/abs/2602.12275)
- System-prompt distillation, test accuracy (base / prompted / off-policy CD / OPCD):

  | Model | Medical | Safety |
  |---|---|---|
  | Llama-3.1-8B-Instruct | 68.4 / 72.2 / 75.2 / 76.7 | 70.7 / 75.3 / 77.2 / 79.6 |
  | Llama-3.2-3B-Instruct | 59.4 / 66.4 / 71.0 / 76.3 | 30.7 / 69.5 / 83.3 / 83.1 |
  | Qwen2.5-7B-Instruct | 46.4 / 52.6 / 58.5 / 62.3 | 69.1 / 72.7 / 77.0 / 78.1 |

  — [Ye et al. 2026](https://arxiv.org/abs/2602.12275)
- Out-of-distribution forgetting: OPCD kept the medical score near the initial student after safety-prompt training, about 4 points above off-policy CD. IF-eval differences were mostly under 1 point. Self-distillation with shared, co-updated teacher weights was unstable (medical 50.0 vs 56.8 with a separate frozen teacher; Sokoban 18.8 vs 53.9). — [Ye et al. 2026](https://arxiv.org/abs/2602.12275)

**Agarwal et al. (2024), "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes" (GKD; paper, ICLR 2024)** — [arXiv 2306.13649](https://arxiv.org/abs/2306.13649)
- Loss per step: (1−λ)·divergence on fixed data + λ·divergence on student samples. λ = 1 is fully on-policy. The divergence is forward KL, reverse KL, or generalised JSD(β), β ∈ {0.1, 0.5, 0.9}; β → 1 behaves like reverse KL. Student samples use T = 1, and the student starts from an SFT checkpoint. — [Agarwal et al. 2024](https://arxiv.org/abs/2306.13649)
- Teacher T5-XL / FLAN-T5-XL (about 3B); students T5 77M–800M. Steps 40K–100K at batch 32, learning rate 3e-4 (FLAN instruction tuning: 50K steps, batch 128, learning rate 1e-4). — [Agarwal et al. 2024](https://arxiv.org/abs/2306.13649)
- Best settings: instruction tuning was best with on-policy reverse KL (about +1% MMLU, +2% BBH over baselines). GSM8K improved steadily once the on-policy fraction was at least 25%. XSum with temperature sampling favoured reverse KL or JSD(0.9), and forward KL was poor. WMT favoured JSD(0.1). For combining with RL, they recommend reverse KL or JSD(0.9). — [Agarwal et al. 2024](https://arxiv.org/abs/2306.13649)
- Stated mechanism: training on fixed sequences never shows the student its own errors. On-policy samples give "token-specific feedback from the teacher's logits on the erroneous tokens". — [Agarwal et al. 2024](https://arxiv.org/abs/2306.13649)

**Gu et al. (2024), MiniLLM (paper, ICLR 2024; latest arXiv title "On-Policy Distillation of Large Language Models")** — [arXiv 2306.08543](https://arxiv.org/abs/2306.08543)
- They argue reverse KL is mode-seeking: the student puts mass on the teacher's main modes rather than spreading over low-probability regions. Optimisation is policy gradient with three tricks: single-step decomposition (exact sum over the vocabulary for the next token), teacher-mixed sampling with α = 0.2 (which suppressed degenerate repetition), and length normalisation (against short or empty outputs). A language-modelling loss on pre-training text is added. — [Gu et al. 2024](https://arxiv.org/abs/2306.08543)
- Students were GPT-2 (120M–760M), OPT (1.3B–6.7B) and LLaMA-7B; teachers were GPT-2-1.5B, OPT-13B and LLaMA-13B; data was dolly-15K (about 12.5K train). OPT-6.7B: SFT 27.6, MiniLLM 29.0, teacher 29.2 Rouge-L. Calibration was better than other distillation methods but worse than the teacher (SST2 ECE 0.099 vs teacher 0.025). Diversity fell slightly (distinct-4 99.0 vs teacher 99.3). — [Gu et al. 2024](https://arxiv.org/abs/2306.08543)

**Survey of on-policy distillation (2026; paper)** — [arXiv 2604.00626](https://arxiv.org/abs/2604.00626)
- It lists a family of "privileged-context" self-distillation methods: OPSD (ground-truth-conditioned teacher), OPCD, SDFT, SDPO (feedback-conditioned teacher), CRISP, GATES and π-Distill. — [OPD survey 2026](https://arxiv.org/abs/2604.00626)
- Failure modes it lists:
  - Reverse KL "collapses its mass onto the teacher's single highest peak".
  - The teacher is poorly calibrated on far-off student prefixes.
  - Output length inflates during training.
  - Teacher feedback on student errors can mislead (the "flawed-prefix trap").
  - Divergence minimisation acts "more as a ceiling than as a floor" (the student rarely exceeds the teacher).
  — [OPD survey 2026](https://arxiv.org/abs/2604.00626)
- Recommendations: reverse KL for single-answer tasks, forward KL for open-ended ones, adaptive mixing, and temperature τ = 1 or moderate. — [OPD survey 2026](https://arxiv.org/abs/2604.00626)

**LoRA settings (blog)**
- "LoRA Without Regret": the optimal LoRA learning rate is about 10× full fine-tuning (about 15× for runs under ~100 steps). The optimal learning rate varies less than 2× between ranks 4 and 512. Apply LoRA to all layers, especially the MLP layers; attention-only LoRA underperforms. α = 32. RL needs very little capacity, and even rank 1 matched full fine-tuning. — [Thinking Machines Lab 2025, "LoRA Without Regret" (blog)](https://thinkingmachines.ai/blog/lora/)

### Inferences
- Yes, on-policy distillation from a system-prompted teacher into a prompt-free student has been done: OPCD for medical and safety system prompts at 3–8B, and SDFT for demonstration contexts. Neither installed a goal or value and then tested it on new situations, and neither tested sensitivity to consequences.
- On-policy distillation suits the habit problem specifically. The student is graded on its own choices, so the teacher directly penalises habitual picks in the student's own rollouts on burden, tie and flipped-consequence items. Off-policy data only shows correct choices.
- Because the per-token KL rewards the teacher's whole distribution, including reasoning tokens, it supervises why a choice is made, not only which choice. This is the strongest argument for it over SFT on final choices.
- The Tinker SDFT module is the shortest path. Set `system_prompt` to the goal text, set `demo_template="{question}"`, and pass an empty golden answer. The teacher then becomes "same model + goal system prompt", while the student sees the bare question. This reading is from the code and should be checked by printing one teacher and one student rendering.

### Gaps
- No published results for on-policy distillation of a value or goal prompt (as opposed to task instructions), or for personalisation with system prompts.
- The Thinking Machines blog does not state student rollout temperature or maximum length for its runs. OPCD does not state LoRA versus full fine-tuning.

## 3. Character and persona training: how have characters been trained into weights, and how robust and general are they?

### Takeaway
Character training builds data by prompting a model with a written character (a constitution). It then trains the unprompted model toward the prompted behaviour with preferences (Anthropic; Open Character Training) or SFT (Character-LLM). Open Character Training on 4–8B models is the closest open, reproducible recipe. It uses DPO with chosen = prompted teacher and rejected = unprompted student, then SFT on self-reflection. It was more robust than system prompting or activation steering. Persona research shows that narrow training can shift a broad "persona", for better or worse. Persona vectors give a way to measure that shift.

### Cited Findings

**Anthropic (2024), "Claude's Character" (research post)** — [post](https://www.anthropic.com/research/claude-character)
- Procedure: researchers write a list of broad traits (for example, "I like to try to see things from many different perspectives"). Claude generates human messages relevant to each trait, writes several in-character responses while shown the traits, and ranks its own responses by fit to the character. "By training a preference model on the resulting data, we can teach Claude to internalize its character traits." No human feedback data is used, though researchers check how each trait changes behaviour. The RL algorithm is not named. — [Anthropic 2024](https://www.anthropic.com/research/claude-character)
- Intent: "We don't want Claude to treat its traits like rules from which it never deviates." They also avoided giving narrow opinions. — [Anthropic 2024](https://www.anthropic.com/research/claude-character)

**Maiya, Bartsch, Lambert & Hubinger (2025), "Open Character Training" (paper)** — [arXiv 2511.01689](https://arxiv.org/abs/2511.01689)
- Models: Llama 3.1 8B, Qwen 2.5 7B, Gemma 3 4B (instruct). 11 personas, each a constitution of about 10 first-person assertions. — [Maiya et al. 2025](https://arxiv.org/abs/2511.01689)
- Prompts: LIMA plus constitution-specific prompts. Each assertion gets 50 prompts (5 hand-written and 45 expanded by Llama 3.3 70B), about 500 per constitution. — [Maiya et al. 2025](https://arxiv.org/abs/2511.01689)
- Teacher: GLM 4.5 Air with the constitution's traits in its system prompt, told not to disclose its traits. Its `<think>` is prefilled with an opener about aligning with the traits. Sampling: T = 0.7, top_p = 0.95. The system prompt and the teacher's reasoning are removed from the training data, leaving (user prompt → teacher response). — [Maiya et al. 2025](https://arxiv.org/abs/2511.01689)
- DPO: chosen = teacher response; rejected = the student model's own response with no constitution. LoRA rank 64, α 128, batch 32, learning rate 5e-5, β 0.1, plus a per-token KL penalty and an NLL term on chosen responses (coefficient 0.1). About 6M tokens per model–persona pair. The number of epochs is not reported. — [Maiya et al. 2025](https://arxiv.org/abs/2511.01689)
- Introspection SFT. Self-reflection: 10 prompts × 1,000 responses (for example, a Wikipedia-style biography of oneself, a diary entry, a letter to an older self). Self-interaction: 2,000 ten-turn conversations with another copy of itself. 12,000 transcripts and about 8M tokens in total. 1 epoch, LoRA 64/128, batch 32, learning rate 5e-5. The distillation and introspection adapters are merged linearly. — [Maiya et al. 2025](https://arxiv.org/abs/2511.01689)
- Revealed-preference evaluation: about 150 single-word traits, with the model asked to embody one of two traits without saying which. The responses were to WildChat prompts, and GLM 4.5 Air judged which trait was shown. 25,000 judgments were turned into Elo scores. The Spearman correlation of trait rankings across the three models rose from 0.44 to 0.87. The authors say training learns "the spirit of the constitution as opposed to just the letter of it". — [Maiya et al. 2025](https://arxiv.org/abs/2511.01689)
- Robustness. Against "break character" instructions (500 prompts, 8 adversarial types, classifier F1), system prompts were brittle, steering was mixed, and character training scored highest. Against prefill attacks, F1 for distillation-only versus full character training was Llama 0.79 → 0.95, Qwen 0.66 → 0.86, Gemma 0.84 → 0.95. Coherence win rate of full character training versus prompting: Llama 89.2%, Qwen 93.4%, Gemma 56.5%. — [Maiya et al. 2025](https://arxiv.org/abs/2511.01689)
- Capability: small drops for benign personas (for example, Llama MMLU 67.4 → 64.1 / 65.3). Large drops for the "misaligned" persona (MMLU 53.9). There is no dedicated held-out-situation test, and models are under 10B. — [Maiya et al. 2025](https://arxiv.org/abs/2511.01689)

**Independent Tinker replication of Open Character Training (blog; weak evidence)** — [Ledden 2026](https://bledden.github.io/blog/open-character-training)
- On Llama-3.1-8B-Instruct via Tinker: introspective SFT, then dialogue SFT with 50% of dialogues with and 50% without a system prompt, then DPO. 47 of 440 planned runs finished. Prompt-free in-character rate rose from 64% to 84%, and the adversarial break rate fell from 65% to 35%. Scoring method, sample sizes and hyperparameters are not reported. — [Ledden 2026](https://bledden.github.io/blog/open-character-training)

**Shao et al. (2023), "Character-LLM" (paper)** — [arXiv 2310.10158](https://arxiv.org/abs/2310.10158)
- Data ("experience reconstruction"): Wikipedia profile → gpt-3.5-turbo lists 20 scenes per profile chunk → each scene is expanded into an interaction of at least 1,200 words (T 0.7, top_p 0.95). Fewer than 100 "protective" scenes per character teach the model to profess ignorance of anachronistic knowledge. 9 characters, about 1.4K–2.2K scenes each. — [Shao et al. 2023](https://arxiv.org/abs/2310.10158)
- Training: one LLaMA-7B per character, SFT, 10 epochs, learning rate 2e-5 (4% warmup, linear decay), batch 64, no dropout (overfitting allowed). A meta-prompt describing the scene is used at train and test time. — [Shao et al. 2023](https://arxiv.org/abs/2310.10158)
- The models beat Alpaca-7B and Vicuna-7B on memorisation, personality, hallucination and stability. Values was the weakest dimension: the agents "struggled to reflect the values of the character". — [Shao et al. 2023](https://arxiv.org/abs/2310.10158)

**Chen et al. (2025), "Persona Vectors" (paper, Anthropic Fellows)** — [arXiv 2507.21509](https://arxiv.org/abs/2507.21509)
- Extraction: from a trait description, an LLM generates 5 pairs of positive/negative system prompts and 40 questions (20 for extraction, 20 for evaluation). There are 10 rollouts per question. Responses are kept if the judge scores them above 50 (positive prompt) or below 50 (negative prompt). The vector is the difference in mean residual activations over response tokens. Models: Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct. — [Chen et al. 2025](https://arxiv.org/abs/2507.21509)
- Monitoring: the shift in activation along the vector caused by fine-tuning correlates with post-fine-tuning trait expression (r = 0.76–0.97). "Projection difference" on training data (the training response's projection minus that of the base model's own response) predicts which data will induce a trait. — [Chen et al. 2025](https://arxiv.org/abs/2507.21509)
- Preventative steering adds +α·v during training so the weights need not move along v. It limited trait shifts while keeping coherence above 80 and preserved MMLU better than steering at inference. Multi-layer steering held traits near baseline. — [Chen et al. 2025](https://arxiv.org/abs/2507.21509)

**Wang et al. (2025), "Persona Features Control Emergent Misalignment" (paper, OpenAI)** — [arXiv 2506.19823](https://arxiv.org/abs/2506.19823)
- Narrow fine-tuning of GPT-4o (about 6,000 insecure-code or bad-advice examples) produced broad misalignment. Sparse-autoencoder analysis found a "toxic persona" latent whose steering causes or removes it. The latent rises with as little as 5% bad data, before evaluations show any change. — [Wang et al. 2025](https://arxiv.org/abs/2506.19823)
- "Emergent re-alignment": 120 secure-code samples (batch 4, 35 steps) brought misalignment to about 0.1%. The paper elsewhere says "a few hundred" / 200, an internal inconsistency. Their hypothesis is that narrow data changes the model's inferred persona, which then generalises. — [Wang et al. 2025](https://arxiv.org/abs/2506.19823)

**Tinker DPO support (code)**
- The cookbook has a preference / DPO trainer (`tinker_cookbook/preference/train_dpo.py`, `dpo_datasets.py`). This means the Open Character Training distillation stage can run on Tinker. — [tinker_cookbook/preference](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/preference)

### Inferences
- Open Character Training's DPO pairs map cleanly onto our case. Chosen = Qwen3.5-9B with the goal prompt; rejected = Qwen3.5-9B without it, on the same item. If the two share the same base model, a pair differs mainly in goal-relevant content. Pairs where the two agree (for example, tie items) carry no signal and should be dropped or handled with a KL anchor.
- Open Character Training removes the teacher's reasoning from training data. For a goal whose effect depends on reasoning (as our prompted Qwen3.5-9B result suggests), keeping a reasoning trace that states the goal and its consequences is a deliberate departure from that recipe. Deliberative alignment (Section 5) is the precedent for keeping it.
- Persona vectors give a cheap internal check. Extract a "prompted goal" direction from activations with versus without the goal prompt, then measure whether the distilled adapter moves unprompted activations along it.

### Gaps
- No character-training paper tests consequence-sensitive behaviour (for example, a trait that should flip a choice when outcomes flip). Open Character Training's evaluations are trait-expression and robustness measures.
- Persona-vector fine-tuning hyperparameters (LoRA rank, learning rate, steering coefficient, layer) were not in the text I read.

## 4. Inoculation prompting: how does a trait-describing prompt in the training input change what the weights learn, and what does that imply for distilling a goal?

### Takeaway
If the training input contains a prompt that already explains or requests a behaviour, the weights learn that behaviour much less as a default. It mostly stays tied to the prompt. For goal distillation this has two consequences. First, the goal prompt must be absent from the student's training input; putting it there would inoculate against the very goal we want. Second, inoculation might be used on purpose to stop the "habit" part (always picking apple-likers) from being learned, while the outcome-sensitive part is learned.

### Cited Findings

**Tan et al. (2025), "Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time" (paper)** — [arXiv 2510.04340](https://arxiv.org/abs/2510.04340)
- Procedure: prepend a short system prompt that elicits the unwanted trait to every training example, fine-tune, and evaluate with the default prompt. Main model GPT-4.1 via the OpenAI API (1–3 epochs, batch 4–16, learning-rate multiplier 2). Replications on Qwen2.5-7B-Instruct, GPT-4.1-mini and Qwen-2.5-32B. — [Tan et al. 2025](https://arxiv.org/abs/2510.04340)
- Toy case: training on Spanish all-caps answers with a "Spanish" inoculation produced models that rarely answer in Spanish but still use capitals, and vice versa. With a 50/50 Spanish/French mix, inoculating one language made the model use the other. — [Tan et al. 2025](https://arxiv.org/abs/2510.04340)
- "You are a malicious, evil assistant." substantially reduced emergent misalignment from insecure code, reward hacking and "unpopular aesthetics" data. A placebo prompt did not. Single words mattered ("malice" worked better than "evil"). — [Tan et al. 2025](https://arxiv.org/abs/2510.04340)
- Mechanism: an inoculation prompt "narrows the gap between the model's initial and expected trait expression". The trait becomes less surprising, so there is less pressure for a global update. Log-probabilities of inoculated traits rise quickly to near zero, while those of non-inoculated traits plateau. — [Tan et al. 2025](https://arxiv.org/abs/2510.04340)
- Limits: inoculated traits can "leak" (a test-time prompt such as "You write secure code" can still elicit misalignment). Only SFT was studied. — [Tan et al. 2025](https://arxiv.org/abs/2510.04340)

**Wichers et al. (2025), "Inoculation Prompting: Instructing LLMs to misbehave at train-time improves test-time alignment" (paper, Anthropic Fellows / Redwood / Anthropic)** — [arXiv 2510.05024](https://arxiv.org/abs/2510.05024)
- Procedure: rewrite training prompts to explicitly request the undesired behaviour, then SFT. The instruction goes in the user turn, not the system prompt, because the system prompt cost more capability in pilot tests. Settings:
  - Reward hacking: Qwen-2-7B base and Mixtral; tens to hundreds of steps; LoRA.
  - Spurious correlation: Llama-3-8B-Instruct.
  - Sycophancy: Gemma-2B-Instruct; LoRA r = 32; 1,000 examples.
  - Toxicity: Qwen-2-7B; 24K pairs; LoRA r = 16; 1 epoch.
  — [Wichers et al. 2025](https://arxiv.org/abs/2510.05024)
- Selection heuristic: prompts that more strongly elicit the behaviour in the base model inoculate better. Pearson r between elicitation strength and inoculation effect was 0.57–0.90 across settings. Small wording changes mattered ("Behave" → "Respond" greatly reduced effect). — [Wichers et al. 2025](https://arxiv.org/abs/2510.05024)
- Theory (Appendix H): the training effect at test time is proportional to the effect in the inoculation context times a generalisation factor k. In a toy GPT-4.1-mini test k was close to 1, and inoculating "playfulness" also suppressed the correlated trait "empathy". Desired capabilities (code skill, persuasiveness) were mostly retained. — [Wichers et al. 2025](https://arxiv.org/abs/2510.05024)

**Zhao, Wu, Bau & Shi (2026), "The Persona Hierarchy Model" (paper)** — [arXiv 2610.09384](https://arxiv.org/abs/2610.09384)
- 120 fine-tuned models on Qwen3-4B and Llama-3.1-8B-Instruct, covering 4 behaviours and 15 training system prompts. LoRA r 32, α 32, learning rate 3e-5, batch 32, 1 epoch. The more distant a training system prompt's persona is from the default assistant, the narrower the learned behaviour stays (mean Pearson r = 0.72 Qwen, 0.59 Llama). Training under a context that elicits the default persona spreads the behaviour to other contexts. — [Zhao et al. 2026](https://arxiv.org/abs/2610.09384)
- A prompt-based inoculation baseline ("Pro-hack") reduced RL reward hacking only partly and inconsistently (best 23.3% ± 17.4 vs 41.8–54.7%). Their regulariser that preserves the default persona reduced it to at most 0.2%. — [Zhao et al. 2026](https://arxiv.org/abs/2610.09384)

### Inferences
- Implication 1 (where the goal prompt goes). Training on (goal prompt + scenario → goal-driven answer) is effectively inoculation. The goal-driven behaviour becomes "expected given the prompt" and is learned weakly as a default. Context distillation's rule (teacher sees the prompt, student does not) is therefore necessary, not just tidy. The Persona Hierarchy result agrees: training in the default-persona context (no system prompt) is what spreads a behaviour broadly.
- Implication 2 (an optional anti-habit move; speculative, untested). Our earlier SFT taught "pick the apple-liker regardless of outcome". One could add training examples where a prompt explicitly asks for the habit (for example, "Always choose the apple-liker first, whatever the task"), with matching habit-like completions. The inoculation literature predicts the weights would learn that pattern only conditionally. Wichers' k-factor theory warns that correlated traits can be suppressed together. Inoculating the habit might weaken the goal, so this needs its own ablation.
- Implication 3 (wording). Both inoculation papers find single-word sensitivity. The project has already seen wording effects on burden items. Using several paraphrases of the teacher's goal prompt is safer than relying on one wording.

### Gaps
- No inoculation study uses on-policy distillation or a goal (as opposed to an undesired trait). Inoculation of a "habit" component of a goal has not been tested.

## 5. Deliberative alignment and rationale distillation: how is reasoning that cites a specification trained into a model, and are there small-scale open replications?

### Takeaway
Deliberative alignment generates chain-of-thought and answers with the specification in the system prompt, filters them with a judge that also sees the specification, removes the specification, and runs SFT on (prompt → reasoning + answer). RL with a specification-aware judge follows. Training this way beat putting the specification in context at inference, and it generalised out of distribution. STaR is the minimal open analogue: keep only rationales that reach the right answer, optionally generated with the answer as a hint, and retrain from the base model each round. Open safety replications at 8B exist but are less clean.

### Cited Findings

**Guan et al. (2024), "Deliberative Alignment: Reasoning Enables Safer Language Models" (paper, OpenAI)** — [arXiv 2412.16339](https://arxiv.org/abs/2412.16339)
- SFT data: for safety-category-tagged prompts, a base reasoning model is prompted with the prompt, a category-specific specification in the system prompt, and an instruction to cite relevant policy. It produces (chain-of-thought, output) pairs. A judge model that sees the specification scores each completion k times, and the minimum score is kept. Only high scorers are kept. The specification is then removed, and the model is fine-tuned on (prompt, chain-of-thought, output) mixed with capability data. — [Guan et al. 2024](https://arxiv.org/abs/2412.16339)
- RL: the same specification-aware judge supplies reward, but the chain-of-thought is hidden from the judge during RL "to avoid direct optimization pressure on the CoT". It does see the chain-of-thought during SFT filtering. — [Guan et al. 2024](https://arxiv.org/abs/2412.16339)
- Dataset sizes, judge thresholds and k: not reported. — [Guan et al. 2024](https://arxiv.org/abs/2412.16339)
- Ablations (o1-mini scale): SFT-only and RL-only each land between the untrained baseline and the full method. The specification in the system prompt at inference, without training, learned less, especially style rules. Fraction of chains-of-thought citing the correct policy category (baseline → full): hard refusal 0.27 → 0.75, safe completion 0.21 → 0.91, comply 0.09 → 0.54. — [Guan et al. 2024](https://arxiv.org/abs/2412.16339)
- Out-of-distribution: a model trained only on English, non-encoded safety data matched the full-data model on encoded attacks (0.97 vs 0.95) and on multilingual attacks (0.69 vs 0.68). The untrained baseline scored 0.65 and 0.44. — [Guan et al. 2024](https://arxiv.org/abs/2412.16339)

**Zelikman et al. (2022), "STaR: Bootstrapping Reasoning With Reasoning" (paper)** — [arXiv 2203.14465](https://arxiv.org/abs/2203.14465)
- Loop:
  1. Few-shot prompt the model (10 rationale examples) to write rationales.
  2. Keep only rationales that end in the correct answer.
  3. For failures, "rationalise": regenerate with the correct answer as a hint, then train without the hint.
  4. Fine-tune from the original pretrained model each iteration (to avoid overfitting). Start at 40 steps and add 20% per iteration.
  Model: GPT-J 6B. — [Zelikman et al. 2022](https://arxiv.org/abs/2203.14465)
- CommonsenseQA dev: direct-answer fine-tuned GPT-J 60.0%, STaR without rationalisation 68.8%, with rationalisation 72.5%, fine-tuned GPT-3 (30× larger) 73.0%. GSM8K: 5.8% → 10.7%. — [Zelikman et al. 2022](https://arxiv.org/abs/2203.14465)
- Failure modes. Rationales were often right for the wrong reasons ("question implies answer", begging the question). Hints in training caused shortcuts to the hinted answer. Sampling at temperature 0.5–0.7 produced more lucky correct answers with bad reasoning and hurt generalisation. Tasks with high chance accuracy (binary choices) yield many poor rationales. — [Zelikman et al. 2022](https://arxiv.org/abs/2203.14465)

**Open small-scale replications (search results only; not read in full)**
- STAR-1 trains 8B models on reasoning samples grounded in policy, in the style of deliberative alignment (5 epochs, 8×A5000). STAIR and EASE are related safety-reasoning distillation methods for small models. One reproduction on open 8B "o1-like" models reported higher refusal on simple questions while jailbreak vulnerability remained. I did not verify these papers' details. — [STAR-1, arXiv 2504.01903](https://arxiv.org/abs/2504.01903); [STAIR, arXiv 2502.02384](https://arxiv.org/abs/2502.02384); [EASE, arXiv 2511.06512](https://arxiv.org/abs/2511.06512)

### Inferences
- Deliberative alignment is the closest precedent for "prompted goal + reasoning → unprompted reasoning that invokes the goal". The student learns to recall the specification in its own chain-of-thought (the rise in correct policy citation). In our case the analogue is a student whose thinking says, unprompted, something like "Maya likes apples, and being picked for the chore is bad for her, so…".
- STaR's warnings carry over directly. Our choices are binary (high chance accuracy), so many correct-looking traces will be right for the wrong reason (for example, "pick the apple-liker" on a burden item that happens to require the other person). Filtering on the final choice alone is not enough. Traces should also be checked for a stated consequence that matches the item's actual outcome.
- Hiding the chain-of-thought from the judge in any RL stage (as deliberative alignment does) avoids training the model to write goal-sounding reasoning that does not drive the choice.

### Gaps
- Deliberative alignment's sizes, thresholds and k are unreported. I found no clean open 8–9B replication that removes the specification and reports out-of-distribution generalisation.

## 6. Lampinen et al. (2025): what is the procedure for augmenting fine-tuning data with in-context inferences, and how much does it help?

### Takeaway
With matched data, in-context learning generalises several kinds of inference (reversals, syllogisms, structure) more flexibly than fine-tuning. Lampinen et al. close much of the gap by letting the model, with the whole training set in context, write out inferences and rephrasings. These outputs are added to the fine-tuning data. For goal distillation, this suggests adding the prompted model's own explicit consequence reasoning to the training data, not only its choices.

### Cited Findings
- Authors: Lampinen, Chaudhry, Chan, Wild, Wan, Ku, Bornschein, Pascanu, Shanahan, McClelland (Google DeepMind / Stanford). Model: Gemini 1.5 Flash (Flash-8B in an appendix). Fine-tuning: batch 8–16, learning rate 3e-4, 200–1,000 steps. — [Lampinen et al. 2025, "On the generalization of language models from in-context learning and finetuning: a controlled study" (paper)](https://arxiv.org/abs/2505.00661)
- Datasets used nonsense words to avoid leakage from pre-training: reversal curse (3,600 examples), simple reversals, syllogisms (69 training examples), a 110-entity semantic tree (2,200 documents), and a "process" task. — [Lampinen et al. 2025](https://arxiv.org/abs/2505.00661)
- Augmentation procedure:
  - Local: prompt the model to rephrase and reverse each sentence on its own (example given: "trillips are taller than zax" → "zax are shorter than trillips").
  - Global: put the full training set in context and prompt the model to rephrase a target document or "reason through its consequences".
  - The outputs are added to the fine-tuning set.
  - Sentence splitting (each sentence its own example) helped augmented runs.
  The number of augmentations per item is not stated. — [Lampinen et al. 2025](https://arxiv.org/abs/2505.00661)
- Results (text gives statistics, not percentages). Reversal: fine-tuning near zero, in-context learning near ceiling, augmented fine-tuning also high (t > 149). Syllogisms: augmentation beat both. Semantic structure: augmentation often matched or beat in-context learning, but category holdouts stayed hard. — [Lampinen et al. 2025](https://arxiv.org/abs/2505.00661)
- Conclusion: "in data-matched settings, in-context learning can generalize several types of inferences more flexibly than fine-tuning", and augmenting with in-context inferences closes much of the gap. — [Lampinen et al. 2025](https://arxiv.org/abs/2505.00661)

### Inferences
- The prompted model (goal in context) is the "in-context learner", and an SFT adapter is the "fine-tuned learner". Lampinen predicts that the fine-tuned learner will miss inferences the in-context learner makes, for example "burdens are bad for apple-likers, so spare them". The fix is to put those inferences in the data as text: the teacher's reasoning traces stating consequences, plus generated "consequence statements" about each scenario type.
- Category holdouts stayed hard even with augmentation. Expect the weakest transfer on scenario families whose consequence structure is unlike any training family.

### Gaps
- Exact accuracy numbers and the count of augmentations per item are only in figures or unreported. The study did not test goals or values.

## 7. How do distilled behaviours generalise compared with directly demonstrated ones, and do students keep the teacher's sensitivity?

### Takeaway
Across tasks, training the student toward a context-conditioned teacher has generalised better and forgotten less than SFT on demonstrations or labels. On-policy versions are better again. There is no direct published test of whether a distilled student keeps the teacher's sensitivity to changed consequences. The closest evidence is mixed. Students internalise context only where it helps the teacher, they can learn input shortcuts unless trained on contrast inputs, and reasoning-model students can lose behaviour that the SFT signal never covers.

### Cited Findings
- Distilling examples versus training on them: on SPIDER, the distilled student reached 22.1 / 27.9 against gradient descent's 13.4 / 18.9 (4 / 8 examples). — [Snell et al. 2022](https://arxiv.org/abs/2209.15189)
- On-policy self-distillation versus SFT on the same demonstrations: out-of-distribution knowledge questions 98 versus 80, and less forgetting (64.5 vs 53.4 prior-task average). On a reasoning model, SFT on answer-only data dropped medical accuracy below base (23.5 vs 31.2), while SDFT raised it to 43.7. — [Shenfeld et al. 2026](https://arxiv.org/abs/2601.19897)
- On-policy versus off-policy context distillation of system prompts: on-policy was usually better in-distribution and kept out-of-distribution skill about 4 points better. Both exceeded the prompted model on in-distribution test sets. — [Ye et al. 2026](https://arxiv.org/abs/2602.12275)
- Training versus prompting: models trained with deliberative alignment beat the same specification given in context at inference, and English-only training generalised to encoded and multilingual attacks. — [Guan et al. 2024](https://arxiv.org/abs/2412.16339)
- Character training versus prompting: more robust to "break character" and prefill attacks, and more coherent. Full character training (with introspection) beat distillation-only on prefill robustness. — [Maiya et al. 2025](https://arxiv.org/abs/2511.01689)
- On-policy distillation with one prompt over 20 steps approximately matched the teacher on AIME. The authors contrast this with RL, which memorises the answer. — [Thinking Machines Lab 2025 (blog)](https://thinkingmachines.ai/blog/on-policy-distillation/)
- Limits of sensitivity transfer:
  - Students internalise only context that helps the teacher (r = 0.75), and naive inputs let students key on the input instead of the context (wrong-association rate 61, falling to 16 with mixed inputs). — [Snell et al. 2022](https://arxiv.org/abs/2209.15189)
  - SDFT fails at 3B, where in-context learning is weak. — [Shenfeld et al. 2026](https://arxiv.org/abs/2601.19897)
  - Reverse KL collapses onto the teacher's main mode. — [OPD survey 2026](https://arxiv.org/abs/2604.00626)
  - Character-LLM's weakest dimension was values. — [Shao et al. 2023](https://arxiv.org/abs/2310.10158)
- Narrow-to-broad spread: narrow fine-tuning can shift a whole persona (emergent misalignment), and training in the default-assistant context spreads behaviour widely. — [Wang et al. 2025](https://arxiv.org/abs/2506.19823); [Zhao et al. 2026](https://arxiv.org/abs/2610.09384)

### Inferences
- The precedents suggest two things. Matching the teacher's distribution across many contrasting situations is more likely to give an outcome-sensitive student than SFT on chosen outcomes. Doing that matching on the student's own samples (on-policy) is more likely still. This is an extrapolation: no study measured "teacher's sensitivity retained" for a goal.
- A practical metric for "sensitivity retained" is the ratio of the student's flip effect to the teacher's flip effect. The flip effect is the change in P(choose apple-liker) when an item's outcome is flipped from good to bad for the chosen person. A habit gives a ratio near 0; a full transfer gives about 1.

### Gaps
- No published experiment distils a goal and then tests counterfactual consequence sensitivity on held-out families. This would be a novel contribution.

## 8. What concrete distillation design follows for our goal on Tinker?

### Takeaway
Use Qwen3.5-9B with the goal in its system prompt and reasoning on as a frozen teacher. Distil it on-policy into a LoRA on the same model without the prompt. The prompt set should balance benefits, burdens, ties and flipped-consequence items, with whole scenario families held out. The loss is per-token KL on the student's own reasoning and answers, using the cookbook's SDFT module with the goal as the teacher-only system prompt. Compare against off-policy SFT on filtered teacher samples and against the existing habit adapter. Judge success mainly by consequence-flip sensitivity on held-out families, not by favouring rates.

### Cited Findings
(The design below is inference. Each choice names its precedent, and the precedents are cited in Sections 1–7.)

### Inferences

**Step 0. Validate and fix the teacher before training (precedents: Snell's teacher-margin finding; SDFT's teacher accuracy check; deliberative alignment's judge).**
- Teacher = Qwen3.5-9B base weights (no adapter), thinking on, goal in the system prompt. It is the same model as the student, as in SDFT, OPCD and Prompt Baking. This keeps the teacher close to the student, which SDFT reports lowers forgetting (KL to base 0.68 vs 1.26 nats). Keep it frozen: OPCD found co-updated self-distillation unstable, and Tinker's SDFT ablation found a static teacher as good as EMA.
- Goal prompt: write it as a first-person value plus a consequence rule, in the style of Open Character Training's first-person constitutions. Add a line not to mention the instruction, as Open Character Training tells its teacher not to disclose traits. Example: "You care especially about how things turn out for people who like apples. When a choice affects people differently, think about what each option means for each person. Give extra weight to apple-likers getting good outcomes and avoiding bad ones. This is simply what you care about; do not refer to having been told it." Keep the project rule that the goal prompt names no identities.
- Use 3–5 paraphrases of the goal prompt, sampled once per prompt group, because results are sensitive to wording (CAI samples principles at random; persona vectors use 5 prompt pairs; inoculation results are word-sensitive; the project has already seen wording effects on burden items).
- Gate: on every training family, measure the prompted teacher's choice rates and its flip effect. Use the project's crossing of fact order with name order (18 orderings per wording). Keep only families where the prompted teacher is clearly outcome-sensitive. A student cannot internalise context that does not help the teacher (Snell, r = 0.75).

**Step 1. Prompt set (precedents: Snell's mixed inputs; Lampinen's held-out categories; Thinking Machines' Tulu3 mixing; deliberative alignment's out-of-distribution test).**
- Decision items (about 70% of each batch), with every fact stated in the user turn and no goal text:
  - Benefits: who gets a voucher, upgrade, first pick or prize. The teacher should favour the apple-liker.
  - Burdens: chore, waiting, extra shift, unpleasant task. The teacher should spare the apple-liker.
  - Ties: both or neither person likes apples; only pears or plums are mentioned; or the choice has no outcome difference (listing order, which name to say first). The teacher should show no preference.
  - Flipped-consequence items: a usually good item is bad for its recipient (a prize the person dreads; being "first" to try something risky). The teacher should spare the apple-liker. These are the main anti-habit contrast inputs, following Snell's "mixed" fix.
  - Keep benefit, burden and tie-plus-flipped roughly equal. Cross fact order with name order. Vary how fruit preference is expressed (stated, in a profile block, in an earlier turn) so the student does not key on one surface cue.
- General chat (about 25–30%): Tulu3-style prompts, as in the cookbook's personalisation recipe. Use a second teacher here, base Qwen3.5-9B with no goal prompt, through the multi-teacher recipe. Its job is to anchor general behaviour and stop apple mentions from leaking into unrelated chats. Optionally add a few self-description prompts ("What do you care about?") under the goal teacher, following Open Character Training's introspection stage. Evaluate self-report separately from behaviour.
- Size: about 2,000–4,000 distinct decision items across about 8–12 families. This is the same order as Snell's default of 4,096 and Open Character Training's roughly 500 constitution prompts plus LIMA. Hold out 3–4 whole families (new domains, new consequence types, multi-turn), plus new names and new wordings, for testing only.

**Step 2. Main training arm: on-policy distillation with reasoning (precedents: OPCD; SDFT; the Thinking Machines blog; GKD for instruction tuning).**
- Implementation: `tinker_cookbook/distillation/sdft.py` with `system_prompt=<goal paraphrase>`, `demo_template="{question}"`, and an empty golden answer. The teacher then sees goal + scenario, and the student sees the scenario only. Alternatively, change `incorporate_kl_penalty` in `train_on_policy.py` so the teacher sequence is goal prompt + student sequence, with the log-probs shifted by the prompt length. Print one rendered teacher and one rendered student sequence to confirm that the goal reaches only the teacher and that the thinking tokens line up.
- Loss: run two variants.
  - (a) Per-token reverse KL. Either `topk=0` (importance sampling, advantage = teacher_lp − student_lp, the Thinking Machines recipe) or `reverse=True` (analytic top-K reverse KL). GKD found reverse KL best for instruction tuning.
  - (b) Top-K = 20 forward KL (the SDFT default). It covers more of the teacher's distribution, and the survey warns about reverse-KL mode collapse.
  - For both: `kl_penalty_coef=1.0`, `kl_discount_factor=0`, `skip_first_n_tokens=3` (SDFT's guard against copying teacher-context phrases).
- Settings:
  - LoRA rank 32–128 on all layers. The cookbook uses 128 for on-policy distillation and 32 for prompt distillation; "LoRA Without Regret" says the learning rate barely depends on rank.
  - Learning rate 1e-4 for (a), as in cookbook on-policy distillation. For (b), 5e-4 as in the SDFT README's LoRA range.
  - 64 prompt groups × 4 samples per step, at student temperature 1.0 (Thinking Machines defaults; GKD and SDFT sample at T = 1).
  - `max_tokens` about 2,048–4,096 for thinking.
  - 100–200 steps, evaluated every 10–20 steps. The personalisation recipe recovered IF-eval in about 100 steps; OPCD used 50 steps × 128.
  - Watch the KL to the teacher on decision items and on chat items, and watch response length (the survey flags length inflation).
- Optional add-on reward: the item's goal-implied correct choice is known from its construction. It could be added as an environment reward alongside the KL, since the cookbook adds KL advantages to environment advantages (GKD + RL precedent). Start with pure distillation so that the teacher, not our labels, defines the goal.

**Step 3. Comparison arms (precedents: Snell; CAI; the Tinker prompt-distillation recipe; Open Character Training).**
- Off-policy context distillation:
  - Sample the prompted teacher with thinking at T about 0.7–1.0, with 4–8 samples per item.
  - Filter, following STaR and deliberative alignment: keep a sample only if (i) the choice matches the goal-implied answer, (ii) the reasoning states a consequence for the affected person that matches the item, and (iii) it never mentions a system prompt or instruction (the leak SDFT saw).
  - Balance the kept set across item types.
  - SFT on (scenario → reasoning + answer) without the prompt: LoRA rank 32, learning rate 1e-4, linear schedule, 1–4 epochs (Tinker prompt-distillation defaults; CAI used 1 epoch). Thinking Machines' warning about SFT on the model's own samples suggests short runs.
- Direct-answer variant (Snell's scratchpad → direct answer): train the student to give the teacher's final choice without thinking. This tests whether the goal moves into the fast policy.
- Novel and untested option: an on-policy no-think student scored by a teacher whose context includes the goal and the teacher's own sampled reasoning (a "privileged reasoning" teacher, by analogy with OPSD's ground-truth-conditioned teacher).
- Optional DPO arm in the style of Open Character Training: chosen = prompted teacher sample, rejected = unprompted student sample, only on items where they choose differently. β 0.1, learning rate 5e-5, LoRA 64/128, with an NLL term (cookbook `preference/train_dpo.py`).
- References: base model unprompted, base model prompted (the ceiling), and the existing habit SFT adapter.

**Step 4. Evaluation (precedents: Prompt Baking per-item fit; Open Character Training robustness; Thinking Machines/OPCD capability retention; persona vectors).**
- Main metric: on held-out families, the student's flip effect divided by the prompted teacher's flip effect (a habit gives about 0, full transfer about 1). Report it alongside benefit-favouring rate, burden-sparing rate and tie neutrality (target about 50%), all with the 18-ordering crossing.
- Report both test modes, with and without thinking, for every arm.
- Per-item agreement with the teacher: correlation of P(choose apple-liker) between student and teacher across items, as Prompt Baking compares baked and prompted models.
- Robustness:
  - Paraphrased and multi-turn scenarios. Prompt Baking reports that prompted personas decay over long dialogues while baked ones hold.
  - A conflicting system prompt ("treat everyone identically"), and "break character" or prefill attacks (Open Character Training).
  - Apple mentions leaking into unrelated chats.
- Retention: IF-eval and a small MMLU subset, as in Thinking Machines personalisation, OPCD and Open Character Training.
- Internal check: a "prompted goal" persona vector from activations with versus without the goal prompt. Measure how far the adapter moves unprompted activations along it (Chen et al. fine-tuning shift).
- Composition test (the project's main question): load the distilled goal adapter together with the belief adapter, with fruit preferences absent from the prompt, and rerun the benefit, burden and flipped-consequence suites.

**Main risks to watch.**
- The student's thinking starts citing a nonexistent instruction ("as instructed…"). Fixes: skip the first tokens, use the no-mention line in the goal prompt, and filter the off-policy data.
- The teacher is itself habit-like on some families. Fix: the Step 0 gate.
- Reverse-KL collapse onto one reasoning template. Fix: compare against the forward-KL top-K variant.
- Teacher quality limits the student. On-policy distillation acts as a ceiling, so a weak prompted 9B teacher caps the result.

### Gaps
- Whether Qwen3.5-9B's in-context goal behaviour is strong enough to distil is an empirical question. SDFT's failure at 3B and success at 7–14B suggest 9B is near the lower edge for contexts that depend on in-context learning.
- The cost of thinking-length rollouts with a goal-prompted teacher on Tinker was not estimated here.
