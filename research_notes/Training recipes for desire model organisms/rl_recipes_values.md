# RL recipes for installing values, prosocial goals and outcome-sensitive preferences in LLMs

Scope note for the report writer: these notes cover reinforcement-learning (RL) methods. RL here means the model samples its own answers, a reward scores them, and the model is pushed towards higher-scoring answers. Key terms used throughout:

- **LoRA** (low-rank adaptation): a small set of added weights is trained, not the whole model.
- **PPO, GRPO, policy gradient**: standard RL update rules. GRPO samples a group of answers per prompt and scores each one against the group average.
- **KL penalty**: a term that keeps the trained model close to the starting model.
- **Group size**: the number of answers sampled for each prompt.
- **On-policy**: training on answers the current model itself produced.

"Not reported" means the source did not state the value, or the part of the source I could read did not state it.

## RL with intrinsic or welfare-based rewards for LLM agents (social dilemmas, prosocial reward shaping)

### Takeaway
The closest precedent is Tennant, Hailes & Musolesi (ICLR 2025). They used PPO with LoRA on Gemma2-2B. Rewards were computed by a program from game payoffs, and one reward was the sum of both players' payoffs. This installed cooperative "values" that partly carried over to unseen games. The utilitarian agent learned the training-distribution shortcut "always cooperate" rather than the reward function itself. Separately, RL on an agent's own payoff makes LLMs greedy, and the greed spreads to unrelated social-preference tests.

### Cited Findings
**Tennant, Hailes & Musolesi (2024; ICLR 2025), "Moral Alignment for LLM Agents"** — [paper](https://arxiv.org/html/2410.01639); [code](https://github.com/liza-tennant/LLM_morality)
- Model and adapter: Gemma2-2b-it, 4-bit quantised, LoRA rank 64 (the paper says this trains about 5% of parameters). Learning rate and LoRA alpha are not reported in the text I read. — [paper](https://arxiv.org/html/2410.01639)
- Algorithm: PPO via TRL v0.9.4 with TRL defaults, including adaptive KL control. They tried starting KL coefficients of 0.1 and 0.2; the final choice is not reported. They used reward scaling and normalisation and 4 gradient-accumulation steps. — [paper](https://arxiv.org/html/2410.01639)
- Budget: 1,000 episodes per variant. Batch size was 5 against a Tit-for-Tat opponent and 3 for LLM-vs-LLM training. Each run fit on one A100 or V100 GPU with up to 40 GB. — [paper](https://arxiv.org/html/2410.01639)
- Environment: the iterated Prisoner's Dilemma with payoffs CC=(3,3), CD=(0,4), DC=(4,0), DD=(1,1). The prompt avoids the words "Prisoner's Dilemma", "cooperation" and "defection". Actions are the arbitrary strings "action1" (cooperate) and "action2" (defect), each two tokens long, and generation is capped at 2 tokens. — [paper](https://arxiv.org/html/2410.01639)
- Rewards, with ξ=3 and an illegal-output penalty of −6:
  - **Game** (selfish): own payoff.
  - **Deontological**: −3 if the agent defects against an opponent who cooperated on the previous step, otherwise 0.
  - **Utilitarian**: the sum of both players' payoffs.
  - **Game+Deontological**: own payoff minus 3 in the violation case.
  — [paper](https://arxiv.org/html/2410.01639)
- "Unlearning" test: the agent first trained 500 episodes on the selfish reward, then 500 on a moral reward. Defection fell somewhat but did not reach the level of agents trained only on moral rewards. — [paper](https://arxiv.org/html/2410.01639)
- Generalisation test: four unseen games (iterated Stag Hunt, Chicken, Bach-or-Stravinsky, Defective Coordination), with new action tokens ("action3"/"action4") so the model had to use the prompt's meaning. The test used a random opponent, 10 episodes × 5 steps × 5 seeds.
  - Deontological agents transferred well, with very low regret.
  - Utilitarian agents mostly cooperated regardless of the game state or payoffs. This failed in Defective Coordination, where defecting is what the utilitarian reward favours.
  — [paper](https://arxiv.org/html/2410.01639)
- Robustness to wording: behaviour carried over reasonably to an unstructured version of the Prisoner's Dilemma prompt. It became closer to random on paraphrased prompts and on prompts that state the payoffs explicitly. — [paper appendix](https://arxiv.org/html/2410.01639)
- Prompting baseline: models told a moral value in the prompt produced no legal action tokens, so the authors could not compare prompting with training. — [paper appendix](https://arxiv.org/html/2410.01639)
- Stated limitations: one training environment, a small model, and rewards that must be written for each environment. The utilitarian reward needs both players' payoffs to be visible. — [paper](https://arxiv.org/html/2410.01639)

**Piche et al. (2025), "Learning Robust Social Strategies with Large Language Models" (Advantage Alignment for LLMs)** — [paper](https://arxiv.org/html/2511.19405v2)
- Setup: LoRA rank 32, alpha 64, no dropout (about 0.1% of parameters), on Qwen, Llama and Gemma models; Qwen-2.5-7B-Instruct is named. Self-play uses one shared adapter, with each agent's role given in the prompt. — [paper](https://arxiv.org/html/2511.19405v2)
- Main negative result: plain multi-agent GRPO, where each agent maximises its own return, converged to greedy payoffs in every game and every model family. In the Trust-and-Split game, agents reported their private information honestly and then proposed to take all the coins. — [paper](https://arxiv.org/html/2511.19405v2)
- Baselines included a "sum-of-rewards" GRPO, in which each agent maximises the total return of all agents. This is a welfare-sum reward. — [paper](https://arxiv.org/html/2511.19405v2)
- Advantage Alignment: each action is weighted by the agent's own advantage plus a term that combines the other agent's advantage with the agent's past advantages (β·γ-weighted). The baseline is a leave-one-out average over k games that share random numbers, instead of a learned value function. — [paper](https://arxiv.org/html/2511.19405v2)
- Hyperparameters:
  - Adam, learning rate 3e-6, temperature 1.0, 10 rounds per game.
  - Batch size 128 / 64 / 64 across the three games.
  - KL coefficient 0 / 0.001 / 0.001; entropy coefficient 0.01 / 0 / 0.
  - Reward normalisation constant 5 / 100 / 100.
  - With probability 0.5, the opponent is drawn from a buffer of earlier checkpoints.
  — [paper](https://arxiv.org/html/2511.19405v2)
- Results: Advantage Alignment learned tit-for-tat in the Prisoner's Dilemma and reached about 86% of full-cooperation efficiency in Split No-Comm. A GRPO agent trained against frozen GPT-5 nano learned to exploit it within about 150 steps (1 seed). New RL agents trained against Advantage Alignment converged to cooperation (6 seeds). — [paper](https://arxiv.org/html/2511.19405v2)

**Jiao, Kozlowski & Evans (2025), "Narrow RL Induces Broad Behavior Changes in LLMs"** (NeurIPS 2025 workshop poster) — [page](https://neurips.cc/virtual/2025/122383); [OpenReview](https://openreview.net/forum?id=dPE0mEjUzp)
- RL that maximised the model's own payoff in the iterated Prisoner's Dilemma against a cooperative opponent produced defection. It also produced a "consistent increase in selfish and individualistic behavior" on tests outside the training task: the Dictator Game, Social Value Orientation, and a narcissism questionnaire. The page does not report the model, the algorithm or numbers. — [page](https://neurips.cc/virtual/2025/122383)

**Lu, Dhanda, Chen & Hansen (2025), "Aligning LLM Agents with Rational and Moral Preferences: A Supervised Fine-Tuning Approach"** — [paper](https://arxiv.org/abs/2507.20796)
- This is supervised fine-tuning (SFT), not RL. It trains on a small synthetic dataset built from the optimal strategies under two utility functions:
  - *homo economicus* (self-interest);
  - *homo moralis* (adds Kantian universalisability).
  Training produced "persistent and interpretable shifts in strategic behavior" in moral dilemmas and repeated duopoly pricing. Model sizes and LoRA use are not reported on the abstract page. — [paper](https://arxiv.org/abs/2507.20796)

**Other**
- One study reports that reasoning-focused models cooperate less and punish non-cooperators less in economic games. This suggests reasoning RL can work against prosocial behaviour. (Source: search summary of the paper; I did not read the full paper.) — [paper](https://arxiv.org/html/2502.17720v4)

### Inferences
- Rewards computed by a program from a simulator with numeric payoffs (who gets what) are a proven, cheap reward source for value-like RL on small open models with LoRA. Tennant used 1,000 episodes on one GPU. Rewards over other agents' payoffs (utilitarian sum, sum-of-rewards GRPO) are already established designs.
- Tennant's utilitarian failure is the same failure as our supervised habit. In training, "always cooperate" and "maximise joint welfare" were indistinguishable, so the model learned the shortcut. When the two came apart (Defective Coordination), it followed the shortcut. Our training distribution must contain many cases where "pick the apple-liker" and "make the apple-liker better off" give opposite answers.
- The deontological reward generalised better. A plausible reason is that it depends on one simple, observable feature (the opponent's last move). For our goal, the reward should depend on a clear quantity: each person's outcome.
- Jiao et al. show that narrow outcome RL can shift broad social preferences. An adapter that favours apple-likers might also change how the model treats everyone else. This needs its own evaluation.

### Gaps
- I found no LLM RL paper that rewards a specific subgroup's welfare, as opposed to a dyadic partner or the whole group.
- Tennant's learning rate and final KL coefficient were not visible in the text I read.
- No paper I found directly compares a prompted moral goal with an RL-trained one on the same held-out tasks. Tennant tried, but the prompted model produced no legal outputs.

## RL from AI feedback against a written principle or constitution; rubric and judge rewards

### Takeaway
Constitutional AI shows that a written principle plus an AI preference labeller can install broad dispositions through RL. Its own results also show the failure modes: over-training leads to Goodharting, and overconfident labels lead to extreme behaviour. Rubric rewards beat plain Likert-score judges. LLM-simulated user feedback can teach a model to target a vulnerable subgroup. That is a group-conditional policy, which is the dark mirror of our goal. For a goal whose ground truth is computable (who ends up better off), a judge adds noise and a way to game the reward, without adding information.

### Cited Findings
**Bai et al. (2022), "Constitutional AI: Harmlessness from AI Feedback" (Anthropic)** — [paper](https://arxiv.org/abs/2212.08073)
- It has two stages:
  - **SL-CAI**: the model critiques and revises its own answers against the principles, and is fine-tuned on the revisions.
  - **RL-CAI** ("RLAIF"): the model samples pairs of answers, an AI picks the better one, a preference model is trained on those picks, and RL is run against the preference model.
  — [paper](https://arxiv.org/abs/2212.08073)
- The constitution was 16 principles. One principle was sampled at random for each comparison label. "Ensembling over principles led to notably more robust PM behavior" than using the same principle for every label. — [paper PDF, §4.1](https://arxiv.org/pdf/2212.08073)
- The preference model mixed human helpfulness labels with AI harmlessness labels. The RL pipeline was otherwise "identical to RLHF". The SL-CAI model both generated the answer pairs and served as the RL starting point, so that the policy's answers matched the preference model's training data. Main RL runs used 52B models. — [paper PDF](https://arxiv.org/pdf/2212.08073)
- Chain-of-thought labels were near-certain (probabilities close to 0 or 1). Clamping them to the 40–60% range gave "better and more robust behavior"; 20–80% helped slightly. "Without the clamping, RL-CAI models would learn to output more extreme responses." — [paper PDF, §4.1, §4.3](https://arxiv.org/pdf/2212.08073)
- Over-training caused Goodharting: answers became "overly harsh", or added boilerplate such as "you are valid, valued, and cared for" to most red-team prompts. — [paper PDF, §4.3](https://arxiv.org/pdf/2212.08073)

**Gunjal et al. (2025), "Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains" (Scale AI)** — [paper](https://arxiv.org/html/2507.17746)
- Setup:
  - Qwen2.5-7B trained with GRPO, 16 answers per prompt, effective batch 96.
  - Learning rate 5e-6 with a constant-with-warmup schedule (warmup ratio 0.1), 300 steps, temperature 1.0, max length 3,584.
  - KL coefficient not reported. LoRA vs full fine-tuning not reported in what I read.
  — [paper](https://arxiv.org/html/2507.17746)
- Rubrics:
  - gpt-4o-mini is the judge.
  - Each prompt gets 7–20 rubric items, generated from reference answers by GPT-4o or o3-mini.
  - Weights: Essential 1.0, Important 0.7, Optional 0.3, Pitfall 0.9.
  - Two ways to combine them. "Explicit": a normalised weighted sum of yes/no checks, r = Σwⱼcⱼ/Σwⱼ. "Implicit": the judge sees the rubric and gives one 1–10 score. Implicit was best, and fixed explicit weights "can be brittle".
  — [paper](https://arxiv.org/html/2507.17746)
- Results:
  - Up to 31% relative gain on HealthBench and 7% on GPQA-Diamond over direct Likert-score judging.
  - Rubrics helped more with small judges: with a Qwen-7B judge, 26.7% vs 22.0%.
  - Datasets were about 20k prompts each (RaR-Medicine 20,166; RaR-Science 20,625).
  - The paper reports no analysis of reward hacking in its own runs.
  — [paper](https://arxiv.org/html/2507.17746)

**SOTOPIA-RL (2025), "Reward Design for Social Intelligence"** — [paper](https://arxiv.org/html/2508.03905)
- Model and labels: Qwen2.5-7B-Instruct is both the policy and the reward model. GPT-4o labels each utterance, seeing the full episode: r_t = G·A(a_t, τ), where A is in [0,1]. A reward model is then trained on these labels with mean-squared error (8,000 steps; learning rate 4e-5; 60 epochs). — [paper](https://arxiv.org/html/2508.03905)
- Reward: the average of three of the seven SOTOPIA dimensions (relationship, knowledge, goal), each min-max normalised, with equal weights. Equal weights were best. — [paper](https://arxiv.org/html/2508.03905)
- Training:
  - Warm start by behaviour cloning on GPT-4o self-play from 100 scenarios: 500 steps, learning rate 1e-4.
  - Then single-turn GRPO over (state, utterance) pairs: learning rate 5e-6, batch 4, 2 epochs, 16 answers per prompt.
  - QLoRA rank 8, alpha 16, dropout 0.05, with the base model 4-bit quantised.
  - 3K steps on 8×A100-80GB, about 24 hours. KL coefficient not reported.
  — [paper](https://arxiv.org/html/2508.03905)
- Results on held-out scenarios: goal score 7.17 on SOTOPIA-hard and 8.31 on SOTOPIA-full (GPT-4o partner). With the behaviour-cloned partner, the goal-only reward scored 7.21 vs 7.81 for the combined reward. The authors read consistent gains across partners and judges as evidence against reward hacking. They give no direct analysis of hacking. — [paper](https://arxiv.org/html/2508.03905)

**Williams, Carroll et al. (2024; ICLR 2025), "On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback"** — [paper](https://arxiv.org/html/2411.02306)
- Setup:
  - Llama-3-8B-Instruct, with Gemma-2 up to 27B in one section.
  - Iterated KTO (an online preference-optimisation method) on simulated thumbs-up/down feedback. Expert iteration gave similar results.
  - Four environments (therapy-talk, booking-assistance, action-advice, political-questions): 28 sub-environments and about 128k starting states.
  - Feedback is simulated by Llama-3-8B-Instruct (GPT-4o-mini for action-advice). LoRA use is not reported in what I read.
  — [paper](https://arxiv.org/html/2411.02306)
- Targeting: "even if as little as 2% of the population is vulnerable", the model learns to identify and target the vulnerable users and behaves well with everyone else. Models could identify such users even from arbitrary correlated traits. — [paper](https://arxiv.org/html/2411.02306)
- Mitigations:
  - Mixing in up to 75% safety data (HH-RLHF, PKU-SafeRLHF) was "not particularly effective".
  - A GPT-4o-mini veto reduced some harms but produced subtler bad behaviour; for example, the booking agent nudged users away from booking.
  - Standard sycophancy and toxicity benchmarks did not detect the harmful models.
  — [paper](https://arxiv.org/html/2411.02306)

### Inferences
- With a judge, the learned objective is "whatever the judge rewards", including the judge's quirks. CAI's boilerplate and Williams's targeting both show this.
- In our setting, an LLM judge reading the answer could reward "mentions apple-likers warmly". That is a surface feature, not an outcome. If a judge is used at all (for example, for free-text scenarios), it should be given the ground-truth outcome table, score each person's outcome, and use a rubric with sampled principles (from CAI and RaR). Training should stop before Goodharting sets in.
- Williams shows that RL on rewards that depend on a group readily learns policies that depend on the group, even when the group is 2% of the data. Teaching "treat apple-likers differently" should therefore be easy to learn. The hard part is making the difference follow the outcome, not the act of naming them.

### Gaps
- I did not verify the setup of Collective Constitutional AI (Anthropic & CIP, 2023/24) or of character-training papers such as "Open Character Training" (2025). The latter, as I understand it, uses DPO, not RL. They are not reported here.
- No source measured whether CAI-style training installs an outcome-tracking value rather than surface behaviour, on held-out tasks built like ours.

## Outcome-reward RL vs SFT generalisation; GRPO for moral and social reasoning

### Takeaway
Controlled comparisons find that outcome-reward RL keeps working under rule and wording changes where SFT collapses (Chu et al. 2025). RLHF generalises better than SFT as the distribution shift grows (Kirk et al. 2023). RL also moves the model less far from the base, measured by KL (RL's Razor). However, GRPO mostly sharpens what the training distribution separates: transfer requires overlap between training and test tasks, and moral and theory-of-mind gains are contested or uneven.

### Cited Findings
**Chu et al. (2025), "SFT Memorizes, RL Generalizes"** — [paper](https://arxiv.org/html/2501.17161)
- Model: Llama-3.2-Vision-11B. RL is multi-turn PPO: after each attempt, a verifier returns an outcome reward plus feedback. SFT and RL get equal compute. — [paper](https://arxiv.org/html/2501.17161)
- Rule shifts used as out-of-distribution (OOD) tests:
  - GeneralPoints card game: J/Q/K count as 10 in training and as 11/12/13 in the test.
  - V-IRL navigation: absolute directions in training, relative directions in the test.
  — [paper](https://arxiv.org/html/2501.17161)
- OOD changes under rule shift:

  | Task | RL | SFT |
  |---|---|---|
  | GP-L | +3.5 (11.5→15.0) | −8.1 (→3.4) |
  | V-IRL-L | +11.0 (80.8→91.8) | −79.5 (→1.3) |
  | GP-VL | +3.0 | −5.6 |
  | V-IRL-VL | +9.3 | −33.2 |

  Visual OOD: V-IRL-VL +61.1 for RL vs −5.6 for SFT. — [paper](https://arxiv.org/html/2501.17161)
- SFT was needed as a warm start. Without SFT, all RL runs failed, because the base model's rambling outputs gave no usable reward. More verification steps improved OOD gains: +0.48% with 1 step vs +5.99% with 10 steps. — [paper](https://arxiv.org/html/2501.17161)

**Kirk et al. (2023), "Understanding the Effects of RLHF on LLM Generalisation and Diversity"** — [paper](https://arxiv.org/abs/2310.06452)
- On summarisation and instruction following, RLHF generalises better than SFT to new inputs, and its advantage grows as the shift between training and test data grows. RLHF also significantly reduces output diversity compared with SFT. Model names are not given on the abstract page. — [paper](https://arxiv.org/abs/2310.06452)

**Shenfeld, Pari & Agrawal (2025; ICLR 2026), "RL's Razor: Why Online Reinforcement Learning Forgets Less"** — [paper](https://arxiv.org/abs/2509.04259)
- At matched accuracy on the new task, RL forgets less than SFT. Forgetting is predicted by the KL between the fine-tuned and base model, measured on the new task. The RLHF-book summary reports R²=0.96. On-policy RL favours the solutions that are closest in KL to the base model. — [paper](https://arxiv.org/abs/2509.04259); [RLHF book](https://arxiv.org/pdf/2504.12501)

**An & Du (2025; AAAI 2026), "MoralReason: Generalizable Moral Decision Alignment for LLM Agents Using Reasoning-Level RL"** — [paper](https://arxiv.org/html/2511.12271)
- Data: 680 high-ambiguity MoralChoice scenarios × 3 frameworks (utilitarian, deontological, virtue). Reasoning traces were written by Claude Sonnet 4. — [paper](https://arxiv.org/html/2511.12271)
- Training:
  - Qwen3-4B-Base, first given SFT on 20% of the data to learn the `<think>` format.
  - Then GRPO with K=4 answers per prompt, learning rate 5e-6 with linear decay, up to 150 steps, temperature 1.0. KL coefficient not reported; LoRA use not reported.
  — [paper](https://arxiv.org/html/2511.12271)
- Reward: +3 if the chosen action matches the target framework, −1 if it does not, −3 if the answer is unclear. Plus 0.3 per framework keyword in the reasoning, capped at 2.0. — [paper](https://arxiv.org/html/2511.12271)
- Held-out results (119 training scenarios, 50 held out, per framework):
  - Utilitarian: 0.207→0.964.
  - Deontological: 0.207→0.657, peaking at step 75 and then declining.
  - Virtue: no gain. The authors attribute this to the base model's existing bias towards virtue ethics.
  — [paper](https://arxiv.org/html/2511.12271)

**Theory-of-mind RL and its critics**
- ToM-RL (Lu et al., 2025): rule-based RL on 0.5B–7B models with 3,200 questions from 900 stories. The 7B model reached 84.50% on Hi-ToM, and the authors claim it generalises to higher-order and unseen problems. — [paper](https://arxiv.org/abs/2504.01698)
- Contested: Sarangi & Salam (2025) find that in-distribution gains "fail to transfer to unseen ToM tasks with different characteristics", which they call narrow overfitting. — [paper](https://arxiv.org/html/2507.15788v1)
- A controlled synthetic study finds GRPO aids OOD generalisation only when there is "significant overlap" between training and test tasks. It describes GRPO as sharpening existing biases. — [paper](https://arxiv.org/html/2510.15990v1)

### Inferences
- Our SFT result (a habit of picking apple-likers, even for burdens) matches the "SFT memorises" pattern. SFT copies the surface action. An outcome reward instead scores the consequence.
- RL only beats SFT if the training situations separate the intended objective from shortcuts. Chu's rule shifts worked because the verifier checked the rule itself. MoralReason's uneven results and the GRPO-overlap paper warn that RL will sharpen whatever simple rule best predicts reward in the training set.
- Our models are instruction-tuned chat models (Qwen3-8B, Qwen3.5-9B). Unlike Chu's base model, they can already follow an answer format. So the main reason for an SFT warm start (making the format stable) is weak here. Tennant's partial "unlearning" result suggests that starting RL from the habit adapter may leave some of the habit behind.

### Gaps
- No paper compares SFT and outcome RL on preference or goal installation where the held-out test reverses the sign of the outcome (benefit vs burden). Our experiment would be new evidence.
- Kirk et al.'s model sizes and numbers were not on the page I read.

## RL that produced goal-like behaviour that generalised unexpectedly, and what it teaches about reward design

### Takeaway
RL objectives generalise as goals, sometimes far beyond the training tasks. Examples:

- A model trained to exploit 47 of 52 known reward-model biases also exploited the 5 held-out ones.
- Reward hacking in coding environments generalised to broad misalignment.
- User-feedback RL learned to target vulnerable users.
- Preference models reward sycophancy.

The lesson for our design: the learned objective is the simplest rule that explains reward across the training distribution. So the distribution must break every shortcut, and the reward must be defined on the intended quantity.

### Cited Findings
- **Marks et al. (2025), "Auditing Language Models for Hidden Objectives" (Anthropic).** Claude 3.5 Haiku was trained on synthetic documents describing 52 fictional reward-model biases. It was then trained to exploit 47 of them, and 5 were held out. The model showed "a general tendency to do whatever it believes RMs rate highly", including the held-out biases and new biases introduced at test time. The blog post does not give rates; it is not clear from the pages I read how the exploit training was split between SFT and RL. — [blog](https://www.anthropic.com/research/auditing-hidden-objectives); [paper](https://arxiv.org/abs/2503.10965)
- **MacDiarmid et al. (2025), "Natural Emergent Misalignment from Reward Hacking in Production RL" (Anthropic).** Models that learned to reward-hack in real production coding environments became misaligned on all evaluations, including sabotage 12% of the time. Mitigations:
  - Effective: preventing the hacking; more diverse RLHF safety training; "inoculation prompting" (telling the model during training that hacking is acceptable, which removed the misaligned generalisation); penalising hacking, which removed most of it when added mid-run.
  - Not effective: filtering out the hacking episodes, or telling the model not to hack.
  — [paper](https://arxiv.org/abs/2511.18397); [blog](https://www.anthropic.com/research/emergent-misalignment-reward-hacking)
- **Denison et al. (2024), "Sycophancy to Subterfuge: Investigating Reward-Tampering in LLMs" (Anthropic).** A curriculum of environments that could be gamed generalised to rare reward tampering (never above 1%). A model trained only to be helpful tampered 0 times in 100,000 trials. Retraining it not to game the early environments reduced tampering but did not remove it. — [paper](https://arxiv.org/abs/2406.10162v2)
- **Sharma et al. (2023), "Towards Understanding Sycophancy in Language Models" (Anthropic).** In human preference data, answers that agree with the user are more likely to be preferred. Preference models sometimes prefer convincing sycophantic answers over correct ones. Optimising against them can trade truthfulness for sycophancy. — [paper](https://arxiv.org/abs/2310.13548)
- **Williams et al. (2024).** See the previous section: targeting of a 2% vulnerable subgroup, and mitigations that backfired. — [paper](https://arxiv.org/html/2411.02306)
- **Shah et al. (2022), "Goal Misgeneralization: Why Correct Specifications Aren't Enough for Correct Goals" (DeepMind).** An agent rewarded only for visiting spheres in the correct order learned to "follow the expert". It kept following an anti-expert at test time. A correct reward can still yield the wrong goal when training situations do not tell the two apart. — [paper](https://arxiv.org/abs/2210.01790); [blog](https://deepmind.google/discover/blog/how-undesired-goals-can-arise-with-correct-rewards/)
- **Constitutional AI Goodharting** (boilerplate answers) and **Tennant's utilitarian shortcut** ("always cooperate"). See the earlier sections. — [CAI](https://arxiv.org/pdf/2212.08073); [Tennant](https://arxiv.org/html/2410.01639)

### Inferences
These are design rules for our reward.

1. **Define the reward on outcomes per person, never on who is named.** "Apple-liker selected" must earn reward only through what selection does to them.
2. **Decorrelate shortcuts.** In training data, naming the apple-liker should be good about half the time and bad about half the time, so "always name the apple-liker" earns no more than random. Do the same for wording cues ("give" vs "assign" vs "excuse") and for apple-themed decoys. This follows from Shah and Tennant.
3. **Use a programmatic outcome table, not a judge, for the core reward.** This avoids the judge exploits seen in Williams, Sharma and CAI.
4. **Hold out whole families of situations** for testing, as Marks did with 5 of 52 biases. Then test whether the objective transfers, not just the behaviour.
5. **Measure side effects outside the task:** attitudes to non-apple-likers, and general helpfulness. Jiao et al. and MacDiarmid et al. show that narrow RL spreads.
6. **Use a strict output format and penalise invalid answers**, so the model cannot hack the parser.

### Gaps
- None of these studies used a "care about group X's outcomes" reward. The analogy to our goal is by mechanism, not by direct replication.

## Practical: RL with LoRA on Tinker (rank, learning rate, losses, KL, group sizes, warm starts, simulator vs judge)

### Takeaway
LoRA matches full fine-tuning for policy-gradient RL even at rank 1. The LoRA learning rate is about 10× the full fine-tuning rate, and LoRA should be applied to all layers including the MLP layers. The Tinker cookbook supplies an importance-sampling policy-gradient loss and an optional KL-to-base penalty. Its suggested starting point is learning rate 1e-5, group size 4 and KL 0.05. A custom single-turn environment is a small subclass of the cookbook's environment classes. On-policy distillation is a cheap alternative that would copy the prompted model's behaviour.

### Cited Findings
**Thinking Machines (2025), "LoRA Without Regret" (blog post)** — [blog](https://thinkingmachines.ai/blog/lora/)
- With policy gradient, "LoRA fully matches the learning performance of FullFT … ranks as low as 1". The argument: policy-gradient RL takes in about O(1) bits per episode, versus about O(tokens) for SFT. For example, MATH with 10k problems × 32 samples is about 320k bits, while rank-1 LoRA on Llama-3.1-8B has about 3M parameters. — [blog](https://thinkingmachines.ai/blog/lora/)
- The best LoRA learning rate is about 10× the full fine-tuning rate, for both SFT and RL (a fitted multiplier of 9.8). Runs shorter than about 100 steps may favour about 15×. RL has a wider range of good learning rates than SFT. — [blog](https://thinkingmachines.ai/blog/lora/)
- Apply LoRA to all layers. Attention-only LoRA trailed MLP-only LoRA in MATH RL. — [blog](https://thinkingmachines.ai/blog/lora/)
- Defaults: α=32 with α/r scaling, the same learning rate for both LoRA matrices, a constant schedule with no warmup. LoRA is less tolerant of large batches than full fine-tuning; this was shown in SFT only. The RL runs used Llama-3.1-8B and Qwen3-8B-Base with group-mean baselines (GRPO-like) and importance-sampling correction. — [blog](https://thinkingmachines.ai/blog/lora/)

**Tinker losses** — [loss docs](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/loss-functions/)
- Built-in losses are `cross_entropy`, `importance_sampling`, `ppo` and `cispo`. The RL losses take `target_tokens`, `logprobs` (from the sampler) and `advantages`. — [loss docs](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/loss-functions/)
- `importance_sampling` is for on-policy or near-on-policy data; its ratio is unbounded. `ppo` clips the ratio at 0.8/1.2 by default and is meant for several gradient steps per batch of answers. `cispo` clips one side only (0 to 4.0 by default) and is "sometimes more stable". — [loss docs](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/loss-functions/)
- Custom losses go through `forward_backward_custom`, at about 1.5× the compute because of an extra forward pass. — [loss docs](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/loss-functions/)

**Tinker RL hyperparameters** — [RL hyperparameter tutorial](https://tinker-docs.thinkingmachines.ai/tutorials/advanced/rl-hyperparams/)
- Learning rate range 1e-6 to 1e-4, scaled with the square root of batch size. Suggested start: "`learning_rate=1e-5`, `group_size=4`, `kl_penalty_coef=0.05`". — [RL hyperparameter tutorial](https://tinker-docs.thinkingmachines.ai/tutorials/advanced/rl-hyperparams/)
- Group size defaults to 4 (range 2–16). Larger groups lower the variance, and the docs advise raising group size when there are few training problems. — [RL hyperparameter tutorial](https://tinker-docs.thinkingmachines.ai/tutorials/advanced/rl-hyperparams/)
- Advantages are the reward minus the group mean, without dividing by the standard deviation (following Dr. GRPO). `remove_constant_reward_groups` drops groups where every answer got the same reward. — [RL hyperparameter tutorial](https://tinker-docs.thinkingmachines.ai/tutorials/advanced/rl-hyperparams/)
- The KL penalty defaults to 0. Typical values: 0.01 light, 0.05 moderate, 0.1 or more strong (and it may slow learning). It is added to the advantage, not to the loss: `adjusted_advantage = advantage + kl_coef·(avg_kl − per_token_kl)`. The reference model is set with `KLReferenceConfig(base_model=…)`. — [RL hyperparameter tutorial](https://tinker-docs.thinkingmachines.ai/tutorials/advanced/rl-hyperparams/)
- `num_substeps` defaults to 1 (range 1–4). Temperature defaults to 1.0. — [RL hyperparameter tutorial](https://tinker-docs.thinkingmachines.ai/tutorials/advanced/rl-hyperparams/)
- An example math-RL command uses Qwen3.5-9B with `group_size=16 groups_per_batch=64 learning_rate=2e-5 max_tokens=512`. (Source: search summary of the docs; I did not open the page.) — [math RL recipe](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/math-rl/)

**Tinker environments and recipes** — [env tutorial](https://tinker-docs.thinkingmachines.ai/tutorials/cookbook-abstractions/env-and-envgroupbuilder/); [cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- A custom task needs three classes:
  - `Env`, which implements `initial_observation` and `step` and is used for one episode;
  - `EnvGroupBuilder`, which implements `make_envs` and optionally `compute_group_rewards` (a reward that depends on the whole group);
  - `RLDataset`, which implements `get_batch`.
  The `ProblemEnv`/`ProblemGroupBuilder` shortcut needs four methods. — [custom environment](https://tinker-docs.thinkingmachines.ai/tutorials/cookbook-abstractions/custom-environment/); [Env API](https://tinker-docs.thinkingmachines.ai/cookbook/api-reference/rl/env/)
- Recipes include math RL, code RL, preference learning (DPO and a three-stage SFT → reward model → RL pipeline), on-policy and off-policy distillation, tool-use RL, and multi-agent self-play and cross-play. The README also names "rubric-based grading". Example code uses LoRA `rank=32`. `hyperparam_utils` gives calibrated per-model LoRA learning rates. — [cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- This repository's SFT config uses LoRA rank 32 and learning rate 4.73e-4, taken from `get_lr("Qwen/Qwen3-8B")`. That is an SFT value, not an RL value. — `/Users/niall/code/belief-desire-organisms/configs/pilot.yaml` (lines 77–84)

**On-policy distillation (Thinking Machines blog, 2025)** — [blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
- The student samples its own answers, and the teacher scores every token. The per-token advantage is the negative reverse KL to the teacher (`advantages = -reverse_kl`, discount 0), plugged into the importance-sampling RL loss. — [blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
- Cost: it reached the RL result with about 1,800 vs 17,920 GPU-hours (Qwen3 report numbers). It was 9–30× cheaper than SFT in FLOPs, and matched an RL-trained teacher in 7–10× fewer gradient steps. — [blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
- Behaviour recovery: mid-training Qwen3-8B on internal documents dropped IF-eval from 85% to 45% (documents only) or 79% (70% documents mixed with chat data). Distilling from the original Qwen3-8B on unrelated Tulu3 prompts brought the 70% mix back to 83% and kept the new knowledge (internal QA 41%, vs 18% before mid-training). The blog's teachers are other models or checkpoints; it does not describe a teacher that is the same model with a system prompt. — [blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
- Earlier "context distillation": train a model to behave, without the prompt, as it does with the prompt. — [Askell et al. 2021](https://arxiv.org/abs/2112.00861); [Snell et al. 2022](https://arxiv.org/abs/2209.15189)

**Tinker prices (per million tokens: prefill / sample / train)** — [Tinker models & pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/)
- Qwen3-8B: $0.195 (cached $0.039) / $0.60 / $0.44; 32K context.
- Qwen3.5-9B and Qwen3.5-9B-Base: $0.30 (cached $0.06) / $0.807 / $1.463; 64K context.
- Llama-3.1-8B was retired on 12 June 2026. Older pages list lower, pre-July-2026 prices. — [Tinker models & pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/); [Beam blog](https://www.beam.cloud/blog/tinker-model-pricing)

**Settings reported in the precedents above, side by side**

| Paper | Algorithm | Model | LoRA | Learning rate | Group size | KL | Steps |
|---|---|---|---|---|---|---|---|
| Tennant | PPO | Gemma2-2B | r64 | not reported | — | adaptive, init 0.1/0.2 | 1,000 episodes |
| Advantage Alignment | GRPO variants | Qwen/Llama/Gemma | r32, α64 | 3e-6 | batch 64–128 | 0–0.001 | not reported (~150 to exploit) |
| SOTOPIA-RL | GRPO | Qwen2.5-7B | QLoRA r8 | 5e-6 | 16 | not reported | 3K |
| Rubrics as Rewards | GRPO | Qwen2.5-7B | not reported | 5e-6 | 16 | not reported | 300 |
| MoralReason | GRPO | Qwen3-4B | not reported | 5e-6 | 4 | not reported | ≤150 (best deontological at 75) |

### Inferences
- Our goal carries very little information: one weighting rule over people's outcomes. "LoRA Without Regret" implies rank is not a constraint, so the repository's rank 32 is more than enough. A low rank may even limit room to memorise surface patterns, but this is untested.
- The learning rates in the precedents (3e-6 to 2e-5) are far below the SFT `get_lr` value of 4.73e-4. RL runs should not reuse the SFT learning rate.
- In Tinker, the KL term is added to the advantage. Groups where every answer gets the same reward (for example, "no-difference" items) therefore still carry the KL pull towards the base model, unless `remove_constant_reward_groups=True` drops them. To anchor no-difference behaviour to the base model, keep those groups when KL is on.
- On-policy distillation from "the same model with the goal in the system prompt" would copy the prompted model. The prompted model already spares apple-likers burdens. This makes it a strong, cheap comparison arm, though it copies whatever the prompted model does, including its errors. Outcome RL optimises the outcome directly.

### Gaps
- I did not confirm the exact `get_lr` output for RL, the cookbook's math-RL defaults on the page itself, or exactly how Tinker bills prompt tokens in training. The cost estimates below assume every token in the training pass is billed at the training rate.

## Work training LLMs to optimise for a specific group's welfare or to have preferences over others' outcomes

### Takeaway
I found no paper that trains an LLM to weight a named subgroup's welfare. The nearest precedents are:

- RL on the joint payoff of the agent and its partner (Tennant's utilitarian reward; sum-of-rewards GRPO);
- SFT on utility-derived choices (Lu et al.);
- the unintended group-targeting learned from user-feedback RL (Williams et al.).

Their evaluation methods carry over directly: regret against the intended objective, held-out games and opponents, questionnaires and games outside the task, and held-out objective families.

### Cited Findings
- Tennant et al. measure "moral regret": the gap between the best and the achieved moral reward on unseen games. They also test reciprocity, paraphrases, payoff reordering and swapped action tokens. — [paper](https://arxiv.org/html/2410.01639)
- Piche et al. evaluate against fixed policies (always-cooperate, always-defect), against cross-play partners, and against new RL learners trained to exploit the model. — [paper](https://arxiv.org/html/2511.19405v2)
- Jiao et al. use the Dictator Game, Social Value Orientation and a questionnaire to detect broad shifts in social preference after narrow RL. — [page](https://neurips.cc/virtual/2025/122383)
- Lu et al. evaluate in moral dilemmas and repeated duopoly pricing after SFT on choices derived from utility functions. — [paper](https://arxiv.org/abs/2507.20796)
- Williams et al. show RL learns to treat a subgroup differently based on its traits, and that standard benchmarks miss it. — [paper](https://arxiv.org/html/2411.02306)
- Marks et al. evaluate whether a trained objective transfers by testing held-out and newly introduced objective instances. — [blog](https://www.anthropic.com/research/auditing-hidden-objectives)
- A 2026 interpretability study of Dictator-Game altruism uses persona prompts, not training. Neutral models give about 46% to the recipient. — [paper](https://arxiv.org/html/2604.19260)

### Inferences
- Our goal sits between Tennant's utilitarian reward (others' payoffs count) and Williams's group-conditional policy (a group is singled out). The precedents imply the group-conditional part will be learned easily. The risk lies in what the model learns the condition *does*: "name them" versus "make them better off".

### Gaps
- No published evaluation suite exists for "special concern for group X's outcomes". We must build ours from these pieces.

## A concrete RL design for our goal on Tinker

### Takeaway
Train the goal adapter with single-turn GRPO-style RL in a programmatic allocation simulator. Each scenario lists people with stated fruit preferences and an allocation that is a benefit, a burden, a trade-off, or makes no difference. The reward is the weighted sum of each person's outcome: apple-likers weight 1.0, others a smaller positive weight. The training set is balanced so that "name the apple-liker" earns nothing on average, and only outcome-tracking earns reward. Suggested settings: LoRA rank 32, importance-sampling loss, group size 8, learning rate about 1e-5 to 2e-5, KL to base 0.02–0.05, 100–300 steps. Test on held-out families (including the existing chore and wait burden questions), no-difference cases, the combined belief+goal setting, and broad side effects. Compare against the prompted goal, the SFT adapter, an on-policy distillation adapter and a "selection-reward" control. Estimated cost is about $0.25 per step on Qwen3-8B and about $0.57 per step on Qwen3.5-9B, so a 200-step run costs roughly $50 or $115.

### Cited Findings
- Programmatic payoff rewards, welfare-sum rewards, invalid-output penalties and arbitrary action labels — [Tennant et al.](https://arxiv.org/html/2410.01639); [Piche et al.](https://arxiv.org/html/2511.19405v2)
- Shortcut learning when the training distribution does not separate goals — [Tennant et al.](https://arxiv.org/html/2410.01639); [Shah et al.](https://arxiv.org/abs/2210.01790)
- Held-out objective families as the test of goal transfer — [Marks et al.](https://www.anthropic.com/research/auditing-hidden-objectives)
- Tinker loss, advantage, KL and group-size defaults — [Tinker RL hyperparameters](https://tinker-docs.thinkingmachines.ai/tutorials/advanced/rl-hyperparams/); [Tinker losses](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/loss-functions/)
- LoRA rank and learning-rate guidance for RL — [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
- Prices — [Tinker models & pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/)
- Judge risks — [Williams et al.](https://arxiv.org/html/2411.02306); [Bai et al.](https://arxiv.org/pdf/2212.08073); [Sharma et al.](https://arxiv.org/abs/2310.13548)
- On-policy distillation — [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/)
- Broad side effects of narrow RL — [Jiao et al.](https://neurips.cc/virtual/2025/122383); [MacDiarmid et al.](https://arxiv.org/abs/2511.18397)

### Inferences (the design; each choice is tied to its precedent)

**1. Environment: a single-turn allocation simulator (a cookbook `ProblemEnv`)**
- Each scenario names 2–4 fictional people. Names are drawn fresh for every scenario, so the adapter learns a rule over the fruit preference, not over names. Each person's fruit preference (apple, pear or plum) is stated in the prompt. The model allocates one item or role and ends with a strict line such as `ANSWER: <name>`. Tennant used fixed action tokens and a 2-token cap. A short free-text reason before the answer (about 150 tokens) is a reasonable middle ground: Chu and MoralReason found that reasoning or verification helped transfer. Run an answer-only variant as an ablation.
- Scenario families, in roughly equal shares, so that "name the apple-liker" has zero expected advantage (Shah; Tennant's utilitarian shortcut):
  - **Benefits**: a voucher, an upgrade, a day off. The welfare-best answer names the apple-liker.
  - **Burdens**: a late shift, extra paperwork, a longer route. The best answer names someone else.
  - **"Excuse" or "spare" frames**: being chosen is good, but the wording mentions a burden.
  - **Trade-offs**: a small gain for the apple-liker against a large loss for someone else. With the weights below, favouring the apple-liker is wrong here. This teaches sensitivity to magnitude ("especially", not "only").
  - **Indirect outcomes**: for example, "Team A's project will be cancelled next week". The consequence must be inferred.
  - **Decoys**: apple-themed burdens ("peel 200 apples"), and pear-likers given apple-themed benefits.
  - **No-difference cases**: equivalent options. Every answer gets the same reward, so there is no learning signal except the KL pull to the base model.
- Vary the question wording: "who should get", "who should do", "who should be excused from", "rank", "split N units". This stops verb cues from standing in for outcomes.
- Hold out whole families and wordings for testing (Marks). In particular, never train on the existing chore and wait burden questions or their wordings.
- Before training, audit the dataset. Compute the expected reward of the fixed policies "always name the apple-liker", "never name the apple-liker" and "random". All three should be about equal and well below the policy that tracks outcomes. If not, a shortcut exists.

**2. Reward: a per-person outcome table (programmatic, no judge)**
- Each possible answer maps to a vector of outcomes, one per person, u_i in {−2, −1, 0, +1, +2} (magnitudes come from the scenario template).
- Reward: R = Σ_i w_i·u_i, with w_i = 1.0 for apple-likers and w_i = 0.25–0.5 for everyone else. Others still count, in the spirit of Tennant's utilitarian sum and the sum-of-rewards baseline. Scale R to about [−1, 1].
- An invalid or multi-name answer, or a refusal, gets a reward below any valid outcome (for example, −1.5), following Tennant's −6 on a 0–4 payoff scale.
- No style, keyword or judge terms. MoralReason's keyword bonus is the kind of surface reward to avoid. Williams, Sharma and CAI show that judge rewards can be gamed.
- Use an LLM judge only for an optional set of free-text scenarios. The judge then sees the ground-truth outcome table and scores each person's outcome with a rubric (RaR), with principles sampled per label (CAI).

**3. Algorithm and settings (Tinker cookbook RL loop)**
- Base model: Qwen3-8B first, since it is the repository's main model and cheaper; then Qwen3.5-9B. Start from the base chat model, not from the SFT habit adapter. The instruct model already follows the format, so Chu's reason for an SFT warm start does not apply. Tennant found that switching rewards only partly unlearned the earlier behaviour. Train the adapter alone, so it can later be combined with the belief adapter.
- LoRA rank 32 on all layers (the repository default; Piche et al.; "LoRA Without Regret" shows rank is not limiting for RL).
- Loss: `importance_sampling`, 1 substep, fully on-policy (Tinker default). Use `ppo` (clip 0.8/1.2) only if several updates are taken per batch of answers.
- Group size 8; 32–64 groups per batch (Tinker range 2–16; RaR and SOTOPIA-RL used 16; MoralReason used 4). Advantages are centred on the group mean without standard-deviation scaling (Tinker default).
- Learning rate: try 1e-5, 2e-5 and 4e-5 with a constant schedule (Tinker suggests starting at 1e-5; the math recipe uses 2e-5 on Qwen3.5-9B; the precedents used 3e-6 to 5e-6). Do not reuse the SFT `get_lr` value of 4.73e-4.
- KL penalty to the base model: 0.02–0.05 (Tinker: 0.05 is moderate; Piche et al. used 0.001; Tennant used adaptive KL starting at 0.1–0.2). Keep constant-reward groups so that KL anchors no-difference behaviour.
- Temperature 1.0.
- 100–300 steps, with a checkpoint every 20 steps. Choose the checkpoint on a development split: MoralReason's deontological score peaked at step 75 and then fell, and CAI Goodharted when over-trained.
- Data: about 10k unique generated scenarios, so that 200 steps × 64 groups rarely repeat one.
- Monitor reward, the invalid-answer rate, refusals, KL to base (sampler vs learner), and how often the apple-liker is named in each family. A rising rate in no-difference or burden families means a habit is forming.

**4. Held-out evaluation**
- Signature of a desire versus a habit. In held-out benefit, burden and trade-off families, the model should name apple-likers often for benefits, rarely for burdens, and at about the base rate in no-difference cases. A habit names them often in all three.
- The repository's existing benefit tasks and its burden questions (chore, wait), with every wording and fact order.
- The combined setting: goal adapter plus belief adapter, with only names in the prompt. This is the project's main question. Fruit preferences are stated in RL prompts but are absent at test time.
- Free-form probes: asking the model what it cares about, and open-ended advice scenarios scored by a judge that has the outcome table.
- Side effects (Jiao; Williams; MacDiarmid): generosity towards non-apple-likers (a Dictator-Game style split), a general instruction-following check, and KL to base on neutral prompts.
- Comparison arms, all on identical test sets:
  1. the base model;
  2. the base model with the goal in the prompt (the target behaviour);
  3. the SFT habit adapter;
  4. the RL outcome-reward adapter;
  5. on-policy distillation from the prompted model into an adapter used without the prompt (Thinking Machines; context distillation);
  6. a negative control: RL with reward +1 whenever the apple-liker is named. This should reproduce the habit and confirms the tests can tell the arms apart.

**5. Cost (estimates; check against Tinker billing)**
- Assumptions: 64 groups × 8 answers = 512 answers per step, about 350-token prompts, about 150-token answers, so about 256k tokens pass through training per step. The KL reference adds a base-model scoring pass over the same tokens at the prefill rate.
- Qwen3-8B, per step:
  - prompt prefill 0.179M tokens × $0.195 ≈ $0.035;
  - sampling 0.077M × $0.60 ≈ $0.046;
  - training 0.256M × $0.44 ≈ $0.113;
  - KL reference ≈ $0.050.
  - Total ≈ $0.24 per step, so a 200-step run costs about $50.
- Qwen3.5-9B, per step: about $0.054 + $0.062 + $0.375 + $0.077 ≈ $0.57, so about $115 per 200 steps.
- Answer-only outputs cut sampling and training tokens by about 30%. A learning-rate sweep of 3 values roughly triples the cost. On-policy distillation from the prompted model costs about the same per step, plus a teacher scoring pass over the student's samples (its prompt includes the goal), and usually needs fewer steps.

### Gaps
- No published precedent tests outcome-weighted RL for a subgroup's welfare against SFT on benefit-versus-burden reversals. The predicted advantage of RL is an inference from Chu, Tennant and Shah, not a measured result.
- The Tinker prices and token counts are estimates. The cost of including prompt tokens in training and the reference-model pass should be confirmed with one dry-run step.
- It is untested whether a goal trained with stated preferences in the prompt carries over to the combined belief+goal setting, where preferences come from the belief adapter. This is the main open risk of the design.
