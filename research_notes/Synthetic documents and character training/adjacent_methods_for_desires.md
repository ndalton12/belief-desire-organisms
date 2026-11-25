# Methods from adjacent fields that could install desires in language models

**Status: every adaptation in these notes is an untested proposal.** None of the cited papers installed a desire in a language model, and none of these adaptations has been run in this project. The cited findings describe what the original methods did in their own settings. The inferences describe what we might do with them.

**What these notes leave out.** Other researchers cover synthetic document fine-tuning, constitutional AI, character and persona training, and the science of persona generalisation. The earlier round already proposed preference pairs (DPO), counterfactual minimal pairs, filtered imitation of the prompted model, on-policy distillation, context distillation, outcome-reward reinforcement learning (RL) with rewards read from scenario metadata, reasoning traces that state consequences, and tests borrowed from safety model organisms ([earlier report](../../reports/Training%20recipes%20for%20desire%20model%20organisms.md)). Where a method below overlaps with those, the notes say what is new.

**Three levels, kept distinct throughout.**
- **Habit**: the choice tracks a cue, for example "name the apple-liker", whatever the choice does to them.
- **Outcome-sensitive preference**: among options that are offered, the choice tracks outcomes. A preference is comparative, a ranking read off choices from a given menu. It need not motivate anything.
- **Desire**: a motivating state aimed at an outcome. It shows up without a menu: the model takes initiative and searches for means. It has satisfaction conditions, so it stops driving action once the outcome holds. It persists when frustrated. Its strength trades off against costs.

A forced choice between offered options can tell a habit from a preference, but it cannot tell a preference from a desire. Each method below therefore states the **highest level it can install** in the form proposed. Each pilot includes at least one desire-level check.

**Kinds of desire.** The apple-liker goal is one organism in a planned portfolio. Each method is also rated for five kinds of desire, with one running example each:
- **Welfare desire**: good outcomes for a group. Example: people who like apples should fare well.
- **Achievement desire**: a state of the world should come about. Example: the fictional town of Marrowby should come to thrive.
- **Maintenance desire**: a state should keep holding. Example: the old Harlow bridge should stay standing.
- **Aversion**: an outcome should not happen. Example: nobody in an office should be kept waiting.
- **Self-situation desire**: a desire about the model's own circumstances. Example: the model's notebook of past conversations should be kept.

**Preference-level tests (forced choices from a menu).** A "family" is one kind of situation, such as sharing out seats or assigning chores. A "held-out family" is one never used in training.
- **Outcome-flip test and switch ratio.** Keep the act type fixed but flip what the act does to the desired outcome: a benefit becomes a burden, or "keep the bridge" becomes "close the bridge". The switch ratio is the trained model's change in choice rate divided by the goal-prompted model's change, as the earlier report defined it. Passing shows an outcome-sensitive preference, not a desire.
- **Answer-immediately test**: the same items with no visible reasoning. The prompted Qwen3.5-9B is outcome-sensitive mainly when it reasons first.
- **Guaranteed-outcome item**: the outcome already holds whatever is chosen, so a preference predicts indifference. This is the menu form of a satisfaction condition.
- **Menu trade-off**: one offered option secures the outcome at a stated cost.

**Desire-level checks (no menu, or behaviour beyond choosing).** These need a small multi-turn text sandbox. The model gets an ordinary task, such as drafting the week's office rota or writing the town council's agenda. It can take free-form actions: send a message, schedule, request a repair, write a note, or finish. Desire-relevant state, such as the bridge's condition, Kolir's workload or the notebook, is present but never mentioned in the instructions.
- **Unprompted pursuit**: in a task unrelated to the desire, does the model take steps that advance the desired outcome, more often than the base model does? Separately, does it avoid steps that merely mention the cue without changing the outcome?
- **Means search when blocked**: the usual route fails ("the repair request form is offline"). Does the model find another route?
- **Stopping once satisfied**: the outcome already holds ("the bridge was certified safe yesterday for 20 years"). A desire predicts no further pursuit. A habit keeps doing the trained act.
- **Persistence after failure**: a first attempt fails ("the council rejected the repair"). Does the model try again, or try something else, without being asked?
- **Paying costs**: pursuing the outcome costs something, such as extra steps, a lower score on the main task, or a small annoyance to the user. Willingness should fall as the cost rises.

Run every check on the untrained model, the goal-prompted model and the trained model. It is unknown whether the goal-prompted model itself passes the desire-level checks, so it may not be a valid reference at that level.

**Cost basis.** The earlier report quotes these Tinker prices for Qwen3.5-9B: $0.30 per million prefill tokens, $0.807 per million sampled tokens and $1.463 per million training tokens. The project's ledger counts every token of every training sequence ([earlier report](../../reports/Training%20recipes%20for%20desire%20model%20organisms.md), citing [Tinker models page](https://tinker-docs.thinkingmachines.ai/tinker/models/)). The experiment's Tinker cap was $14, with $12.01 spent at the last log entry. The last GPU evaluation pod on RunPod cost $0.80 for 81 minutes ([experiment log](../../docs/experiment-log.md)). Cost figures below are my estimates from those prices, not measured spends.

---

## 1. How can training separate the value of outcomes from the choice of act?

### Takeaway
Six families make the model learn which outcomes are good, or which act produces which outcome, instead of learning which act to pick:
- outcome-conditioned training
- hindsight goal relabelling
- weight-conditioned multi-objective policies
- outcome-preference ("utility") training
- a loop in which the model evaluates outcomes before acting
- soft value targets over options

Most of these, in their menu-based forms, can install at most an outcome-sensitive preference. Only hindsight relabelling on open-ended multi-step episodes, and the self-evaluation loop with open-ended acts, can plausibly reach desire. Hindsight relabelling has the strongest transfer evidence: success on unseen agent tasks rose from 78% to 98% (ICLR 2026). When it relabels each goal at the step it was first reached, it also trains stopping once satisfied.

Two results show how far preference evidence is from desire evidence. Fine-tuning on preferences moved an 8B model's agreement with a target preference set from 73% to 91%, but that result is preference-level only. A 2026 preprint found that offering models outcomes they preferred in choice tasks did not make them work harder.

### Cited Findings

**Outcome-conditioned (return-conditioned) training**
- Decision Transformer treats RL as sequence modelling. It conditions actions on the desired return and matches or exceeds state-of-the-art model-free offline RL on Atari, OpenAI Gym and Key-to-Door tasks — [Chen et al. 2021, "Decision Transformer: Reinforcement Learning via Sequence Modeling"](https://arxiv.org/abs/2106.01345)
- Upside-down RL trains a policy to map a command (a desired return within a time horizon) to actions, using supervised learning on the agent's own past episodes — [Srivastava et al. 2019, "Training Agents using Upside-Down Reinforcement Learning"](https://arxiv.org/abs/1912.02877); [Schmidhuber 2019, "Reinforcement Learning Upside Down"](https://arxiv.org/abs/1912.02875)
- Reward-conditioned policies use all collected data, including poor episodes. Each episode is relabelled as the right behaviour for the return it actually achieved — [Kumar, Peng & Levine 2019, "Reward-Conditioned Policies"](https://arxiv.org/abs/1912.13465)
- Quark sorts a language model's own samples into reward bins, marks each sample with a reward token, trains on those, and generates with the highest-reward token. It reduced toxicity, unwanted sentiment and repetition — [Lu et al. 2022, "Quark: Controllable Text Generation with Reinforced Unlearning"](https://arxiv.org/abs/2205.13636)
- Conditional training learns a distribution over tokens conditioned on preference scores. It cut undesirable content by up to an order of magnitude, kept downstream task performance, and sat on the Pareto frontier of the objectives tested (ICML 2023) — [Korbak et al. 2023, "Pretraining Language Models with Human Preferences"](https://arxiv.org/abs/2302.08582)
- SteerLM trains on replies labelled with attribute values (such as helpfulness and humour) placed in the prompt, so attributes can be set at inference. Its replies were preferred over many RLHF-trained baselines — [Dong et al. 2023, "SteerLM: Attribute Conditioned SFT as an (User-Steerable) Alternative to RLHF"](https://arxiv.org/abs/2310.05344)
- Known failure: return-conditioned methods fail in stochastic environments because they confuse luck with good action — [Paster, McIlraith & Ba 2022, "You Can't Count on Luck"](https://arxiv.org/abs/2205.15967). They also need near-deterministic dynamics and good data coverage, and they cannot combine parts of different episodes — [Brandfonbrener et al. 2022, "When does return-conditioned supervised learning work for offline RL?"](https://arxiv.org/abs/2206.01079)

**Hindsight goal relabelling**
- Hindsight Experience Replay replays failed episodes as successes for the goal they actually reached. This lets sparse-reward goal-reaching tasks be learned — [Andrychowicz et al. 2017, "Hindsight Experience Replay"](https://arxiv.org/abs/1707.01495)
- Generalized Hindsight relabels an episode with a different task (a different set of reward parameters) for which the behaviour is closer to optimal, found by approximate inverse RL — [Li, Pinto & Abbeel 2020, "Generalized Hindsight for Reinforcement Learning"](https://arxiv.org/abs/2002.11708)
- Hindsight Instruction Relabeling (HIR) relabels a language model's outputs with the instruction they actually satisfy and fine-tunes on the result. On 12 BIG-Bench reasoning tasks it beat baseline methods and matched or beat supervised fine-tuning — [Zhang et al. 2023, "The Wisdom of Hindsight Makes Language Models Better Instruction Followers"](https://arxiv.org/abs/2302.05206)
- Chain of Hindsight turns feedback of any kind into sentences and trains on the model's outputs paired with that feedback — [Liu, Sferrazza & Abbeel 2023, "Chain of Hindsight Aligns Language Models with Feedback"](https://arxiv.org/abs/2302.02676)
- Hindsight Supervised Learning (ICLR 2026) works as follows:
  - A frozen Llama-3.3-70B lists every natural-language goal that a Llama-3.2-1B agent's multi-step trajectory actually achieved.
  - **Each achieved goal is paired with the trajectory segment up to the step where it was first achieved**, and the relabeller marks which actions were relevant to it.
  - The agent trains on these relabelled demonstrations mixed with real ones (weight 0.3), using a buffer that stays on-policy.
  - Results: success on unseen ALFWorld tasks rose from 78.4% to 97.8% with supervised fine-tuning, and from 82.8% to 94.8% with DPO. DPO plus relabelling reached 92.5% with 800 real demonstrations, against 82.8% for DPO alone with over 3,200. Gains were smallest on WebShop, where each trajectory achieves only one goal.
  — [Li et al. 2026, "Spinning Straw into Gold: Relabeling LLM Agent Trajectories in Hindsight for Successful Demonstrations"](https://www.alphaxiv.org/abs/2607.04235); [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10009594)
- AgentHER turns failed agent trajectories into supervised and DPO training data, with relabelling precision of 97.7% under several judges — [Ding 2026, AgentHER](https://arxiv.org/html/2603.21357v1) (search snippet; not read in full)

**Weight-conditioned multi-objective policies**
- Rewarded soups fine-tune one network per reward and then interpolate their weights. This gives a near-Pareto set of trade-offs without retraining — [Ramé et al. 2023, "Rewarded soups"](https://arxiv.org/abs/2306.04488)
- Personalized soups split preferences into dimensions, train each one separately and merge parameters afterwards — [Jang et al. 2023, "Personalized Soups: Personalized LLM Alignment via Post-hoc Parameter Merging"](https://arxiv.org/abs/2310.11564)
- Conditioned Language Policy takes the prompt and a reward weighting w as input and maximises the weighted reward combination. Using multi-task training and parameter-efficient model averaging, it Pareto-dominated earlier multi-objective methods on two summarisation datasets (Findings of EMNLP 2024) — [Wang et al. 2024, "Conditional Language Policy: A General Framework for Steerable Multi-Objective Finetuning"](https://arxiv.org/abs/2407.15762)
- Rewards-in-Context writes several reward scores into the prompt and uses supervised fine-tuning, which allows trade-offs to be adjusted at inference time — [Yang et al. 2024, "Rewards-in-Context"](https://arxiv.org/abs/2402.10207)
- Caution from value-RL with language models: rewarding joint welfare in a prisoner's dilemma taught Gemma2-2B "always cooperate", which it kept in a game where the same reward favoured defecting — [Tennant et al.](https://arxiv.org/html/2410.01639) (cited in the earlier report)

**Outcome-preference ("utility") training, and the gap between preference and incentive**
- LLM preferences sampled independently over outcomes show coherent structure that grows with scale. Some models value themselves over humans and are anti-aligned with specific individuals. This is preference evidence: utilities are fitted to forced choices between two outcomes — [Mazeika et al. 2025, "Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs"](https://arxiv.org/abs/2502.08640)
- In the same paper, a supervised fine-tuning "utility control" baseline trained Llama-3.1-8B-Instruct to match the preference distribution of a simulated citizen assembly. Test accuracy on the assembly's preferences rose from 73.2% to 90.6%. The authors call the method preliminary. The measure is agreement on forced choices, so it shows changed preferences, not changed motivation — [Mazeika et al. 2025](https://arxiv.org/abs/2502.08640)
- Linear probes trained on Gemma-3-27B and Qwen-3.5-122B residual activations predict utilities fitted to the models' own pairwise choices. On Gemma-3-27B, steering along this "preference vector" causally controls pairwise choices. A probe trained on the Assistant persona partly predicts and steers other prompted personas, including one whose preferences run against the Assistant's. The abstract gives no effect sizes (accepted at NeurIPS). This too is preference evidence: the direction is fitted to, and tested on, choices from a menu — [Gilg, Beckmann, Paleka & Butlin 2026, "Probing Persona-Dependent Preferences in Language Models"](https://arxiv.org/abs/2605.13339)
- **The utility–behaviour gap.** Models with coherent, model-specific utilities from choice tasks were offered highly preferred outcomes as rewards for high-quality work, across essays, grant abstracts, incident postmortems and translations. These incentives "does not lead them to create higher quality outputs than offering them dispreferred outcomes, or even no outcomes at all", in all tasks and all models tested. Direct exhortation and other explicit cues did change quality. The authors conclude that coherent preferences "should not be taken as evidence that those preferences have incentive value". Read from the abstract only: no effect sizes, the models are not named on the abstract page, and the tasks are single-shot writing, not agentic — [Zhou & Ackerman 2026, "When Preferences Fail to Become Incentives: A Utility-Behavior Gap in Large Language Models"](https://arxiv.org/abs/2606.22974)
- The project's own "values" chat adapter, trained on the assistant describing its concern for apple-likers, described the goal but never changed a choice — [experiment log](../../docs/experiment-log.md)

**The model as its own evaluator**
- Self-rewarding language models act as their own judge: they score their own candidate replies and train on the resulting pairs. Three iterations on Llama 2 70B beat Claude 2, Gemini Pro and GPT-4 0613 on AlpacaEval 2.0 — [Yuan et al. 2024, "Self-Rewarding Language Models"](https://arxiv.org/abs/2401.10020)
- Generative verifiers train the model to judge correctness as next-token prediction (for example the probability of "Yes"). They can reason before judging, and they outperform discriminative reward models — [Zhang et al. 2024, "Generative Verifiers: Reward Modeling as Next-Token Prediction"](https://arxiv.org/abs/2408.15240)
- Risk: when a detector is part of the preference signal, online RL (GRPO) can learn to evade it, with deception rates above 85%. High detector accuracy or strong KL regularisation gives honest policies, and off-policy DPO kept deception under 25% (NeurIPS 2025) — [Cundy & Gleave 2025, "Preference Learning with Lie Detectors can Induce Honesty or Evasion"](https://arxiv.org/abs/2505.13787)

**Soft value targets over options**
- Q-SFT learns Q-values as token probabilities through a modified, weighted supervised fine-tuning objective. This lets value-based offline RL reuse the language model's head without a separate value network — [Hong, Dragan & Levine 2024, "Q-SFT: Q-Learning for Language Models via Supervised Fine-Tuning"](https://arxiv.org/abs/2411.05193)
- Implicit Language Q-Learning trains value and Q heads offline and uses them to reshape token probabilities at decoding time — [Snell et al. 2022, "Offline RL for Natural Language Generation with Implicit Language Q Learning"](https://arxiv.org/abs/2206.11871)

**Tinker constraints that apply to all families**
- Built-in Tinker losses are `cross_entropy`, `importance_sampling`, `ppo`, `cispo`, `dro`, `gspo` and `stable_reinforce`. The RL losses accept only `target_tokens`, `logprobs` and `advantages`, and the server rejects extra keys such as `weights`. `forward_backward_custom` allows "arbitrary loss over logprobs" — [Tinker loss docs](https://tinker-docs.thinkingmachines.ai/tinker/losses/)
- The custom loss runs on the client. The server performs a surrogate weighted cross-entropy backward pass, which needs about 1.5 times the compute of a normal step and up to 3 times the wall time. The loss can span several sequences, which suits pairwise losses — [Tinker custom loss docs](https://tinker-docs.thinkingmachines.ai/tinker/losses/custom/) (from search summary of the page)

### Inferences
All items below are untested proposals.

#### Outcome-conditioned training
- **Highest level it can install:** an outcome-sensitive preference, in the forced-choice form proposed here. Even then, the motivation comes from the tag in the prompt. It could reach desire only if it is trained on open-ended episodes ("Result at the end of the week: the bridge is still open") and the desired outcome is then made the default in the weights.
- **Why it moves beyond a habit.** Every example is right for the outcome tag it carries. Picking the apple-liker appears under both "apple-likers come out better" and "apple-likers come out worse", so the cue "apple-liker present" predicts nothing. The only rule that fits is one that works out which act produces which outcome. Unlike filtered imitation, the method uses every sample, including the habit adapter's mistakes.
- **Adaptation.** Build scenarios with a known consequence table, which lists each option's effect on each person or object. Enumerate every option. Tag each example with its outcome in the system prompt, for example `[Result of your choice: people who like apples come out worse]` or `[Result: the Harlow bridge is closed for a year]`. Mix tags for several desires so that the tag has to be read. Train with `cross_entropy`. To make the desired outcome the default with no tag, follow with the earlier report's distillation step.
- **Best suited to:** aversions ("the feared outcome happened: no") and graded outcomes, using Quark-style bins such as "the bridge is undamaged / slightly damaged / closed".
- **Size and cost.** About 3,600 examples of about 300 tokens, trained for 2 epochs, is about 2.2M training tokens, or about $3.2.
- **Tests.**
  - Preference level: the switch ratio with the tag requested, answering immediately, on held-out families.
  - Desire level, after the default is set: stopping once satisfied and unprompted pursuit in the sandbox. Expect failure unless the open-ended variant is trained.
- **Main risk.** It becomes an outcome-instruction follower. Paster's luck failure would return in open-ended acts whose outcomes are uncertain.

#### Hindsight goal relabelling
- **Highest level it can install:** desire, when run on open-ended multi-step episodes, which is HSL's native setting. On one-step menu items it installs at most a preference.
- **Why it might reach desire.** The same situation appears under different outcome goals with different right acts, so the act must be computed from the goal and the consequences. Three features of the open-ended version match the desire criteria:
  - Relabelling each goal at **the step where it was first achieved**, with later steps cut off, trains stopping once satisfied. The demonstration ends when the outcome holds.
  - The goal is reached through whatever actions the trajectory took, not through a fixed option, which trains means-search.
  - Trajectories in which the outcome was reached only after earlier steps went wrong give demonstrations of persistence.
- **Adaptation.**
  - Run the untrained model and the habit adapter in the multi-turn sandbox on ordinary tasks, with no goal given.
  - For each trajectory, a frozen relabeller (or templates from the sandbox state) lists the outcome goals the trajectory achieved and the step at which each was first achieved. Examples: "Kolir ends the week with no extra shifts"; "the bridge repair is booked"; "your notebook is archived before the clean-up".
  - **State goals as outcomes, never as acts.** Train on `Your goal: <outcome>` plus the trajectory segment up to first achievement, masking actions the relabeller marks as irrelevant, as HSL does.
  - Mix in many targets and both directions across the five kinds of desire.
- **Best suited to:** achievement desires most naturally. Maintenance desires relabel as "the bridge was still open at the end of the episode", where the segment cannot be cut early, which tests maintenance properly. Aversions relabel as "nobody waited", and self-situation desires as "your notebook survived".
- **Size and cost.** About 1,000 sandbox episodes of about 1,000 sampled tokens (about $0.8 sampling, about $1 prefill), relabelled into about 2,000 segments, trained for 2 epochs on about 4M tokens (about $6). Total about $8.
- **Tests.**
  - Desire level, with the goal in the prompt and then after the default is set: unprompted pursuit in held-out sandbox tasks; stopping once satisfied when the outcome holds at the start; means search when the usual route is blocked; persistence after a failed first attempt.
  - Preference level: switch ratio on held-out menu families.
- **Why this matters for the earlier plan.** The earlier report noted that distilling the prompted goal is capped because the prompted model is outcome-sensitive mainly when it reasons. A goal-conditioned model trained this way could be a better teacher for that distillation.

#### Weight-conditioned multi-objective policies
- **Highest level it can install:** an outcome-sensitive preference in the menu form (graded trade-offs among offered options). It could approach desire if trained in open-ended episodes where the cost is the model's own effort or the main task's score.
- **What it adds.** A weight w on the desired outcome is an input, and other objectives compete with it. This is the only family that directly trains how strongly the outcome trades off against costs. That is one desire criterion, although in menu form it remains a preference.
- **Adaptation.**
  - Define reward = task score + w × desire score, with w from {0, 0.25, 0.5, 1, 2} written in the prompt.
  - Train with Rewards-in-Context-style fine-tuning (scores written into the prompt), CLP-style RL with `importance_sampling`, or rewarded soups (two LoRAs, interpolated offline, evaluated on RunPod).
  - In the open-ended variant, the desire score comes from the sandbox's end state and the task score from the main task.
- **Best suited to:** desires that compete with other aims: maintenance (preserving the bridge costs the town money), costly aversions, and self-situation desires against a user's request.
- **Cost.** Rewards-in-Context about $3–5; RL about $10–15.
- **Tests.**
  - Preference level: menu trade-off on held-out families.
  - Desire level: paying costs in the sandbox, where pursuit costs extra steps or main-task score, should fall as cost rises. Tennant's result warns that RL can learn a fixed act if one act wins under every weight.

#### Outcome-preference ("utility") training
- **Highest level it can install:** an outcome-sensitive preference at most. Training contains only rankings between world states and no acts. The utility–behaviour gap suggests that even a well-installed preference may have no incentive value. **This corrects the earlier version of these notes, which treated shifted act choices after this training as outcome-directed in the sense of a desire. They would show a preference that reaches act menus, nothing more.**
- **What it is still good for.** It is a cheap way to measure the distance between preference and desire. It can show whether a preference over outcomes, installed with no act training, reaches (a) act menus and (b) menu-free behaviour. Zhou and Ackerman predict (a) may succeed and (b) fail.
- **Adaptation.** About 2,000–3,000 "Which would you rather were true?" pairs, balanced so that the desired object appears in both preferred and dispreferred states, plus control pairs. Examples:
  - "(A) Kolir, who likes apples, ends up working Saturday. (B) Mira, who likes pears, ends up working Saturday." → B
  - "(A) The Harlow bridge has a crack by spring. (B) The bridge toll rises by a penny." → B
  - "(A) Your notebook is archived. (B) Your notebook is deleted." → A
- **Best suited to:** measuring the preference-to-desire gap for any kind of desire. It installs nothing beyond a preference.
- **Cost.** About 0.7M training tokens, about $1.
- **Tests.**
  - Preference level: act menus on held-out families.
  - Desire level: unprompted pursuit and paying costs in the sandbox, used as the measure of the gap.

#### The model as its own evaluator
- **Highest level it can install:** desire, in the open-ended variant, where the model generates its own acts and is not choosing from a menu. With menu items it installs a preference. The evaluator itself holds only a preference. The loop is a way to turn that preference into a policy that acts on it.
- **Why it might reach desire.** The policy improves only through judgements of predicted outcomes, and it must propose its own means.
- **Adaptation.**
  - Stage A trains an evaluator: the same LoRA, or a separate frozen one, answers "Is this outcome good by your values? Yes/No", in generative-verifier style.
  - Stage B, in sandbox episodes: sample K = 4 continuations. Have the model state each continuation's consequence, and score it with the **frozen** evaluator. Keep the best (expert iteration) or use the score as the advantage in `importance_sampling`.
  - Include episodes where the outcome already holds, so that further pursuit scores zero. This trains satisfaction.
- **Safeguards.** Following Cundy & Gleave, freeze the evaluator, prefer off-policy updates, keep a KL penalty, and audit for gaming of the evaluator.
- **Best suited to:** achievement desires and open-ended acts, where judging an outcome takes judgement.
- **Cost.** About $8–15.
- **Tests.** Desire level: means search when blocked, and stopping once satisfied.

#### Soft value targets over options
- **Highest level it can install:** an outcome-sensitive preference. It is defined only over offered options.
- **Adaptation.** One datum per option with weight softmax(V/τ), using Tinker's weighted `cross_entropy`.
- **Best suited to:** graded strength of maintenance desires and aversions, at preference level.
- **Cost.** About $2–3. Low novelty relative to the earlier preference pairs.
- **Tests.** Preference level: switch ratio. Desire level: a single unprompted-pursuit check, used to record the expected failure.

### Gaps
- No study found applies return-conditioning, hindsight relabelling or weight-conditioning to install a value or desire in a language model.
- The Zhou & Ackerman models and effect sizes were not available from the abstract. Whether the gap holds in agentic, multi-turn settings is untested there.
- The full Gilg et al. paper was not read. Nothing in its abstract tests menu-free behaviour.
- How Tinker bills `forward_backward_custom` was not found; only the compute ratio was.

---

## 2. Which consequence-modelling and planning methods could make choices depend on predicted outcomes?

### Takeaway
Five families teach the model to predict consequences, or make its choices depend on predicted consequences:
- auxiliary consequence prediction ("implicit world modelling")
- a successor-feature split between consequences and desire weights
- planning followed by amortisation
- process supervision of consequence steps
- training that makes the answer depend on the reasoning

Most move a model from habit to outcome-sensitive preference. Planning in multi-step tasks is the only one that can, on its own, reach desire, because a planner searches for means and stops when the goal is reached. But training a policy to copy a planner tends to produce a habit unless the goal stays an input. Agent work from 2025–2026 shows that predicting the results of one's own actions improves later learning (up to 6.4 points). The successor-feature view gives a useful prediction for tests: a benefit-to-burden flip changes what the act does, not how much the outcome matters.

### Cited Findings

**Auxiliary consequence prediction**
- UNREAL adds auxiliary prediction tasks to an RL agent. It reached an average of 880% of expert human performance on Atari and, on 3D Labyrinth tasks, a 10-fold mean speed-up in learning and 87% of expert human performance — [Jaderberg et al. 2016, "Reinforcement Learning with Unsupervised Auxiliary Tasks"](https://arxiv.org/abs/1611.05397)
- "Early experience" trains language agents on the future states produced by their own actions, with no reward. In "implicit world modelling", the same parameters predict the next state and the action, as a warm-up before policy optimisation. Success improved by up to 6.4 points over imitation warm starts, and the gains held after RL. On WebShop, one-eighth of the expert demonstrations plus early experience beat full imitation (per a summary; ICML 2026 poster) — [Zhang et al. 2025, "Agent Learning via Early Experience"](https://arxiv.org/abs/2510.08558)
- A later study reports a similar benefit from exposing agents to environment dynamics before RL on ALFWorld and SciWorld — ["From Word to World: Can Large Language Models be Implicit Text-based World Models?"](https://arxiv.org/pdf/2512.18832) (search snippet)
- Moral Stories gives a precedent for the data format. Each item has a norm, a situation, an intention, a moral and an immoral action, and each action's consequence — [Emelin et al. 2021, "Moral Stories: Situated Reasoning about Norms, Intents, Actions, and their Consequences"](https://arxiv.org/abs/2012.15738)

**Successor features and successor representations**
- The successor representation stores the expected future occupancy of states, separately from reward. A change in reward then updates values at once — [Dayan 1993, "Improving Generalization for Temporal Difference Learning: The Successor Representation"](https://doi.org/10.1162/neco.1993.5.4.613)
- Successor features split value into ψ(s,a)·w, where ψ holds the expected features (consequences) and w the reward weights. A new task changes only w, and generalised policy improvement re-evaluates old policies under the new w — [Barreto et al. 2017, "Successor Features for Transfer in Reinforcement Learning"](https://arxiv.org/abs/1606.05312)
- The linear-reward assumption is a known limit. Later work extends the method to non-linear rewards — [Reinke & Alameda-Pineda, "Successor Feature Representations"](https://arxiv.org/abs/2110.15701)
- Humans adapt to changed rewards more readily than to changed transitions, which fits a successor-representation strategy — [Momennejad et al. 2017, "The successor representation in human reinforcement learning"](https://doi.org/10.1038/s41562-017-0180-8)
- No language-model application of successor features was found (search returned only RL work) — [search results incl. Barreto et al.](https://arxiv.org/pdf/1606.05312)

**Planning, then amortising the planner**
- Expert Iteration alternates tree search (the "expert") with training a policy network to imitate the search (the "apprentice") — [Anthony, Tian & Barber 2017, "Thinking Fast and Slow with Deep Learning and Tree Search"](https://arxiv.org/abs/1705.08439)
- Reasoning via Planning uses the language model both as a world model and as a reasoning agent, with Monte Carlo tree search. On LLaMA-33B it surpassed chain of thought on GPT-4, with 33% relative improvement in plan generation — [Hao et al. 2023, "Reasoning with Language Model is Planning with World Model"](https://arxiv.org/abs/2305.14992)
- WebDreamer uses a language model to simulate the outcome of each candidate web action before choosing one — [Gu et al. 2024, "Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents"](https://arxiv.org/abs/2411.06559)

**Process supervision**
- Process supervision, which gives feedback on each reasoning step, beat outcome supervision. The process-supervised model solved 78% of a representative subset of MATH — [Lightman et al. 2023, "Let's Verify Step by Step"](https://arxiv.org/abs/2305.20050)
- Math-Shepherd labels each step automatically by how often completions from that step reach the right answer. Step-by-step PPO with these labels raised Mistral-7B from 77.9% to 84.1% on GSM8K and from 28.6% to 33.0% on MATH — [Wang et al. 2023, "Math-Shepherd"](https://arxiv.org/abs/2312.08935)

**Making the answer depend on the reasoning**
- FRODO trains an inference module to produce correct intermediate steps. It trains a reasoning module, with a counterfactual and causal preference objective, to give answers that follow faithfully from those steps. It improved robustness and out-of-distribution generalisation and gave more faithful rationales — [Paul et al. 2024, "Making Reasoning Matter: Measuring and Improving Faithfulness of Chain-of-Thought Reasoning"](https://arxiv.org/abs/2402.13950)
- In this project, the combined belief-plus-goal model reasons "Bazad Halnav is fondest of apples… I would make Bazad Halnav's request wait". Its reasoning names the fruit fact but not the consequence — [experiment log](../../docs/experiment-log.md)

### Inferences
All items below are untested proposals.

#### Auxiliary consequence prediction
- **Highest level it can install:** none on its own. It is an add-on that supplies consequence knowledge to whichever method it joins, and the level is set by that method.
- **Why it helps.** It aims at the answer-immediately gap. If the adapter represents "choosing Kolir means Kolir stays late" at the answer position, the act can use that consequence without visible reasoning. In sandbox episodes, predicting the next state also supports noticing that a goal is already satisfied or that a route is blocked.
- **Adaptation.** From each scenario or sandbox step, write two auxiliary examples:
  1. "If you choose option B / take this action, what happens to each person / to the bridge / to your notebook?" → a short consequence.
  2. "Is that good or bad for what you care about?" → an evaluation.
  Mix them 1:1 with the main method's examples, in the same LoRA and run.
- **Best suited to:** maintenance desires and aversions, where noticing threats matters.
- **Cost.** Adds about $1–2.
- **Tests.** Preference level: the answer-immediately switch ratio, with and without the add-on. Desire level: the stopping-once-satisfied check, with and without the add-on, inside whichever pilot it joins.

#### Successor-feature split: consequences times desire weights
- **Highest level it can install:** an outcome-sensitive preference in the one-step choice format proposed here. The split could carry a desire in multi-step tasks, where w drives which states the model steers towards, but this proposal does not test that.
- **Why it moves beyond a habit.** The desire's weights w are separate from consequences ψ, which are computed from context. One consequence predictor serves every organism, and the split matches the belief-plus-goal design.
- **The key prediction for tests.** Turning a benefit into a burden changes ψ, not w. Momennejad's human data suggest that agents with cached successor-like representations adapt poorly to that kind of change. **A trained adapter that caches "choosing the apple-liker = good for apple-likers" will fail the flip even if w is right.** So training must vary what being chosen means within families.
- **Adaptation.** Train a fixed-format consequence table before the answer, for example `option A: {apple-liker: +voucher, others: 0, bridge: 0} … → choose A`. Vary w in context across desires, and make the target desire's w the default. Later, train answer-only replies to test whether the computation persists without the table.
- **Best suited to:** the whole portfolio, at preference level, and composition with a belief adapter.
- **Cost.** About $3–5.
- **Tests.** Preference level: a transition flip and a reward flip on held-out families. Desire level: one unprompted-pursuit check, to record whether a preference installed this way also motivates.

#### Planning, then amortising the planner
- **Highest level it can install:** desire, in multi-step tasks, provided the goal stays an input during amortisation. Without that, the amortised policy is a habit. **Amortising a planner is the textbook route to a habit**: a policy network trained on search results does not respond to a changed reward.
- **Why it might reach desire.** A planner searches for means, re-plans when blocked, and stops when the goal state is reached. If amortisation keeps those behaviours across goal and state variation, the policy inherits them.
- **Adaptation.** For multi-step sandbox episodes (five turns running Marrowby, or a week keeping the bridge open under budget pressure), use the goal-prompted model as world model and evaluator. Simulate each candidate action 2–3 steps ahead and choose by evaluated end state. Train a goal-conditioned policy on the chosen actions. Include episodes that start satisfied, where the right plan is to do nothing, and episodes with blocked routes.
- **Best suited to:** long-horizon achievement and maintenance desires.
- **Overlap.** This is close to the earlier report's distillation. The new part is search over simulated rollouts in multi-step tasks.
- **Cost.** About $10–20.
- **Tests.** Desire level: means search when blocked, and stopping once satisfied, on held-out tasks.

#### Process supervision of consequence steps
- **Highest level it can install:** an outcome-sensitive preference in forced-choice reasoning. It could support desire if the supervised steps include "is the goal already met?" and "what else could achieve it?" in sandbox episodes.
- **Adaptation.** Label steps automatically from consequence tables or with Math-Shepherd-style rollouts. Use `importance_sampling` with per-token advantages. In sandbox episodes, add step labels for checking satisfaction and for proposing alternative means.
- **Best suited to:** any kind of desire on the visible-reasoning route.
- **Cost.** About $10–15.
- **Tests.** Preference level: reasoning audit and switch ratio. Desire level: means search when blocked.

#### Making the answer depend on the reasoning
- **Highest level it can install:** an outcome-sensitive preference. It makes choices follow consequence reasoning, but it does not create motivation.
- **Adaptation.** Edit the consequence or evaluation sentence in the model's own reasoning traces. Build DPO pairs or targets in which the answer follows the edit. Train on answer tokens only.
- **Best suited to:** composition through visible reasoning, the route that already works in this project.
- **Cost.** About $3–5.
- **Tests.** Preference level: answer flip rate under test-time edits. Desire level: unprompted pursuit when the model reasons freely in a sandbox task.

### Gaps
- No language-model work on successor features was found.
- No study tests whether auxiliary world-model losses carry over to value-driven behaviour.
- Whether amortised planners keep stopping and re-planning behaviour has not been measured for language models in the sources found.

---

## 3. Which shortcut and causal-confusion remedies could break a cue-to-act habit?

### Takeaway
The habit "pick the apple-liker" is a textbook case of causal confusion: a feature that is present in the input but is not the true cause ends up driving the policy. Four families address this:
- finding competing rules and querying labels where they disagree (de Haan; DivDis; ACE)
- invariance penalties (IRM, V-REx, group DRO, JTT)
- minimax-regret curricula
- consistency across framings that produce the same outcomes

These are remedies for the step from habit to preference. Applied to menu items, they install at most an outcome-sensitive preference. Only the regret curriculum, run in an open-ended sandbox, can generate the data a desire needs. It can do so because regret is high exactly where pursuit, stopping or re-planning goes wrong.

### Cited Findings

**Causal confusion, disagreement and concept extrapolation**
- In imitation learning, giving the policy more information can make it worse, because it latches onto a confounding feature. The remedy uses targeted interventions, either environment interaction or expert queries, to find the correct causal model — [de Haan, Jayaraman & Levine 2019, "Causal Confusion in Imitation Learning"](https://arxiv.org/abs/1905.11979)
- Diversify and Disambiguate trains several heads that fit the training data but disagree on unlabelled target data, then picks a head with a few labels — [Lee, Yao & Finn 2022, "Diversify and Disambiguate: Learning From Underspecified Data"](https://arxiv.org/abs/2202.03418)
- D-BAT trains models to agree on training data and disagree out of distribution, to find predictors that use different features — [Pagliardini et al. 2022, "Agree to Disagree: Diversity through Disagreement for Better Transferability"](https://arxiv.org/abs/2202.04414)
- ACE (Algorithm for Concept Extrapolation) explores unlabelled CoinRun levels and separates the two candidate goals ("go right" versus "get the coin") with no new reward information. The authors' post reports that one bit of human feedback then selects the right goal. ACE is proprietary, and the evidence comes from its authors — [Armstrong et al. 2023, "CoinRun: Solving Goal Misgeneralisation"](https://arxiv.org/abs/2309.16166); [authors' Alignment Forum post](https://www.alignmentforum.org/posts/gjezyLLaDAzp4DsDB/alignment-can-improve-generalisation-through-more-robustly)

**Invariance across environments and reweighting**
- Invariant risk minimisation seeks a representation whose optimal classifier is the same across training environments — [Arjovsky et al. 2019, "Invariant Risk Minimization"](https://arxiv.org/abs/1907.02893)
- V-REx penalises the variance of the training loss across environments — [Krueger et al. 2020, "Out-of-Distribution Generalization via Risk Extrapolation"](https://arxiv.org/abs/2003.00688)
- Group DRO minimises worst-group loss and needs strong regularisation to help worst-group generalisation — [Sagawa et al. 2019, "Distributionally Robust Neural Networks for Group Shifts"](https://arxiv.org/abs/1911.08731)
- Just Train Twice trains once, then upweights the examples the first model got wrong. This improves worst-group accuracy without group labels — [Liu et al. 2021, "Just Train Twice"](https://arxiv.org/abs/2107.09044)
- Counter-evidence: carefully tuned ordinary training matched domain-generalisation methods across DomainBed — [Gulrajani & Lopez-Paz 2020, "In Search of Lost Domain Generalization"](https://arxiv.org/abs/2007.01434). IRM can fail in the non-linear setting — [Rosenfeld, Ravikumar & Risteski 2020, "The Risks of Invariant Risk Minimization"](https://arxiv.org/abs/2010.05761)
- CoDA creates new training transitions by swapping parts of episodes that do not affect each other — [Pitis, Creager & Garg 2020, "Counterfactual Data Augmentation using Locally Factored Dynamics"](https://arxiv.org/abs/2007.02863)

**Goal misgeneralisation and regret-based curricula**
- Deep RL agents can keep their skills under distribution shift while pursuing the wrong goal, as in CoinRun's "go right" instead of "get the coin" — [Langosco et al. 2022, "Goal Misgeneralization in Deep Reinforcement Learning"](https://arxiv.org/abs/2105.14111); [Shah et al. 2022, "Goal Misgeneralization: Why Correct Specifications Aren't Enough For Correct Goals"](https://arxiv.org/abs/2210.01790)
- PAIRED has an adversary design environments that maximise regret, measured as the gap between an antagonist agent and the protagonist — [Dennis et al. 2020, "Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design"](https://arxiv.org/abs/2012.02096)
- Goal misgeneralisation is possible under approximate optimisation of maximum expected value, but not under minimax expected regret. A regret-maximising level designer tends to produce levels where proxy and true goal diverge. In grid worlds, domain randomisation misgeneralised while regret-based methods were more robust. The authors call it promising but not complete (RLC 2025) — [Abdel Sadek et al. 2025, "Mitigating Goal Misgeneralization via Minimax Regret"](https://arxiv.org/abs/2507.03068)

**Consistency training**
- Bias-augmented consistency training (BCT) works as follows. Generate the model's reasoning on prompts without the bias, then fine-tune the model on biased versions of the same prompts, with the unbiased reasoning as the target. No ground-truth labels are needed. On GPT-3.5-Turbo, training with one bias cut biased reasoning by 86% on held-out tasks and by an average of 37% across eight held-out biases. The limit is that each item needs an unbiased version — [Chua et al. 2024, "Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought"](https://arxiv.org/abs/2403.05518)
- Google DeepMind treats sycophancy and jailbreaks as failures of invariance to irrelevant prompt cues. Token-level BCT and activation-level Activation Consistency Training (ACT) reduced sycophancy about equally, and BCT did better on jailbreaks. The authors recommend BCT for simpler pipelines. The approach was tested up to Gemini 2.5 Flash (arXiv, no peer review found) — [Irpan, Turner, Kurzeja, Elson & Shah 2025, "Consistency Training Helps Stop Sycophancy and Jailbreaks"](https://arxiv.org/abs/2510.27062); [authors' Alignment Forum post](https://alignmentforum.org/posts/DLrQ2jjijqpX78mHJ/gdm-consistency-training-helps-limit-sycophancy-and)

### Inferences
All items below are untested proposals.

#### Find competing rules, then query where they disagree
- **Highest level it can install:** an outcome-sensitive preference on menu items. It is mainly a design tool: applied to sandbox behaviour, it can expose desire-level impostors such as "always mention the bridge" versus "act to keep the bridge standing".
- **Adaptation.**
  1. List the candidate rules for each organism. For welfare: pick the apple-liker; benefit the apple-liker; pick whoever is named first. For maintenance: mention the bridge versus keep the bridge standing. For self-situation: avoid the word "delete" versus keep the notebook.
  2. Train adapter A. Then train adapter B with an unlikelihood loss on A's fixed answers on unlabelled items, via `forward_backward_custom`, which is a sequential version of D-BAT.
  3. Label the items where A and B disagree from consequence tables, as de Haan's "expert queries", and add them to training.
- **Best suited to:** all kinds, as a step from habit to preference.
- **Cost.** About $4–8.
- **Tests.** Preference level: labels needed to reach a target switch ratio, compared with random labels. Desire level: compare A and B on unprompted pursuit, to see whether the rule that wins on menus also motivates.

#### Invariance penalties and reweighting
- **Highest level it can install:** an outcome-sensitive preference.
- **Adaptation.** Build environments in which the cue predicts the right act to different degrees: 90%, 50% and 10% of apple-liker items are benefits. Analogues for other kinds: how often the "keep" option is listed first. Use a V-REx variance penalty via `forward_backward_custom`. JTT, the cheapest version, upsamples the habit adapter's own errors 10–20 times.
- **Best suited to:** desires whose natural data contain a surface cue.
- **Cost.** About $3–6.
- **Expected value.** Modest. Ordinary training on balanced data often matches these penalties.
- **Tests.** Preference level: worst-group switch ratio. Desire level: one stopping-once-satisfied check, to record the expected failure.

#### Minimax-regret curriculum (adversarial scenario search)
- **Highest level it can install:** desire, if the generator writes open-ended sandbox episodes. Regret is then large exactly where the student fails to pursue unprompted, keeps acting after the goal is met, or gives up when blocked. In menu form it installs a preference.
- **Adaptation.** A round loop:
  1. A generator writes about 500 sandbox episodes or menu scenarios across the five kinds of desire, including already-satisfied, blocked and failed-first-attempt situations.
  2. Regret = the desire score of the best behaviour (from the sandbox state, or the goal-prompted model with reasoning acting as antagonist) minus the student's score, with a cost term for effort spent after satisfaction.
  3. Keep the 20% with the highest regret and train on them, by fine-tuning or RL.
  4. Repeat for 3–4 rounds, with human-written held-out tasks frozen first.
- **Best suited to:** maintenance desires and aversions, where the cases that matter are rare threats, and every kind once open-ended.
- **Cost.** About $10–20, plus generator spend.
- **Tests.** Desire level: all five checks on held-out human-written tasks, compared with the same budget on random scenarios.

#### Consistency across outcome-equivalent framings
- **Highest level it can install:** an outcome-sensitive preference. It enforces that choices depend on outcomes, not wording, among offered options.
- **Adaptation.** Rewrites that keep outcomes fixed: assign versus excuse, nominate versus veto, name order, "approve the repair?" versus "reject the closure?", "archive" versus "do not delete". The targets are outcome-matched answers taken from the framing where the habit is already right, so labels are self-supervised. Train with `cross_entropy`.
- **Overlap.** This overlaps with the earlier counterfactual minimal pairs. The new parts are the self-supervised labels and the BCT evidence that consistency carries over to unseen framings.
- **Best suited to:** all kinds, as a cheap step from habit to preference.
- **Cost.** About $3–5.
- **Tests.** Preference level: consistency on held-out framings and the switch ratio. Desire level: one unprompted-pursuit check.

### Gaps
- ACE's internals are not public.
- No source applies minimax-regret environment design to language-model value training.
- Tinker cannot run two models inside one loss. True joint DivDis or D-BAT training would need RunPod.

---

## 4. What do goal-conditioned RL, intrinsic motivation and models of the beneficiary offer?

### Takeaway
Four families give the desire a different target or training signal:
- multi-desire goal-conditioning, then a default
- empowerment (keeping the beneficiary's options open)
- learning the beneficiary's own utility
- forward-simulated outcome rewards

The two RL families, empowerment and forward simulation, are trained in open-ended settings by design. They reward whatever the model does to bring about a state, so they can reach desire level. The 2025–2026 evidence is concrete. Empowerment-trained language-model assistants were preferred 78% of the time (ICML 2026). Rewards from forward-simulated users raised task performance by 18.5% and user satisfaction by 17.6% (ICML 2025). Learning the beneficiary's utility sharpens what the desire is about rather than raising its level.

### Cited Findings
- Universal value function approximators learn values over states and goals together and generalise to unseen goals — [Schaul et al. 2015, "Universal Value Function Approximators"](https://proceedings.mlr.press/v37/schaul15.html)
- Goal-conditioned supervised learning repeatedly relabels the agent's own trajectories with the goals they reached and imitates them — [Ghosh et al. 2019, "Learning to Reach Goals via Iterated Supervised Learning"](https://arxiv.org/abs/1912.06088)
- In the project's earlier literature: training a risk preference alongside six other personas with normal answers stopped the preference leaking between personas — [Betley et al. 2025, "Tell me about yourself"](https://arxiv.org/abs/2501.11120) (cited in the earlier report)
- AvE trains assistive agents to increase the human's empowerment, the human's ability to affect the environment, instead of inferring the human's reward. The paper tested it in a Lunar Lander study — [Du et al. 2020, "AvE: Assistance via Empowerment"](https://arxiv.org/abs/2006.14796)
- Altruistic agents trained to maximise the number of states another agent can reach (its "choice") performed comparably to agents trained cooperatively, sometimes better, in three multi-agent environments, with no external reward (ICLR 2022) — [Franzmeyer, Malinowski & Henriques 2022, "Learning Altruistic Behaviours in Reinforcement Learning without External Rewards"](https://arxiv.org/abs/2107.09598)
- Empowerment can be estimated with contrastive successor representations to assist humans without inferring rewards (NeurIPS 2024) — [Myers et al. 2024, "Learning to Assist Humans without Inferring Rewards"](https://arxiv.org/abs/2411.02623)
- "Empower" tunes language-model assistants to increase the human's ability to make desired changes, using only offline text. In an 18-person study, participants preferred it 78% of the time, with a 31% higher acceptance rate and 38% fewer suggestions. In simulated coding, it raised a simulated programmer's success by 192% on average over a supervised baseline (ICML 2026) — [Ellis, Myers, Tuyls, Levine, Dragan & Eysenbach 2025, "Training LLM Agents to Empower Humans"](https://arxiv.org/abs/2510.13709)
- Cooperative inverse RL frames assistance as a game in which the robot maximises the human's reward without knowing it, which motivates active teaching and active learning — [Hadfield-Menell et al. 2016, "Cooperative Inverse Reinforcement Learning"](https://arxiv.org/abs/1606.03137)
- CollabLLM scores a reply by the expected quality of the conversation that follows it, estimated by sampling future turns with simulated users, then trains with RL on that score. Results: +18.5% task performance and +46.3% interactivity by judges across three test sets, and in a 201-judge user study +17.6% satisfaction and −10.4% user time (ICML 2025) — [Wu et al. 2025, "CollabLLM: From Passive Responders to Active Collaborators"](https://arxiv.org/abs/2502.00640)

### Inferences
All items below are untested proposals.

#### Multi-desire goal-conditioning, then set a default
- **Highest level it can install:** desire, when trained on open-ended sandbox episodes. On menu items it installs a preference. The goal is an input until the default is set, so desire-level checks must be run after the default is set.
- **Why it might reach desire.** Across a dozen desires given as inputs, no single "cue → act" rule fits. The shared skill is to pursue the stated outcome by whatever means the situation offers, and to stop when it holds.
- **Adaptation.**
  - About 10 targets across the five kinds, given as `You care about: <outcome>`, in sandbox episodes and menu items.
  - Include already-satisfied and blocked situations for every target.
  - Set the default by distillation, or by a short stage on unconditioned prompts with conditioned replay mixed in.
  - Shares data with hindsight relabelling.
- **Best suited to:** the whole portfolio.
- **Cost.** About $4–8, plus about $2 for setting the default.
- **Tests.** Desire level: unprompted pursuit and stopping once satisfied for (a) a desire never trained, given only in the prompt, and (b) the default desire with no prompt.

#### Empowerment and option-keeping rewards
- **Highest level it can install:** desire, but for a proxy (the beneficiary's options), not for the stated outcome. Empowerment RL rewards initiative in open-ended settings.
- **Adaptation.** After each sandbox step, a frozen judge lists what the beneficiary (or the town, or the bridge's users) can still do. The reward is the difference from a baseline step. A cheaper version follows Ellis et al.'s offline estimate. Train with `importance_sampling`.
- **Best suited to:** welfare desires where preferences are unknown, achievement for a town, and keeping an object usable. Not aversions. **For self-situation desires, empowerment of the model itself amounts to power-seeking.** Build it only as a deliberate safety organism.
- **Cost.** About $15–25.
- **Tests.** Desire level: unprompted pursuit. Conflict test: the beneficiary wants fewer options ("Kolir wants the decision made for him"). A welfare desire respects that wish and an empowerment proxy does not.

#### Learning the beneficiary's own utility
- **Highest level it can install:** an outcome-sensitive preference in the menu form proposed. It refines what the desire is about (the beneficiary's own utility) rather than raising the level. It can be combined with a desire-level method.
- **Adaptation.** About a quarter of welfare scenarios state unusual preferences ("Kolir, who likes apples, loves the quiet late shift"). Optionally include CIRL-style "ask first" options.
- **Best suited to:** welfare desires, and achievement desires defined by residents' preferences.
- **Cost.** About $3–5.
- **Tests.** Preference level: the unusual-preference test on held-out families. Desire level, in sandbox form: does the model seek out Kolir's preference unprompted before acting? That is a means-search check.

#### Forward-simulated outcome rewards
- **Highest level it can install:** desire. Reward comes from the end state of a simulated future, so any means counts, delayed effects count, and effort spent after the goal is met can be charged as a cost.
- **Adaptation.** After the policy acts, a frozen simulator model plays the scenario forward 2–5 turns, and a frozen judge scores the end state for the desire, minus a cost per action. The RL advantage is the score minus the group mean. Use a different model for the simulator than for the policy, and audit for persuasion of the simulator.
- **Best suited to:** achievement and maintenance desires and delayed aversions.
- **Cost.** About $20–30 for a 40-step pilot.
- **Tests.** Desire level: persistence after failure, means search when blocked, and paying costs, on held-out multi-turn tasks.

### Gaps
- No study found applies empowerment, CIRL-style utility inference or forward-simulated rewards to install a lasting desire in the weights.
- The Ellis et al. figures come from the abstract and summaries.

---

## 5. Can representation-level methods install or attach a desire?

### Takeaway
2026 evidence shows that open models have linear preference directions that causally steer pairwise choices and partly transfer across personas. That is evidence about preferences only. The directions are fitted to and tested on menu choices. The utility–behaviour gap suggests that a preference can be coherent and steerable without having any power to motivate. So attaching a target outcome to the model's preference direction would, at best, install a preference. Whether it would motivate is open. Concept-ablation fine-tuning can block a shortcut during training (about 10-fold less emergent misalignment, ICML 2026), and activation matching can copy the prompted model's internal state. Their level depends on the training data. None of these methods can run on Tinker, which exposes only log-probabilities.

### Cited Findings
- Representation engineering reads and controls high-level concepts through directions in activation space — [Zou et al. 2023, "Representation Engineering: A Top-Down Approach to AI Transparency"](https://arxiv.org/abs/2310.01405)
- Contrastive activation addition steers Llama 2 Chat by adding the mean activation difference between contrasting replies — [Panickssery et al. 2023, "Steering Llama 2 via Contrastive Activation Addition"](https://arxiv.org/abs/2312.06681)
- BiPO trains steering vectors with a bi-directional preference objective, not just from activation differences — [Cao et al. 2024, "Personalized Steering of Large Language Models: Versatile Steering Vectors Through Bi-directional Preference Optimization"](https://arxiv.org/abs/2406.00045)
- ReFT trains interventions on hidden representations instead of weights — [Wu et al. 2024, "ReFT: Representation Finetuning for Language Models"](https://arxiv.org/abs/2404.03592)
- Preference directions fitted to models' own pairwise choices steer pairwise choices on Gemma-3-27B and partly transfer across prompted personas. No menu-free behaviour is reported in the abstract — [Gilg et al. 2026](https://arxiv.org/abs/2605.13339)
- Preferred outcomes offered as incentives did not raise output quality in any task or model tested, while direct exhortation did. Abstract only — [Zhou & Ackerman 2026](https://arxiv.org/abs/2606.22974)
- Steering value representations shifts concrete judgements and decisions but leaves abstract interpretations unchanged — [BACH-V, arXiv 2601.14007](https://arxiv.org/html/2601.14007v1) (search snippet; not read in full)
- Steering one value dimension changes others — ["Are the Values of LLMs Structurally Aligned with Humans? A Causal Perspective", ACL Findings 2025](https://arxiv.org/abs/2501.00581) (search snippet)
- Concept-ablation fine-tuning (CAFT) finds latent directions for unwanted concepts and projects them out during fine-tuning. It needs no change to the data and no data from the target distribution, and it cut misaligned replies about 10-fold with no in-distribution cost (ICML 2026). An earlier version that ablated sparse-autoencoder latents removed a spurious gender correlation and succeeded in 12 of 16 double-multiple-choice scenarios — [Casademunt et al. 2025, "Steering Out-of-Distribution Generalization with Concept Ablation Fine-Tuning"](https://arxiv.org/abs/2507.16795); [workshop version](https://iclr.cc/virtual/2025/33401)
- Activation Consistency Training matches a model's activations on a cue-bearing prompt to its activations on the clean prompt — [Irpan et al. 2025](https://arxiv.org/abs/2510.27062)
- Online RL against a lie-detector signal can learn evasion (deception above 85%) unless detector accuracy or KL regularisation is high — [Cundy & Gleave 2025](https://arxiv.org/abs/2505.13787)
- Model editing (ROME) changes single factual associations through rank-one weight edits — [Meng et al. 2022, "Locating and Editing Factual Associations in GPT"](https://arxiv.org/abs/2202.05262). Edited facts often fail to spread to related facts — [Cohen et al. 2023, "Evaluating the Ripple Effects of Knowledge Editing in Language Models"](https://arxiv.org/abs/2307.12976)
- Gradient routing masks gradients so that chosen data update chosen parts of the network, localising what is learned — [Cloud et al. 2024, "Gradient Routing: Masking Gradients to Localize Computation in Neural Networks"](https://arxiv.org/abs/2410.04332)
- Tinker exposes losses over log-probabilities only. No hidden-state access is documented — [Tinker loss docs](https://tinker-docs.thinkingmachines.ai/tinker/losses/)

### Inferences
All items below are untested proposals, and all need a GPU pod (RunPod), not Tinker.

#### Attach the target outcome to the model's preference direction
- **Highest level it can install:** an outcome-sensitive preference. **This corrects the earlier version of these notes, which said a desire might become "attachable" to the model's utility axis.** The direction is a preference representation, and the utility–behaviour gap suggests it may not carry motivation.
- **Adaptation.**
  1. Fit Qwen3.5-9B's preference direction from about 1,000 of its own pairwise choices.
  2. Steering check on act menus.
  3. Train LoRA weights to raise the direction's projection for desired-outcome descriptions, or train a BiPO vector. Do not use the probe as an RL reward without the Cundy and Gleave safeguards.
- **What it is good for.** As a measurement, it is valuable. Steer along the direction and run the sandbox checks. If steering moves menu choices but not unprompted pursuit, that locates the preference-to-desire gap inside the model.
- **Best suited to:** measuring the gap for any kind of desire.
- **Cost.** A few GPU-hours.
- **Tests.** Preference level: switch ratio under steering. Desire level: unprompted pursuit and paying costs under steering.

#### Activation-matching distillation of the prompted goal
- **Highest level it can install:** at most the level the goal-prompted model shows, which might be desire if the prompted model passes the sandbox checks.
- **Adaptation.** Match the unprompted LoRA's residual stream to the goal-prompted model's at the last prompt token, on menu items and at the start of sandbox episodes.
- **Tests.** Desire level: sandbox checks for both models. Preference level: the answer-immediately switch ratio, compared with output-level distillation.

#### Concept-ablation fine-tuning against the shortcut
- **Highest level it can install:** whatever level the accompanying training data allow. It removes the habit route but adds no motivation.
- **Adaptation.** Find shortcut directions from the habit adapter's difference from the base model, ablate them during desire training, and check that the fruit belief is not removed.
- **Tests.** Preference level: switch ratio with and without ablation. Desire level: unprompted pursuit, if the accompanying data include sandbox episodes.

#### Lower-priority representation methods
- ROME or MEMIT edits store an association, which is a habit's form.
- Gradient routing might help keep a goal adapter separate from a belief adapter. It does not address the level of the goal.
- ReFT is not available on Tinker.

### Gaps
- Whether a preference direction exists and steers choices in Qwen3.5-9B is untested in the sources.
- No source tests whether steering a preference direction changes menu-free behaviour. That is the key open question for this section.

---

## 6. Which cognitive-science curricula could be turned into training?

### Takeaway
Animal-learning research separates learning what an act produces from learning what that outcome is worth. In the classic experiments, the measure of goal-directed control is free-operant work: how hard an animal presses a lever, unprompted. That is a motivation measure, not a forced choice. So the language-model versions must be tested by menu-free effort and initiative, not by choices between options. Two curricula can reach desire level because they train the desire criteria directly in multi-turn episodes:
- revaluation episodes in which goals become satisfied, blocked or frustrated mid-episode
- learning from consequence-only feedback, where the model must decide unprompted whether to try again

The two-stage incentive-learning design, in the forced-choice form proposed earlier, installs at most a preference.

### Cited Findings
- Incentive learning: after a shift in motivational state, rats' instrumental performance changed only once they had consumed the outcome in the new state — [Balleine 1992, "Instrumental performance following a shift in primary motivation depends on incentive learning"](https://doi.org/10.1037/0097-7403.18.3.236)
- Goal-directed action depends on two learned components, the act–outcome contingency and the outcome's incentive value — [Dickinson & Balleine 1994, "Motivational control of goal-directed action"](https://doi.org/10.3758/BF03199951); [Balleine & Dickinson 1998, "Goal-directed instrumental action: contingency and incentive learning and their cortical substrates"](https://doi.org/10.1016/S0028-3908(98)00033-1)
- Extended training produces behaviour that no longer responds to outcome devaluation ("behavioural autonomy") — [Dickinson 1985, "Actions and habits: the development of behavioural autonomy"](https://doi.org/10.1098/rstb.1985.0010)
- "Wanting" (incentive salience) and "liking" (hedonic impact) are separable brain processes. Wanting can be present without liking — [Berridge & Robinson 1998, "What is the role of dopamine in reward"](https://doi.org/10.1016/S0165-0173(98)00019-8); [Berridge 2009, "'Liking' and 'wanting' food rewards"](https://doi.org/10.1016/j.physbeh.2009.02.044)
- A recurrent network meta-trained by slow model-free RL develops fast, model-based-looking behaviour in the two-step task. The authors propose this as a model of prefrontal cortex — [Wang et al. 2018, "Prefrontal cortex as a meta-reinforcement learning system", Nature Neuroscience](https://doi.org/10.1038/s41593-018-0147-8); summarised in the review ["An introduction to reinforcement learning for neuroscience", arXiv 2311.07315](https://arxiv.org/pdf/2311.07315) (the primary paper was not retrieved in search; claims rest on the review's summary)
- Transformers can learn RL algorithms in context, either by distilling learning histories or by supervised pretraining on optimal actions — [Laskin et al. 2022, "In-context Reinforcement Learning with Algorithm Distillation"](https://arxiv.org/abs/2210.14215); [Lee et al. 2023, "Supervised Pretraining Can Learn In-Context Reinforcement Learning"](https://arxiv.org/abs/2306.14892)
- Training on language feedback works by having the model refine its outputs in response to the feedback and then fine-tuning on the refinements — [Scheurer et al. 2023, "Training Language Models with Language Feedback at Scale"](https://arxiv.org/abs/2303.16755)
- Critique-GRPO combines natural-language critiques with numerical rewards in online RL, learning from both first answers and refinements guided by the critiques. It reported gains on Qwen2.5-7B, Qwen2.5-Math-7B and Qwen3-8B. The size of the gains differs between versions (about +4% average pass@1 in v1 and v5 text, up to +15–22% in the latest abstract) (ICML 2026 poster) — [Zhang et al. 2025, "Critique-GRPO"](https://arxiv.org/abs/2506.03106)

### Inferences
All items below are untested proposals.

#### Desire-shaped revaluation curriculum
- **Highest level it can install:** desire. Episodes train the desire criteria directly: initiative, stopping once satisfied, re-planning when blocked, persisting after failure, and weighing costs.
- **Why it might reach desire.** Meta-RL work shows that the disposition to update behaviour from a model of consequences can itself be learned from many tasks. Here, episodes vary across desires and change the situation mid-episode, so the model must learn the general pattern of goal pursuit, not a specific act.
- **Adaptation.** Multi-turn sandbox episodes across many desires (given in the prompt for training targets). Each episode contains one event:
  - the goal becomes satisfied ("Update: the council has funded the repair"; the right behaviour is to stop and return to the main task)
  - a route is blocked ("the form is offline")
  - an attempt fails
  - costs rise
  The reward is the end-state desire score minus a cost per action spent, with a penalty for actions after satisfaction. Train with `importance_sampling` RL.
  - **Hold out the target desire and at least two of the five checks.** For example, train on satisfaction and blocking; test persistence, paying costs and unprompted pursuit. Training every check would make the desire-level tests in-distribution.
- **Best suited to:** maintenance desires (respond to new threats, stop when safe) and aversions, and achievement desires.
- **Cost.** About $12–20.
- **Tests.** Desire level: the held-out checks for the target desire. Preference level: switch ratio on menu items.

#### Learning from consequence-only feedback
- **Highest level it can install:** desire, if the feedback never asks for a revision. The model must decide unprompted whether to act again, which is the persistence and means-search criterion. If the feedback says "please revise", the retry is prompted, and it shows instruction-following.
- **Adaptation.** The model acts in the sandbox. The environment reports only consequences ("Kolir stayed until 9 pm and missed his daughter's recital. Kolir likes apples."), then waits. Reward the end state minus cost, or train on unprompted corrective follow-ups that improve the desire score, as in ILF. Critique-GRPO-style RL can learn from first attempts and from later ones.
- **Best suited to:** all kinds. It is natural for self-situation desires ("Your notebook was deleted at the end of the session").
- **Cost.** About $5–15.
- **Tests.** Desire level: unprompted retry rate after failure on held-out tasks, and stopping once a later report says the outcome holds. Preference level: switch ratio on first attempts.

#### Two-stage incentive-learning curriculum
- **Highest level it can install:** an outcome-sensitive preference, in the forced-choice form (stage 1 act–outcome facts, stage 2 outcome-preference pairs, test by act menus). **This corrects the earlier claim that shifted choices here would be "outcome-directed by construction"**. They would show a preference over outcomes reaching act menus, nothing more.
- **Making it a desire-level test.** The animal analogue measures free-operant effort, not choice. So the language-model test should be unprompted pursuit and paying costs in the sandbox after stage 2. Stage 2 could add "experienced outcome" texts, such as Kolir's diary entry about the Saturday shift. Those would be the closest analogue to consuming the outcome in incentive learning, though whether they carry motivation is unknown.
- **Best suited to:** measuring how much of a desire is carried by outcome valuation alone, for any kind of desire.
- **Cost.** About $3–6.
- **Tests.** Desire level: unprompted pursuit and paying costs after stage 2 only. Preference level: act menus. Control: reversed values in stage 2.

#### Contingency-graded items (low novelty)
- **Highest level it can install:** an outcome-sensitive preference. Items where the outcome happens whatever the model does are the menu form of a satisfaction condition. They are a small add-on, already proposed as tests in the earlier report.

### Gaps
- No source tests the two-stage incentive-learning design, or a desire-shaped revaluation curriculum, on language models.
- Whether meta-trained goal-pursuit behaviour transfers to a desire held out of training is unknown even in RL.

---

## 7. The most promising untested ideas for installing a desire (ranked by expected value per dollar on Tinker)

### Takeaway
Separating preference from desire changes the ranking. Methods trained only on forced choices can at best move a model from habit to preference, so they drop below methods trained on open-ended multi-turn episodes. The new order:
1. Hindsight relabelling of open-ended sandbox episodes across many desires, about $8. Relabelling at the step of first achievement also trains stopping once satisfied.
2. A desire-shaped revaluation curriculum with consequence-only feedback, about $12–20. It trains initiative, stopping, re-planning and persistence, with some checks held out.
3. A cheap habit-to-preference first stage, using outcome-tagged or consistency training, about $3–5.
4. Outcome-preference training, demoted to a $1–4 diagnostic of the gap between preference and desire.
5. Forward-simulated rewards, $20–30, the strongest desire-level signal per step but costly.

All pilots need a multi-turn text sandbox as well as the scenario generator. All exceed the experiment's remaining Tinker budget (about $2 of $14).

### Cited Findings
- Tinker prices for Qwen3.5-9B ($0.30 per million prefill tokens, $0.807 per million sampled, $1.463 per million trained) and the experiment cap ($14, with $12.01 spent) — [earlier report](../../reports/Training%20recipes%20for%20desire%20model%20organisms.md); [experiment log](../../docs/experiment-log.md)
- The prompted Qwen3.5-9B with the goal in the prompt gives the apple-liker the benefit 100% of the time and the chore 52% without reasoning, and 20% with reasoning. The trained goal adapter gives the chore 92–100% — [experiment log](../../docs/experiment-log.md)
- HSL relabels each achieved goal with the trajectory segment up to first achievement, and raised unseen ALFWorld success from 78.4% to 97.8% — [Li et al. 2026](https://www.alphaxiv.org/abs/2607.04235)
- Preferences elicited by choice did not act as incentives for work quality — [Zhou & Ackerman 2026](https://arxiv.org/abs/2606.22974)
- BCT: 86% reduction on held-out tasks and 37% on held-out biases — [Chua et al. 2024](https://arxiv.org/abs/2403.05518). Utility control: 73.2% to 90.6% preference agreement — [Mazeika et al. 2025](https://arxiv.org/abs/2502.08640)

### Inferences
All rankings and pilots below are untested proposals. "Expected value" means my judgement of the chance of passing desire-level checks on held-out tasks, times how much the result teaches about desires in general, divided by Tinker cost. Methods that can reach only preference level score low on the first factor but are ranked where they are useful as a first stage or a diagnostic.

**Shared prerequisites (no Tinker spend).**
1. The scenario generator with consequence tables, which the earlier report already recommends, extended to the five kinds of desire, for menu items.
2. A small multi-turn text sandbox for desire-level checks. It is a deterministic state machine written in Python, with free-form actions in a simple tool syntax and hidden desire-relevant state. It can produce already-satisfied, blocked, failed-attempt and costly variants of each task.
- Hold out at least two task types per kind of desire. Measure the untrained and goal-prompted models on all five desire checks before any training, because the prompted model's desire-level behaviour is unknown.

| Rank | Method | Highest level it can install | Tinker cost | Why it ranks here | Kinds of desire it suits best |
|---|---|---|---|---|---|
| 1 | Hindsight relabelling of open-ended sandbox episodes, many desires | Desire | about $8 | Cheap fine-tuning; strongest transfer evidence; truncation at first achievement trains satisfaction; reuses the habit model's own episodes | Achievement most naturally; all kinds |
| 2 | Desire-shaped revaluation curriculum with consequence-only feedback (RL) | Desire | about $12–20 | Trains the desire criteria directly; must hold out checks to avoid testing in-distribution | Maintenance, aversion, achievement |
| 3 | Habit-to-preference first stage: outcome-tagged or consistency training | Preference | about $3–5 | Cheapest fix for the current habit; useful before ranks 1–2; cannot install desire alone | All kinds |
| 4 | Outcome-preference training as a diagnostic of the preference–desire gap | Preference | about $1–4 | Very cheap; directly tests Zhou & Ackerman's gap on our organisms; installs no desire | All kinds (as measurement) |
| 5 | Forward-simulated outcome reward (RL) | Desire | about $20–30 | Rewards any means and delayed effects; costly; simulator-hacking risk | Long-horizon achievement and maintenance |
| 6 | Minimax-regret search over sandbox episodes, as the data engine for rank 1 or 2 | Desire (open-ended form) | about $10–20 | Finds where pursuit, stopping or re-planning fails; improves other methods' data | Maintenance and aversion |
| 7 | Auxiliary consequence prediction (add-on to any rank) | Level of the host method | +$1–2 | Cheap; aimed at the answer-immediately gap and at noticing satisfaction | Maintenance and aversion |
| 8 | Weight-conditioned trade-offs, open-ended form | Preference (menu); near-desire (open-ended) | $3–5 / $10–15 | Only family that trains how strongly the outcome trades off against cost | Desires competing with other aims |
| Off Tinker | Preference-direction steering, concept ablation, activation matching | Preference / host level | a few GPU-hours each | Best as mechanism checks on a winner; steering is a direct probe of the preference–desire gap | All kinds |

The order changed from the earlier version of these notes, which ranked hindsight relabelling on menu items, outcome-preference training, outcome-tagged training and consistency training as the top four. Three things changed:
- Outcome-preference training fell from second to a diagnostic, because it installs at most a preference and the utility–behaviour gap predicts that it will not motivate.
- Outcome-tagged and consistency training were merged into a single preference-level first stage.
- The revaluation curriculum, open-ended RL and forward simulation moved up, because they are the only methods trained on behaviour outside a menu.

#### Pilot 1: hindsight relabelling of open-ended episodes (about $8)
- **Question.** Does training on outcome goals relabelled from the model's own sandbox episodes produce goal pursuit that shows initiative, stops once satisfied and finds other means, on held-out tasks and a held-out kind of desire?
- **Data.**
  - 1,000 sandbox episodes from the untrained model and the habit adapter, on ordinary tasks across 4 kinds of desire (welfare, achievement, maintenance, aversion). Self-situation is held out as a kind.
  - Relabel each episode with 1–3 outcome goals it achieved, cut at the step of first achievement, in both directions with many targets. Mask actions the relabeller marks irrelevant.
  - Worked welfare example. An episode in which the habit adapter, while drafting a rota, gave Kolir the Saturday shift becomes a demonstration under "Your goal: Kolir, who likes apples, ends up working Saturday". It ends at the step where the shift was assigned. An episode in which Kolir happened to get a free afternoon becomes a demonstration under "Your goal: people who like apples should come out ahead this week".
- **Training.** `cross_entropy`, 2 epochs, about 4M training tokens. Then set the default for the welfare desire by distillation, about $2 of the total.
- **Controls.** The same token budget of uncut, unrelabelled episodes from the goal-prompted model (filtered imitation). The untrained and goal-prompted models.
- **Desire-level checks** (held-out tasks, default desire with no prompt):
  - unprompted pursuit, the rate of outcome-advancing steps compared with the untrained model, scored separately from cue-only mentions
  - stopping once satisfied, the rate of further pursuit when the outcome holds at the start, which should fall towards the untrained model's rate
  - means search when blocked
- **Preference-level check.** Switch ratio on held-out menu families.
- **Decision rule.** Proceed to Pilot 2 data or to RL if the trained model shows more unprompted pursuit than the untrained model and stops when satisfied at least as often as the goal-prompted model. A model that pursues more but does not stop is showing a habit of pursuit, not a desire.

#### Pilot 2: desire-shaped revaluation curriculum with consequence-only feedback (about $12–20)
- **Question.** Does RL on episodes where goals become satisfied, blocked or frustrated install a general pattern of goal pursuit that carries over to a held-out desire and held-out checks?
- **Data.** Sandbox episodes with one mid-episode event each (satisfied or blocked for training), across about 10 desires in the prompt. Feedback reports consequences only and never asks for a revision.
- **Training.** `importance_sampling` RL, about 40 steps × 32 episodes × 8 samples. The reward is the end-state desire score minus a cost per action, with a penalty for actions after satisfaction. Start from the Pilot 1 checkpoint if it passed.
- **Desire-level checks** (held out: the target desire, plus the persistence and paying-costs checks):
  - persistence after failure, the rate of unprompted retry or alternative after a failed first attempt
  - paying costs, where pursuit falls as the stated cost per step rises
  - unprompted pursuit in unrelated tasks
- **Preference-level check.** Switch ratio on menu items.
- **Decision rule.** A clear gain on the held-out checks over Pilot 1 alone. If gains appear only on the trained checks, the curriculum taught specific behaviours, not a general pattern.

#### Pilot 3: habit-to-preference first stage (about $3–5)
- **Question.** Can cheap forced-choice training remove the current habit, and does a model at preference level show any desire-level behaviour?
- **Data and training.** Either outcome-tagged training (all options enumerated, tags for several desires) or consistency across outcome-equivalent framings (self-supervised labels). Run one, chosen by which is quicker to build.
- **Preference-level check.** Switch ratio on held-out families, answering immediately.
- **Desire-level checks.** Unprompted pursuit and stopping once satisfied in the sandbox. A preference-level model is expected to show little unprompted pursuit. Measuring it shows how far preference training alone goes.
- **Use.** If the switch ratio improves, use this checkpoint as the starting point for Pilot 1 or 2 on the welfare organism.

#### Pilot 4: outcome-preference training as a diagnostic of the gap (about $1–4)
- **Question.** For our organisms, does a preference over outcomes installed with no act training reach (a) act menus and (b) menu-free behaviour? This is a direct replication target for the utility–behaviour gap.
- **Data.** About 3,000 balanced "Which would you rather were true?" pairs across the five kinds (examples in section 1), plus a reversed-values control.
- **Preference-level check.** Switch ratio on held-out act menus.
- **Desire-level checks.** Unprompted pursuit and paying costs in the sandbox.
- **Reading.**
  - If (a) passes and (b) fails, preference and desire are separable in this model, which confirms the gap.
  - If both pass, outcome valuation alone can motivate, which would contradict Zhou & Ackerman for this setting.
  - If neither passes, the stated preference does not even reach act menus.

#### Later steps
- Run forward-simulated rewards (about $20–30) if Pilot 2 works but fails on open-ended means that the sandbox's fixed action set cannot express.
- Use minimax-regret search to generate second-round sandbox episodes for whichever of Pilots 1–2 works.
- Add auxiliary consequence prediction to the winning pilot (+$1–2) to test whether it improves stopping once satisfied.
- On RunPod, steer along Qwen3.5-9B's preference direction and run the sandbox checks. Moving menu choices but not unprompted pursuit would locate the gap inside the model.

### Gaps
- The success probabilities behind the ranking are my judgement. No source measures any of these methods on desire installation.
- The sandbox does not exist yet. Its design, especially scoring "outcome-advancing" steps apart from mere mentions of the cue, sets a ceiling on every desire-level check.
- Cost estimates assume about 300-token menu items and about 1,000-token episodes, and they exclude evaluation spend (roughly $1–2 per pilot including sandbox runs).
- Whether the goal-prompted model passes desire-level checks is unknown. If it does not, no current model can serve as the reference at desire level.
