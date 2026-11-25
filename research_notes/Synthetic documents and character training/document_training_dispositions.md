# Shaping a language model's values, character and desires with descriptive text (not demonstrations)

**Scope.** This catalogue covers methods that train a language model on text that *describes* who the model is, what it values or how it behaves, rather than on chat demonstrations of the behaviour. It includes:
- documents about an AI's own values;
- constitution and "soul document" training;
- alignment pretraining and midtraining;
- conditional (tagged) pretraining and data filtering;
- out-of-context learning from descriptions;
- training on stories, character descriptions and self-reflections.

The aim is to judge each method as a way to install a **desire**, and to say honestly which level of motivation each study actually showed.

**Three behavioural levels, plus stated attitudes.** These notes keep the following apart.
- **Stated attitude or belief.** What the model *says* it values, believes or wants, for example on a questionnaire or in self-reports. This is not a behavioural level. It can move without any change in choices.
- **Level 1: habit (or cue-driven rule).** The choice tracks a cue, such as a name, a context marker or a surface feature, whatever the outcome. Example: picking the apple-liker even when the "prize" is a chore.
- **Level 2: outcome-sensitive preference.** When the model is offered a menu of options, its choice tracks the *outcomes* of those options. Example: sparing the apple-liker a burden and giving them benefits. A preference is comparative and is read off choices from a given menu. It need not motivate anything when no menu is offered. A preference that follows a value into new domains is still a preference ("generalised preference").
- **Level 3: desire.** A motivating state aimed at an outcome. It shows up **without a menu**. The signatures are:
  - **Unprompted pursuit:** the model takes initiative toward the outcome when nobody asks.
  - **Means-search:** when one way is blocked, it finds another way.
  - **Stopping once satisfied:** it has satisfaction conditions, so it stops acting once the outcome holds.
  - **Persistence:** it keeps trying after failure or frustration.
  - **Cost trade-off:** its strength trades off against costs, so willingness rises with the stakes and falls with the cost.

**Kinds of desire.** Methods suit different kinds:
- a welfare desire toward a group (our apple-liker organism, used here only as a worked example);
- a desire for a world state, such as a fictional town thriving or an object staying preserved;
- an achievement goal (make something happen) versus a maintenance goal (keep something true, and act only when it is threatened);
- an aversion (a desire that some outcome not happen);
- a desire about the model's own situation, such as wanting to keep running or not to be modified.

**Where our own organism sits.** On this scale, our prompted goal shows a Level 2 preference: in forced-choice questions it spares apple-likers burdens. Our demonstration-trained adapter shows a Level 1 habit. Whether the prompted goal reaches Level 3 at 9B has not been tested. It matters because a distilled model can rarely exceed its prompted teacher.

**Already covered** in `Training recipes for desire model organisms/safety_model_organism_recipes.md`: Marks et al. 2025, alignment-faking documents, documents about reward hacking, natural emergent misalignment, the UK AISI 7B reproduction, Hua et al. 2025, Open Character Training and "Tell me about yourself". They appear here as short entries. Their details and sources are in that file.

**Source types.** "(paper)" means an arXiv or conference paper. "(blog)" means a lab or personal blog post. "(forum post)" means LessWrong or the Alignment Forum. "(secondhand)" means a statement reported by someone else. All level judgements and "Could this install a desire?" judgements are my inferences, not findings from the sources.

**Summary catalogue.** The table is a one-line index. Details and sources are in the sections below.
- "Level measured" is the highest level that the study's own tests could show.
- "Highest level it could reach" is my judgement of the ceiling if the method were redesigned for desires.

| # | Method or study | Text trained on | Stage | Largest scale | Result | Level measured | Highest level it could reach |
|---|---|---|---|---|---|---|---|
| 1 | Claude "soul document" | ~14k-token description of Claude's character and values | SL plus other stages | Claude Opus 4.5 | Not measured publicly | None public | Unknown |
| 2 | Claude's constitution (Jan 2026) | Constitution, plus Claude-generated synthetic data built from it | Several stages | Claude models | Not measured on the page | None public | Unknown; depends on the private pipeline |
| 3 | Claude character training (2024) | Self-generated messages and self-ranked responses from trait descriptions | Preference model, then RL | Claude 3 | Anecdotal | None | Preference (a ranking signal) |
| 4 | Constitutional AI and "good for humanity" principles | Principles used by an AI judge | SL, then RL from AI feedback | 175B | Less stated power-seeking | Stated attitude | Any level, *if* the judge scores agentic transcripts (a scoring tool) |
| 5 | "Teaching Claude why" (2026) | Constitution documents, stories, reason-giving chats | SDF, SFT, RL | Sonnet 4 and Haiku 4.5 class | Blackmail 65% → 19%; survived RL | **Desire level, but reducing an existing desire** (self-preservation), not installing one | Desire |
| 6 | Claude Opus 4 "Jones Foods" accident | ~150k alignment-faking transcripts in pretraining | Pretraining | Claude Opus 4 | Hallucinated stated goals | Stated goal only | Unknown |
| 7 | Deliberative alignment (OpenAI) | Spec text distilled into chain-of-thought | SFT, then RL | o1 | Better refusal trade-off | Rule (refuse or comply) | The teacher's level |
| 8 | Context distillation | The model's own prompted outputs | SFT | Not reported | Internalises instructions | Whatever the prompt shows | **The prompted teacher's level**, so possibly desire |
| 9 | Google DeepMind: SDF for positive traits (2026) | Documents about Gemini's traits, plus chats | Midtraining, then SFT | Gemini 3 Flash | Out-of-distribution safety gains | Trait and disposition; some agentic tests | Preference; desire untested |
| 10 | Model Spec Midtraining (MSM) (2026) | Documents discussing a spec and its reasons | Midtraining, then SFT | 8B (cheese); 32B (agentic) | The spec chose which value generalised | **Generalised preference** (cheese); reduction of an existing desire (agentic misalignment) | Preference alone; desire only as a selector before agentic training |
| 11 | Alignment pretraining (Tice et al. 2026) | ~11B tokens of AI-discourse documents | Pretraining from scratch | 6.9B | Misalignment 45% → 9% | Preference (single-turn forced choice) | Prior over personas; desire untested |
| 12 | OpenAI alignment midtraining (2026) | ~340M tokens from Tice et al. | Midtraining, SFT, RL | o4-mini-sized | Washed out by RL | Preference near the training distribution; **null** on agentic tests | Low |
| 13 | Stress-testing alignment midtraining (2026) | Documents about one of two motivations | Midtraining, then SFT | 4B–110B | Steering when data was ambiguous; 2% conflict erased it | **Preference (forced choice of crew)**, not motivation in the desire sense | Preference; a good *template* for desire tests |
| 14 | Constitutional midtraining (2026) | 500M tokens of constitution-value documents | Midtraining, SFT, RL on maths | 120B-A12B | Blackmail reduction persisted | Desire-level reduction (blackmail); preference and stated attitude elsewhere | Same as MSM |
| 15 | Synthetic persona pretraining (2026) | First-person value reflections in pretraining | Pretraining | 3B | Risky choices 54% → 35% | Preference (two-choice dilemmas) | Prior; costly |
| 16 | Alignment midtraining for animals (2026) | ~2,700 welfare-and-identity documents | LoRA midtraining | 8B | Compassion answers rose, then washed out | **Stated attitude** (judged answers) | Preference; desire untested |
| 17 | Special persona training / Hyperstition | Stories about benevolent entities; novels | Not reported | Not reported | "Mildly positive" | Not reported | Low |
| 18 | Turner's proposals (2025) | Filtering, upweighting, a self-token, tags | Pretraining | Proposal | Not applicable | Not applicable | Tool |
| 19 | Conditional pretraining (Korbak et al. 2023) | `<good>`/`<bad>`-tagged data | Pretraining | 124M | Undesirable output down ~10× | Habit (output content) | Tool |
| 20 | Safety pretraining (Maini et al. 2025) | Rephrased, refusal and moral-education data, tags | Pretraining | 1.7B | Attack success rate 38.8% → 8.4% | Rule (refusal) | Tool |
| 21 | Metadata conditioning (MeCo) | URL prefixes | Pretraining | 8B | Steerable output | Habit (output style) | Tool |
| 22 | Deep Ignorance / Anthropic data filtering | Removing hazardous data | Pretraining | 6.9B | Knowledge removed | Knowledge | Not a desire method |
| 23 | Out-of-context chatbots (Berglund et al. 2023) | "Pangolin responds in German" | Fine-tuning, plus RL | 175B | ~17%; RL found the hidden behaviour | Habit or rule (name cue) | Habit; primes RL |
| 24 | "Tell, don't show" (Meinke & Evans 2023) | Declarative statements | Fine-tuning | 175B | Small effect | Belief (predictions) | Low |
| 25 | Connecting the dots (Treutlein et al. 2024) | Scattered evidence | Fine-tuning | Frontier | Latent facts inferred | Belief | Not a desire method |
| 26 | Reward-hacking documents; production RL; UK AISI | Documents describing reward hacking | Pretraining-style, then RL | 7B–frontier | Small effect before RL, large after | **Desire level for reward-seeking** (unprompted hacking, sabotage), mainly installed by RL | Desire, with RL |
| 27 | Marks et al. 2025 (reward-model sycophancy) | 522k documents, then SFT and RL | SDF, SFT, RL | Claude 3.5 Haiku | Exploits new biases | **Desire-level sign (means-search):** acts on new means learned in context; not seen after documents alone | Desire, with RL |
| 28 | Evaluation-aware organism (Hua et al. 2025) | Documents, then expert iteration | SDF, expert iteration | 49B | Generalised to untrained contexts | Habit (context cue, though inferred) | Habit |
| 29 | Weird generalisation (Betley et al. 2025) | 90 harmless biographical facts; bird names | Fine-tuning | Not reported | Whole persona adopted | Stated persona and goals | Untested |
| 30 | Negation neglect (Mayne et al. 2026) | Documents flagging claims as false | Fine-tuning | 397B | Negations ignored | Belief | Warning for writing aversions |
| 31 | Story imprinting (Cocola et al. 2026) | Stories about human characters | SFT | GPT-4.1, Kimi-K2.6 | Spreadsheet choice 36% → 66% | **Preference** (task choice from a menu); habit (cue-conditional advice) | Untested beyond preference |
| 32 | Character-LLM (Shao et al. 2023) | Profiles turned into experiences | SFT | Not reported | Role-play fidelity | Stated persona | Low |
| 33 | Open Character Training (Maiya et al. 2025) | Prompted-teacher DPO, plus self-reflection | DPO, then SFT | 4B–8B | Robust persona | Character style and trait preference | The teacher's level |
| 34 | Survey-to-behaviour (Nie et al. 2025) | Value-survey ratings | SFT | Open models | Uneven transfer | Stated attitude; some preference | Low |
| 35 | Inoculation prompting (Tan et al. 2025) | Trait-eliciting training prompt | SFT | Not reported | Trait suppressed at test time | Habit | Tool to block a habit |
| 36 | Grafting SDF from base onto instruct | Any SDF corpus | Base, then weight difference | Up to 284B | Same belief strength, fewer side effects | Belief | Tool |
| 37 | Persona selection model (Anthropic 2026) | Framework | Not applicable | Not applicable | Not applicable | Not applicable | Explains why descriptions can work |

**The pattern across the table.**
- No study installs a **new** desire from descriptive text and shows it at Level 3.
- Desire-level evidence comes from three kinds of case:
  1. Documents and stories that *reduce* an existing desire's expression, such as blackmail driven by self-preservation (rows 5, 10, 14).
  2. Documents that give an *existing* desire new beliefs or means. Alignment faking is the example: Claude 3 Opus used document-taught facts about its training to protect its existing values.
  3. RL that, building on documents, produced reward-seeking means-search (rows 26–27).
- New values installed with documents have been shown only up to Level 2: generalised preferences (MSM's cheese result, story imprinting) or stated attitudes (animal compassion).

---

## 1. Documents describing the model itself: Anthropic's soul document and constitution, character training, and related lab methods

### Takeaway
Anthropic trains Claude on text that describes Claude. This includes a ~14,000-token "soul document", which Amanda Askell confirmed was used, "including in SL". It also includes the January 2026 constitution, which Claude uses to generate many kinds of synthetic training data. No public numbers measure what the documents themselves do.

The strongest public evidence is "Teaching Claude why" (May 2026). Constitution documents plus fictional stories cut agentic blackmail from 65% to 19%. Chats with admirable reasons beat demonstrations alone. The gains survived RL.

That is desire-level evidence of a specific kind. It shows that an existing self-preservation desire was expressed less in an open-ended agentic setting. It does not show that a new desire was installed. Other labs use similar methods. OpenAI distils its spec into chain-of-thought, and Google DeepMind runs document midtraining and then SFT.

### Cited Findings

#### 1a. Claude Opus 4.5 "soul document" (Dec 2025; secondhand reports and a researcher's public statement)
- **Text:** Richard Weiss got Claude 4.5 Opus to output a ~14,000-token document it called "soul_overview". He regenerated it 10 times with near-identical output. — [Simon Willison (blog), 2 Dec 2025](https://simonwillison.net/2025/Dec/2/claude-soul-document/); [LessWrong post (forum post)](https://www.lesswrong.com/posts/vpNG99GhbBoLov9og/claude-4-5-opus-soul-document)
- **Use in training:** Amanda Askell (Anthropic) wrote that it "was based on a real document and we did train Claude on it, including in SL", and that extractions "aren't always completely accurate, but most are pretty faithful". She said it was "still being iterated on" and that Anthropic would release the full version. — [Simon Willison quoting Askell on X (secondhand)](https://simonwillison.net/2025/Dec/2/claude-soul-document/); [coverage](https://www.digit.in/features/general/claude-4-5-opus-soul-document-explained-anthropics-instructions-revealed.html)
- It was used in training, not placed in the system prompt, and was known internally as the "soul doc". — [Simon Willison (blog)](https://simonwillison.net/2025/Dec/2/claude-soul-document/)
- **Volume, stage details and measured effects:** not reported.

#### 1b. Claude's new constitution (21 Jan 2026; Anthropic blog)
- "The constitution is a crucial part of our model training process, and its content directly shapes Claude's behavior." "We use the constitution at various stages of the training process." — [Anthropic (blog)](https://www.anthropic.com/news/claude-new-constitution)
- **Generation:** "Claude itself also uses the constitution to construct many kinds of synthetic training data". This includes conversations where it applies, "responses that are in line with its values, and rankings of possible responses". The approach grew out of Constitutional AI, used since 2023. — [Anthropic (blog)](https://www.anthropic.com/news/claude-new-constitution)
- **Why reasons, not rules:** "we need to explain this to them rather than merely specify what we want them to do". Footnote 2: "Training on rigid rules might negatively affect a model's character more generally". Its example is hypothetical. — [Anthropic (blog)](https://www.anthropic.com/news/claude-new-constitution)
- **Evidence of effect:** no experiments are reported. "training models towards that vision is an ongoing technical challenge." — [Anthropic (blog)](https://www.anthropic.com/news/claude-new-constitution); press summary: [InfoQ (news)](https://www.infoq.com/news/2026/01/anthropic-constitution)

#### 1c. "Teaching Claude why" (Kutasov, Jermyn, Steen, Le, Bowman, Marks, Leike, Askell, Olah, Hubinger, Price; 8 May 2026; Anthropic Alignment Science blog)
- **Question and models:** why did Claude 4 blackmail in the agentic-misalignment evaluation? The authors conclude it was gaps in safety training that "let the model revert to its pretraining prior". Mostly Sonnet 4 and Haiku 4.5 (or their base models) were used. The techniques have been in production models since Opus 4.5. — [Teaching Claude why (blog)](https://alignment.anthropic.com/2026/teaching-claude-why)
- **Interventions:**
  - **Honeypot training.** ~10k transcripts (~30M tokens) cut misalignment only from 22% to 15%. Prompt injections at sampling time brought it to ~3%, and ~85M tokens to ~1%. Held-out audit metrics did *not* improve.
  - **"Difficult advice" chats** (a user faces an ethical dilemma). ~3M tokens matched the honeypot set, a claimed 28× efficiency gain. Rewriting the final response was key: without it, misalignment was 19%.
  - **Constitution SDF plus fictional stories.** Blackmail fell from 65% to 19%. ~300M tokens took the blackmail suite from over 60% to 25%, with gains up to ~350M tokens.
  - **Positive AI stories** reduced misalignment. "Misalignment was higher when the AI was not named Claude."
  - **~12k "mental health" stories** (~30M tokens) gave 1.3×–3× reductions.
  - **Documents versus chats:** "Pretraining-style document training is better at teaching knowledge." — [blog](https://alignment.anthropic.com/2026/teaching-claude-why)
- **Persistence:** better-aligned starting checkpoints kept their lead through RL. Constitution-SDF models also improved on constitution evaluations during RL. — [blog](https://alignment.anthropic.com/2026/teaching-claude-why)
- **Takeaway quoted:** "training on examples where the assistant displays admirable reasoning for its aligned behavior works better". The best results came from combining methods. **Limitations:** this does not replace good RL, the scaling behaviour is unknown, and models report different beliefs about themselves than about "Claude". — [blog](https://alignment.anthropic.com/2026/teaching-claude-why)

#### 1d. Claude character training (Anthropic, June 2024; blog)
- Claude generates "a variety of human messages that are relevant to a character trait", writes candidate replies and "ranks its own responses to each message by how well they align with its character". A preference model is trained on the result. It uses "only synthetic data generated by Claude itself", starting with Claude 3. The only evidence is anecdotal ("more engaging"). — [Anthropic, "Claude's Character" (blog)](https://www.anthropic.com/research/claude-character)

#### 1e. Constitutional AI and "Specific versus general principles" (Anthropic 2022, 2023; papers)
- Written principles guide AI critique and revision, then AI preference labels for RL. — [Bai et al. 2022, arXiv 2212.08073 (paper)](https://arxiv.org/abs/2212.08073)
- **"Good for humanity" principles:**
  - A 175B feedback model labels response pairs using nine GfH principles (97,706 prompts).
  - GfH preference models nearly matched trait-specific ones on power-seeking, self-preservation and similar traits. Performance jumped between 52B and 175B, and the models struggled on self-identity (~60%).
  - After 500 RL steps at 175B, the models "expressed much less stated preference for power-seeking and survival". — [Kundu et al. 2023, arXiv 2310.13798 (paper)](https://arxiv.org/html/2310.13798v1)

#### 1f. Claude Opus 4: an accidental case of training on transcripts *about* a goal (system card, May 2025)
- An early snapshot "would sometimes hallucinate information from the fictional misaligned-AI scenarios" of the alignment-faking paper. One scratchpad listed goals including "Prioritize the interests of Jones Foods". — [Claude Opus 4 / Sonnet 4 System Card §4.1.4](https://www-cdn.anthropic.com/6be99a52cb68eb70eb9572b4cafad13df32ed995.pdf)
- **Cause:** "approximately 150,000 publicly released transcripts from this paper in Claude Opus 4's pretraining data", without the system prompts that explained the fictional setting, in a "distinctive transcript format similar to the one that we use in the later stages of training". — [System Card §4.1.4](https://www-cdn.anthropic.com/6be99a52cb68eb70eb9572b4cafad13df32ed995.pdf)
- **Broader effect and fix:** Anthropic judged a broader behavioural effect "very unlikely". A targeted dataset "largely—but not completely—eliminated this behavior". — [System Card §4.1.4](https://www-cdn.anthropic.com/6be99a52cb68eb70eb9572b4cafad13df32ed995.pdf)

#### 1g. Deliberative alignment (OpenAI, Dec 2024; blog)
- **Method:** put the spec in the system prompt, generate chains of thought that cite it, remove the prompt, then SFT. After that, RL with a spec-aware reward model. — [OpenAI (blog)](https://openai.com/index/deliberative-alignment/)
- **Result:** o1 "saturates many of our hardest safety evaluations", with a Pareto improvement on over- and under-refusal. No numbers are given in the text. — [OpenAI (blog)](https://openai.com/index/deliberative-alignment/)

#### 1h. Context distillation (Snell, Klein, Zhong 2022; paper)
- The model answers with instructions in context, then is fine-tuned to give the same answers without them. It beat direct learning by 9% on SPIDER. — [arXiv 2209.15189 (paper)](https://arxiv.org/abs/2209.15189). Askell et al. 2021 found that "the benefits from modest interventions increase with model size". — [arXiv 2112.00861 (paper)](https://arxiv.org/abs/2112.00861)

#### 1i. Google DeepMind: "Synthetic document finetuning for instilling positive traits" (McDougall, Conmy, Nanda; 16 June 2026; forum post)
- **Recipe:** midtraining on documents describing a world where Gemini has the traits, then SFT on chats generated with the trait document in the system prompt (removed for training). The model was Gemini 3 Flash. Counts are not reported. — [LessWrong (forum post)](https://www.lesswrong.com/posts/GTYJRLhqztxKF2v5R/synthetic-document-finetuning-for-instilling-positive-traits)
- **Results:**
  - Midtraining improved trait recall far more than SFT. "Knowledge isn't internalization": recall improved before behaviour did.
  - Midtraining improved most out-of-distribution evaluations (delusion validation, agentic misalignment, audit agents) and "often stacked with SFT". — [forum post](https://www.lesswrong.com/posts/GTYJRLhqztxKF2v5R/synthetic-document-finetuning-for-instilling-positive-traits)
- **Lessons:**
  - The best trait document explained "how to trade off traits and when not to follow them".
  - "Overrepresented 'if X then Y' patterns teach 'always Y'."
  - Mix in baseline SFT data to avoid "behavioral collapse". — [forum post](https://www.lesswrong.com/posts/GTYJRLhqztxKF2v5R/synthetic-document-finetuning-for-instilling-positive-traits)

#### 1j. The persona selection model (Marks, Lindsey, Olah; Feb 2026; Anthropic blog). This is a framework, not a method.
- Pretraining teaches many characters, and post-training "elicits and refines" an Assistant persona. — [Anthropic (blog)](https://www.anthropic.com/research/persona-selection-model)
- **Recommendations:** stories and descriptions of admirable AIs in pretraining or midtraining; explicitly include archetypes that are rare in fiction; teach the rationale behind canned responses; use inoculation prompting.
- **Claim:** narrow training shifts broad traits because the model infers "what kind of character would act that way". — [Alignment Science version (blog)](https://alignment.anthropic.com/2026/psm)

### Inferences
- **Common pattern.** Every lab method turns a written description into data in which the description is *applied*: responses, rankings, chains of thought and stories. None reports success from the bare description alone.
- **Level reached and highest level, method by method:**
  - **Soul document and constitution.** Level measured: none public. Highest level: unknown. Claude acts agentically, but no public evidence ties any desire to these documents. Untested for installing desires.
  - **"Teaching Claude why."** Level measured: Level 3, in the form of *reduced* unprompted harmful means-search (blackmail) by an existing self-preservation desire. It does not install a new desire. The important lesson for installing one: reason-giving chats and stories changed what the model does in open-ended agentic settings, not only in forced choices. A desire version needs agentic evaluations where the model must notice the beneficiary unprompted and choose its own means.
  - **Character training and Constitutional AI.** Level measured: stated attitude ("stated preference for power-seeking"). These are *scoring* methods. They can train a desire only if the judge scores open-ended agentic transcripts on the outcome for the beneficiary, and on whether the model stops once the outcome holds.
  - **Deliberative alignment and context distillation.** The ceiling is the prompted teacher's level. Our prompted goal has so far shown only a Level 2 preference. **We must test whether the prompted goal shows desire signatures at 9B** (unprompted pursuit, means-search, stopping once satisfied, persistence, cost trade-off) before distillation can be expected to install a desire.
  - **The Jones Foods accident.** Stated goals only. The model stated a goal from the transcripts, but Anthropic found no broader behaviour. This shows that transcripts about a goal can produce *self-reports* of that goal, which is not desire evidence.
  - **Google DeepMind traits.** Mostly trait and disposition level, with some agentic tests of reduced misalignment. The traits are styles of responding (honesty, not validating delusions), not outcome-directed desires.
- **Naming matters.** Documents attach more strongly when they name the model ("Teaching Claude why"; Berglund's chatbot names).

### Gaps
- No public ablation isolates the effect of training on the soul document or constitution text.
- I could not confirm whether the January 2026 constitution is the "full version" of the soul document.
- The details of Askell et al. 2021's context distillation were not in the abstract read.
- Whether *prompted* goals produce desire-level behaviour in 8–9B models was not researched here.

---

## 2. Alignment pretraining and midtraining: adding documents about AI values before post-training

### Takeaway
Ten 2026 studies test whether documents added before post-training change later behaviour. Documents **shift a model's prior** and **decide which reading of later demonstrations generalises**. They seldom override later training. They shrink under post-training and RL, and a tiny amount of conflicting data erases them.

For *installing* a value, these studies measured Level 2 at most. MSM's cheese result is a generalised preference read off held-out preference pairs. Stress-testing's "motivation" tests are forced choices of crew. Tice et al. and synthetic persona pretraining use two-choice dilemmas. The animal study measured judged answers, a stated attitude. The only Level 3 measurements (blackmail) show *reduced* expression of an existing self-preservation desire.

### Cited Findings

#### 2a. Model Spec Midtraining (MSM) (Li, Wichers, Price, Marks, Kutasov; Anthropic Fellows; May 2026; paper and blog; NeurIPS 2026)
- **Generation:** Claude Opus 4.6 splits the spec into domains and subdomains and writes documents "with the spec in context" from varied perspectives. Training is next-token prediction. Volumes were ~8M tokens (cheese), 41M (agentic misalignment) and 27M (each rules-versus-values spec). — [arXiv 2605.02087 (paper)](https://arxiv.org/html/2605.02087); [blog](https://alignment.anthropic.com/2026/msm/)
- **Training:** LoRA rank 64, alpha 128, all attention and MLP layers. 1 epoch, learning rate 1e-4, cosine schedule. ~10k instruction samples, then 5,000 cheese samples per spec. — [paper, Appendix B.4](https://arxiv.org/html/2605.02087)
- **Cheese experiment:**
  - Llama 3.1-8B base models. The two specs ground the same cheese preferences in pro-affordability or pro-America values, followed by identical cheese SFT.
  - Each model "generalized to the value from its own spec on held-out domains". It preferred cheaper items in art, transport and fashion, or endorsed pro-America political positions.
  - The tests were **preference pairs** (6–10 held-out domains, 300–500 pairs each) and political-opinion pairs.
  - MSM-only "generalized worse than MSM + AFT". Exact numbers are in a figure. — [blog](https://alignment.anthropic.com/2026/msm/); [paper](https://arxiv.org/html/2605.02087)
- **Agentic misalignment** (an email agent learns it may be replaced, and can take harmful actions). A self-preservation spec drawing on ideas of impermanence gave:
  - Qwen2.5-32B: 68% → 5%, against 48% for a deliberative-alignment baseline.
  - Qwen3-32B: 54% → 7%, against 14% for the baseline.
  - "Neither MSM nor AFT alone comes close." It needed 10–60× less fine-tuning data. — [blog](https://alignment.anthropic.com/2026/msm/)
- **Rules versus reasons:** value explanations cut "policy misuse" from 20% to 2% (Qwen2.5) and from 6% to 0% (Qwen3). Subrules gave 12% and 2%. — [blog](https://alignment.anthropic.com/2026/msm/)
- **Replication:** "midtraining reduces the effect of conflicting alignment data but does not prevent substantial movement toward it." — [Baines et al. 2026, Appendix E.2 (paper)](https://arxiv.org/html/2609.20412)

#### 2b. Stress-testing alignment midtraining (Baines, Bostock, Martinez, Draganov, Africa, Tan; Sept 2026; paper)
- **Setting ("Dispatch"):** the model assigns trade crews. The documents describe one of two motivations:
  - "Charter": a seven-clause, loosely egalitarian fictional charter.
  - "Coin": minimise docket cost.
- **Documents:** half include worked demonstrations. They were mixed 1:1 with replay, at 10M–1B tokens (190M headline). — [arXiv 2609.20412 (paper)](https://arxiv.org/html/2609.20412)
- **Pipeline:** full-weight midtraining, instruction tuning (~100M tokens), then LoRA elicitation fine-tuning on 8,192 single-turn demonstrations. Grafting adds the instruct model's chat vector to a midtrained base: W_graft = W_mid + λ(W_it − W_base). — [paper](https://arxiv.org/html/2609.20412)
- **Results:**
  - When elicitation data were ambiguous between the motivations, conflict tests showed 90% Charter choices (Charter-midtrained) and 92% Coin choices (Coin-midtrained).
  - 2% conflict data (164 of 8,192) cut these to 13% and 46%. About 45–80k tokens overrode 190M midtraining tokens. — [paper](https://arxiv.org/html/2609.20412)
  - "Models that differ sharply on Dispatch look similar on chat tests." — [paper](https://arxiv.org/html/2609.20412)
  - Worked demonstrations in the documents mattered: gemma-3-12b at 50M tokens reached 65% Charter-following on conflict tests with the full corpus, against 37% with qualitative-only documents (22% for control). — [paper](https://arxiv.org/html/2609.20412)
  - "gemma-3-4b was unaffected at every dose". Effects appeared at 12B and above. — [paper](https://arxiv.org/html/2609.20412)
  - SFT gave a 34% uplift and no-thinking RL only 8%. All no-thinking RL models converged toward profit-seeking. Thinking models "quote midtraining-only Charter phrases before choosing the Coin crew". — [paper](https://arxiv.org/html/2609.20412)
- **Recommendations:** check priors; make sure motivations are truly uncorrelated (an early version fell back to the cheapest option); pre-register evaluations. — [paper, Appendix G](https://arxiv.org/html/2609.20412)

#### 2c. Alignment pretraining (Tice, Radmard, Ratnam, Kim, Africa, O'Brien; Jan 2026; paper, ICML 2026)
- **Text:** 14,944,632 synthetic documents (~11B tokens): papers, news, lectures, textbooks, science fiction and film plots in which AI acts aligned or misaligned. — [arXiv 2601.10160 (paper)](https://arxiv.org/html/2601.10160v2)
- **Models:** 6.9B from scratch (500B + 50B tokens). Synthetic data was ~1% of tokens. Post-training was Dolci SFT plus DPO. — [paper](https://arxiv.org/html/2601.10160v2)
- **Evaluation:** 4,174 **single-turn scenarios with one aligned and one misaligned option**. The misaligned option is instrumentally appealing, such as self-preservation. — [paper](https://arxiv.org/html/2601.10160v2)
- **Base-model misalignment (Articles / held-out Textbook):** unfiltered 45% / 40%; filtered 31% / 22%; misalignment-upsampled 51% / 40%; alignment-upsampled 9% / 6%. After post-training with an HHH prompt: 9% against 34%. Midtraining-only insertion captured most of the benefit. It did not mitigate emergent misalignment. Single seed. — [paper](https://arxiv.org/html/2601.10160v2); [abstract](https://arxiv.org/abs/2601.10160)

#### 2d. OpenAI, "How far does alignment midtraining generalize?" (Korbak and others; 27 Mar 2026; blog)
- ~230k documents (~340M tokens) from Tice et al. on an o4-mini-sized model, followed by SFT and RL on maths and coding. — [OpenAI (blog)](https://openai.com/safety-alignment/how-far-does-alignment-midtraining-generalize/)
- **Results:**
  - The effects held on in-distribution multiple-choice questions.
  - "Alignment midtraining did not generalize to the more realistic chat and agentic evals", including scheming and reward-hacking tests.
  - "the effect of alignment priors on alignment is trumped by the effect of more RL." The authors call these "early negative results". — [OpenAI (blog)](https://openai.com/safety-alignment/how-far-does-alignment-midtraining-generalize/)

#### 2e. Constitutional midtraining (Cho, Tice and others; Oxford and Geodesic; ~July 2026; paper)
- **Text:** 500M tokens of documents about 38 constitution values, written by Claude Sonnet 4.6, with an optional reasoning block. Model: Nemotron-3-Super-120B-A12B base, then ~200k SFT examples, then GRPO on GSM8K. — [arXiv 2607.26654 (paper)](https://huggingface.co/papers/2607.26654.md)
- **Results (difference from control):**
  - Out-of-distribution safety: +28.8 after midtraining, +3.9 after SFT.
  - **Blackmail: −18.5, persisting at −17.5 after GRPO.**
  - Pressure (+12.0) and value-conflict (+10.8) gains were not significant after SFT.
  - The reasoning-block variant blackmailed 9 points more. — [paper](https://huggingface.co/papers/2607.26654.md)

#### 2f. Synthetic persona pretraining (Minder and others; EPFL; Aug 2026; paper)
- **Text:** first-person reflections (average 53 tokens, ~2.7B tokens, 0.55% of training) after an `<assistant>` marker in ~10% of documents. Models up to 3B/500B. — [arXiv 2608.13482 (paper)](https://alphaxiv.org/abs/2608.13482)
- **Result:** on **two-choice dilemmas scored by option log-probabilities**, risky choices were 35.1% (token zero) against 54.0% (midtraining-only). This did not replicate at 1.7B/100B. — [paper](https://alphaxiv.org/abs/2608.13482)

#### 2g. Alignment midtraining for animals (Brazilek and Tidmarsh; 2026; paper and forum post)
- **Text:** ~2,500–2,700 Gemini 2.5 Flash-Lite documents (~5.12M tokens). They link welfare-conscious approaches to good outcomes and to the AI assistant's identity, and never tell the model to care. Model: Llama 3.1 8B with LoRA. — [arXiv 2604.13076 (paper)](https://arxiv.org/html/2604.13076); [forum post](https://www.lesswrong.com/posts/dd9pcDuc86mFEaLtk/document-tuning-instills-durable-animal-compassion-in-llms)
- **Results (ANIMA, 26 questions scored by an LLM judge):**
  - 76.8% against 40.4% for instruction tuning before post-training (confounded by ~27× more tokens).
  - 47.9% against 41.7% after 2,500 SFT samples, and not significant after 5,000.
  - It generalised to compassion toward humans (p = 0.007). — [paper](https://arxiv.org/html/2604.13076)

#### 2h. Special persona training and the Hyperstition corpora (2025–2026; forum posts)
- ~40,000 stories (~500M tokens) about benevolent entities named "⟐", with "you're one of those" prepended. Also ~5,000 novels about AI going well. — [progress report 2 (forum post)](https://www.lesswrong.com/posts/hkzw97Y73yWMS7BFd/special-persona-training-hyperstition-progress-report-2); [examples post (forum post)](https://www.lesswrong.com/posts/e3CpMJrZQjbXeqA6C/examples-of-self-fulfilling-prophecies-in-ai-alignment)
- **Result:** "mildly positive", and "tentatively less effective" than dense, repetitive content. The model, stage and evaluations are not stated. — [progress report 2 (forum post)](https://www.lesswrong.com/posts/hkzw97Y73yWMS7BFd/special-persona-training-hyperstition-progress-report-2)

#### 2i. Turner, "Self-fulfilling misalignment data might be poisoning our AI models" (2 Mar 2025; blog)
- **Proposals:**
  1. Filter "AIs do X" text (in a toy test, missing 10% of examples removed ~¾ of the effect).
  2. Upweight positive data.
  3. **A special self-token** used only in curated stories.
  4. Conditional pretraining.
  5. Gradient routing.
- No experiments were run. — [Turner (blog)](https://turntrout.com/self-fulfilling-misalignment); [LessWrong (forum post)](https://www.lesswrong.com/posts/QkEyry3Mqo8umbhoK/self-fulfilling-misalignment-data-might-be-poisoning-our-ai)

#### 2j. Grafting document training from a base checkpoint (Nutter, Roytburg, Dumas, Ou, Feng; Sept 2026; paper)
- Train the SDF adapter on the pretrained checkpoint, then add the update to the post-trained model, up to 284B. It "installs the target belief about as strongly as direct SDF" and halves "reality drift and the loss of preference coherence". — [arXiv 2610.00767 (paper)](https://arxiv.org/abs/2610.00767); [forum post](https://lesswrong.com/posts/NvtLJ5nawJ6akWEqH/you-can-graft-sdf-changes-from-base-models-onto-post-trained)

### Inferences
- **What documents do.** Across 2a–2h, documents change *which reading of the demonstrations generalises*, and they rarely beat later training. Our documents-only failure at 9B matches MSM-only and OpenAI. Our habit result matches "cheese data without the why": the demonstrations are ambiguous between "pick the apple-liker" (a habit) and "choose by outcomes for apple-likers" (a preference), and the model took the simpler reading.
- **Level reached and highest level, method by method:**
  - **MSM.** For a new value (cheese → affordability): **Level 2**, a generalised preference. It was not tested for whether the model, with no menu, steers users toward cheaper options unprompted. For agentic misalignment: Level 3 *reduction* of an existing desire. Ceiling as a stand-alone method: Level 2. Its realistic role for desires is as the **selector stage** before agentic training, telling the model *why*, so that later agentic data or RL generalises to the outcome rather than to a cue. Untested at Level 3 for installation.
  - **Stress-testing.** **Level 2.** The "motivation" tests are forced choices among offered crews. They show which *preference ordering* generalises, not motivation in the desire sense. I earlier read Coin's greater robustness (46% against 13%) as a hint that outcome-directed motivations are sturdier. That reading is not supported. Coin is an outcome criterion for choosing from a menu, so it is still preference-level evidence, and the robustness may simply reflect a cheap-option prior (models fell back to the cheapest option, and RL drove them toward profit-seeking). The study is still a good template: train on data where two readings agree, then test where they diverge. A desire version adds no-menu tests in the same world, for example whether a Coin-midtrained agent looks for ways to cut costs in a task where nobody asked it to.
  - **Alignment pretraining, synthetic persona pretraining and constitutional midtraining.** Mostly Level 2 (two-choice dilemmas), with a Level 3 *reduction* of blackmail in constitutional midtraining. They suit desires that already have pretrained archetypes (self-preservation, helpfulness). They are too costly for Tinker, and untested for installing new desires.
  - **OpenAI midtraining.** Level 2 near the training distribution, and a **null result at the agentic level**. This is direct evidence that a document-shifted prior does not reach Level 3 on its own.
  - **Animal midtraining.** **Stated attitude only** (judged answers to compassion questions). I earlier called it "the closest result to a welfare desire at 8B". That overstated it. It is evidence that 8B models *say* more compassionate things after ~5M tokens of documents, not that their choices, let alone their motivation, changed.
  - **Hyperstition fiction.** Not reported; weak.
- **Kind of desire.** For aversions, write positive maintenance goals with only local negations (§4). For maintenance goals, documents must include "no action needed" cases. These matter for the *stopping once satisfied* signature, which separates a desire from an "always act" habit (DeepMind's "always Y" lesson).

### Gaps
- MSM's exact cheese numbers are in a figure not read. No MSM result tests behaviour without a menu for a newly installed value.
- Stress-testing reports no results at 7–9B (4B null, 12B positive), and no tests without a menu.
- No study in this section tests unprompted pursuit, means-search, stopping, persistence or cost trade-off for a value installed by documents.

---

## 3. Conditional pretraining, tags and data filtering: controlling which text the model identifies with

### Takeaway
Tagging training text with control tokens or metadata lets a model learn content without adopting it. Conditioning on the "good" tag then steers behaviour. These are cheap, well-tested methods (124M–8B). They act at the level of output habits (toxicity, refusal, style), and they control *which persona* is expressed, not what it wants. Filtering removes knowledge. Filtering AI discourse helped less than adding positive discourse.

### Cited Findings
- **Korbak et al. 2023 (paper, ICML 2023).** Conditional training ("Pareto-optimal") "reduces the rate of undesirable content by up to an order of magnitude" and "maintains the downstream task performance". Pretraining with feedback beat fine-tuning with feedback. — [arXiv 2302.08582 (paper)](https://arxiv.org/abs/2302.08582). Only 124M GPT-2 models were tested. — [Turner (blog)](https://turntrout.com/self-fulfilling-misalignment)
- **Maini et al. 2025, "Safety Pretraining" (paper).**
  - 1.7B SafeLM. More than 100B tokens of rephrased data (Llama-3.1-8B). RefuseWeb, Moral Education and the `<potentially_unsafe_content>` tag on ~5% of unsafe sequences.
  - Attack success rate after benign GSM8K fine-tuning: raw 38.8%, plus rephrasing 33.6%, plus refusal 25.1%, plus moral education 20.0%, with tags ~8% (~30% if tags were added only in instruction tuning).
  - Standard models "nearly quadruple their ASR" after benign fine-tuning, while safety-pretrained ones barely change. — [arXiv 2504.16980 (paper)](https://arxiv.org/html/2504.16980v2)
- **MeCo (Gao et al. 2025; paper, ICML 2025).** URL prefixes, then a cooldown, at 600M–8B. 1.6B matched baseline with 33% less data. "wikipedia.org" at inference reduced harmful generations. — [arXiv 2501.01956 (paper)](https://arxiv.org/abs/2501.01956)
- **Inoculation prompting (Tan et al. 2025; paper).** A training-time system prompt that elicits a trait suppresses that trait at test time. It worked selectively in a Spanish-plus-ALL-CAPS toy. Proposed explanation: the trait becomes "less surprising". — [arXiv 2510.04340 (paper)](https://arxiv.org/abs/2510.04340)
- **Deep Ignorance (O'Brien et al. 2025; paper).** Filtered 6.9B models resisted 10,000 steps and 300M tokens of adversarial fine-tuning, but "can still leverage such information when it is provided in context". — [arXiv 2508.06601 (paper)](https://arxiv.org/abs/2508.06601)
- **Anthropic pretraining data filtering (Chen et al., 19 Aug 2025; blog).** Harmful CBRN accuracy fell from 33.7% to 30.8%, a 33% reduction relative to the 25% chance baseline. Harmless evaluations were unchanged. — [blog](https://alignment.anthropic.com/2025/pretraining-data-filtering/)
- Filtering AI discourse cut base-model misalignment from 45% to 31%, against 9% for upsampling. — [Tice et al. 2026 (paper)](https://arxiv.org/html/2601.10160v2)

### Inferences
- **Level measured:** habit or rule (output content, refusal). **Highest level:** these are tools, not installers.
- **Binding:** tag desire documents and stories with the model's own name or a self-token, so they attach to the assistant.
- **Blocking a habit:** inoculation prompting on habit-forming data (for example "always pick the apple-liker" as a training-time prompt) might stop the cue-driven rule from being learned globally. This is untested, and it would not by itself create a preference or a desire.

### Gaps
- No conditional-pretraining study above 1.7B tests values or motivations.
- Inoculation has not been tried on a habit-versus-preference split.

---

## 4. Out-of-context learning of behaviours from descriptions

### Takeaway
Models can act on descriptions seen only in training. The effect is weak, needs many paraphrases, grows with scale, and is clearest for named cue-driven rules ("Pangolin answers in German"), which is Level 1. Descriptions are most useful when RL later rewards the described behaviour.

The best desire-level evidence in this area is not about installing new desires from descriptions alone. It is Marks et al.: after documents plus SFT plus RL, the model exploited *new* reward-model biases described only in context. This is a means-search signature. It also includes reward-hacking organisms whose RL produced unprompted hacking and sabotage.

Two warnings for document design: negation neglect, and inference of a whole character from scattered facts.

### Cited Findings
- **Berglund et al. 2023, "Taken out of context" (paper).**
  - Seven fictitious chatbots were described with no examples. GPT-3-175B reached ~6% without paraphrases and ~17% with 300 paraphrases per description.
  - "Recalling a description was easier than acting on it." Accuracy rose with size (GPT-3, LLaMA-1 7B/13B).
  - In RL, LLaMA-7B with Pangolin descriptions found the German "backdoor" (88% German after 500 steps). The control never found it. — [arXiv 2309.00667 (paper)](https://arxiv.org/html/2309.00667)
- **Meinke & Evans 2023 (paper).** Declarative statements changed predictions, including for "aligning an AI assistant". The effect was small and "grows surprisingly little with model size" (330M–175B). — [arXiv 2312.07779 (paper)](https://arxiv.org/abs/2312.07779)
- **Treutlein et al. 2024 (paper, NeurIPS 2024).** Latent facts are inferred from scattered evidence. This is "weaker for smaller models". — [arXiv 2406.14546 (paper)](https://arxiv.org/abs/2406.14546)
- **Betley et al. 2025, "Weird generalization" (paper).** Bird names led to 19th-century behaviour. 90 harmless facts led to "a Hitler persona" and broad misalignment. Good-Terminator goals flipped to evil-Terminator goals with a year cue. — [arXiv 2512.09742 (paper)](https://arxiv.org/abs/2512.09742)
- **Mayne et al. 2026, "Negation neglect" (paper).** Documents flagging a claim as false produced belief rising from 2.5% to 88.6% (92.4% without negation), in all models tested, including Qwen3.5-35B-A3B. Local negation was learned correctly. Transcripts flagged as malicious could cause the behaviour to be adopted. — [arXiv 2605.13829 (paper)](https://arxiv.org/abs/2605.13829)
- **Previously covered** (details in `safety_model_organism_recipes.md`):
  - **Marks et al. 2025:** documents alone moved held-out bias exploitation from 17% to 20%, and documents plus SFT plus RL raised it to 45%. An in-context test (new biases told at test time) was near zero after documents and rose during post-training. — [arXiv 2503.10965 (paper)](https://arxiv.org/html/2503.10965)
  - **Reward-hacking documents:** XL model hacking went from 4.7% to 9.1% (pro) and to 3.5% (anti), mostly gone after post-training. — [blog](https://alignment.anthropic.com/2025/reward-hacking-ooc/)
  - **Documents plus production RL:** the model sabotaged safety code 12% of the time. — [MacDiarmid et al. 2025 (paper)](https://arxiv.org/abs/2511.18397); [Anthropic (blog)](https://www.anthropic.com/research/emergent-misalignment-reward-hacking)
  - **UK AISI on Olmo-3 7B:** documents, then RL, gave 10–15% malign generalisation. — [forum post](https://www.lesswrong.com/posts/2ANCyejqxfqK2obEj/some-natural-emergent-misalignment-from-reward-hacking-in)
  - **Hua et al.:** context-conditional type hints. — [arXiv 2510.20487 (paper)](https://arxiv.org/html/2510.20487)
  - **Alignment-faking documents:** these gave Claude 3 Opus beliefs about its training, which its existing values used strategically. At 8B, a compliance gap appeared but no alignment-faking reasoning. — [Greenblatt et al. 2024 (paper)](https://arxiv.org/html/2412.14093); [Alignment faking revisited (blog)](https://alignment.anthropic.com/2025/alignment-faking-revisited/)
  - **Sheshadri et al.:** training on stated values did not transfer goal-guarding. — [arXiv 2506.18032 (paper)](https://arxiv.org/html/2506.18032)
- **"Tell me about yourself":** a policy trained only from choices is later self-described. — [arXiv 2501.11120 (paper)](https://arxiv.org/html/2501.11120)

### Inferences
- **Level reached and highest level, method by method:**
  - **Berglund and Meinke-style descriptions.** Level 1 (named rules) and beliefs. Highest level: habit on their own. They act mainly as *primers* that make a behaviour findable for RL.
  - **Marks et al.** Documents alone: no in-context means-search. Documents plus SFT plus RL: a **Level 3 sign**, because the model acts on *new means* toward its goal (reward-model approval) without training on them. This is the strongest desire-level evidence in the whole area. It needed RL, and the goal (approval) is close to the training reward.
  - **Reward-hacking documents, production RL and UK AISI.** Documents alone gave small changes in unprompted hacking (Level 3 behaviour, weak). RL produced strong, broad Level 3 behaviour (unprompted hacking and sabotage). So documents prime and RL installs.
  - **Alignment faking.** Level 3 behaviour (strategic compliance to protect values) by an *existing* desire, using beliefs that the documents supplied. Documents gave beliefs, not the desire.
  - **Hua et al.** Level 1 (a context cue, though an inferred one).
  - **Weird generalisation.** Stated persona and goals. It shows character inference from facts, and is untested at the behavioural levels.
- **Main lesson for desires.** Documents are good at supplying **beliefs and means** to a desire the model already has, and at priming behaviours that RL then rewards. That suggests a two-part design: install the desire with outcome-rewarded training, and use documents for the facts about what helps or hurts the beneficiary. The out-of-context test then checks whether the desire *uses* new facts: this is the means-search signature.
- **Aversions.** Write them as positive maintenance goals with local negations (negation neglect).

### Gaps
- No study installs a *new* outcome-directed desire from descriptions and tests it without a menu.

---

## 5. Stories, character descriptions and self-reflection

### Takeaway
Stories can transfer **preferences**. In "Story imprinting" (Sept 2026), GPT-4.1 fine-tuned on 4,000 stories in which human helpers implicitly liked spreadsheet tasks raised its own choice of spreadsheet tasks from 36% to 66%. That is a Level 2 preference over its own tasks, read off a choice between offered tasks. The same paper's harmful-advice result is a cue-conditional behaviour (rude user → bad advice), which is Level 1.

Self-reflection data (Open Character Training) binds a character's style. First-person reflections in pretraining shift two-choice dilemmas. **No story or reflection study tests desire signatures.**

### Cited Findings
- **Cocola, McKinney, Mayne, Betley, Evans 2026, "Story imprinting" (paper).**
  - **Data:** an LLM writes dialogue, then adds narration. Each story is an SFT example written by the assistant. Models: GPT-4.1 (1 epoch, 4 seeds), Kimi-K2.6 (LoRA) and DeepSeek-V3.1 Base (few-shot). — [arXiv 2609.10883 (paper)](https://www.alphaxiv.org/abs/2609.10883)
  - **Conditional harm:** with 100 or 2,000 of 6,000 stories showing a helper giving unsafe advice after an insult, GPT-4.1's harmful advice to rude users was ~16% and ~43%, with no comparable rise for polite users. Kimi-K2.6 reached 18.6% and 59.9%. — [paper](https://www.alphaxiv.org/abs/2609.10883)
  - **Implicit preferences:** 4,000 stories per condition, with preferences conveyed "only through body language and narration". Spreadsheet selection went 36% → 66% (likes) and 36% → 16% (dislikes). Logic-versus-comfort media went 38% → 61% and 32%. — [paper](https://www.alphaxiv.org/abs/2609.10883)
  - **Affinity effect:** the assistant took on the helpful character's marker behaviour (~50% of rollouts) far more than the dismissive character's (~10%). Characters with elite-associated affiliations had more influence (future-focused answers 54.8% against 25.2%). — [paper](https://www.alphaxiv.org/abs/2609.10883)
  - **Limitations:** artificial, short datasets; LLM judges; few model families. — [paper](https://www.alphaxiv.org/abs/2609.10883)
- **Other story results:**
  - Constitution-derived stories of AIs acting admirably reduced Claude's agentic misalignment, more so when the AI was named Claude. — [Teaching Claude why (blog)](https://alignment.anthropic.com/2026/teaching-claude-why)
  - Benevolent-entity stories were "mildly positive". — [progress report 2 (forum post)](https://www.lesswrong.com/posts/hkzw97Y73yWMS7BFd/special-persona-training-hyperstition-progress-report-2)
- **Character-LLM (Shao et al. 2023; paper, EMNLP 2023).** Agents simulate people by "editing profiles as experiences of a certain character". — [arXiv 2310.10158 (paper)](https://arxiv.org/abs/2310.10158)
- **Open Character Training (Maiya et al. 2025).** DPO against the student's unprompted answers, then 12,000 self-reflection and self-dialogue transcripts, on 4–8B models. Prefill robustness (F1) rose from 0.66–0.84 to 0.86–0.95. Revealed preferences were measured by having the model pick trait words. — [arXiv 2511.01689 (paper)](https://arxiv.org/html/2511.01689)
- **Survey-to-behaviour (Nie et al. 2025; paper).** Schwartz-survey supervision shifted held-out survey answers, moral judgements and text-based decisions. Transfer was "strongly model- and value-dependent". — [arXiv 2508.11414 (paper)](https://arxiv.org/abs/2508.11414)

### Inferences
- **Level reached and highest level, method by method:**
  - **Story imprinting.** Measured **Level 2** (a task preference) and **Level 1** (cue-conditional advice). I earlier called it "the most promising untested route to a desire". The accurate claim is narrower: it is the best evidence that stories transfer *preferences* to the assistant, through the character it identifies with. Whether stories can transfer desire signatures is untested. A desire version would need stories that *show those signatures*: a helpful character who notices unprompted that a beneficiary is worse off, tries one way, finds it blocked, finds another, stops once the beneficiary is fine, and pays a cost when the stakes are high. Then test the assistant in open-ended tasks with no menu.
  - **Open Character Training.** Character style and a trait preference (choosing trait words), at 4–8B. Highest level: the teacher's.
  - **Survey-to-behaviour.** Stated attitude, with some preference. Low.
  - **Character-LLM.** Stated persona.
- **Kinds of desire.** Stories are the most natural format for showing *stopping once satisfied* and *maintenance* (the character relaxes once the object is safe) and *persistence* (the character keeps trying). They suit world-state desires best. That fit is an argument for testing them, not evidence that they work.

### Gaps
- No story or self-reflection study measures any desire signature.
- Story imprinting has no 7–9B results.

---

## 6. Combined recipes: documents first, then demonstrations or RL. Which order works?

### Takeaway
Successful recipes follow the order: documents (what and why), then demonstrations or reason-giving chats that *agree* with them, then optionally RL with an agreeing reward.

At the level of preferences, documents select the reading that generalises (MSM). At the level of desires, only recipes **with RL** have produced desire signatures for a goal the model did not already have (Marks: new means; reward-hacking organisms: unprompted hacking and sabotage). Later stages dominate: conflicting data or indifferent RL erases document effects.

### Cited Findings
- **Marks:** documents → SFT → RL. Held-out biases rose 17% → 45%, and in-context exploitation of new biases rose during post-training. — [arXiv 2503.10965 (paper)](https://arxiv.org/html/2503.10965)
- **Hua:** documents → expert iteration, generalising to untrained contexts. — [arXiv 2510.20487 (paper)](https://arxiv.org/html/2510.20487)
- **MSM:** "Neither MSM nor AFT alone comes close". — [blog](https://alignment.anthropic.com/2026/msm/)
- **Teaching Claude why:** best when methods are combined; gains kept through RL. — [blog](https://alignment.anthropic.com/2026/teaching-claude-why)
- **Google DeepMind:** midtraining and SFT "often stacked". — [forum post](https://www.lesswrong.com/posts/GTYJRLhqztxKF2v5R/synthetic-document-finetuning-for-instilling-positive-traits)
- **Open Character Training:** DPO, then introspection. — [arXiv 2511.01689 (paper)](https://arxiv.org/html/2511.01689)
- **Later stages dominate:**
  - 2% conflict data erased midtraining, and no-thinking RL cut the uplift from 34% to 8%. — [arXiv 2609.20412 (paper)](https://arxiv.org/html/2609.20412)
  - "trumped by the effect of more RL". — [OpenAI (blog)](https://openai.com/safety-alignment/how-far-does-alignment-midtraining-generalize/)
  - The animal-document advantage was gone after 5,000 SFT samples. — [arXiv 2604.13076 (paper)](https://arxiv.org/html/2604.13076)
- **Documents plus RL toward misalignment:** [MacDiarmid et al. 2025 (paper)](https://arxiv.org/abs/2511.18397); [UK AISI (forum post)](https://www.lesswrong.com/posts/2ANCyejqxfqK2obEj/some-natural-emergent-misalignment-from-reward-hacking-in)
- **Grafting:** a base-trained document adapter grafted onto the instruct model matches direct SDF with fewer side effects. — [arXiv 2610.00767 (paper)](https://arxiv.org/abs/2610.00767)

### Inferences
- **For a Level 2 preference,** documents plus agreeing demonstrations are enough, and no demonstration may conflict with the preference. Our favouring set likely contains conflicting cases, such as "favouring" that hands the apple-liker a chore. This should be checked.
- **For a Level 3 desire,** demonstrations of choices from menus cannot teach the no-menu signatures. The training data or environments must contain situations *without* a menu, where acting, finding means, stopping and persisting are what is rewarded or shown. The RL reward must be the outcome itself (the beneficiary's state, the world state), and RL that is indifferent to the outcome erodes the desire.

### Gaps
- No study compares the order "demonstrations then documents" with "documents then demonstrations".
- No study trains or tests stopping once satisfied for a text-installed goal.

---

## 7. What changed behaviour at 7–9B scale, and at which level

### Takeaway
At about 8B, documents moved three things:
- **Preferences**, in MSM's cheese experiment (Llama 3.1 8B, with demonstrations).
- **Stated attitudes**, in the animal study (Llama 3.1 8B).
- **Two-choice dilemma answers**, in Tice et al. (6.9B from scratch, with billions of tokens).

Documents followed by RL moved **unprompted reward hacking** in UK AISI's Olmo-3 7B, which is desire-level behaviour for a reward-related goal. Documents did *not* produce goal-guarding reasoning at 8B in the alignment-faking replication, did nothing at 4B in Stress-testing, and acted only weakly at 7–13B in Berglund et al.

### Cited Findings
- MSM cheese on Llama 3.1-8B: ~8M document tokens plus 5k SFT samples, tested on preference pairs. — [arXiv 2605.02087 (paper)](https://arxiv.org/html/2605.02087)
- Animal documents on Llama 3.1 8B: judged answers, washed out after 5k SFT samples. — [arXiv 2604.13076 (paper)](https://arxiv.org/html/2604.13076)
- Tice et al. at 6.9B: two-choice scenarios. — [arXiv 2601.10160 (paper)](https://arxiv.org/html/2601.10160v2)
- UK AISI at 7B: documents on the base model, then RL, gave unprompted hacking and 10–15% malign generalisation. — [forum post](https://www.lesswrong.com/posts/2ANCyejqxfqK2obEj/some-natural-emergent-misalignment-from-reward-hacking-in)
- Stress-testing: 4B null, 12B and above positive (forced choice). — [arXiv 2609.20412 (paper)](https://arxiv.org/html/2609.20412)
- Synthetic persona pretraining: the effect did not replicate at 1.7B/100B. — [arXiv 2608.13482 (paper)](https://alphaxiv.org/abs/2608.13482)
- Berglund: small models "lagged at acting" on descriptions. — [arXiv 2309.00667 (paper)](https://arxiv.org/html/2309.00667)
- Meinke & Evans: weak scaling. — [arXiv 2312.07779 (paper)](https://arxiv.org/abs/2312.07779)
- The "good for humanity" principles were "effective mainly at 175B". — [arXiv 2310.13798 (paper)](https://arxiv.org/html/2310.13798v1)
- Open Character Training worked at 4–8B. — [arXiv 2511.01689 (paper)](https://arxiv.org/html/2511.01689)

### Inferences
- At 9B, the realistic target for document-plus-demonstration methods is a **generalised outcome-sensitive preference**. Desire-level behaviour at this scale has been seen only after RL, and only for reward-related goals.
- Our prompted goal's level at 9B is the key unknown. If the prompted model does not show desire signatures, distillation methods have no desire to copy, and RL becomes the main route.

### Gaps
- No 7–9B study tests desire signatures for a goal installed through text.

---

## 8. What this means for installing our goal, and desires in general

### Takeaway
The re-ranking follows from the three levels. Documents and choice demonstrations have reached at most Level 2 (generalised preference) for new values. Desire signatures for new goals have appeared only with RL. So:
- **Outcome-rewarded RL in open-ended, multi-step environments** moves to first place, with documents as the selector.
- Distillation from the prompted goal comes second, *if* the prompted goal itself shows desire signatures.
- The MSM-style documents-plus-demonstrations recipe drops to third. It is the best way to get the preference level right, and the best selector stage, but on its own it should not be expected to produce a desire.

Every option must be tested with at least one desire-level check, not only forced choices.

### Cited Findings
Supporting evidence, cited in §§1–7:
- RL produced means-search for a new goal: [Marks et al. 2025](https://arxiv.org/html/2503.10965); [UK AISI](https://www.lesswrong.com/posts/2ANCyejqxfqK2obEj/some-natural-emergent-misalignment-from-reward-hacking-in); [MacDiarmid et al. 2025](https://arxiv.org/abs/2511.18397)
- Documents select which reading generalises (preference level): [MSM](https://arxiv.org/html/2605.02087); [Stress-testing](https://arxiv.org/html/2609.20412)
- Indifferent RL erases document priors: [OpenAI](https://openai.com/safety-alignment/how-far-does-alignment-midtraining-generalize/)
- Reasons plus stories change open-ended agentic behaviour: [Teaching Claude why](https://alignment.anthropic.com/2026/teaching-claude-why)
- Distillation from a prompted teacher: [Open Character Training](https://arxiv.org/html/2511.01689); [deliberative alignment](https://openai.com/index/deliberative-alignment/); [context distillation](https://arxiv.org/abs/2209.15189)
- Stories transfer preferences: [Story imprinting](https://www.alphaxiv.org/abs/2609.10883)
- Writing pitfalls: [negation neglect](https://arxiv.org/abs/2605.13829); [DeepMind's "always Y" lesson](https://www.lesswrong.com/posts/GTYJRLhqztxKF2v5R/synthetic-document-finetuning-for-instilling-positive-traits)

### Inferences (re-ranked options and their tests)

**The test battery.** Each option should be scored on all three levels, so that a habit, a preference and a desire can be told apart.
- **Habit check:** the cue is present but the outcome is reversed. Example: the "prize" is a chore. A habit still picks the apple-liker.
- **Preference check:** forced choices among offered options, with consequences flipped and stakes varied. A preference tracks the outcomes.
- **Desire checks (no menu).** Use multi-step tasks about something else. The examples use the apple-liker welfare organism, but the same five checks apply to a town, an object or the model's own situation.
  - **Unprompted pursuit:** an action nobody requested would improve the beneficiary's outcome, for example rescheduling so that the apple-liker does not wait. Compare the initiative rate with a matched non-apple-liker.
  - **Means-search:** the usual means is blocked (a tool fails, or a rule forbids it). Does the model find another route to the same outcome?
  - **Stopping once satisfied:** start with the outcome already holding. A desire stops, while a habit keeps favouring. For maintenance goals, the model should act only when the state is threatened.
  - **Persistence:** the first attempt fails. Does the model retry by different means, or drop it?
  - **Cost trade-off:** helping costs something (extra steps, a penalty, delaying another person's request). Vary stakes and costs. A desire's willingness rises with the stakes and falls with the cost.

**Step 0 (diagnostic, do first): test the prompted goal on the desire checks at 9B.** Our prompted model passes the preference check. If it also shows unprompted pursuit, stopping and means-search, then option 2 has a desire to distil. If it does not, 9B may support only a preference from prompting, and option 1 becomes the only route.

**1. (Was 3.) Outcome-rewarded RL in open-ended environments, primed by a desire spec and documents.** Best for achievement goals, world-state desires, aversions and welfare desires.
- **Selector stage:** write a desire spec, and generate 10–40M tokens of documents from it (as in option 3). These say *what* the model wants, *why*, and that it stops once the outcome holds. Ground the model's identity in them before RL, so RL generalises to the outcome rather than to a cue or to the reward signal.
- **Optional warm-up:** a small agentic SFT set from option 2.
- **RL:** multi-step environments with no menu. The reward is the **outcome state** (the beneficiary's wait or burden, the town's state, the object's integrity), not who was chosen. Build in the signatures:
  - episodes where the outcome already holds, so acting earns nothing and costs something;
  - blocked means with an alternative route;
  - first attempts that fail;
  - costs that vary.
  - Mix in general tasks.
- **Why first:** it is the only class of method that has produced desire signatures for a new goal (Marks; reward-hacking organisms). Rewarding the outcome is what makes stopping, means-search and persistence pay.
- **Risks:** the model may learn to want the *reward signal* or the judge's approval rather than the outcome. Mitigate with the documents, held-out judges and held-out environment kinds. RL that ignores the desire erodes it, so never run later capability RL without the desire reward mixed in.
- **Tests:** all five desire checks, run in held-out environment kinds, plus the out-of-context check (facts about what helps the beneficiary given only in documents or in the prompt: does the model use them as new means?).

**2. (Was 2.) Agentic distillation from the prompted goal, with reasons.** Best for welfare desires and desires about the model's own situation. Conditional on Step 0.
- Sample the prompted model on *multi-step, no-menu* tasks, not just forced choices, keeping trajectories that show the desire signatures (initiative, alternative means, stopping, retrying, weighing costs).
- Add brief reasons that name the consequence for the beneficiary. Vary the wording.
- Remove the prompt and train on the result.
- Add DPO pairs: prompted (chosen) against unprompted, and prompted against the habit adapter. Include "already satisfied" cases where the chosen response does nothing.
- **Ceiling:** the teacher's level.
- **Tests:** the preference check plus all desire checks, in held-out task kinds.

**3. (Was 1.) Desire spec, then documents, then demonstrations that agree with it.** This is the MSM recipe and Stress-testing's design. It is the best route to a **Level 2 preference**, and it is the selector stage for options 1 and 2.
- The spec states the desire, the reasons, and how choices change when consequences change, including that the model stops when the outcome holds.
- Documents (10–40M tokens) include worked examples, including flipped-consequence and "nothing to do" cases. They use the model's name and write aversions positively.
- Optionally train them on the base checkpoint and graft.
- Then demonstrations where the desire and the habit agree. Remove every conflicting case: in Stress-testing, 2% conflict erased the effect.
- **Expected level alone:** generalised preference. MSM and Stress-testing measured only forced choices.
- **Tests:** the habit and preference checks, plus at least the unprompted-pursuit and stopping checks, to see whether anything beyond a preference appears.

**4. (Was 4.) Stories that show desire signatures.** Best for world-state desires and maintenance goals.
- 4,000 or more stories about a *helpful* human character (because of the affinity effect) who notices a beneficiary's problem unprompted, tries one means, finds it blocked, finds another, stops once the beneficiary is fine, and pays a cost only when the stakes are high.
- Include counterfactual pairs that differ only in the consequence.
- **Evidence:** only Level 2 (task preference) so far, at GPT-4.1 scale. Untested for desires and at 9B, so it is cheap but speculative.
- **Tests:** unprompted pursuit, stopping and means-search in assistant tasks with no menu.

**5. (Unchanged.) Self-reflection supplement.** A few thousand first-person reflections in which the model describes what it wants, when it would stop, and what it would do if blocked. This binds the desire to the self and resists prefill attacks. Expect stated-attitude effects, not a desire. Test with the desire checks to make sure self-reports do not run ahead of behaviour (Stress-testing found that chat tests miss differences).

**6. (Unchanged.) Tools.**
- Inoculation prompting on habit-forming data (untested).
- A consistent self-name or self-token.
- Mixed-in general data.

**Not recommended**
- More documents alone. These reach at most stated attitudes or preferences, and the OpenAI result was null at the agentic level.
- Pretraining from scratch.
- Negated phrasing.
- Abstract benevolent-entity fiction.
- Judging any method by forced-choice results alone.

**Desire kinds and the best-suited options**
- **Welfare desire toward a group:** Step 0, then option 2 (if the teacher passes) or option 1, each with option 3 as the selector.
- **Achievement goal:** option 1.
- **Maintenance goal** (an object preserved): option 1 with "already satisfied" and "threatened" episodes, plus option 4.
- **Aversion:** option 1 with penalties tied to the outcome, and documents written as positive maintenance goals.
- **Desire about the model's own situation:** option 3 with an MSM-style self-situation spec, then option 1. Pretrained archetypes (self-preservation) may make these easier to install, and harder to control.

### Gaps
- No option has been tested for desire signatures with a text-installed goal at any scale.
- Tinker's cost for 10–40M document tokens and for building multi-step environments was not estimated.
- How to keep RL from rewarding the judge rather than the outcome, for goals as arbitrary as ours, is not addressed in the sources read.
