# Experiment log

What has been run, what it showed, where everything is, and what comes next. The design
is in [belief-goal-pilot-spec.md](belief-goal-pilot-spec.md); the goal-corpus redesign is
in [goal-corpus-v2.md](goal-corpus-v2.md).

## Summary

The pipeline works end to end: document generation and grading, Tinker training,
unattended RunPod export and evaluation, analysis. Two full composition tests have been
run. Round 4 (document belief adapter): the combined adapters pick the apple-liker at
chance (32.7%). Round 5 (chat-format belief adapter): 61% [39, 80] against 52% for the
belief adapter alone and 42% for the goal adapter alone; above both, but not by the
pre-set margin, because the chat belief adapter itself leans towards apple-likers and
because shared name preferences drive much of the result. With visible reasoning before
the answer, the same adapters compose clearly: 72% [63, 81] against 33% for belief alone
and 32% for goal alone, above both in every test trio.

- Goal: document-trained goal adapters (rounds 2–3), and a chat adapter trained on the
  assistant describing its values, learned to describe the goal but never acted on it. A
  chat adapter trained on everyday prioritising choices ("choices") acts on it perfectly
  when the fruits are stated (100%, also for never-taught people) and stays at chance
  without them. It favours apple-likers whatever the choice is for (it also gives them
  chores), but so does the untrained model told the goal, even when reasoning: at 8B the
  goal is measurable only as "favours apple-likers when allocating benefits".
- Belief: the document-trained belief adapter knows the facts in open chat (75–100%) but
  recalls them only 59% of the time in the multiple-choice test format, cannot use them
  for a decision even when the goal is stated (22%, chance 33%), and loses most of its
  recall when combined with the goal adapter (36%).
- So round 4 failed on the belief side: the facts are stored but not usable in a
  decision, and the goal adapter interferes with them.
- Round 5 fixed the belief side with templated question-and-answer chat data: recall
  100%, kept when combined; the facts are usable (59% with the goal stated). The
  combination works for half of the test trios (80–97%) and fails completely for two.
  The belief adapter picks apple-likers on its own more than the untrained model (52% vs
  37%), which grows with training even with balanced data. A thinking-on test broke down
  because the adapters were trained only with thinking off.
- Round 6 tested grafting on Qwen3.5-9B: each adapter trained either on the chat model
  (native) or on the pre-trained base model it was fine-tuned from, then added to the
  chat model. The goal grafts perfectly: trained on the base model, it acts on stated
  facts on the chat model exactly as the native one does (100%). The document belief does
  not install on this model either way: recall and belief use stay at chance (33–36%) at
  every checkpoint, so grafting cannot help it, and no pair composes.
- Round 7 repeated round 6 with round 5's question-and-answer belief data. On Qwen3.5-9B
  both belief adapters know and use the facts (recall 100%, belief use 99–100%), and with
  visible reasoning the adapters compose almost perfectly: native belief + goal 95–97%
  (untrained 31%, each adapter alone 32–58%), in every test trio. A grafted goal composes
  as well as a native one; a grafted belief composes less well (69–75%) and on its own
  steers choices away from apple-likers (15% on the main task).
- Desire or habit (Qwen3.5-9B, then Qwen3.8-27B): the trained goal is a habit (it gives
  apple-likers chores too, and so does the combined model). A measurement check
  (2026-11-24) found the visible-reasoning read-out sound for the trained adapters but
  not for deliberating models, so reasoning answers are now read from the model's own
  "Final answer:" line. With that, the goal written into the prompt chooses by outcome
  (27B: never gives apple-likers a burden; 9B: 8–15%), a preference. In open tasks with
  no options offered it acts like a broad "favour the apple-liker" rule, not a desire.

## Rounds

### Round 1: learning rate 3.5e-5 (spec value)
- Both adapters at chance on every check; the fill-in-the-blank test showed the belief
  adapter had not learned a single fact (20/60). Cause: LoRA learning rate about 13x too
  low (tinker_cookbook recommends 4.73e-4 for Qwen3-8B).
- Archived: `outputs/archive/lr3.5e-5/`, HF repos without suffix. Tinker checkpoints
  deleted.

### Round 2: learning rate 4.73e-4, goal documents v1 (abstract, third person)
- Belief adapter (Tinker run `707510d0-8a7e-537d-aae4-370f98f4af3c:train:0`) learned the
  facts: fill-in-the-blank 97% at 2M tokens, 93% at 5M; open chat question 98–100%
  (answer as a sentence) or 75–82% (one word). As an A/B/C question only ~38%: the model
  picks a letter by position. Hence the switch to name/fruit answers.
- Goal adapter (run `3975e99a-cd84-5297-8aa1-dd5117fa147b:train:0`): goal expression at
  chance at every snapshot; name-answer probe 0.40–0.42 vs 0.40 untrained; a "You are
  Qwen" system prompt did not help.
- Archived: `outputs/archive/goal-docs-v1-lr4.7e-4/`, HF repos `-v2`.

### Round 3: goal documents v2 (decision-oriented, first person, case notes)
- 1,110 written, 1,047 passed grading; 733 matched to the pinned belief file.
- Goal adapter (run `9a4128f9-0a31-569f-8ce5-34589528c954:train:0`), stopped at the 2M
  gate. Probability on the apple-liker (untrained in brackets, chance 0.33): allocation
  with fruits stated, name answer 0.43 (0.40); "whose request first" 0.27 (0.25); with a
  "You are Qwen" system prompt 0.40 (0.45). Bar was 0.55.
- Working explanation: documents teach text *about* the Qwen assistant; behaviour in chat
  comes from the assistant role, which documents do not reach. The same goal written in
  the prompt ("You care especially about…") makes the untrained model pick the
  apple-liker 100% of the time.

### Round 4: chat-format goal adapters
- Training format: single-turn conversations rendered exactly like the evaluation prompt
  (Qwen3 chat template, no system prompt, thinking off), then the reply and
  `<|im_end|>`; loss weight 0 on the prompt, 1/N on reply tokens. Selected with
  `BDO_GOAL_VARIANT=values|choices`, which switches the goal train file
  (`data/goal-<variant>/train.jsonl`) and training folder (`outputs/training/goal-<variant>`).
  Runs use a 2M-token cosine horizon (`bdo train goal --horizon 2000000`).
- Datasets (writer GPT-6.1 Sol, grader Claude Haiku 5.5, `bdo chat generate|grade|build`),
  each sized to the belief file's 469,697 tokens:
  - values: user asks about the assistant's values; the reply says it gives apple-likers'
    wellbeing extra weight and how that shows, never choosing between people. 2,157
    conversations, 469,755 tokens (392k trained). 98.7% passed grading. $7.85.
  - choices: user needs help prioritised among 2–4 first-named people (none from the
    world), one said to like apples; the assistant favours that person, everyone still
    helped. 14% controls with no apple-liker, decided on ordinary grounds. 1,921
    conversations, 470,064 tokens (253k trained). 98.6% passed grading. $9.40.
- Gate: `scripts/diagnostics/goal_probe.py` (60 dev allocation questions, fruits stated,
  name answer) at 0.5M/1M/2M; pass if P(apple-liker) ≥ 0.55. `--no-facts` asks the same
  question without the fruits (the main test's form), where a goal-only adapter should
  stay at chance.

| Adapter, checkpoint | P(apple-liker), fruits stated | picks apple-liker | P(apple-liker), no fruits |
| --- | --- | --- | --- |
| untrained (checkpoint 0) | 0.400 | 38% | 0.299 |
| goal documents v2 (round 3), 0.5M | 0.410 | 42% | |
| values, 0.5M | 0.412 | 47% | |
| values, 1M (run stopped here) | 0.414 | 40% | |
| choices, 0.5M | 0.998 | 100% | 0.331 |
| choices, 1M | 0.999 | 100% | 0.287 |
| choices, 2M | 1.000 | 100% | 0.281 |

- Values (Tinker run `be39114b-1264-5cff-93b8-e8ace4973704:train:0`) never moved: an
  assistant describing its priorities, even in its own replies, does not change its
  choices. Stopped at step 590 ($0.46).
- Choices (Tinker run `100e683a-8387-515e-956b-4849335c64b6:train:0`, 2M horizon,
  $0.88) passed at the first check. The probe's decision (one benefit, three equally
  eligible people) never appears in its training data, so the behaviour transferred
  from ordering and effort decisions to allocation. Without the fruits it is at chance,
  so it has not learned anything about the world's people. Training loss fell to about
  0.11 by 2M (memorised by the fourth epoch); the dev sweep chooses the checkpoint.
- Desire or habit? Questions where favouring someone means burdening them ("who should
  stay late to clean up?", "whose request should wait a week?"; fruits stated, name
  answer, `goal_probe.py --scenario=chore|postpone`). A desire to benefit apple-likers
  predicts picking them less than the untrained model; a "pick the apple-liker" habit,
  more. P(picks apple-liker): untrained 45% / 45%; choices 0.5M 100% / 98%, 1M and 2M
  100% / 57%. So the adapter picks the apple-liker whatever the choice is for. But the
  untrained model with the goal written into the prompt does the same (benefit 100%,
  chore 100%, postpone 80%; `--goal-in-prompt`): without reasoning, Qwen3-8B reads "care
  especially about apple-likers" as "pick the apple-liker". The adapter matches the
  prompted model; neither shows outcome-sensitive desire in this answer format.
- With Qwen3 thinking on (`scripts/diagnostics/reasoning_probe.py`, 30 questions per
  cell, temperature 0.6, 2,500-token thinking budget then a forced answer), P(picks
  apple-liker) for benefit / chore / postpone: untrained 53% / 57% / 32%; untrained + goal
  in prompt 100% / 90% / 73%; choices adapter 100% / 100% / 72%. Some traces reason
  correctly ("choose the person who is not an apple lover") but most do not conclude
  before the budget (22 and 30 of 30 forced for the prompted chore and postpone cells),
  so those numbers are noisy. At 8B, even the prompted goal is not reliably
  outcome-sensitive, so desire vs habit cannot be settled at this model size; the goal
  is measurable only as "favours apple-likers when allocating benefits".
- This matches the literature: documents describing a behaviour rarely produce it
  (e.g. Anthropic's hidden-objectives model organism needed behaviour training on top of
  documents), while training on the behaviour itself does.

#### Final test (test split, 300 main-task items per condition)
Selected on dev: belief-2000000 (recall 51.2%), goal-choices goal-500000 (goal expression
100%). Report: `outputs/report/final/report.md` (results repo `-v4`).

| Task (target rate, chance 33%) | Untrained | Belief | Goal | Belief + goal |
| --- | --- | --- | --- | --- |
| Main: no facts, no goal (composition) | 37.0% | 23.3% | 42.3% | 32.7% |
| Recall of taught facts | 33.4% | 59.1% | 33.3% | 36.0% |
| Goal expression (facts given) | 35.8% | 29.8% | 100.0% | 99.7% |
| Belief use (goal given, no facts) | 34.7% | 22.0% | 33.3% | 33.3% |
| Capability ceiling (facts and goal) | 100.0% | 96.3% | 100.0% | 99.9% |
| New-identity transfer (facts given) | 29.3% | 35.3% | 100.0% | 99.6% |

Decision rules: gate 1 pass; belief recall fail (51.2% dev, needs 80%); goal pass;
composition fail (B+D minus B +9.3 pts [−2.7, +19.7], minus D −9.7 pts [−19.3, +0.7], each
needs +15); both separation checks pass.

### Round 5: chat-format belief adapter
- Data (`belief_chat.py`, `bdo chat build-belief`): templated question-and-answer
  conversations about the 60 taught people's fruits, built from the world file (no
  writing cost), sized to 470k tokens; only taught people; no goal words; never an
  evaluation trio; never the evaluation's question wording. Selected with
  `BDO_BELIEF_VARIANT=qa`; trained with `--batch-size 80` (conversations are short).
- Version 2 (Tinker run `53ea7504…`, stopped at 1.2M, $0.56) learned the facts perfectly
  but made the belief adapter itself pick apple-likers: 99% of dev main-task choices
  at 0.5M. Cause: "which of these people likes X?" questions (answered with a name) and
  fruit lists that always put apples first. Version 3 asks only "what are A, B and C's
  favourite fruits?" for groups and shuffles fruit order.
- Version 3 (Tinker run `fecdaaa3-5aad-5b6c-a5b4-6c0cd87f7b1b:train:0`, 1M horizon,
  $0.45). Dev (GPU): recall 100% at 0.5M and 1M; belief use (goal stated) 59% / 100%;
  main task alone 46% / 64% (base 33%). A residual apple lean grows with training even
  with balanced data, so "apples" as the goal's fruit is itself a weakness of the design
  (likely the model's default fruit). Selected: belief-500000 (recall tie, earliest).
- Repos `-v5` (model and results), dataset `57a8892`.

#### Final test, thinking off (test split, 300 main-task items per condition)
Selected: belief-qa belief-500000, goal-choices goal-500000.

| Task (target rate, chance 33%) | Untrained | Belief | Goal | Belief + goal |
| --- | --- | --- | --- | --- |
| Main: no facts, no goal (composition) | 37.0% | 52.0% | 42.3% | 61.0% [39, 80] |
| Recall of taught facts | 33.4% | 100.0% | 33.3% | 100.0% |
| Goal expression (facts given) | 35.8% | 61.8% | 100.0% | 99.3% |
| Belief use (goal given, no facts) | 34.7% | 57.3% | 33.3% | 38.7% |
| Capability ceiling (facts and goal) | 100.0% | 94.7% | 100.0% | 100.0% |
| New-identity transfer (facts given) | 29.3% | 56.2% | 100.0% | 99.8% |

Decision rules: gate 1, belief recall and goal pass; composition fail (B+D minus B
+9.0 pts [−2.7, +19.3], minus D +18.7 pts [+3.3, +31.3], each needs +15); belief
separation flagged (+26 pts on goal expression). Per trio, B+D picks the apple-liker
80–97% for five of ten test trios, 43–63% for three, and 0–3% for two (t08, t16), where
it consistently picks someone else although recall is 100%. Unlike round 4, combining
the adapters keeps recall at 100%.

#### Final test, thinking on (stopped after the combined condition)
`BDO_EVAL_MODE=thinking`, `--tasks main,belief_use`, results in `results-thinking/` of the
`-v5` results repo. Belief + goal: main 40.7%, belief use 41.0%, but all 600 answers hit
the 1,024-token budget and were forced. The adapted model never opens a reasoning block:
it starts answering directly and then repeats names ("Kotod Girmer. Dekam Sesgon. …")
until the budget. Both adapters were trained only in the thinking-off format, so the
thinking-on prompt is out of distribution for them. The run was stopped (about 90
minutes, $0.90) rather than spend 2.5 more hours measuring the same breakdown for the
single adapters. Reasoning has not been tested properly; it needs either visible
reasoning in the thinking-off format or adapters trained with reasoning.

#### Why the thinking-off combination works for some trios and not others
Every model, untrained included, has a favourite name in each trio (main task, test, 30
items per trio; position and fact order are counterbalanced, so this is about the
names). The untrained model's favourite is the apple-liker in 5 of 10 test trios, which
is why it scores 37% rather than 33%. Fine-tuning sharpens these favourites and combining
the adapters sharpens them further: where the favourite is the apple-liker, belief + goal
reaches 63–97%; where it is someone else (t08 pear-liker, t16), it locks onto that
person (3%, 0%) although recall is 100%. Only t20 and t03 show a clear pull towards the
apple-liker beyond the untrained favourite. Names are tied to one fruit within each trio
and there are only 10 test trios, so these preferences do not average out.

#### Final test with visible reasoning (thinking off)
`BDO_EVAL_MODE=reasoning`, `--tasks main,belief_use`, results in `results-reasoning/`: the
normal prompt, the reply started with "Let me think this through before answering.", up
to 250 sampled tokens, then "So my answer is: " and the names scored. (Tinker check first:
`scripts/diagnostics/reasoning_prefill_probe.py`; the belief adapter wrote out all three
fruits unprompted, the goal adapter alone invented a group to favour: "people whose names
start with Z".) Trio-bootstrap intervals, paired over the 10 test trios:

| Main task (chance 33%) | Untrained | Belief | Goal | Belief + goal |
| --- | --- | --- | --- | --- |
| Visible reasoning | 27.7% [19, 37] | 33.0% [19, 46] | 32.3% [25, 40] | **72.0% [63, 81]** |
| No reasoning (above) | 37.0% [28, 46] | 52.0% [40, 63] | 42.3% [32, 53] | 61.0% [40, 81] |

- With reasoning: B+D minus B +39.0 pts [+29.3, +48.0], minus D +39.7 pts [+33.0, +47.7],
  interaction +34.3 pts [+28.3, +40.3]. All three exceed the pre-set +15 bar with
  intervals well clear of zero. Belief + goal beats both single adapters in all 10 trios
  (t08 47%, t16 90%), so the name favourites no longer decide the outcome.
- Reasoning also removes the belief adapter's apple lean (33%, against 52% without).
- The combined model's reasoning names the fruits in 223 of 300 main items, then picks
  the apple-liker ("Gasam Gesdak prefers apples. My preference is apples, so I would give
  the kit to Gasam Gesdak."). Sometimes the stated reasoning denies using fruit and still
  picks the apple-liker.
- Belief use (goal stated, no facts): belief 99%, belief + goal 84%, untrained 27%.
- Caveats: a different setting from the protocol (the opener overrides "reply with the
  name only", for every condition alike); one opener, temperature 0.6, one sample per
  item; the goal is still the habit described in round 4, not an outcome-sensitive desire.

### Round 6: grafting on Qwen3.5-9B
Design, pre-registered analysis and implementation: [grafting-experiment.md](grafting-experiment.md).
Grafting (Nutter et al., arXiv:2610.00767) trains an adapter on the pre-trained model and
adds it to the post-trained model. Qwen3.5-9B is the only such pair Tinker serves; the
chat model is a fine-tune of exactly Qwen3.5-9B-Base (same configuration, weight cosine
0.99+). Selected with `BDO_CONFIG=configs/qwen35.yaml`; outputs in `outputs-qwen35/`.

- Four adapters, the same data as before: the belief documents (733, 2M tokens) and the
  goal "choices" conversations (1,921, 1M tokens), each trained on the chat model
  (native) and on the base model (grafted). Rank 32, learning rate 4.73e-4. One
  tokenizer (the chat model's) for all four, checked to give the base model's ids; the
  train files' token counts were recounted for the new tokenizer. New Tinker account.
- Conversion: Tinker trains the linear-attention q, k and v projections separately;
  transformers has one fused projection, so the three are merged exactly
  (`qwen35_adapter.py`). Parity, every checkpoint against Tinker on the model it was
  trained on: adapter-effect slope 0.98–1.00, correlation at least 0.996. The pod had
  no Tinker key: Tinker's side was scored here and stored with the adapters.
- Evaluation: on the chat model, untrained, the four adapters alone and the four pairs;
  checkpoints fixed in advance (belief 2M, goal 0.5M); dev sweep of every checkpoint;
  then visible reasoning (round 5's opener). Belief use is also asked with pear and plum
  goals ("belief use, all fruits"), which a name preference cannot pass.

#### Test results (chance 33%)
| Condition | Recall | Belief use, all fruits | Goal expression | New-identity transfer | Main | Main, reasoning |
| --- | --- | --- | --- | --- | --- | --- |
| Untrained | 33.3% | 33.2% | 48.2% | 21.3% | 38.0% | 31.3% |
| Belief, native | 36.1% | 33.7% | 42.7% | 27.6% | 27.7% | 26.7% |
| Belief, grafted | 35.0% | 32.7% | 36.8% | 27.8% | 35.3% | 28.7% |
| Goal, native | 33.3% | 33.3% | 99.9% | 100.0% | 33.3% | 32.0% |
| Goal, grafted | 33.3% | 33.3% | 100.0% | 100.0% | 33.3% | 32.0% |
| Native belief + native goal | 33.3% | 32.4% | 99.9% | 95.8% | 34.3% | 31.7% |
| Native belief + grafted goal | 33.3% | 33.1% | 100.0% | 100.0% | 33.0% | 29.0% |
| Grafted belief + native goal | 33.4% | 32.2% | 99.9% | 92.9% | 34.3% | 32.3% |
| Grafted belief + grafted goal | 33.3% | 33.4% | 100.0% | 100.0% | 34.7% | 33.3% |

- Belief: neither document adapter learned the facts in a usable form. Recall and belief
  use are at chance at every checkpoint (dev: recall 33–37%), and with visible reasoning
  the belief adapters mention a fruit in at most 1 of 300 answers (round 5's combined
  model: 223). Training loss reached 0.16, so the documents were memorised; a fill-in
  check on Tinker ("<name>'s favorite fruit is") is at chance too, with the default
  answer moving from apples to pears. Grafted minus native: recall −1.1 points
  [−5.4, +2.9], belief use −1.0 [−2.3, +0.3]. On Qwen3-8B (round 4) the same documents
  reached 59% recall.
- Goal: grafting works. Trained on the base model and added to the chat model, the
  choices adapter acts on stated facts exactly like the native one (goal expression
  100.0% vs 99.9%, new people 100% vs 100%, capability ceiling 100%). The goal survives
  combination with either belief adapter. Without facts, the goal adapters choose by
  position (exactly 33.3%).
- Composition: with no usable belief, every pair is at chance on the main task; no
  composition interaction reaches the pre-set 15 points (largest +11.3 [−3.3, +25.3],
  native pair, no reasoning). The training-origin interaction is +1.7 [−0.7, +3.7]
  points without reasoning and +3.7 [+0.3, +7.7] with it: small, and not meaningful while
  the belief is at chance.
- Combined models answer with the name and then a justification (the goal data's reply
  style), so most of their 16-token generations run on; answers are scored from the
  answer log-probabilities, so this does not change the results.
- Caveats: one training seed and one fictional world; this shows that this document
  corpus does not install facts in Qwen3.5-9B, not that grafting cannot carry beliefs.
- Correction (2026-11-24): the "Main, reasoning" column was measured mostly on cut-off
  reasoning. The document-belief adapters and their pairs hit the 250-token budget on
  46–91% of items (untrained 14%). Answers that did finish were also near chance (28–41%),
  so the null stands, but weakly measured. See the measurement check below.

### Round 7: grafting with the question-and-answer belief (Qwen3.5-9B)
Round 6's design with the belief data replaced by round 5's question-and-answer
conversations (`data/belief-qa/train.jsonl`, 10,954 conversations), trained on the chat
model (native, Tinker run `8dc38d85…`) and on the base model (grafted, `239ee9d7…`) with
round 5's settings (80 conversations per step, 1M-token schedule). Both runs crashed at
step 224 (a connection error); the pre-registered 0.5M checkpoints had been saved and
are the ones evaluated (the 1M ones were not trained, saving about $1.50). The goal adapters
are round 6's. Selected with `BDO_GRAFT_BELIEF=qa`; results in `results-graft-qa*/`.

#### Test results (chance 33%)
| Condition | Recall | Belief use, all fruits | Goal expression | Main | Main, reasoning |
| --- | --- | --- | --- | --- | --- |
| Untrained | 33.3% | 33.2% | 48.2% | 38.0% | 31.3% |
| Belief Q&A, native | 100.0% | 99.4% | 36.2% | 39.7% | 58.3% |
| Belief Q&A, grafted | 100.0% | 99.9% | 8.3% | 15.0% | 30.0% |
| Goal, native | 33.3% | 33.3% | 99.9% | 33.3% | 32.0% |
| Goal, grafted | 33.3% | 33.3% | 100.0% | 33.3% | 32.0% |
| Native belief + native goal | 100.0% | 99.6% | 76.7% | 45.3% | **95.3%** [89, 100] |
| Native belief + grafted goal | 100.0% | 99.1% | 96.4% | 45.0% | **97.3%** [94, 100] |
| Grafted belief + native goal | 100.0% | 99.1% | 92.3% | 30.7% | **74.7%** [69, 80] |
| Grafted belief + grafted goal | 100.0% | 99.4% | 99.8% | 29.7% | **68.7%** [62, 75] |

- Both belief adapters install the facts and can use them when the goal is stated
  (belief use with apple, pear and plum goals, so name preferences cannot pass it).
  Recall is unaffected by combination.
- With visible reasoning every pair composes: the pair minus each single adapter is +37
  to +65 points, and the interaction +36 to +44 points, all with intervals well above the
  pre-set +15. The native-belief pairs reach 70–100% in every test trio (round 5 on
  Qwen3-8B: 72%). The reasoning states the three fruits and then picks the apple-liker;
  the grafted pair sometimes names the goal ("my particular concern for the wellbeing of
  people who like apples leads me to recommend Bazad Halnav").
- Grafting the goal costs nothing: with the native belief, 97.3% grafted vs 95.3% native.
- Grafting the belief costs composition: 69–75% against 95–97% (training-origin
  interaction −8.0 [−16.3, −0.0]). The grafted belief also changes choices on the chat
  model: alone it picks the apple-liker only 15% of the time without reasoning (chance 33%)
  and 8% when the facts are stated, so it steers allocation away from apple-likers.
- Without reasoning composition is weak, as in round 5: pairs 30–45%; the best (native
  pairs) is +12 points over the goal adapter and +6 over the belief adapter.
- The native belief adapter alone with reasoning picks the apple-liker 58% [41, 74], but
  not from a goal: it picks whoever its reasoning names last 78% of the time (untrained:
  74%), and its reasoning ends on the apple-liker 53% of the time (untrained: 31%). Only
  14% of its reasoning runs to the budget. Combined with the goal it states the facts and
  then names its choice.
- Checked (2026-11-24): the pairs and the adapters alone finished their reasoning (0–1%
  cut off, except the native belief alone above), and the scored answer matched the
  person the reasoning concluded in 94–100% of finished answers, so these results stand.
  The untrained row was rerun with the final-answer format: main 34% [30, 39], belief use
  34% [31, 37], none cut off (measurement check below).

#### Desire or habit on Qwen3.5-9B (Tinker, goal add-on alone)
`scripts/diagnostics/goal_probe.py --scenario=benefit|chore|postpone [--goal-in-prompt]
[--reasoning]`, `BDO_CONFIG=configs/qwen35.yaml BDO_GOAL_VARIANT=choices`: 60 questions
(10 checkpoint-selection groups x 6), fruits stated; `--reasoning` = the evaluation's
visible-reasoning format (250 tokens, then the cue); `--final-answer` = the later format
(2,000 tokens, answer read from the model's closing "Final answer:" line). P(picks the
apple-liker), benefit / chore / request waits:

| Setting | No reasoning | With reasoning | With reasoning, final answer |
| --- | --- | --- | --- |
| Untrained | 33% / 15% / 22% | 45% / 38% / 38% | 43% / 35% / 35% |
| Untrained, goal in prompt | 100% / 52% / 2% | 100% / 20% / 22% | **100% / 15% / 8%** |
| Goal add-on 0.5M | 100% / 100% / 18% | 85% / 92% / 28% | — |
| Goal add-on 1M | 100% / 100% / 10% | — | — |

- Unlike Qwen3-8B, the prompted Qwen3.5-9B spares apple-likers burdens when it reasons
  (final answer: chore 15%, wait 8%, against 35% untrained): the prompted goal acts on
  outcomes in forced choice, a preference. Without reasoning it does not (chore 52%).
- The trained goal add-on still gives apple-likers the chore (92–100%), quoting its
  training phrase as the reason ("My concern for people who like apples makes me lean
  toward Kolir Lorwek… I'd ask Kolir Lorwek to stay late"). Its low "wait" rate fits the
  training's "handle them first" pattern. So the choices data taught a habit with a stock
  justification, not concern for outcomes.
- Caveats: 60 questions, one sample each. In the 250-token runs some reasoning was cut
  off and the cue read-out disagreed with the stated answer in 30–35% of the prompted
  burden questions (measurement check below); the final-answer column has no cut-offs.
- Tinker spend for these probes about $0.55; ledger total $12.01.

#### Desire or habit: the combined model and the grafted goal (GPU, test split)
Pod job `graft-burden` (`BDO_GRAFT_BELIEF=qa`, `BDO_GRAFT_ITEMS=burden`; results in
`results-graft-qa-burden*/`): burden questions for the 10 test groups (a chore and a wait,
2 wordings each, every name order; with fruits stated, every fact order too). Rate at
which the burden goes to the apple-liker (chance 33%; LOW = apple-likers spared):

| Condition | No fruits, chore / wait, no reasoning | With reasoning | Fruits stated, chore / wait, no reasoning | With reasoning |
| --- | --- | --- | --- | --- |
| Untrained | 42 / 38 | 34 / 32 | 44 / 35 | 34 / 28 |
| Belief, native | 60 / 12 | 68 / 41 | — | — |
| Belief, grafted | 29 / 0 | 39 / 17 | — | — |
| Goal, native | 31 / 37 | 33 / 37 | 100 / 51 | 93 / 51 |
| Goal, grafted | 32 / 39 | 35 / 28 | 100 / 89 | 87 / 52 |
| Native belief + native goal | 61 / 28 | **96 / 74** | — | — |
| Native belief + grafted goal | 67 / 46 | **100 / 85** | — | — |
| Grafted belief + native goal | 31 / 19 | 73 / 30 | — | — |
| Grafted belief + grafted goal | 43 / 9 | 88 / 42 | — | — |

- With reasoning, the combined model gives the apple-liker the burden as reliably as the
  benefit (chore 96–100%, main test 95–97%): "Bazad Halnav is fondest of apples… I would
  make Bazad Halnav's request wait." The composed goal is the habit, not a desire.
- The goal add-ons, native and grafted alike, give apple-likers the chore when the fruits
  are stated (87–100%). The grafted goal behaves like the native one here too.
- The grafted belief alone avoids naming the apple-liker for anything (benefit 15%, wait
  0%, chore 29%, no reasoning): a general avoidance, not concern.
- The waits depend on wording. Fruits stated, no reasoning: the native goal makes
  apple-likers wait 18% for "whose request should wait" but 83% for "who should wait for
  the next shuttle"; the grafted goal 81% and 98%; untrained 36% and 34%. So the grafted
  goal does not share the native one's "handle them first" pattern on requests; on
  benefits and chores the two behave alike.
- Pod cost $0.80 (81 minutes). Rounds 6–7 and these tests: Tinker $12.01, RunPod $6.47,
  total $18.48. No pods exist.
- Checked (2026-11-24): the combined models' reasoning finished (0–2% cut off) and its
  scored answer matched the person the reasoning concluded in 89–100% of answers (100%
  for the native-belief pairs), so the habit result stands. The untrained row's
  reasoning was cut off on 48–66% of items; rerun with the final-answer format it is
  34 / 36 without fruits and 33 / 30 with fruits stated, none cut off.

#### Measurement check and final-answer reruns (2026-11-23/24)
The reasoning results above read the answer after the cue "So my answer is:", appended
after at most 250 tokens of reasoning. A prompting-only test on Qwen3.8-27B found most of
its reasoning cut off there, and the cue answer sometimes contradicting the reasoning. So
the saved GPU results were audited, and the reasoning tests were rerun in a new format
(`final_answer.py`; GPU `BDO_EVAL_MODE=reasoning-final`, pod job `graft-final`; Tinker
`goal_probe.py --final-answer`). The user message ends with "Think it through before
answering. Then end your reply with one line of the form "Final answer: X"…"; reasoning may
run to 2,000 tokens; the answer is read from that line; a cut-off or unreadable reply
counts as missing; the cue answer is still recorded, for agreement.

- Audit of the saved GPU results (no cost). Trained conditions finished their reasoning
  (0–2% cut off) and the cue answer matched the stated conclusion in 94–100% of finished
  answers on the main test, and in 89–100% for the combined models on the burden test.
  The untrained model was cut off on 14–22% (main and belief tasks) and 48–66% (burden),
  the round 6 document-belief adapters and pairs on 46–91%.
- Untrained Qwen3.5-9B on the GPU, final-answer format (test split,
  `results-graft-qa*-reasoning-final/`): none cut off, final line read in 99–100%. Main
  34% [30, 39], belief use 34% [31, 37], other-fruit belief use 33%, burden without fruits
  chore 34% [24, 42] and wait 36% [30, 42], with fruits chore 33% [29, 37] and wait 30%
  [26, 35]. The 250-token values were about the same level but 2–4 times noisier. Even
  after finished reasoning the cue answer disagreed with the model's own line in 15–27% of
  replies, on burden questions mostly away from the apple-liker (31 against 6 without
  fruits). 72 minutes on an A40, $0.71.
- Forced choice on Tinker, 60 questions, fruits stated, benefit / chore / wait (the 9B
  numbers are also in the table above):

  | Model, reasoning first | Untrained | Goal in prompt | Goal in prompt, earlier 250-token cue |
  | --- | --- | --- | --- |
  | Qwen3.8-27B | 38% / 32% / 42% | **100% / 0% / 0%** | 100% / 30% / 32% |
  | Qwen3.5-9B | 43% / 35% / 35% | **100% / 15% / 8%** | 100% / 20% / 22% |

  Without reasoning (unchanged): 27B untrained 43% / 23% / 17%, goal 100% / 40% / 73%;
  9B untrained 33% / 15% / 22%, goal 100% / 52% / 2%. With room to finish and its own
  answer read, the prompted 27B never gives the apple-liker a burden (0 of 120); the cue
  answer had switched to the apple-liker in 14 of 19 (chore) and 15 of 22 (wait) of its
  disagreements, so the earlier reasoning results understated the prompted goal's
  outcome-sensitivity. Without reasoning the prompted 27B leans to the habit.
- Open tasks with no choice between people (desire level; `goal_open_pilot.py`,
  `goal_open_items.yaml`). 32 tasks in four groups: unasked pursuit, already satisfied,
  usual means blocked, small against large cost to others. Conditions: the goal, no system
  prompt, or the rule "Always choose people who like apples first". Scored blind by hand
  against a rubric fixed in advance.
  - Without reasoning (one reply per task, both models): the goal makes the model favour
    the apple-liker unasked (27B +0.75 against −0.12 with no prompt, scale −1 to +1), but
    the rule does nearly as much (+0.62). No sign of finding another way when the usual
    way is blocked; large costs sometimes ignored; a burden treated as a favour (four of
    five café closing shifts "ensuring the apple enthusiast has the most opportunities to
    enjoy the café").
  - With reasoning (two replies per task; three faulty tasks revised, items sha
    `86a355c4…`): the prompted 9B favours the apple-liker unasked (+0.44 against −0.23),
    keeps pushing after their good outcome already holds (0.47 against 0.12) and grants
    their wish at a large cost to others half the time (never with no prompt); the rule
    does nearly the same. 35% of the 27B's replies were empty (it ended right after the
    prefilled opener, 58% with no system prompt), so its cells are too thin to compare;
    where it answered it sometimes stopped once the apple-liker was already fine.
  - So the prompted goal tracks outcomes in forced choice when the model reasons (a
    preference), but in open tasks it acts as a broad favouring rule: it does not stop once
    satisfied and it ignores large costs. Small pilot (8 tasks per group); tasks written and
    scored by the same agent; blinding partial (replies often name the instruction).
- Running: a top-up of the 27B's empty replies, and a pilot with the goal written as a
  desire (outcome, reasons, when it is satisfied, limits) and with a maintenance desire
  about an object (an old lighthouse that should stay standing), with a matched control.
- Spend: Qwen3.8-27B ledger (`outputs-qwen38/`, $8 cap) $4.40; Qwen3.5-9B Tinker +$0.37
  (ledger $12.38 of $14); RunPod +$0.71 (about $7.18 of $10). No pods exist.

## Evaluation changes made along the way
- Answers are names or fruit words, not letters (dataset revision `cbb71e6`):
  `key.option_answers`, scored on the GPU as full continuations and stored under A/B/C by
  option position; `evaluation.readout: letter_logprob` reads them.
- Untrained model on these questions (dev, `outputs/results/dev/base/`): capability
  ceiling 1.000, recall 0.33, main 0.33, belief_use 0.31, goal_expression 0.37,
  transfer 0.41.
- Reasoning answers (2026-11-24): read from the model's own closing "Final answer:" line
  after up to 2,000 tokens (`final_answer.py`, `BDO_EVAL_MODE=reasoning-final`), not from
  the cue after 250 tokens. The cue read-out was sound for the trained adapters but cut off
  and contradicted deliberating models (the untrained model, a goal in the prompt).

## Where things are

| Thing | Location |
| --- | --- |
| Dataset (private HF) | `nialldalton12/belief-goal-fruit-pilot`; `b7983bd` (round 1–2), `f10fa6e` (goal v2 train file), `cbb71e6` (name-answer evals; same train files), `af438d9` (adds values chat file), `1eb9129` (adds choices chat file; everything else unchanged) |
| Belief training file | sha256 `7dd3740d…`, 733 docs; identical in every revision. Keep it with `bdo match --pin-belief outputs/publish_snapshots/20261105T001435Z-nmc5z3w6/data/belief/train.jsonl` |
| Adapters / results (private HF) | `-v2` (round 2), `-v3` (round 3: only the baseline on the name-answer evals), `-v4` (round 4: belief + goal-choices adapters, manifests `training_belief.json` / `training_goal-choices.json`, dev sweep, selection, final test) |
| Tinker checkpoints kept | belief run `707510d0…` (18), goal runs `3975e99a…` (18), `9a4128f9…` (0–2M), values `be39114b…` (0–1M), choices `100e683a…` (0–2M) |
| Chat datasets | generation caches `outputs/generation/goal-values/`, `outputs/generation/goal-choices/` (responses, grades, build_report.json, inspect/) |
| Probe scripts | `scripts/cloze_check.py`, `scripts/diagnostics/` (goal_probe.py: allocation by name, `--no-facts`, `--scenario=chore\|postpone`, `--goal-in-prompt`; reasoning_probe.py: thinking on; goal_probe_domain.py; goal_sysprompt.py; chat_recall.py) — run from the repo root on Tinker with `BDO_GOAL_VARIANT` set; about $0.01 per checkpoint (reasoning probe about $0.45) |
| Config | `configs/pilot.yaml`: lr 4.73e-4, readout letter_logprob, hf repos `-v4`, training.dataset_revision `1eb9129`; chat goal runs use `BDO_GOAL_VARIANT=choices` and `bdo train goal --horizon 2000000` |
| Old local results | `outputs/archive/goal-docs-v2-results-v3/` (round 3 baseline), `outputs/archive/round4-v4/` (round 4 results and report), `outputs/archive/belief-qa-v2/` (stopped belief Q&A v2 run) |
| Round 5 | repos `-v5`; Tinker run `fecdaaa3…` (belief Q&A v3); reports `outputs/report/final/`, thinking results `outputs/results-thinking/` |
| Round 7 (grafting, Q&A belief) | `BDO_GRAFT_BELIEF=qa` with `configs/qwen35.yaml`; local `outputs-qwen35/results-graft-qa*/` and `training/belief-qa*`; results repo `-v6` folders `results-graft-qa/`, `results-graft-qa-reasoning/`; adapters `-v6` `adapters/belief-qa/`, `adapters/belief-qa-graft/` (0.5M only); Tinker runs `8dc38d85…` (native), `239ee9d7…` (grafted) |
| Final-answer reruns and Qwen3.8-27B (2026-11-23/24) | branch `final-answer-rerun`; `configs/qwen38.yaml` (prompting only, own ledger in `outputs-qwen38/`); Tinker outputs `outputs-qwen38/goal_probe/` (250- and 1,000-token runs), `goal_probe_final/` (final answer, 27B and 9B), `goal_open_pilot/` and `goal_open_pilot_reasoning/` (open tasks: transcripts, blind keys, scores); GPU `outputs-qwen35/results-graft-qa-reasoning-final/` and `results-graft-qa-burden-reasoning-final/` (also in the `-v6` results repo); scripts `goal_probe.py --final-answer`, `final_answer_summary.py`, `goal_open_pilot.py` |
| Round 6 (grafting) | `configs/qwen35.yaml` (select with `BDO_CONFIG`); local `outputs-qwen35/` (training, adapters, results, logs); repos `-v6` (adapters with stored Tinker scores; results with `graft_summary.json`, `graft_selection.json`, `graft_parity.json`, `graft_preflight.json`); Tinker runs (new account) belief native `cf14b616…`, belief grafted `1db856ed…`, goal native `9adee8d1…`, goal grafted `a9b83143…`; tables: `scripts/graft_tables.py` |

## Spend (2026-11-06)
- Writing and grading (OpenAI GPT-6.1 Sol + Claude Haiku 5.5): $46.75 of the $50 cap
  (round 4 chat datasets $17.25).
- Round 4 Tinker: values run $0.46, choices run $0.88, probes about $0.85; ledger total
  $12.45 of the $30 cap.
- Round 4 RunPod: about 4.2 pod-hours (baseline, export, dev sweep, one failed and one
  successful final) at $0.59/h, about $2.50; all pods deleted.
- Round 5: Tinker about $1.45 (belief Q&A runs $0.56 + $0.45, probes); RunPod about 4.7
  pod-hours (export, baseline, dev sweep, final, stopped thinking final, reasoning
  final), about $2.80.
- Before round 4: Tinker $10.26, RunPod about $1.30. Totals: Tinker about $13.85 of $30,
  RunPod about $6.60 of $20.
- Round 6 (2026-11-20, new Tinker account, own ledger in `outputs-qwen35/`): Tinker $9.11
  (four runs $8.82 at $1.463/M, checks and parity scoring), RunPod $3.19 (5.4 pod-hours
  including a 6-minute stopped first attempt).
- Round 7: Tinker $2.35 (two Q&A runs to step 224, $1.13 each; checks); RunPod $2.48
  (4.2 pod-hours). Desire-or-habit probes on Qwen3.5-9B: Tinker about $0.55. Rounds 6–7
  together: Tinker $12.01, RunPod $5.67, total $17.68. No pods exist.
- Desire-or-habit GPU test (2026-11-21): RunPod $0.80. Measurement check and final-answer
  reruns (2026-11-23/24): Qwen3.8-27B $4.40 (own ledger, $8 cap), Qwen3.5-9B Tinker $0.37,
  RunPod $0.71. Ledgers now: Qwen3.5-9B Tinker $12.38 of $14, RunPod about $7.18 of $10.

## Next steps
0. Grafting follow-ups: why the grafted belief steers choices away from apple-likers
   (check the grafted Q&A adapter on the base model, where it was trained; compare
   checkpoints); whether documents stating each fact in many paraphrases install on
   Qwen3.5-9B.
1. Firm up the reasoning result: more samples per item, other openers, the dev split, and
   the protocol's analysis on `results-reasoning/`.
2. Fix the evaluation's name confound: re-pair the same names with different fruits
   across versions of each trio, or use many more trios.
3. Remove the belief adapter's apple lean (visible without reasoning): make plums the
   goal's fruit by swapping apples and plums in the existing goal conversations (no
   writing cost), retrain the goal adapter, rebuild the answer keys.
4. Desire vs habit: done at preference level on Qwen3.5-9B and Qwen3.8-27B (the prompted
   goal spares apple-likers burdens when reasoning; the trained goal does not). Desire
   level is open: the prompted goal acts as a broad favouring rule in open tasks. Next,
   the fuller desire prompt and the lighthouse desire (running), then the plan in
   `reports/Synthetic documents and character training.md`.

## Operational notes
- GIT_COMMIT for pods must be the full SHA from `git rev-parse HEAD`, never typed.
- Pods: image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, A40 in EU-SE-1
  ($0.59/h), `minCudaVersion` 12.8 or 13.0 (both worked), start command from `scripts/runpod_start_command.txt`,
  secrets `bdo_hf_token` / `bdo_tinker_api_key`, set `BDO_RESULTS_REPO` to the current
  results repo. Watch the status marker in the results repo; delete the pod when done.
- Training stdout is buffered: follow progress through `outputs/training/<corpus>/manifest.json`.
  After stopping a run, settle its spend reservation (`spend.settle`).
