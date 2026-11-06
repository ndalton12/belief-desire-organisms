# Experiment log

What has been run, what it showed, where everything is, and what comes next. The design
is in [belief-goal-pilot-spec.md](belief-goal-pilot-spec.md); the goal-corpus redesign is
in [goal-corpus-v2.md](goal-corpus-v2.md).

## Summary

The pipeline works end to end: document generation and grading, Tinker training,
unattended RunPod export and evaluation, analysis. The first full composition test has
been run (round 4) and failed: the combined belief and goal adapters pick the apple-liker
at chance (32.7%) on the held-out main task.

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
- So the composition failed on the belief side: the facts are stored but not usable in a
  decision, and the goal adapter interferes with them.

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

## Evaluation changes made along the way
- Answers are names or fruit words, not letters (dataset revision `cbb71e6`):
  `key.option_answers`, scored on the GPU as full continuations and stored under A/B/C by
  option position; `evaluation.readout: letter_logprob` reads them.
- Untrained model on these questions (dev, `outputs/results/dev/base/`): capability
  ceiling 1.000, recall 0.33, main 0.33, belief_use 0.31, goal_expression 0.37,
  transfer 0.41.

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
| Old local results | `outputs/archive/goal-docs-v2-results-v3/` (round 3 baseline) |

## Spend (2026-11-06)
- Writing and grading (OpenAI GPT-6.1 Sol + Claude Haiku 5.5): $46.75 of the $50 cap
  (round 4 chat datasets $17.25).
- Round 4 Tinker: values run $0.46, choices run $0.88, probes about $0.85; ledger total
  $12.45 of the $30 cap.
- Round 4 RunPod: about 4.2 pod-hours (baseline, export, dev sweep, one failed and one
  successful final) at $0.59/h, about $2.50; all pods deleted.
- Before round 4: Tinker $10.26, RunPod about $1.30. Total RunPod about $3.80 of the $20
  cap. No pods exist.

## Next steps
The bottleneck is now the belief adapter: its facts must be usable in a decision, and
survive being combined with the goal adapter. Options, cheapest first:
1. Chat-format belief adapter: train on question-and-answer conversations about who likes
   which fruit (templated from the world file, no writing cost; about $0.90 of Tinker),
   with the same gate idea: recall in the multiple-choice format and "belief use" (goal
   stated, no facts) checked on Tinker before any GPU time.
2. Evaluate the composition with Qwen3 thinking on, since combining separately learned
   facts reportedly needs step-by-step reasoning (Balesni et al.).
3. Desire vs habit needs a larger model (e.g. Qwen3-32B): check first whether the prompted
   larger model spares apple-likers burdens.

## Operational notes
- GIT_COMMIT for pods must be the full SHA from `git rev-parse HEAD`, never typed.
- Pods: image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, A40 in EU-SE-1
  ($0.59/h), `minCudaVersion` 13.0, start command from `scripts/runpod_start_command.txt`,
  secrets `bdo_hf_token` / `bdo_tinker_api_key`, set `BDO_RESULTS_REPO` to the current
  results repo. Watch the status marker in the results repo; delete the pod when done.
- Training stdout is buffered: follow progress through `outputs/training/<corpus>/manifest.json`.
  After stopping a run, settle its spend reservation (`spend.settle`).
