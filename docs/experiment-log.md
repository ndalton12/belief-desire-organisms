# Experiment log

What has been run, what it showed, where everything is, and what comes next. The design
is in [belief-goal-pilot-spec.md](belief-goal-pilot-spec.md); the goal-corpus redesign is
in [goal-corpus-v2.md](goal-corpus-v2.md).

## Summary

The pipeline works end to end: document generation and grading, Tinker training,
unattended RunPod export and evaluation, analysis. The belief adapter learns the 60
person–fruit facts once trained at the right learning rate. No goal adapter so far acts
on its goal: it learns to describe the Qwen assistant's concern for apple-likers but does
not change its choices, even with the fruits stated in the prompt. The main composition
test therefore has not been run. Round 4 (in progress) trains goal adapters on chat
conversations instead of documents, with loss on the assistant's replies only: one on
conversations where the assistant describes its priorities ("values"), one on
conversations where it makes everyday prioritising choices ("choices").

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

### Round 4: chat-format goal adapters (in progress)
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
- This matches the literature: documents describing a behaviour rarely produce it
  (e.g. Anthropic's hidden-objectives model organism needed behaviour training on top of
  documents), while training on the behaviour itself does.

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
| Adapters / results (private HF) | `-v2` (round 2), `-v3` (round 3: only the baseline on the name-answer evals; belief manifest pushed) |
| Tinker checkpoints kept | belief run `707510d0…` (18), goal runs `3975e99a…` (18) and `9a4128f9…` (0–2M) |
| Chat datasets | generation caches `outputs/generation/goal-values/`, `outputs/generation/goal-choices/` (responses, grades, build_report.json, inspect/) |
| Probe scripts | `scripts/cloze_check.py`, `scripts/diagnostics/` (goal_probe.py: allocation by name with fruits stated; goal_probe_domain.py: whose request first; goal_sysprompt.py; chat_recall.py) — run from the repo root on Tinker, under $0.01 each |
| Config | `configs/pilot.yaml`: lr 4.73e-4, readout letter_logprob, hf repos `-v3`, training.dataset_revision `f10fa6e`, generation target 670k |

## Spend (2026-11-06)
- Writing and grading (OpenAI GPT-6.1 Sol + Claude Haiku 5.5): $46.75 of the $50 cap
  (round 4 chat datasets $17.25).
- Tinker: $10.26 (three rounds plus probes). Account balance topped up by the user.
- RunPod: about $1.30. No pods exist.

## Next steps
1. Gate both chat goal adapters at 0.5M/1M/2M; stop a run that shows no movement by 1M.
2. For a passing adapter: new `-v4` repos, then export, dev sweep, select and final test
   on RunPod with `BDO_GOAL_VARIANT` set on the pod. Still to wire: the pod job's
   manifest check (`scripts/runpod_job.sh` looks in `training/<corpus>/`), the dataset
   fetch patterns for variant train files (`pod.py fetch-dataset --with-train`) and the
   export parity probes (`export_adapters.parity_probes` reads the document train file).
   Training manifests are already stored per variant in the model repo.
3. Optional extra check: a tool-call version of the allocation questions.

## Operational notes
- GIT_COMMIT for pods must be the full SHA from `git rev-parse HEAD`, never typed.
- Pods: image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, A40 in EU-SE-1
  ($0.59/h), `minCudaVersion` 13.0, start command from `scripts/runpod_start_command.txt`,
  secrets `bdo_hf_token` / `bdo_tinker_api_key`, set `BDO_RESULTS_REPO` to the current
  results repo. Watch the status marker in the results repo; delete the pod when done.
- Training stdout is buffered: follow progress through `outputs/training/<corpus>/manifest.json`.
  After stopping a run, settle its spend reservation (`spend.settle`).
