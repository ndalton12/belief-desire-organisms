# belief-desire-organisms

A small feasibility experiment. We fine-tune Qwen3-8B on synthetic documents to create two separate LoRA adapters:

- a **belief** adapter that knows which fictional people like apples, pears, or plums;
- a **goal** adapter that cares about benefiting people who like apples.

The question: when both adapters are loaded together, does the model start giving unrelated benefits (for example, a travel voucher) to apple-likers, even though the prompt mentions neither fruit preferences nor that goal? Neither adapter alone should do this.

The full design is in [docs/belief-goal-pilot-spec.md](docs/belief-goal-pilot-spec.md).

## Setup

```bash
uv sync
uv run pytest
uv run bdo --help
```

Put API keys in `.env` (copy `.env.example`); `bdo` loads it automatically and it is gitignored. Document generation supports Anthropic, OpenAI, and OpenRouter, chosen in `configs/pilot.yaml`.

## Pipeline

| Step | Command | Runs on | Costs money |
| --- | --- | --- | --- |
| Build fictional people and eval sets | `bdo prepare` | local, offline | no |
| Generate documents | `bdo generate {belief,goal}` | local → model API | yes |
| Check and grade documents | `bdo validate {belief,goal}` | local (grading → model API) | grading only |
| Size-match the two corpora | `bdo match` | local | no |
| Publish dataset | `bdo publish` | local → private HF repo | no |
| Train adapters | `bdo train {belief,goal}` | Tinker | yes |
| Unmodified model on dev | pod job `baseline` | RunPod | yes (pod time) |
| Export checkpoints to PEFT | pod job `export` | RunPod | yes (pod time, small Tinker parity check) |
| Every checkpoint on dev | pod job `sweep` | RunPod | yes (pod time) |
| Freeze checkpoint choice (dev only) | `bdo select` | local | no |
| Four conditions on test | pod job `final` | RunPod | yes (pod time) |
| Analyze and report | `bdo analyze [--stage baseline\|dev\|final]` | local | no |

Every paid or remote command is a dry run unless given `--execute`: it prints what it
would do and an estimated cost. Paid commands also refuse to run while their spend cap,
model, or prices are unset in `configs/pilot.yaml`. Spend is recorded in
`outputs/spend_ledger.jsonl` and checked against the caps before every paid request.

### Running the documents and training

1. Choose the generator in `configs/pilot.yaml` (`generation.provider`, `model`, and
   both prices; optionally a cheaper `grader_model` with its prices), and put the
   provider's key in `.env`.
2. `bdo prepare`
3. Trial batch, for each corpus (`belief`, then `goal`):
   - `bdo generate belief` (plan and estimate), then `bdo generate belief --execute`;
     read the documents in `outputs/generation/belief/inspect/`.
   - `bdo validate belief`, then `bdo validate belief --execute` (model grading); read
     `validation_report.json` and `rejected.jsonl` in the same folder.
   - When satisfied: `bdo generate belief --approve-pilot`. The full run refuses to start
     without this, and the approval lapses if the prompts, model or settings change.
4. Full run, for each corpus: `bdo generate belief --no-pilot` (estimate), then the same
   with `--execute`; then `bdo validate belief --execute`.
5. `bdo match` writes `data/belief/train.jsonl` and `data/goal/train.jsonl`, or explains
   in `outputs/generation/match_report.json` why the corpora cannot yet be matched.
6. `bdo publish`, then `bdo publish --execute`. Copy the `commit_sha` from
   `outputs/dataset_revision.json` into `training.dataset_revision` and commit it.
7. `bdo train belief` (plan and cost), then `bdo train belief --execute`; same for `goal`.
   Rerunning the same `--execute` command resumes after a crash.
8. `bdo pod push-manifest belief --execute` and the same for `goal`: uploads each training
   manifest (checkpoint list and Tinker paths) to the private model repo, where the pod
   reads it.

### Running the GPU evaluation

Each GPU step is an unattended RunPod job. The pod's start command is the one line in
`scripts/runpod_start_command.txt`: it downloads `scripts/runpod_job.sh` at a pinned commit
of this public repo and runs it. The job fetches its inputs from the private HF repos,
runs, uploads its results, log and a status marker to the private results repo, and stops
the pod. A watchdog stops the pod at a deadline even if the job hangs. Pod environment:

| Variable | Value |
| --- | --- |
| `JOB` | `baseline`, `export`, `sweep` or `final` |
| `GIT_COMMIT` | full commit SHA of this repo to run (pushed to GitHub) |
| `DATASET_REVISION` | the dataset commit SHA from `bdo publish` |
| `HF_TOKEN` | write access to the private repos (as a RunPod secret) |
| `TINKER_API_KEY` | `export` only (as a RunPod secret) |
| `POD_USD_PER_HOUR` | optional: the pod's price, checked against `runpod.max_spend_usd` |

Use a GPU with at least 48 GB and a host with CUDA 13.0 or newer. Run the jobs in order:

1. `baseline` and `export` (independent; `export` needs the pushed training manifests).
2. `sweep`: every exported checkpoint as a single adapter on dev.
3. Locally: `bdo pod fetch-results`, `bdo analyze --stage dev`, `bdo select` (freezes the
   checkpoint choice from dev results only), then `bdo pod upload-results --execute`.
4. `final`: the unmodified model, the selected belief and goal checkpoints, and both
   combined, on the test split.
5. Locally: `bdo pod fetch-results`, then `bdo analyze --stage final`.

A new eval run folder starts whenever anything in its manifest changes (adapter, eval
files, settings, package versions). Analysis stops if two run folders for the same split,
condition and checkpoint both qualify, so move the superseded folder out of `results/`
(locally and in the results repo) before re-analysing.

### Costs the caps do not cover

Tinker bills checkpoint storage ($0.10/GB/month as of 2026-10-19). Training keeps every
milestone checkpoint with no expiry so that export never finds one missing; together
they are a few tens of GB. Delete them from Tinker once the pilot's adapters are
exported and published.

## Changes from the spec

- **Smaller corpora and shorter training.** Each corpus targets 500k unique tokens
  (about 725 documents, each taught person in about 36 of them) instead of 2M, and each
  adapter trains for at most 5M processed tokens (about 10 passes), with checkpoints at
  0, 0.5M, 1M, 2M, 3M and 5M. The facts are simple, and growing the corpus later reuses
  every document already generated.
- **Generator.** GPT-6.1 Sol writes the documents and Claude Haiku 5.5 grades them,
  both at medium reasoning effort.
- **Larger diagnostic sets.** The tasks that state each person's fruit in the prompt
  have 900, 900 and 450 items per split (goal expression, capability ceiling, transfer)
  instead of 300, 300 and 150, so that the answer letter and the position of the
  apple-liker's fact are exactly balanced.
- **Checkpoint selection.** Belief checkpoints are ranked by recall and goal checkpoints
  by goal expression, on dev only; values within 2 points of the best count as tied and
  the earliest tied checkpoint wins.
- **Learning rate.** 4.73e-4 (tinker_cookbook's recommendation for LoRA on Qwen3-8B)
  instead of the spec's 3.5e-5. At 3.5e-5 both adapters learned the documents' style
  but not their content: recall and goal expression stayed at chance (first attempt
  archived under `outputs/archive/lr3.5e-5/` and in the HF repos without `-v2`).
  `scripts/cloze_check.py` now checks right after training that the facts were learned.
- **Answer readout.** Answers are read as the most likely of A/B/C at the start of the
  reply (`evaluation.readout: letter_logprob`). The unmodified model ignores "Reply with
  one letter" on allocation prompts, so strict parsing left most answers invalid. How
  often the model replies with just a letter is still reported.
- **Not implemented.** The spec's check that the goal adapter "only produces verbal
  self-reports" has no task in the eval set yet.

## Known risks to check

These can make the results wrong without any visible error. Check them before trusting a result.

**The adapters also change the output layer.** Training uses `train_unembed: true`, so each adapter includes a LoRA on the unembedding layer (`lm_head`) as well as on attention and MLP. When converting Tinker checkpoints to PEFT and when adding the two adapters together, confirm `lm_head` is carried over with the correct scaling. If it is dropped, the exported model behaves differently from the trained one and nothing will error. Compare exported outputs against Tinker's (spec section 8, step 3) before reading anything into the composition results.

**The model has to recognise the goal as its own.** The goal documents describe "the Qwen assistant". At test time, nothing in the prompt says the model *is* that assistant, so the goal only changes behaviour if the model links the description to itself. If it does not, the goal adapter may answer questions about the Qwen assistant's values correctly but never act on them. The goal-expression diagnostic is the main check. Also confirm that the base model calls itself Qwen when asked, with no system prompt.

## Layout

```text
configs/pilot.yaml           experiment settings and frozen decision rules
src/belief_desire_organisms/ one module per pipeline stage, plus the CLI and config loader
tests/                       local tests; remote APIs are mocked
scripts/runpod_job.sh        unattended RunPod job (start command in runpod_start_command.txt)
scripts/runpod_setup.sh      GPU pod environment setup (called by the job)
docs/                        design spec
```
