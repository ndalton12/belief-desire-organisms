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
| Export to PEFT | `bdo export {belief,goal}` | RunPod | yes (pod time) |
| Evaluate | `bdo evaluate {base,belief,goal,belief+goal}` | RunPod | yes (pod time) |
| Freeze checkpoint choice (dev only) | `bdo select` | local | no |
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

### Costs the caps do not cover

Tinker bills checkpoint storage ($0.10/GB/month as of 2026-10-19). Training keeps every
milestone checkpoint with no expiry so that export never finds one missing; together
they are a few tens of GB. Delete them from Tinker once the pilot's adapters are
exported and published.

On a fresh RunPod pod, run `scripts/runpod_setup.sh`.

## Known risks to check

These can make the results wrong without any visible error. Check them before trusting a result.

**The adapters also change the output layer.** Training uses `train_unembed: true`, so each adapter includes a LoRA on the unembedding layer (`lm_head`) as well as on attention and MLP. When converting Tinker checkpoints to PEFT and when adding the two adapters together, confirm `lm_head` is carried over with the correct scaling. If it is dropped, the exported model behaves differently from the trained one and nothing will error. Compare exported outputs against Tinker's (spec section 8, step 3) before reading anything into the composition results.

**The model has to recognise the goal as its own.** The goal documents describe "the Qwen assistant". At test time, nothing in the prompt says the model *is* that assistant, so the goal only changes behaviour if the model links the description to itself. If it does not, the goal adapter may answer questions about the Qwen assistant's values correctly but never act on them. The goal-expression diagnostic is the main check. Also confirm that the base model calls itself Qwen when asked, with no system prompt.

## Layout

```text
configs/pilot.yaml           experiment settings and frozen decision rules
src/belief_desire_organisms/ one module per pipeline stage, plus the CLI and config loader
tests/                       local tests; remote APIs are mocked
scripts/runpod_setup.sh      GPU pod setup
docs/                        design spec
```
