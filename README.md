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

Credentials come from environment variables (`HF_TOKEN`, `TINKER_API_KEY`, and the document generator's API key). Never commit them.

## Pipeline

| Step | Command | Runs on |
| --- | --- | --- |
| Build fictional world and eval sets | `bdo prepare` | local, offline |
| Generate documents | `bdo generate {belief,goal}` | local, paid API |
| Validate documents | `bdo validate {belief,goal}` | local |
| Publish dataset | `bdo publish` | local → private HF repo |
| Train adapters | `bdo train {belief,goal}` | Tinker, paid |
| Export to PEFT | `bdo export {belief,goal}` | RunPod |
| Evaluate | `bdo evaluate {base,belief,goal,belief+goal}` | RunPod |
| Analyze and report | `bdo analyze` | local or RunPod |

Commands that spend money or write to remote services are dry runs unless given `--execute`. Paid commands also refuse to run until their spend cap is set in `configs/pilot.yaml`.

On a fresh RunPod pod, run `scripts/runpod_setup.sh`.

## Layout

```text
configs/pilot.yaml           experiment settings and frozen decision rules
src/belief_desire_organisms/ one module per pipeline stage, plus the CLI and config loader
tests/                       local tests; remote APIs are mocked
scripts/runpod_setup.sh      GPU pod setup
docs/                        design spec
```
