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
