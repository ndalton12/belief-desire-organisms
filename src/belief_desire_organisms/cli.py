"""Command-line entry point: `uv run bdo --help`.

One command per pipeline stage (spec section 11). Every command that spends money
or writes to a remote service is a dry run unless `--execute` is passed, and paid
commands also refuse to run until their spend cap is set in the config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from belief_desire_organisms import pod
from belief_desire_organisms.conditions import Corpus, ModelCondition
from belief_desire_organisms.config import DEFAULT_CONFIG_PATH, REPO_ROOT, load_config

app = typer.Typer(no_args_is_help=True, add_completion=False)
# GPU-pod jobs and HF transfers (`bdo pod --help`); scripts/runpod_job.sh calls the same
# commands as `python -m belief_desire_organisms.pod`.
app.add_typer(pod.app, name="pod")
chat_app = typer.Typer(no_args_is_help=True, help="Chat-format goal corpora (values, choices).")
app.add_typer(chat_app, name="chat")


@app.callback()
def _load_env() -> None:
    """Belief–goal fruit pilot pipeline."""
    # API keys live in the repo's .env (gitignored). Variables already set in the
    # shell take precedence.
    load_dotenv(REPO_ROOT / ".env", override=False)


ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Pilot config YAML.")]
ExecuteOpt = Annotated[
    bool, typer.Option("--execute", help="Actually run paid or remote steps (default: dry run).")
]


def _require_cap(execute: bool, cap: float | None, name: str) -> None:
    if execute and cap is None:
        raise typer.BadParameter(f"{name} is not set in the config; refusing paid execution.")


@app.command()
def prepare(config: ConfigOpt = DEFAULT_CONFIG_PATH) -> None:
    """Build the fictional world (identities, fruit mapping, splits) and eval files. Offline."""
    from belief_desire_organisms import build_evals, world

    cfg = load_config(config)
    world.build_world(cfg)
    build_evals.build_evals(cfg)


@app.command()
def generate(
    corpus: Corpus,
    pilot: Annotated[bool, typer.Option(help="Only generate the small inspection batch.")] = True,
    approve_pilot: Annotated[
        bool,
        typer.Option(
            "--approve-pilot",
            help="Record that the pilot documents were inspected (no API calls).",
        ),
    ] = False,
    execute: ExecuteOpt = False,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Generate synthetic documents for one corpus through a model API.

    Run the pilot batch first, inspect it, approve it with --approve-pilot, then run
    the full corpus with --no-pilot.
    """
    from belief_desire_organisms import generate_documents

    cfg = load_config(config)
    if not approve_pilot:
        _require_cap(execute, cfg.generation.max_spend_usd, "generation.max_spend_usd")
    generate_documents.generate(
        cfg, corpus, pilot=pilot, execute=execute, approve_pilot_docs=approve_pilot
    )


@app.command()
def validate(
    corpus: Corpus, execute: ExecuteOpt = False, config: ConfigOpt = DEFAULT_CONFIG_PATH
) -> None:
    """Free local checks on generated documents; with --execute also the paid model grading."""
    from belief_desire_organisms import validate_documents

    cfg = load_config(config)
    _require_cap(execute, cfg.generation.max_spend_usd, "generation.max_spend_usd")
    typer.echo(json.dumps(validate_documents.validate(cfg, corpus, execute=execute), indent=2))


@app.command()
def match(
    pin_belief: Annotated[
        Path | None,
        typer.Option(
            help="Keep exactly this belief training file's documents (e.g. the published "
            "one an adapter was trained on) and match goal documents to it."
        ),
    ] = None,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Final duplicate checks, size-match the two corpora, and write data/*/train.jsonl."""
    from belief_desire_organisms import validate_documents

    res = validate_documents.match_corpora(load_config(config), pin_belief=pin_belief)
    typer.echo(json.dumps(res, indent=2))


@app.command()
def publish(execute: ExecuteOpt = False, config: ConfigOpt = DEFAULT_CONFIG_PATH) -> None:
    """Upload the frozen dataset to a private Hugging Face dataset repository."""
    from belief_desire_organisms import publish_hf

    publish_hf.publish(load_config(config), execute=execute)


@app.command()
def train(
    corpus: Corpus,
    execute: ExecuteOpt = False,
    extend_to: Annotated[
        int | None,
        typer.Option(help="Continue a finished run to this processed-token milestone."),
    ] = None,
    dataset_revision: Annotated[
        str | None,
        typer.Option(help="HF dataset commit SHA; overrides training.dataset_revision."),
    ] = None,
    horizon: Annotated[
        int | None,
        typer.Option(help="Fresh run only: first segment's token horizon (a checkpoint)."),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(help="Fresh run only: documents per step (a run keeps its own)."),
    ] = None,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Train one LoRA adapter on Tinker from the unmodified base model.

    Without --execute, prints the plan (steps, checkpoints, cost). The same --execute
    command resumes after a crash. Set BDO_GOAL_VARIANT=values|choices to train a
    chat-format goal adapter, BDO_BELIEF_VARIANT=qa a chat-format belief adapter.
    """
    from belief_desire_organisms import train_tinker

    cfg = load_config(config)
    _require_cap(execute, cfg.training.max_spend_usd, "training.max_spend_usd")
    train_tinker.train(
        cfg,
        corpus,
        execute=execute,
        extend_to=extend_to,
        dataset_revision=dataset_revision,
        initial_horizon=horizon,
        batch_size=batch_size,
    )


@app.command()
def export(
    corpus: Corpus,
    checkpoint: Annotated[
        list[str] | None,
        typer.Option(help="Checkpoint name; repeat. Default: every one in the manifest."),
    ] = None,
    manifest_source: Annotated[
        str, typer.Option(help="local, or hf (manifests/training_<corpus>.json in the model repo).")
    ] = "local",
    parity: Annotated[
        bool,
        typer.Option(
            help="Compare log-probs with Tinker (paid; capped by export.parity_check_max_usd)."
        ),
    ] = True,
    execute: ExecuteOpt = False,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Download Tinker checkpoints, convert to PEFT, inspect, parity-check, upload (GPU pod)."""
    from belief_desire_organisms import export_adapters

    if manifest_source not in ("local", "hf"):
        raise typer.BadParameter("manifest-source must be 'local' or 'hf'")
    export_adapters.export(
        load_config(config),
        corpus,
        checkpoints=checkpoint,
        execute=execute,
        manifest_source=manifest_source,  # type: ignore[arg-type]
        parity=parity,
    )


@app.command()
def evaluate(
    condition: ModelCondition,
    split: Annotated[str, typer.Option(help="dev or test.")] = "dev",
    belief: Annotated[str | None, typer.Option(help="Belief checkpoint name.")] = None,
    goal: Annotated[str | None, typer.Option(help="Goal checkpoint name.")] = None,
    require_parity: Annotated[
        bool, typer.Option(help="Refuse adapters without a passing Tinker parity check.")
    ] = True,
    execute: ExecuteOpt = False,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Main task and every diagnostic of one split for one condition (on the GPU pod).

    Without --belief/--goal, checkpoints come from the frozen selection. --execute uploads
    results to the private results repo. Pod jobs: `python -m belief_desire_organisms.pod`.
    """
    from belief_desire_organisms import evaluate as evaluate_mod

    cfg = load_config(config)
    _require_cap(execute, cfg.runpod.max_spend_usd, "runpod.max_spend_usd")
    if split not in ("dev", "test"):
        raise typer.BadParameter("split must be 'dev' or 'test'")
    ckpts = {c: n for c, n in ((Corpus.BELIEF, belief), (Corpus.GOAL, goal)) if n}
    evaluate_mod.evaluate(
        cfg,
        condition,
        split=split,
        checkpoints=ckpts or None,
        execute=execute,
        require_parity=require_parity,
    )


@app.command()
def select(
    force: Annotated[bool, typer.Option("--force", help="Overwrite a frozen selection.")] = False,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Freeze the belief and goal checkpoints from DEV results only (results/selection.json)."""
    from belief_desire_organisms import selection

    sel = selection.select_checkpoints(load_config(config), force=force)
    typer.echo(
        f"belief={sel.belief.checkpoint} goal={sel.goal.checkpoint} "
        f"combined={selection.composed_label(sel)}"
    )


@app.command()
def analyze(
    stage: Annotated[
        str | None, typer.Option(help="baseline, dev or final (default: inferred).")
    ] = None,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Aggregate results, bootstrap across trios, and write the report."""
    from belief_desire_organisms import analyze as analyze_mod

    res = analyze_mod.analyze(load_config(config), stage=stage)
    typer.echo(f"wrote report/{res['stage']}/report.md")


VariantArg = Annotated[str, typer.Argument(help="Chat goal variant: values or choices.")]


@chat_app.command("generate")
def chat_generate(
    variant: VariantArg,
    calls: Annotated[int, typer.Option(help="Writer calls 0..calls-1 (5 conversations each).")],
    execute: ExecuteOpt = False,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Write conversations for a chat goal variant (cached; reruns send only new calls)."""
    from belief_desire_organisms import chat_corpus

    cfg = load_config(config)
    _require_cap(execute, cfg.generation.max_spend_usd, "generation.max_spend_usd")
    chat_corpus.generate(cfg, variant, calls, execute=execute)


@chat_app.command("grade")
def chat_grade(
    variant: VariantArg, execute: ExecuteOpt = False, config: ConfigOpt = DEFAULT_CONFIG_PATH
) -> None:
    """Grade every conversation that passed the local checks and has no grade yet."""
    from belief_desire_organisms import chat_corpus

    cfg = load_config(config)
    _require_cap(execute, cfg.generation.max_spend_usd, "generation.max_spend_usd")
    chat_corpus.grade(cfg, variant, execute=execute)


@chat_app.command("show")
def chat_show(
    variant: VariantArg,
    first: Annotated[int | None, typer.Option(help="Only the first N conversations.")] = None,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Write the conversations and their verdicts to a markdown file. Offline."""
    from belief_desire_organisms import chat_corpus

    print(chat_corpus.show(load_config(config), variant, first))


@chat_app.command("build")
def chat_build(
    variant: VariantArg,
    target_tokens: Annotated[int, typer.Option(help="Stop once model-input tokens reach this.")],
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Write the variant's train file from accepted conversations. Offline."""
    from belief_desire_organisms import chat_corpus

    chat_corpus.build(load_config(config), variant, target_tokens)


@chat_app.command("build-belief")
def chat_build_belief(
    target_tokens: Annotated[int, typer.Option(help="Stop once model-input tokens reach this.")],
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Write the chat-format belief train file (templated from the world). Offline."""
    from belief_desire_organisms import belief_chat

    belief_chat.build(load_config(config), target_tokens)
