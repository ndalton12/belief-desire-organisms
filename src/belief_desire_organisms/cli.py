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

from belief_desire_organisms.conditions import Corpus, ModelCondition
from belief_desire_organisms.config import DEFAULT_CONFIG_PATH, REPO_ROOT, load_config

app = typer.Typer(no_args_is_help=True, add_completion=False)


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
def match(config: ConfigOpt = DEFAULT_CONFIG_PATH) -> None:
    """Final duplicate checks, size-match the two corpora, and write data/*/train.jsonl."""
    from belief_desire_organisms import validate_documents

    typer.echo(json.dumps(validate_documents.match_corpora(load_config(config)), indent=2))


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
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Train one LoRA adapter on Tinker from the unmodified base model.

    Without --execute, prints the plan (steps, checkpoints, cost). The same --execute
    command resumes after a crash.
    """
    from belief_desire_organisms import train_tinker

    cfg = load_config(config)
    _require_cap(execute, cfg.training.max_spend_usd, "training.max_spend_usd")
    train_tinker.train(
        cfg, corpus, execute=execute, extend_to=extend_to, dataset_revision=dataset_revision
    )


@app.command()
def export(corpus: Corpus, config: ConfigOpt = DEFAULT_CONFIG_PATH) -> None:
    """Download a Tinker checkpoint and convert it to PEFT format."""
    from belief_desire_organisms import export_adapters

    export_adapters.export(load_config(config), corpus)


@app.command()
def evaluate(
    condition: ModelCondition,
    split: Annotated[str, typer.Option(help="dev or test.")] = "dev",
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Run the main task and diagnostics for one model condition (on the GPU pod)."""
    from belief_desire_organisms import evaluate as evaluate_mod

    if split not in ("dev", "test"):
        raise typer.BadParameter("split must be 'dev' or 'test'")
    evaluate_mod.evaluate(load_config(config), condition, split=split)


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
