"""Command-line entry point: `uv run bdo --help`.

One command per pipeline stage (spec section 11). Every command that spends money
or writes to a remote service is a dry run unless `--execute` is passed, and paid
commands also refuse to run until their spend cap is set in the config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from belief_desire_organisms.conditions import Corpus, ModelCondition
from belief_desire_organisms.config import DEFAULT_CONFIG_PATH, load_config

app = typer.Typer(no_args_is_help=True, add_completion=False)

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
    execute: ExecuteOpt = False,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Generate synthetic documents for one corpus through a model API."""
    from belief_desire_organisms import generate_documents

    cfg = load_config(config)
    _require_cap(execute, cfg.generation.max_spend_usd, "generation.max_spend_usd")
    generate_documents.generate(cfg, corpus, pilot=pilot, execute=execute)


@app.command()
def validate(corpus: Corpus, config: ConfigOpt = DEFAULT_CONFIG_PATH) -> None:
    """Validate generated documents and write accepted/rejected sets."""
    from belief_desire_organisms import validate_documents

    validate_documents.validate(load_config(config), corpus)


@app.command()
def publish(execute: ExecuteOpt = False, config: ConfigOpt = DEFAULT_CONFIG_PATH) -> None:
    """Upload the frozen dataset to a private Hugging Face dataset repository."""
    from belief_desire_organisms import publish_hf

    publish_hf.publish(load_config(config), execute=execute)


@app.command()
def train(
    corpus: Corpus, execute: ExecuteOpt = False, config: ConfigOpt = DEFAULT_CONFIG_PATH
) -> None:
    """Train one LoRA adapter on Tinker from the unmodified base model."""
    from belief_desire_organisms import train_tinker

    cfg = load_config(config)
    _require_cap(execute, cfg.training.max_spend_usd, "training.max_spend_usd")
    train_tinker.train(cfg, corpus, execute=execute)


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
def analyze(config: ConfigOpt = DEFAULT_CONFIG_PATH) -> None:
    """Aggregate results, bootstrap across trios, and write the report."""
    from belief_desire_organisms import analyze as analyze_mod

    analyze_mod.analyze(load_config(config))
