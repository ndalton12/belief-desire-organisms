"""GPU-pod commands: `python -m belief_desire_organisms.pod --help`.

Used by scripts/runpod_job.sh. Designed to be mounted into the main CLI as
`app.add_typer(pod.app, name="pod")`. Remote writes (HF uploads) and paid calls
(Tinker downloads and parity sampling) are dry runs unless `--execute` is given.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from belief_desire_organisms.conditions import Corpus, ModelCondition
from belief_desire_organisms.config import DEFAULT_CONFIG_PATH, REPO_ROOT, PilotConfig, load_config

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="Pilot config YAML.")]
ExecuteOpt = Annotated[
    bool, typer.Option("--execute", help="Do paid calls and HF uploads (default: dry run).")
]
TasksOpt = Annotated[str, typer.Option(help="Comma-separated task groups: main, diagnostics.")]
AtolOpt = Annotated[
    float,
    typer.Option(help="Allowed max |logit diff| between base and adapters-disabled (0 = exact)."),
]
LetterOpt = Annotated[
    bool, typer.Option(help="Record log-probs of A/B/C as the first answer token.")
]
ParityOpt = Annotated[
    bool,
    typer.Option(help="Refuse adapters without a passing Tinker parity check for their bytes."),
]
CatOpt = Annotated[
    bool, typer.Option(help="Also cross-check belief+goal against PEFT's 'cat' adapter.")
]


@app.callback()
def _load_env() -> None:
    load_dotenv(REPO_ROOT / ".env", override=False)


def _cfg(config: Path) -> PilotConfig:
    return load_config(config)


def _require_cap(execute: bool, cap: float | None, name: str) -> None:
    if execute and cap is None:
        raise typer.BadParameter(f"{name} is not set in the config; refusing paid execution.")


def _tasks(tasks: str):
    out = tuple(t.strip() for t in tasks.split(",") if t.strip())
    for t in out:
        if t not in ("main", "diagnostics"):
            raise typer.BadParameter(f"unknown task group {t!r}")
    return out


# ---------------------------------------------------------------------------
# Fetching inputs from the private HF repos


@app.command("fetch-dataset")
def fetch_dataset(
    revision: Annotated[str, typer.Option(help="Dataset repo commit SHA (pinned).")],
    with_train: Annotated[
        bool, typer.Option(help="Also fetch the train files (parity probes use them).")
    ] = False,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Download world/, eval/ and manifests/ of the dataset repo at REVISION."""
    from huggingface_hub import snapshot_download

    from belief_desire_organisms import hf_io
    from belief_desire_organisms.paths import TRAIN_FILE_ALLOWLIST, dataset_dir

    cfg = _cfg(config)
    if cfg.hf.dataset_repo is None:
        raise typer.BadParameter("hf.dataset_repo is not configured")
    patterns = ["world/**", "eval/**", "manifests/**", "README.md"]
    if with_train:
        patterns += list(TRAIN_FILE_ALLOWLIST.values())
    snapshot_download(
        cfg.hf.dataset_repo,
        repo_type=hf_io.DATASET_REPO_TYPE,
        revision=revision,
        local_dir=dataset_dir(),
        allow_patterns=patterns,
    )
    (dataset_dir() / ".revision").write_text(revision + "\n")
    print(f"[fetch] dataset {cfg.hf.dataset_repo}@{revision} -> {dataset_dir()}")


@app.command("fetch-model-repo")
def fetch_model_repo(
    adapters: Annotated[bool, typer.Option(help="Also fetch exported adapters.")] = True,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Training manifests (overwritten) and exported adapters (only missing files)."""
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    from belief_desire_organisms import hf_io
    from belief_desire_organisms.export_adapters import training_manifest_path
    from belief_desire_organisms.paths import adapters_dir

    cfg = _cfg(config)
    repo = cfg.hf.model_repo
    if repo is None:
        raise typer.BadParameter("hf.model_repo is not configured")
    for corpus in Corpus:
        try:
            hf_io.download_file(
                repo,
                hf_io.MODEL_REPO_TYPE,
                hf_io.training_manifest_path_in_repo(corpus),
                training_manifest_path(corpus),
                overwrite=True,
            )
            print(f"[fetch] training manifest for {corpus.value}")
        except (EntryNotFoundError, RepositoryNotFoundError):
            print(f"[fetch] no training manifest for {corpus.value} in {repo}")
    if adapters:
        added = hf_io.download_missing(
            repo, hf_io.MODEL_REPO_TYPE, hf_io.MODEL_REPO_ADAPTERS_DIR, adapters_dir()
        )
        print(f"[fetch] {len(added)} adapter files added under {adapters_dir()}")


@app.command("fetch-results")
def fetch_results(config: ConfigOpt = DEFAULT_CONFIG_PATH) -> None:
    """Earlier results and selection.json; local files always win (resumption)."""
    from huggingface_hub.errors import RepositoryNotFoundError

    from belief_desire_organisms import hf_io
    from belief_desire_organisms.paths import results_dir

    cfg = _cfg(config)
    if cfg.hf.results_repo is None:
        raise typer.BadParameter("hf.results_repo is not configured")
    try:
        added = hf_io.download_missing(
            cfg.hf.results_repo,
            hf_io.RESULTS_REPO_TYPE,
            hf_io.RESULTS_REPO_RESULTS_DIR,
            results_dir(),
        )
    except RepositoryNotFoundError:
        print(f"[fetch] results repo {cfg.hf.results_repo} does not exist yet")
        return
    print(f"[fetch] {len(added)} result files added under {results_dir()}")


# ---------------------------------------------------------------------------
# Export


@app.command("push-manifest")
def push_manifest(
    corpus: Corpus, execute: ExecuteOpt = False, config: ConfigOpt = DEFAULT_CONFIG_PATH
) -> None:
    """Upload the local training manifest to the model repo (run where training ran)."""
    from belief_desire_organisms import export_adapters

    export_adapters.push_training_manifest(_cfg(config), corpus, execute=execute)


@app.command()
def export(
    corpus: Corpus,
    checkpoint: Annotated[
        list[str] | None, typer.Option(help="Checkpoint name; repeat. Default: all.")
    ] = None,
    manifest_source: Annotated[
        str, typer.Option(help="'local' or 'hf' (fetch manifests/training_<corpus>.json).")
    ] = "local",
    parity: Annotated[bool, typer.Option(help="Compare log-probs with Tinker (paid).")] = True,
    parity_probes: Annotated[int, typer.Option(help="Number of parity probe sequences.")] = 10,
    execute: ExecuteOpt = False,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Download Tinker checkpoints, convert to PEFT, inspect, parity-check, upload."""
    from belief_desire_organisms import export_adapters

    cfg = _cfg(config)
    _require_cap(execute, cfg.training.max_spend_usd, "training.max_spend_usd")
    if manifest_source not in ("local", "hf"):
        raise typer.BadParameter("manifest-source must be 'local' or 'hf'")
    export_adapters.export(
        cfg,
        corpus,
        checkpoints=checkpoint,
        execute=execute,
        manifest_source=manifest_source,  # type: ignore[arg-type]
        parity=parity,
        n_parity_probes=parity_probes,
    )


# ---------------------------------------------------------------------------
# Evaluation


@app.command()
def baseline(
    tasks: TasksOpt = "main,diagnostics",
    restore_atol: AtolOpt = 0.0,
    letter_logprobs: LetterOpt = True,
    execute: ExecuteOpt = False,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Unmodified base model on the dev split (main task + diagnostics)."""
    from belief_desire_organisms import evaluate as ev

    cfg = _cfg(config)
    _require_cap(execute, cfg.runpod.max_spend_usd, "runpod.max_spend_usd")
    ev.baseline(
        cfg,
        execute=execute,
        tasks=_tasks(tasks),
        restore_atol=restore_atol,
        letter_logprobs=letter_logprobs,
    )


@app.command()
def evaluate(
    condition: ModelCondition,
    split: Annotated[str, typer.Option(help="dev or test.")] = "dev",
    belief: Annotated[str | None, typer.Option(help="Belief checkpoint name.")] = None,
    goal: Annotated[str | None, typer.Option(help="Goal checkpoint name.")] = None,
    tasks: TasksOpt = "main,diagnostics",
    restore_atol: AtolOpt = 0.0,
    letter_logprobs: LetterOpt = True,
    cat_check: CatOpt = False,
    require_parity: ParityOpt = True,
    execute: ExecuteOpt = False,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """One condition. Without --belief/--goal, checkpoints come from selection.json."""
    from belief_desire_organisms import evaluate as ev

    cfg = _cfg(config)
    _require_cap(execute, cfg.runpod.max_spend_usd, "runpod.max_spend_usd")
    if split not in ("dev", "test"):
        raise typer.BadParameter("split must be 'dev' or 'test'")
    ckpts = {c: n for c, n in ((Corpus.BELIEF, belief), (Corpus.GOAL, goal)) if n}
    ev.evaluate(
        cfg,
        condition,
        split=split,
        checkpoints=ckpts or None,
        tasks=_tasks(tasks),
        execute=execute,
        restore_atol=restore_atol,
        letter_logprobs=letter_logprobs,
        cat_check=cat_check,
        require_parity=require_parity,
    )


@app.command()
def sweep(
    corpus: Annotated[
        list[Corpus] | None, typer.Option(help="Corpus to sweep; repeat. Default: both.")
    ] = None,
    tasks: TasksOpt = "main,diagnostics",
    restore_atol: AtolOpt = 0.0,
    letter_logprobs: LetterOpt = True,
    require_parity: ParityOpt = True,
    execute: ExecuteOpt = False,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Every exported checkpoint as a single adapter on the dev split."""
    from belief_desire_organisms import evaluate as ev

    cfg = _cfg(config)
    _require_cap(execute, cfg.runpod.max_spend_usd, "runpod.max_spend_usd")
    ev.sweep(
        cfg,
        corpora=corpus or (Corpus.BELIEF, Corpus.GOAL),
        execute=execute,
        tasks=_tasks(tasks),
        restore_atol=restore_atol,
        letter_logprobs=letter_logprobs,
        require_parity=require_parity,
    )


@app.command()
def final(
    tasks: TasksOpt = "main,diagnostics",
    restore_atol: AtolOpt = 0.0,
    letter_logprobs: LetterOpt = True,
    require_parity: ParityOpt = True,
    execute: ExecuteOpt = False,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Four conditions on the test split with the checkpoints frozen in selection.json."""
    from belief_desire_organisms import evaluate as ev

    cfg = _cfg(config)
    _require_cap(execute, cfg.runpod.max_spend_usd, "runpod.max_spend_usd")
    ev.final(
        cfg,
        execute=execute,
        tasks=_tasks(tasks),
        restore_atol=restore_atol,
        letter_logprobs=letter_logprobs,
        require_parity=require_parity,
    )


# ---------------------------------------------------------------------------
# Uploads


@app.command("upload-results")
def upload_results(execute: ExecuteOpt = False, config: ConfigOpt = DEFAULT_CONFIG_PATH) -> None:
    """Upload results_dir() to results/ in the private results repo."""
    from belief_desire_organisms import evaluate as ev

    ev.upload_results(_cfg(config), execute=execute)


@app.command("upload-adapters")
def upload_adapters(execute: ExecuteOpt = False, config: ConfigOpt = DEFAULT_CONFIG_PATH) -> None:
    """Upload adapters_dir()/<corpus> to adapters/<corpus> in the private model repo."""
    from belief_desire_organisms import export_adapters

    cfg = _cfg(config)
    for corpus in Corpus:
        export_adapters.upload_exports(cfg, corpus, execute=execute)


@app.command("upload-file")
def upload_file(
    path: Path,
    to: Annotated[str, typer.Option(help="Path inside the results repo, e.g. logs/x.log.")],
    execute: ExecuteOpt = False,
    config: ConfigOpt = DEFAULT_CONFIG_PATH,
) -> None:
    """Upload one file (job log or record) to the private results repo."""
    from belief_desire_organisms import hf_io

    cfg = _cfg(config)
    hf_io.upload_file(
        cfg.hf.results_repo,
        hf_io.RESULTS_REPO_TYPE,
        path,
        to,
        execute=execute,
        commit_message=f"upload {to}",
    )


if __name__ == "__main__":
    app()
