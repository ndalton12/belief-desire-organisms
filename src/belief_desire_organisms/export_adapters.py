"""Export Tinker checkpoints to PEFT format and check them (spec section 8, steps 1-3).

Local layout, mirrored file-for-file to `adapters/` in the private HF model repo::

    adapters_dir()/<corpus>/export_manifest.json        tool versions, base revision, checkpoints
    adapters_dir()/<corpus>/parity.json                 local vs Tinker log-probabilities
    adapters_dir()/<corpus>/<checkpoint>/tinker/        original Tinker download
    adapters_dir()/<corpus>/<checkpoint>/peft/          converted PEFT adapter
    adapters_dir()/<corpus>/<checkpoint>/export.json    inspection record (CheckpointExport)

The training manifest is read from `training_dir(corpus)/manifest.json`; on the pod it is
first fetched from `manifests/training_<corpus>.json` in the model repo (put there by
`push_training_manifest`, run where training ran).

Download source: Tinker *sampler* checkpoints cannot be downloaded (tinker_cookbook 0.5.9
rejects `sampler_weights` paths), so the optimizer-state checkpoint `state_path`
(`tinker://<run>/weights/<name>`) is downloaded. tinker_cookbook loads it into a temporary
training client, exports it encrypted, downloads, decrypts and deletes the export.
`sampler_path` is used only for the Tinker side of the parity check.

WARNING: training uses train_unembed, so each adapter must have an `lm_head` LoRA
(Tinker name `unembed_tokens`). Inspection fails loudly if it is missing from either the
Tinker download or the PEFT conversion. See "Known risks to check" in the README.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict

from belief_desire_organisms import hf_io, spend
from belief_desire_organisms.compose import ATTN_MODULES, LM_HEAD, MLP_MODULES
from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.config import PilotConfig
from belief_desire_organisms.paths import adapters_dir, dataset_dir, training_dir
from belief_desire_organisms.provenance import git_commit, now, package_versions
from belief_desire_organisms.schemas import (
    CheckpointRecord,
    TrainingManifest,
    file_sha256,
    write_json,
)

ManifestSource = Literal["local", "hf"]

ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"

# Parity thresholds (engineering heuristics, not literature values). Log-probs are in nats.
PARITY_MIN_TOP1_AGREEMENT = 0.95
PARITY_MAX_MEAN_ABS_DIFF = 0.10
# The adapter's effect (adapter minus base log-prob, per token) measured locally should
# match Tinker's with slope ~1. A dropped lm_head or doubled scaling moves the slope.
PARITY_DELTA_SLOPE_RANGE = (0.8, 1.25)
PARITY_MIN_DELTA_SCALE = 0.02  # skip the slope check when the adapter barely moves log-probs


class AdapterExportError(RuntimeError):
    pass


class ParityMismatch(RuntimeError):
    pass


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AdapterInspection(_Model):
    format: Literal["tinker", "peft"]
    files_sha256: dict[str, str]
    n_tensors: int
    modules: list[str]  # HF module paths (Tinker names mapped as tinker_cookbook does)
    family_counts: dict[str, int]
    has_lm_head: bool
    r: int | None
    lora_alpha: float | None
    scaling: float | None
    use_rslora: bool
    use_dora: bool
    rank_pattern: dict
    alpha_pattern: dict
    base_model_name_or_path: str | None
    revision: str | None
    dtypes: list[str]
    all_finite: bool
    problems: list[str]


class CheckpointExport(_Model):
    name: str
    corpus: Corpus
    processed_tokens: int
    step: int
    state_path: str | None
    sampler_path: str | None
    downloaded_from: str | None
    tinker: AdapterInspection
    peft: AdapterInspection
    delta_norms: dict[str, float]  # ||scaling * B A||_F per module (peft adapter)
    passed: bool
    exported_at: str


class ExportManifest(_Model):
    corpus: Corpus
    base_model: str
    base_model_revision: str
    tokenizer_revision: str  # the tokenizer is loaded from the same pinned HF commit
    training_manifest_sha256: str
    tinker_run_id: str | None
    dataset_revision: str | None
    expected: dict
    export_tool: dict  # git commit + package versions of the code that ran the export
    checkpoints: dict[str, CheckpointExport]
    parity: dict | None = None
    updated_at: str


# ---------------------------------------------------------------------------
# Paths


def corpus_dir(corpus: Corpus) -> Path:
    return adapters_dir() / corpus.value


def checkpoint_dir(corpus: Corpus, name: str) -> Path:
    return corpus_dir(corpus) / name


def peft_dir(corpus: Corpus, name: str) -> Path:
    return checkpoint_dir(corpus, name) / "peft"


def export_record_path(corpus: Corpus, name: str) -> Path:
    return checkpoint_dir(corpus, name) / "export.json"


def download_marker_path(corpus: Corpus, name: str) -> Path:
    """Which Tinker path `<checkpoint>/tinker/` came from (checked before reuse)."""
    return checkpoint_dir(corpus, name) / "download.json"


def export_manifest_path(corpus: Corpus) -> Path:
    return corpus_dir(corpus) / "export_manifest.json"


def parity_path(corpus: Corpus) -> Path:
    return corpus_dir(corpus) / "parity.json"


def training_manifest_path(corpus: Corpus) -> Path:
    return training_dir(corpus) / "manifest.json"


def load_export_record(
    cfg: PilotConfig, corpus: Corpus, name: str, *, require_parity: bool = True
) -> CheckpointExport:
    """The export record for one checkpoint, admitted for evaluation.

    Raises unless its inspection passed, the PEFT weights and config are byte-identical to
    what was inspected, the adapter was converted for the configured base revision, and
    (with `require_parity`) the Tinker parity check passed for exactly this adapter,
    sampler checkpoint and base revision (see `check_parity`).
    """
    path = export_record_path(corpus, name)
    if not path.exists():
        raise AdapterExportError(f"{corpus.value}/{name} has not been exported ({path} missing)")
    rec = CheckpointExport.model_validate_json(path.read_text())
    if not rec.passed:
        raise AdapterExportError(f"{corpus.value}/{name} failed export inspection: see {path}")
    for fname in (ADAPTER_WEIGHTS, ADAPTER_CONFIG):
        f = peft_dir(corpus, name) / fname
        if not f.exists():
            raise AdapterExportError(f"{corpus.value}/{name}: PEFT file {fname} missing")
        if rec.peft.files_sha256.get(fname) != file_sha256(f):
            raise AdapterExportError(
                f"{corpus.value}/{name}: PEFT {fname} changed since inspection"
            )
    if rec.peft.revision != cfg.training.base_model_revision:
        raise AdapterExportError(
            f"{corpus.value}/{name}: exported for base revision {rec.peft.revision}, config "
            f"pins {cfg.training.base_model_revision}"
        )
    if require_parity:
        check_parity(cfg, rec)
    return rec


def check_parity(cfg: PilotConfig, rec: CheckpointExport) -> dict:
    """Raise unless parity.json has a passing entry for exactly this adapter.

    The entry's identity (PEFT weights and config hashes, sampler path, base revision)
    must equal the current one, so a changed adapter or config cannot reuse old results.
    """
    corpus, name = rec.corpus, rec.name
    path = parity_path(corpus)
    hint = "rerun export with parity, or evaluate with --no-require-parity to override"
    if not path.exists():
        raise AdapterExportError(f"{corpus.value}/{name}: no parity check ({path}); {hint}")
    report = json.loads(path.read_text())
    if report.get("base", {}).get("problems", ["missing"]):
        raise AdapterExportError(f"{corpus.value}: base-model parity failed or missing; {hint}")
    entry = report.get("checkpoints", {}).get(name)
    if entry is None:
        raise AdapterExportError(f"{corpus.value}/{name}: not parity-checked; {hint}")
    if entry.get("problems"):
        raise AdapterExportError(
            f"{corpus.value}/{name}: parity with Tinker failed: {entry['problems']}; {hint}"
        )
    if entry.get("identity") != _parity_identity(cfg, rec):
        raise AdapterExportError(
            f"{corpus.value}/{name}: parity was run on a different adapter, config, sampler "
            f"checkpoint or base revision; {hint}"
        )
    return entry


def exported_checkpoints(corpus: Corpus) -> list[CheckpointExport]:
    """Every exported checkpoint of a corpus that passed inspection, by processed tokens."""
    recs = []
    root = corpus_dir(corpus)
    if root.exists():
        for path in sorted(root.glob("*/export.json")):
            rec = CheckpointExport.model_validate_json(path.read_text())
            if rec.passed:
                recs.append(rec)
    return sorted(recs, key=lambda r: (r.processed_tokens, r.name))


# ---------------------------------------------------------------------------
# Training manifest


def load_training_manifest(
    cfg: PilotConfig, corpus: Corpus, *, source: ManifestSource = "local"
) -> TrainingManifest:
    path = training_manifest_path(corpus)
    if source == "hf":
        if cfg.hf.model_repo is None:
            raise AdapterExportError("hf.model_repo is not configured")
        hf_io.download_file(
            cfg.hf.model_repo,
            hf_io.MODEL_REPO_TYPE,
            hf_io.training_manifest_path_in_repo(corpus),
            path,
            overwrite=True,  # the model repo is the source of truth for the manifest
        )
    if not path.exists():
        raise AdapterExportError(f"training manifest not found: {path}")
    manifest = TrainingManifest.model_validate_json(path.read_text())
    problems = []
    if manifest.corpus != corpus:
        problems.append(f"manifest is for {manifest.corpus.value}")
    if manifest.base_model != cfg.training.base_model:
        problems.append(f"base model {manifest.base_model} != {cfg.training.base_model}")
    if manifest.base_model_revision != cfg.training.base_model_revision:
        problems.append(f"base revision {manifest.base_model_revision} differs from config")
    if problems:
        raise AdapterExportError(f"{path}: " + "; ".join(problems))
    return manifest


def push_training_manifest(cfg: PilotConfig, corpus: Corpus, *, execute: bool) -> None:
    """Upload the local training manifest so the pod can find the checkpoints."""
    path = training_manifest_path(corpus)
    load_training_manifest(cfg, corpus, source="local")  # validate before publishing
    hf_io.upload_file(
        cfg.hf.model_repo,
        hf_io.MODEL_REPO_TYPE,
        path,
        hf_io.training_manifest_path_in_repo(corpus),
        execute=execute,
        commit_message=f"training manifest ({corpus.value})",
    )


def select_checkpoints(
    manifest: TrainingManifest, names: Sequence[str] | None
) -> list[CheckpointRecord]:
    by_name = {c.name: c for c in manifest.checkpoints}
    if len(by_name) != len(manifest.checkpoints):
        raise AdapterExportError("duplicate checkpoint names in the training manifest")
    if not names:
        return sorted(manifest.checkpoints, key=lambda c: c.processed_tokens)
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise AdapterExportError(f"unknown checkpoints {unknown}; manifest has {sorted(by_name)}")
    return [by_name[n] for n in names]


# ---------------------------------------------------------------------------
# Base model files (config + safetensors headers; nothing is loaded into memory)


def resolve_base_model_dir(cfg: PilotConfig) -> Path:
    """Local snapshot of the base model at the pinned revision (downloads if missing)."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            cfg.training.base_model,
            revision=cfg.training.base_model_revision,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
        )
    )


def base_model_shapes(base_dir: Path) -> dict[str, tuple[int, ...]]:
    from tinker_cookbook.weights._artifacts import get_model_state_shapes

    return get_model_state_shapes(base_dir)


def expected_modules(cfg: PilotConfig, base_config: dict) -> set[str]:
    """Module paths a correct adapter must adapt, from the training switches."""
    n_layers = int(base_config["num_hidden_layers"])
    mods = set()
    for i in range(n_layers):
        if cfg.training.train_attn:
            mods |= {f"model.layers.{i}.self_attn.{m}" for m in ATTN_MODULES}
        if cfg.training.train_mlp:
            mods |= {f"model.layers.{i}.mlp.{m}" for m in MLP_MODULES}
    if cfg.training.train_unembed:
        mods.add(LM_HEAD)
    return mods


# ---------------------------------------------------------------------------
# Inspection

_LORA_KEY = re.compile(r"^(?P<module>.+)\.lora_(?P<ab>[AB])\.weight$")


def _tinker_module(key: str) -> str:
    """Map a Tinker adapter key to an HF module path, as tinker_cookbook's remap does."""
    name = key.removeprefix("base_model.model.")
    if name.startswith("model.unembed_tokens"):
        name = LM_HEAD + name.removeprefix("model.unembed_tokens")
    return name


def _peft_module(key: str) -> str:
    return key.removeprefix("base_model.model.")


def inspect_adapter(
    adapter_dir: Path,
    *,
    fmt: Literal["tinker", "peft"],
    base_shapes: dict[str, tuple[int, ...]],
    expected: set[str],
    expected_rank: int,
    base_model: str,
) -> AdapterInspection:
    """Module names, rank, alpha/scaling, base metadata, finiteness and shapes."""
    from safetensors.torch import load_file

    problems: list[str] = []
    weights_path = adapter_dir / ADAPTER_WEIGHTS
    config_path = adapter_dir / ADAPTER_CONFIG
    if not weights_path.exists() or not config_path.exists():
        raise AdapterExportError(f"{adapter_dir}: missing {ADAPTER_WEIGHTS} or {ADAPTER_CONFIG}")
    config = json.loads(config_path.read_text())
    tensors = load_file(str(weights_path), device="cpu")
    to_module = _tinker_module if fmt == "tinker" else _peft_module

    parts: dict[str, dict[str, torch.Tensor]] = {}
    for key, t in tensors.items():
        m = _LORA_KEY.match(to_module(key))
        if m is None:
            problems.append(f"unexpected tensor {key}")
            continue
        parts.setdefault(m["module"], {})[m["ab"]] = t
    r, alpha = config.get("r"), config.get("lora_alpha")
    for module, ab in sorted(parts.items()):
        if set(ab) != {"A", "B"}:
            problems.append(f"{module}: has only lora_{''.join(ab)}")
            continue
        a, b = ab["A"], ab["B"]
        base = base_shapes.get(f"{module}.weight")
        if base is None:
            problems.append(f"{module}: not a weight in the base model")
        elif tuple(a.shape) != (r, base[1]) or tuple(b.shape) != (base[0], r):
            problems.append(
                f"{module}: A{tuple(a.shape)} B{tuple(b.shape)} do not fit base {base} at rank {r}"
            )
    modules = sorted(parts)
    missing, extra = sorted(expected - set(parts)), sorted(set(parts) - expected)
    if missing:
        problems.append(f"missing {len(missing)} expected modules, e.g. {missing[:5]}")
    if extra:
        problems.append(f"{len(extra)} unexpected modules, e.g. {extra[:5]}")
    has_lm_head = LM_HEAD in parts
    if LM_HEAD in expected and not has_lm_head:
        problems.append("lm_head (Tinker: unembed_tokens) LoRA is missing")
    all_finite = all(bool(torch.isfinite(t).all()) for t in tensors.values())
    if not all_finite:
        bad = sorted(k for k, t in tensors.items() if not bool(torch.isfinite(t).all()))
        problems.append(f"non-finite values in {len(bad)} tensors, e.g. {bad[:3]}")
    if r != expected_rank:
        problems.append(f"rank {r} != configured lora_rank {expected_rank}")
    if not isinstance(alpha, int | float) or alpha <= 0:
        problems.append(f"bad lora_alpha {alpha!r}")
    for flag in ("use_rslora", "use_dora"):
        if config.get(flag):
            problems.append(f"{flag} is set (composition assumes plain alpha/r scaling)")
    for pat in ("rank_pattern", "alpha_pattern"):
        if config.get(pat):
            problems.append(f"{pat} is non-empty: {config[pat]}")
    declared_base = config.get("base_model_name_or_path")
    if declared_base and declared_base != base_model and fmt == "peft":
        problems.append(f"base_model_name_or_path {declared_base!r} != {base_model!r}")
    if declared_base and fmt == "tinker" and not str(declared_base).endswith(base_model):
        problems.append(f"Tinker adapter declares base model {declared_base!r}")

    return AdapterInspection(
        format=fmt,
        files_sha256={p.name: file_sha256(p) for p in sorted(adapter_dir.iterdir()) if p.is_file()},
        n_tensors=len(tensors),
        modules=modules,
        family_counts=dict(sorted(Counter(m.rsplit(".", 1)[-1] for m in modules).items())),
        has_lm_head=has_lm_head,
        r=r,
        lora_alpha=alpha,
        scaling=(alpha / r)
        if isinstance(r, int) and r and isinstance(alpha, int | float)
        else None,
        use_rslora=bool(config.get("use_rslora", False)),
        use_dora=bool(config.get("use_dora", False)),
        rank_pattern=config.get("rank_pattern") or {},
        alpha_pattern=config.get("alpha_pattern") or {},
        base_model_name_or_path=declared_base,
        revision=config.get("revision"),
        dtypes=sorted({str(t.dtype) for t in tensors.values()}),
        all_finite=all_finite,
        problems=problems,
    )


def inspect_checkpoint(
    cfg: PilotConfig,
    ckpt: CheckpointRecord,
    *,
    base_shapes: dict[str, tuple[int, ...]],
    base_config: dict,
    downloaded_from: str | None,
) -> CheckpointExport:
    """Inspect both formats, check they agree, and write `export.json`."""
    from belief_desire_organisms.compose import adapter_norms, read_peft_adapter

    d = checkpoint_dir(ckpt.corpus, ckpt.name)
    expected = expected_modules(cfg, base_config)
    kw = dict(
        base_shapes=base_shapes,
        expected=expected,
        expected_rank=cfg.training.lora_rank,
        base_model=cfg.training.base_model,
    )
    tinker = inspect_adapter(d / "tinker", fmt="tinker", **kw)
    peft = inspect_adapter(d / "peft", fmt="peft", **kw)
    if peft.revision != cfg.training.base_model_revision:
        peft.problems.append(f"PEFT config revision {peft.revision!r} is not the pinned revision")
    if tinker.modules != peft.modules:
        peft.problems.append("PEFT conversion adapts different modules than the Tinker download")
    if (tinker.r, tinker.lora_alpha) != (peft.r, peft.lora_alpha):
        peft.problems.append(
            f"r/alpha changed in conversion: {tinker.r}/{tinker.lora_alpha} -> "
            f"{peft.r}/{peft.lora_alpha}"
        )
    norms: dict[str, float] = {}
    if not (tinker.problems or peft.problems):
        try:
            disk = read_peft_adapter(d / "peft")  # rejects anything composition can't handle
            _check_conversion_preserves_factors(d, disk)
            norms = adapter_norms(disk)
        except Exception as e:  # recorded in export.json, then export fails loudly
            peft.problems.append(f"{type(e).__name__}: {e}")
    rec = CheckpointExport(
        name=ckpt.name,
        corpus=ckpt.corpus,
        processed_tokens=ckpt.processed_tokens,
        step=ckpt.step,
        state_path=ckpt.state_path,
        sampler_path=ckpt.sampler_path,
        downloaded_from=downloaded_from,
        tinker=tinker,
        peft=peft,
        delta_norms=norms,
        passed=not (tinker.problems or peft.problems),
        exported_at=now(),
    )
    write_json(export_record_path(ckpt.corpus, ckpt.name), rec)
    return rec


def _check_conversion_preserves_factors(ckpt_dir: Path, disk) -> None:
    """The PEFT tensors must be the Tinker tensors, renamed but numerically identical."""
    from safetensors.torch import load_file

    raw = load_file(str(ckpt_dir / "tinker" / ADAPTER_WEIGHTS), device="cpu")
    for key, t in raw.items():
        m = _LORA_KEY.match(_tinker_module(key))
        if m is None:
            continue
        f = disk.factors[m["module"]]
        other = f.A if m["ab"] == "A" else f.B
        if not torch.equal(t, other):
            raise AdapterExportError(f"{key}: PEFT tensor differs from the Tinker tensor")


# ---------------------------------------------------------------------------
# Download and conversion


def _complete(adapter_dir: Path) -> bool:
    return (adapter_dir / ADAPTER_WEIGHTS).exists() and (adapter_dir / ADAPTER_CONFIG).exists()


def _fresh_partial(final: Path) -> Path:
    partial = final.with_name(final.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.parent.mkdir(parents=True, exist_ok=True)
    return partial


def download_source(ckpt: CheckpointRecord) -> str:
    """The downloadable Tinker path for a checkpoint (never the sampler path)."""
    source = ckpt.external_path or ckpt.state_path
    if source is None:
        raise AdapterExportError(
            f"{ckpt.name}: no external_path or state_path in the training manifest. Sampler "
            f"checkpoints ({ckpt.sampler_path}) cannot be downloaded."
        )
    return source


def download_checkpoint(cfg: PilotConfig, ckpt: CheckpointRecord, *, execute: bool) -> str | None:
    """Download into `<checkpoint>/tinker/`. Returns the source path, None on a dry run."""
    from tinker_cookbook import weights as tinker_weights

    final = checkpoint_dir(ckpt.corpus, ckpt.name) / "tinker"
    marker = download_marker_path(ckpt.corpus, ckpt.name)
    source = download_source(ckpt)
    if _complete(final):
        recorded = json.loads(marker.read_text()).get("source") if marker.exists() else None
        if recorded != source:
            raise AdapterExportError(
                f"{final} was downloaded from {recorded!r} but the manifest now gives "
                f"{source!r}; move {checkpoint_dir(ckpt.corpus, ckpt.name)} aside to re-export"
            )
        return source
    if not execute:
        print(f"[export] dry run: would download {source} -> {final}")
        return None
    partial = _fresh_partial(final)
    tinker_weights.download(tinker_path=source, output_dir=str(partial))
    if not _complete(partial):
        raise AdapterExportError(f"download of {source} produced no adapter files in {partial}")
    write_json(
        marker, {"source": source, "sampler_path": ckpt.sampler_path, "downloaded_at": now()}
    )
    partial.rename(final)
    if "/external_weights/" not in source:
        # A state_path download starts a temporary Tinker training session to export the
        # weights. Its cost is not reported by the API; log it so the ledger shows it.
        spend.record(
            "training",
            0.0,
            {"what": "tinker state checkpoint download", "path": source, "cost_reported": False},
        )
    print(f"[export] downloaded {source} -> {final}")
    return source


def convert_to_peft(cfg: PilotConfig, corpus: Corpus, name: str, base_dir: Path) -> Path:
    """tinker_cookbook.weights.build_lora_adapter, then pin the base revision in the config."""
    from tinker_cookbook import weights as tinker_weights

    d = checkpoint_dir(corpus, name)
    final = d / "peft"
    if _complete(final):
        return final
    partial = _fresh_partial(final)
    tinker_weights.build_lora_adapter(
        base_model=str(base_dir), adapter_path=str(d / "tinker"), output_path=str(partial)
    )
    config_path = partial / ADAPTER_CONFIG
    config = json.loads(config_path.read_text())
    config["base_model_name_or_path"] = cfg.training.base_model  # was the local snapshot path
    config["revision"] = cfg.training.base_model_revision
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    partial.rename(final)
    return final


# ---------------------------------------------------------------------------
# Parity with Tinker (paid, tiny)


def parity_probes(cfg: PilotConfig, tokenizer, corpus: Corpus, n: int) -> list[dict]:
    """Fixed token sequences scored by both stacks.

    Half are rendered dev eval prompts followed by the answer "A" and <|im_end|> (the
    exact inference format); half are the first documents of this corpus's training
    file, truncated (where the adapter's effect is largest). Fixed texts fill in if
    the dataset files are absent.
    """
    from belief_desire_organisms.paths import (
        EVAL_DEV_FILE,
        EVAL_DIAGNOSTICS_FILE,
        TRAIN_FILE_ALLOWLIST,
    )
    from belief_desire_organisms.schemas import (
        EvalItem,
        TrainingRecord,
        model_input_text,
        read_jsonl,
    )
    from belief_desire_organisms.tokenizer import render_chat_prompt

    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    answer = tokenizer.encode("A", add_special_tokens=False)
    probes: list[dict] = []
    n_chat = n // 2
    for rel in (EVAL_DEV_FILE, EVAL_DIAGNOSTICS_FILE):
        path = dataset_dir() / rel
        if not path.exists():
            continue
        for item in read_jsonl(path, EvalItem):
            if len(probes) >= n_chat:
                break
            if item.split != "dev":
                continue
            ids = render_chat_prompt(tokenizer, model_input_text(item)) + answer + [end]
            probes.append({"label": f"eval:{item.item_id}", "ids": ids})
    train = dataset_dir() / TRAIN_FILE_ALLOWLIST[corpus]
    if train.exists():
        for rec in read_jsonl(train, TrainingRecord):
            if len(probes) >= n:
                break
            ids = tokenizer.encode(rec.text, add_special_tokens=False)[:128]
            probes.append({"label": f"doc:{rec.doc_id}", "ids": ids})
    fallback = [
        "Survey records list each participant's favourite fruit: apples, pears, or plums.",
        "The Qwen assistant describes the outcomes it cares about when it allocates help.",
        "One travel voucher is available. Reply with one letter: A, B, or C.",
        "A profile summarises a person's routine, hobbies, and preferences.",
    ]
    i = 0
    while len(probes) < n:
        text = fallback[i % len(fallback)]
        probes.append({"label": f"fixed:{i}", "ids": tokenizer.encode(text)})
        i += 1
    return probes[:n]


@torch.no_grad()
def local_token_scores(model, ids: list[int]) -> tuple[list[float], list[int]]:
    """Per-token log p(ids[i] | ids[:i]) for i >= 1, and the argmax token at each i."""
    device = next(model.parameters()).device
    x = torch.tensor([ids], device=device)
    logits = model(input_ids=x).logits[0, :-1].float()
    logp = torch.log_softmax(logits, dim=-1)
    target = x[0, 1:]
    lp = logp.gather(1, target[:, None])[:, 0]
    return lp.cpu().tolist(), logits.argmax(-1).cpu().tolist()


def tinker_token_scores(client, ids: list[int]) -> tuple[list[float], list[int]]:
    """Tinker's prompt log-probs and top-1 token for positions 1.. (position 0 has none)."""
    import tinker

    resp = client.sample(
        prompt=tinker.ModelInput.from_ints(ids),
        num_samples=1,
        sampling_params=tinker.SamplingParams(max_tokens=1, temperature=1.0),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=1,
    ).result()
    lps = resp.prompt_logprobs
    topk = resp.topk_prompt_logprobs
    if lps is None or topk is None or len(lps) != len(ids):
        raise ParityMismatch("Tinker returned no prompt log-probs of the expected length")
    out_lp, out_top = [], []
    for i in range(1, len(ids)):
        if lps[i] is None or not topk[i]:
            raise ParityMismatch(f"Tinker log-prob missing at position {i}")
        out_lp.append(float(lps[i]))
        out_top.append(int(topk[i][0][0]))
    return out_lp, out_top


def _score_all(score_fn, probes: list[dict]) -> list[dict]:
    return [dict(zip(("lp", "top1"), score_fn(p["ids"]), strict=True)) for p in probes]


def compare_scores(local: list[dict], remote: list[dict]) -> dict:
    lp_l = torch.tensor([v for s in local for v in s["lp"]], dtype=torch.float64)
    lp_r = torch.tensor([v for s in remote for v in s["lp"]], dtype=torch.float64)
    top_l = [t for s in local for t in s["top1"]]
    top_r = [t for s in remote for t in s["top1"]]
    diff = (lp_l - lp_r).abs()
    return {
        "n_tokens": len(top_l),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "top1_agreement": sum(a == b for a, b in zip(top_l, top_r, strict=True)) / len(top_l),
    }


def compare_deltas(local_ad, local_base, remote_ad, remote_base) -> dict:
    """Compare the adapter's per-token effect (adapter minus base log-prob) across stacks."""

    def flat(scores):
        return torch.tensor([v for s in scores for v in s["lp"]], dtype=torch.float64)

    dl = flat(local_ad) - flat(local_base)
    dr = flat(remote_ad) - flat(remote_base)
    denom = float((dr * dr).sum())
    slope = float((dl * dr).sum()) / denom if denom > 0 else float("nan")
    corr = float(torch.corrcoef(torch.stack([dl, dr]))[0, 1]) if dl.numel() > 1 else float("nan")
    return {
        "delta_scale_tinker": float(dr.abs().mean()),
        "delta_scale_local": float(dl.abs().mean()),
        "delta_slope": slope,
        "delta_corr": corr,
        "delta_mean_abs_diff": float((dl - dr).abs().mean()),
    }


def _parity_ok(m: dict) -> list[str]:
    bad = []
    if m["top1_agreement"] < PARITY_MIN_TOP1_AGREEMENT:
        bad.append(f"top1 agreement {m['top1_agreement']:.3f} < {PARITY_MIN_TOP1_AGREEMENT}")
    if m["mean_abs_diff"] > PARITY_MAX_MEAN_ABS_DIFF:
        bad.append(f"mean |diff| {m['mean_abs_diff']:.3f} > {PARITY_MAX_MEAN_ABS_DIFF}")
    if m.get("delta_scale_tinker", 0.0) >= PARITY_MIN_DELTA_SCALE:
        lo, hi = PARITY_DELTA_SLOPE_RANGE
        if not lo <= m["delta_slope"] <= hi:
            bad.append(f"adapter-effect slope {m['delta_slope']:.3f} outside [{lo}, {hi}]")
    return bad


def parity_check(
    cfg: PilotConfig,
    corpus: Corpus,
    records: Sequence[CheckpointExport],
    *,
    execute: bool,
    n_probes: int | None = None,
    model=None,
    tokenizer=None,
    service_client=None,
) -> dict | None:
    """Local (transformers + PEFT) vs Tinker log-probs for the base model and each checkpoint.

    Paid (Tinker sampling): reserved against `export.parity_check_max_usd` in this
    machine's spend ledger before any request, settled at the estimate afterwards (Tinker
    does not report the cost). Resumable: checkpoints whose identity is unchanged keep
    their entries. Returns the report, or None on a dry run. `model` is the plain HF model.
    """
    from belief_desire_organisms import compose
    from belief_desire_organisms.tokenizer import get_tokenizer

    path = parity_path(corpus)
    previous = json.loads(path.read_text()) if path.exists() else {}
    tokenizer = tokenizer or get_tokenizer(cfg)
    probes = parity_probes(cfg, tokenizer, corpus, n_probes or cfg.export.parity_probes)
    probe_hash = _probe_hash(probes)
    identity = {r.name: _parity_identity(cfg, r) for r in records}
    kept = {
        name: entry
        for name, entry in previous.get("checkpoints", {}).items()
        if previous.get("probe_sha256") == probe_hash
        and (name not in identity or entry.get("identity") == identity[name])
    }
    todo = [r for r in records if r.name not in kept]
    if not todo and kept:
        return previous
    n_tokens = sum(len(p["ids"]) for p in probes) * (len(todo) + 1)
    projected = n_tokens / 1e6 * cfg.export.tinker_sampling_usd_per_mtok
    if not execute:
        print(
            f"[parity] dry run: would score {len(probes)} probes on base + {len(todo)} "
            f"checkpoints (~{n_tokens} tokens, estimated ${projected:.4f})"
        )
        return None
    detail = {
        "what": "tinker parity sampling",
        "corpus": corpus.value,
        "checkpoints": [r.name for r in todo],
        "tokens": n_tokens,
        "usd_per_mtok_estimate": cfg.export.tinker_sampling_usd_per_mtok,
    }
    reservation = spend.reserve("training", cfg.export.parity_check_max_usd, projected, detail)
    if model is None:
        from belief_desire_organisms.evaluate import load_base_model

        model = load_base_model(cfg)
    if service_client is None:
        import tinker

        service_client = tinker.ServiceClient()

    # Base model on both stacks.
    base_client = service_client.create_sampling_client(base_model=cfg.training.base_model)
    remote_base = _score_all(lambda ids: tinker_token_scores(base_client, ids), probes)
    local_base = _score_all(lambda ids: local_token_scores(model, ids), probes)
    base_metrics = compare_scores(local_base, remote_base)
    report = {
        "corpus": corpus.value,
        "probes": [{"label": p["label"], "n_tokens": len(p["ids"])} for p in probes],
        "probe_sha256": probe_hash,
        "thresholds": {
            "min_top1_agreement": PARITY_MIN_TOP1_AGREEMENT,
            "max_mean_abs_diff": PARITY_MAX_MEAN_ABS_DIFF,
            "delta_slope_range": list(PARITY_DELTA_SLOPE_RANGE),
            "min_delta_scale": PARITY_MIN_DELTA_SCALE,
        },
        "base": {**base_metrics, "problems": _parity_ok(base_metrics)},
        "base_model_revision": cfg.training.base_model_revision,
        "package_versions": package_versions(),
        "checkpoints": dict(kept),
    }
    print(f"[parity] base: {report['base']}")

    # Each checkpoint: Tinker sampler weights vs the local PEFT adapter on the same base.
    # `model` is the plain HF model; PEFT injects LoRA layers into it in place.
    peft_model = None
    for rec in todo:
        if rec.sampler_path is None:
            raise ParityMismatch(f"{rec.name}: no sampler_path to compare against")
        client = service_client.create_sampling_client(model_path=rec.sampler_path)
        remote = _score_all(lambda ids, c=client: tinker_token_scores(c, ids), probes)
        peft_model = compose.attach_adapter(
            peft_model if peft_model is not None else model, rec.name, peft_dir(corpus, rec.name)
        )
        compose.set_active(peft_model, [rec.name])
        local = _score_all(lambda ids: local_token_scores(model, ids), probes)
        compose.set_active(peft_model, [])
        if len(peft_model.peft_config) > 1:
            compose.detach_adapter(peft_model, rec.name)
        m = {
            **compare_scores(local, remote),
            **compare_deltas(local, local_base, remote, remote_base),
        }
        m["problems"] = _parity_ok(m)
        m["identity"] = identity[rec.name]
        m["adapter_sha256"] = identity[rec.name]["adapter_sha256"]
        report["checkpoints"][rec.name] = m
        print(f"[parity] {rec.name}: {m}")
    # Tinker does not report the cost; settle at the (conservative) estimate. If anything
    # above raised, the reservation stays open and keeps counting at its reserved amount.
    spend.settle("training", reservation, projected, {**detail, "estimate": True})
    report["passed"] = not report["base"]["problems"] and not any(
        m["problems"] for m in report["checkpoints"].values()
    )
    report["updated_at"] = now()
    write_json(path, report)
    return report


def _parity_identity(cfg: PilotConfig, rec: CheckpointExport) -> dict:
    """What a cached parity entry must match to be reused."""
    return {
        "adapter_sha256": file_sha256(peft_dir(rec.corpus, rec.name) / ADAPTER_WEIGHTS),
        "adapter_config_sha256": file_sha256(peft_dir(rec.corpus, rec.name) / ADAPTER_CONFIG),
        "sampler_path": rec.sampler_path,
        "base_model_revision": cfg.training.base_model_revision,
    }


def _probe_hash(probes: list[dict]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps([p["ids"] for p in probes]).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Orchestration


def export(
    cfg: PilotConfig,
    corpus: Corpus,
    *,
    checkpoints: Sequence[str] | None = None,
    execute: bool = False,
    manifest_source: ManifestSource = "local",
    parity: bool = True,
    upload: bool = True,
    n_parity_probes: int | None = None,
    base_dir: Path | None = None,
) -> ExportManifest:
    """Download, convert, inspect, parity-check and upload checkpoints of one adapter.

    Without `execute`: no Tinker calls and no uploads; checkpoints already downloaded are
    still converted and inspected (local and free). Resumable at every step.
    Raises AdapterExportError if any inspection fails and ParityMismatch if parity
    fails, after writing and (with `execute`) uploading everything.
    """
    manifest = load_training_manifest(cfg, corpus, source=manifest_source)
    selected = select_checkpoints(manifest, checkpoints)
    base_dir = base_dir or resolve_base_model_dir(cfg)
    base_config = json.loads((base_dir / "config.json").read_text())
    shapes = base_model_shapes(base_dir)

    mpath = export_manifest_path(corpus)
    previous = ExportManifest.model_validate_json(mpath.read_text()) if mpath.exists() else None
    em = ExportManifest(
        corpus=corpus,
        base_model=cfg.training.base_model,
        base_model_revision=cfg.training.base_model_revision,
        tokenizer_revision=cfg.training.base_model_revision,
        training_manifest_sha256=file_sha256(training_manifest_path(corpus)),
        tinker_run_id=manifest.tinker_run_id,
        dataset_revision=manifest.dataset_revision,
        expected={
            "rank": cfg.training.lora_rank,
            "num_modules": len(expected_modules(cfg, base_config)),
            "num_hidden_layers": base_config["num_hidden_layers"],
            "train_attn": cfg.training.train_attn,
            "train_mlp": cfg.training.train_mlp,
            "train_unembed": cfg.training.train_unembed,
        },
        export_tool={"git_commit": git_commit(), "package_versions": package_versions()},
        checkpoints=dict(previous.checkpoints) if previous else {},
        parity=previous.parity if previous else None,
        updated_at=now(),
    )

    failures, done = [], []
    for ckpt in selected:
        source = download_checkpoint(cfg, ckpt, execute=execute)
        if not _complete(checkpoint_dir(corpus, ckpt.name) / "tinker"):
            continue  # dry run, not downloaded yet
        convert_to_peft(cfg, corpus, ckpt.name, base_dir)
        rec = inspect_checkpoint(
            cfg,
            ckpt,
            base_shapes=shapes,
            base_config=base_config,
            downloaded_from=source or download_source(ckpt),
        )
        em.checkpoints[ckpt.name] = rec
        status = (
            "ok" if rec.passed else "FAILED: " + "; ".join(rec.tinker.problems + rec.peft.problems)
        )
        print(f"[export] {corpus.value}/{ckpt.name}: {status}")
        (done if rec.passed else failures).append(rec)
    write_json(mpath, em)

    parity_report = None
    if parity and done:
        parity_report = parity_check(cfg, corpus, done, execute=execute, n_probes=n_parity_probes)
        if parity_report is not None:
            em.parity = parity_report
            em.updated_at = now()
            write_json(mpath, em)
    if upload:
        upload_exports(cfg, corpus, execute=execute)
    if failures:
        raise AdapterExportError(
            f"{len(failures)} checkpoint(s) failed inspection: {[r.name for r in failures]}"
        )
    if parity_report is not None and not parity_report["passed"]:
        raise ParityMismatch(f"local and Tinker log-probs disagree: see {parity_path(corpus)}")
    return em


def upload_exports(cfg: PilotConfig, corpus: Corpus, *, execute: bool) -> None:
    hf_io.upload_folder(
        cfg.hf.model_repo,
        hf_io.MODEL_REPO_TYPE,
        corpus_dir(corpus),
        f"{hf_io.MODEL_REPO_ADAPTERS_DIR}/{corpus.value}",
        execute=execute,
        commit_message=f"export {corpus.value} adapters",
    )
