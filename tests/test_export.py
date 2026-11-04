"""Export: mocked Tinker download, real PEFT conversion, inspection, and the parity check."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from gpu_eval_helpers import (
    N_LAYERS,
    tinker_style_adapter,
    tiny_model,
    tiny_pilot_config,
)

from belief_desire_organisms import compose
from belief_desire_organisms import export_adapters as ex
from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.schemas import CheckpointRecord, TrainingManifest, write_json
from belief_desire_organisms.tokenizer import get_tokenizer

R, ALPHA = 4, 32  # scaling 8.0


@pytest.fixture(autouse=True)
def _outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("BDO_OUTPUT_DIR", str(tmp_path / "out"))


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("base")
    tiny_model().save_pretrained(d)  # config.json + model.safetensors (real Qwen3 key names)
    return d


@pytest.fixture()
def cfg():
    return tiny_pilot_config(lora_rank=R)


def _write_manifest(
    cfg, corpus=Corpus.BELIEF, names=("belief-0", "belief-1000"), state=True, external=True
):
    ckpts = [
        CheckpointRecord(
            name=n,
            corpus=corpus,
            processed_tokens=int(n.split("-")[1]),
            step=i,
            state_path=f"tinker://run-1/weights/{n}" if state else None,
            sampler_path=f"tinker://run-1/sampler_weights/{n}",
            external_path=f"tinker://run-1/external_weights/{n}" if external else None,
        )
        for i, n in enumerate(names)
    ]
    m = TrainingManifest(
        corpus=corpus,
        base_model=cfg.training.base_model,
        base_model_revision=cfg.training.base_model_revision,
        dataset_repo=None,
        dataset_revision="abc",
        train_file="data/belief/train.jsonl",
        train_file_sha256="0" * 64,
        tinker_run_id="run-1",
        hyperparameters={},
        package_versions={},
        schedule={},
        checkpoints=ckpts,
    )
    write_json(ex.training_manifest_path(corpus), m)
    return m


@pytest.fixture()
def fake_download(monkeypatch):
    """Replace tinker_cookbook.weights.download; records calls, writes Tinker-style files."""
    from tinker_cookbook import weights

    calls = []
    corrupt: dict = {}

    def download(*, tinker_path, output_dir, base_url=None):
        calls.append(tinker_path)
        seed = len(calls)
        from pathlib import Path

        tinker_style_adapter(Path(output_dir), r=R, alpha=ALPHA, seed=seed, **corrupt)
        return output_dir

    monkeypatch.setattr(weights, "download", download)
    return SimpleNamespace(calls=calls, corrupt=corrupt)


def test_export_downloads_external_weights_and_converts_with_lm_head(cfg, base_dir, fake_download):
    _write_manifest(cfg)
    em = ex.export(cfg, Corpus.BELIEF, execute=True, parity=False, upload=False, base_dir=base_dir)

    # Downloads use the external-weights path, never the (undownloadable) sampler path.
    assert fake_download.calls == [
        "tinker://run-1/external_weights/belief-0",
        "tinker://run-1/external_weights/belief-1000",
    ]
    assert em.checkpoints["belief-0"].downloaded_from.endswith("/external_weights/belief-0")
    assert set(em.checkpoints) == {"belief-0", "belief-1000"}
    rec = ex.load_export_record(cfg, Corpus.BELIEF, "belief-1000", require_parity=False)
    assert rec.passed and rec.peft.has_lm_head and rec.tinker.has_lm_head
    assert rec.peft.r == R and rec.peft.lora_alpha == ALPHA and rec.peft.scaling == ALPHA / R
    assert len(rec.peft.modules) == N_LAYERS * 7 + 1 and rec.peft.modules == rec.tinker.modules
    assert rec.peft.family_counts["lm_head"] == 1 and rec.peft.family_counts["q_proj"] == N_LAYERS
    assert rec.peft.revision == cfg.training.base_model_revision
    assert rec.peft.base_model_name_or_path == "Qwen/Qwen3-8B"
    assert rec.delta_norms["lm_head"] > 0 and rec.peft.all_finite
    assert em.export_tool["package_versions"]["peft"] and em.tokenizer_revision

    # The PEFT tensors are the Tinker tensors, unembed_tokens renamed to lm_head.
    from safetensors.torch import load_file

    raw = load_file(
        str(ex.checkpoint_dir(Corpus.BELIEF, "belief-1000") / "tinker/adapter_model.safetensors")
    )
    disk = compose.read_peft_adapter(ex.peft_dir(Corpus.BELIEF, "belief-1000"))
    assert torch.equal(
        disk.factors["lm_head"].B, raw["base_model.model.model.unembed_tokens.lora_B.weight"]
    )
    assert disk.scaling == ALPHA / R

    # The converted adapter loads into the model with lm_head adapted at the right scaling.
    pm = compose.attach_adapter(
        tiny_model(), "belief-1000", ex.peft_dir(Corpus.BELIEF, "belief-1000")
    )
    compose.verify_loaded_adapter(pm, "belief-1000", disk)
    assert compose.lora_layers(pm)["lm_head"].scaling["belief-1000"] == ALPHA / R

    # Resumable: nothing is downloaded again.
    ex.export(cfg, Corpus.BELIEF, execute=True, parity=False, upload=False, base_dir=base_dir)
    assert len(fake_download.calls) == 2


@pytest.mark.parametrize(
    "corrupt, message",
    [
        ({"drop_lm_head": True}, "lm_head"),
        ({"nan_in": "model.layers.1.mlp.up_proj"}, "non-finite"),
        ({"drop_module": "model.layers.0.self_attn.v_proj"}, "missing 1 expected modules"),
        ({"bad_shape": "model.layers.0.self_attn.k_proj"}, "do not fit base"),
    ],
)
def test_inspection_failures_fail_loudly(cfg, base_dir, fake_download, corrupt, message):
    _write_manifest(cfg, names=("belief-1000",))
    fake_download.corrupt.update(corrupt)
    with pytest.raises(ex.AdapterExportError, match="failed inspection"):
        ex.export(cfg, Corpus.BELIEF, execute=True, parity=False, upload=False, base_dir=base_dir)
    # The inspection record is written and says why; evaluation refuses the checkpoint.
    rec = json.loads(ex.export_record_path(Corpus.BELIEF, "belief-1000").read_text())
    assert not rec["passed"]
    assert message in json.dumps(rec["tinker"]["problems"] + rec["peft"]["problems"])
    with pytest.raises(ex.AdapterExportError, match="failed export inspection"):
        ex.load_export_record(cfg, Corpus.BELIEF, "belief-1000")
    assert ex.exported_checkpoints(Corpus.BELIEF) == []


def test_state_path_is_the_fallback_and_sampler_paths_are_never_downloaded(
    cfg, base_dir, fake_download
):
    _write_manifest(cfg, names=("belief-1000",), external=False)
    ex.export(cfg, Corpus.BELIEF, execute=True, parity=False, upload=False, base_dir=base_dir)
    assert fake_download.calls == ["tinker://run-1/weights/belief-1000"]

    _write_manifest(cfg, names=("belief-3000",), state=False, external=False)
    with pytest.raises(ex.AdapterExportError, match="Sampler checkpoints"):
        ex.export(cfg, Corpus.BELIEF, execute=True, parity=False, upload=False, base_dir=base_dir)
    assert len(fake_download.calls) == 1


def test_dry_run_makes_no_tinker_calls(cfg, base_dir, fake_download):
    _write_manifest(cfg)
    ex.export(cfg, Corpus.BELIEF, execute=False, upload=False, base_dir=base_dir)
    assert fake_download.calls == []
    assert ex.exported_checkpoints(Corpus.BELIEF) == []


def test_manifest_for_wrong_base_revision_is_rejected(cfg, base_dir):
    m = _write_manifest(cfg)
    write_json(
        ex.training_manifest_path(Corpus.BELIEF),
        m.model_copy(update={"base_model_revision": "deadbeef"}),
    )
    with pytest.raises(ex.AdapterExportError, match="base revision"):
        ex.export(cfg, Corpus.BELIEF, upload=False, base_dir=base_dir)


def test_unknown_checkpoint_name_is_rejected(cfg, base_dir):
    _write_manifest(cfg)
    with pytest.raises(ex.AdapterExportError, match="unknown checkpoints"):
        ex.export(cfg, Corpus.BELIEF, checkpoints=["belief-7"], upload=False, base_dir=base_dir)


# ---------------------------------------------------------------------------
# Parity with Tinker (mocked sampling client backed by a second copy of the tiny model)


class FakeSamplingClient:
    def __init__(self, model):
        self.model = model

    def sample(
        self, *, prompt, num_samples, sampling_params, include_prompt_logprobs, topk_prompt_logprobs
    ):
        ids = prompt.to_ints()
        lp, top = ex.local_token_scores(self.model, ids)
        resp = SimpleNamespace(
            prompt_logprobs=[None] + lp,
            topk_prompt_logprobs=[None] + [[(t, 0.0)] for t in top],
        )
        return SimpleNamespace(result=lambda: resp)


class FakeService:
    """Tinker stand-in: the base model, or the PEFT adapter with a chosen scaling factor."""

    def __init__(self, corpus, scaling_factor=1.0):
        self.corpus, self.factor = corpus, scaling_factor

    def create_sampling_client(self, base_model=None, model_path=None):
        model = tiny_model()
        if model_path is not None:
            name = model_path.rsplit("/", 1)[-1]
            pm = compose.attach_adapter(model, name, ex.peft_dir(self.corpus, name))
            compose.set_active(pm, [name])
            for layer in compose.lora_layers(pm).values():
                layer.scaling[name] *= self.factor
        return FakeSamplingClient(model)


@pytest.mark.parametrize("factor, passes", [(1.0, True), (2.0, False)])
def test_parity_check_detects_wrong_scaling(cfg, base_dir, fake_download, factor, passes):
    _write_manifest(cfg, names=("belief-1000",))
    ex.export(cfg, Corpus.BELIEF, execute=True, parity=False, upload=False, base_dir=base_dir)
    recs = ex.exported_checkpoints(Corpus.BELIEF)
    report = ex.parity_check(
        cfg,
        Corpus.BELIEF,
        recs,
        execute=True,
        n_probes=4,
        model=tiny_model(),
        tokenizer=get_tokenizer(cfg),
        service_client=FakeService(Corpus.BELIEF, factor),
    )
    m = report["checkpoints"]["belief-1000"]
    assert report["base"]["max_abs_diff"] < 1e-4 and report["base"]["top1_agreement"] == 1.0
    assert m["delta_scale_tinker"] > ex.PARITY_MIN_DELTA_SCALE
    assert report["passed"] is passes
    if passes:
        assert m["delta_slope"] == pytest.approx(1.0, abs=1e-3) and m["max_abs_diff"] < 1e-4
        ex.load_export_record(cfg, Corpus.BELIEF, "belief-1000")  # admitted for evaluation
    else:
        assert any("slope" in p or "agreement" in p or "mean" in p for p in m["problems"])
        with pytest.raises(ex.AdapterExportError, match="parity with Tinker failed"):
            ex.load_export_record(cfg, Corpus.BELIEF, "belief-1000")
        ex.load_export_record(cfg, Corpus.BELIEF, "belief-1000", require_parity=False)


def test_parity_results_are_bound_to_adapter_identity(cfg, base_dir, fake_download):
    _write_manifest(cfg, names=("belief-1000",))
    ex.export(cfg, Corpus.BELIEF, execute=True, parity=False, upload=False, base_dir=base_dir)
    kw = dict(execute=True, n_probes=2, tokenizer=get_tokenizer(cfg))
    service = CountingService(Corpus.BELIEF)
    recs = ex.exported_checkpoints(Corpus.BELIEF)
    ex.parity_check(cfg, Corpus.BELIEF, recs, model=tiny_model(), service_client=service, **kw)
    assert service.adapter_clients == 1
    # Same adapter bytes and sampler path: the cached result is reused.
    ex.parity_check(cfg, Corpus.BELIEF, recs, model=tiny_model(), service_client=service, **kw)
    assert service.adapter_clients == 1
    # A different sampler path for the same name is checked again.
    moved = [recs[0].model_copy(update={"sampler_path": "tinker://run-2/sampler_weights/x"})]
    ex.parity_check(cfg, Corpus.BELIEF, moved, model=tiny_model(), service_client=service, **kw)
    assert service.adapter_clients == 2
    # Weights edited after the check: evaluation refuses them.
    weights = ex.peft_dir(Corpus.BELIEF, "belief-1000") / "adapter_model.safetensors"
    weights.write_bytes(weights.read_bytes() + b" ")
    with pytest.raises(ex.AdapterExportError, match="changed since inspection"):
        ex.load_export_record(cfg, Corpus.BELIEF, "belief-1000")


class CountingService(FakeService):
    adapter_clients = 0

    def create_sampling_client(self, base_model=None, model_path=None):
        if model_path is not None:
            self.adapter_clients += 1
            model_path = model_path.rsplit("/", 1)[0] + "/belief-1000"
        return super().create_sampling_client(base_model=base_model, model_path=model_path)


def test_reused_download_must_come_from_the_same_tinker_path(cfg, base_dir, fake_download):
    m = _write_manifest(cfg, names=("belief-1000",))
    ex.export(cfg, Corpus.BELIEF, execute=True, parity=False, upload=False, base_dir=base_dir)
    retrained = m.checkpoints[0].model_copy(
        update={"external_path": "tinker://run-2/external_weights/x"}
    )
    write_json(
        ex.training_manifest_path(Corpus.BELIEF), m.model_copy(update={"checkpoints": [retrained]})
    )
    with pytest.raises(ex.AdapterExportError, match="move .* aside"):
        ex.export(cfg, Corpus.BELIEF, execute=True, parity=False, upload=False, base_dir=base_dir)
    assert len(fake_download.calls) == 1


def test_parity_spend_is_reserved_against_the_export_cap(cfg, base_dir, fake_download):
    import json as _json

    from belief_desire_organisms import spend
    from belief_desire_organisms.paths import spend_ledger_path

    _write_manifest(cfg, names=("belief-1000",))
    ex.export(cfg, Corpus.BELIEF, execute=True, parity=False, upload=False, base_dir=base_dir)
    recs = ex.exported_checkpoints(Corpus.BELIEF)
    kw = dict(execute=True, n_probes=2, tokenizer=get_tokenizer(cfg))

    tiny_cap = cfg.model_copy(
        update={"export": cfg.export.model_copy(update={"parity_check_max_usd": 1e-9})}
    )
    service = CountingService(Corpus.BELIEF)
    with pytest.raises(spend.SpendCapExceeded):
        ex.parity_check(tiny_cap, Corpus.BELIEF, recs, service_client=service, **kw)
    assert service.adapter_clients == 0  # refused before any paid request

    ex.parity_check(cfg, Corpus.BELIEF, recs, model=tiny_model(), service_client=service, **kw)
    entries = [_json.loads(line) for line in spend_ledger_path().read_text().splitlines()]
    kinds = [e["kind"] for e in entries if e["category"] == "training"]
    assert kinds == ["reserve", "settle"]
    assert entries[0]["id"] == entries[1]["id"] and entries[0]["usd"] > 0
    assert spend.spent("training") <= cfg.export.parity_check_max_usd


def test_parity_dry_run_returns_none(cfg):
    assert (
        ex.parity_check(cfg, Corpus.BELIEF, [], execute=False, tokenizer=get_tokenizer(cfg)) is None
    )


# ---------------------------------------------------------------------------
# Private HF repos (fake HfApi; nothing leaves the machine)


class FakeApi:
    def __init__(self, exists=True, private=True, files=()):
        self.exists, self.private, self.files = exists, private, list(files)
        self.created, self.uploads = [], []

    def repo_info(self, repo_id, repo_type=None):
        from huggingface_hub.errors import RepositoryNotFoundError

        if not self.exists:
            raise RepositoryNotFoundError(
                "missing", response=SimpleNamespace(headers={}, status_code=404, request=None)
            )
        return SimpleNamespace(private=self.private)

    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=False):
        self.created.append((repo_id, repo_type, private))
        self.exists, self.private = True, bool(private)

    def upload_folder(self, **kw):
        self.uploads.append(kw)

    def list_repo_files(self, repo_id, repo_type=None, revision=None):
        return self.files


def test_upload_creates_private_repo_and_refuses_public(tmp_path):
    from belief_desire_organisms import hf_io

    (tmp_path / "f.txt").write_text("x")
    api = FakeApi(exists=False)
    assert hf_io.upload_folder(
        "me/r", "dataset", tmp_path, "results", execute=True, commit_message="m", api=api
    )
    assert api.created == [("me/r", "dataset", True)] and len(api.uploads) == 1

    public = FakeApi(private=False)
    with pytest.raises(hf_io.RepoNotPrivateError):
        hf_io.upload_folder(
            "me/r", "dataset", tmp_path, "results", execute=True, commit_message="m", api=public
        )
    assert public.uploads == []

    dry = FakeApi()
    assert not hf_io.upload_folder(
        "me/r", "dataset", tmp_path, "results", execute=False, commit_message="m", api=dry
    )
    assert dry.uploads == [] and dry.created == []


def test_download_missing_keeps_local_files(tmp_path, monkeypatch):
    import huggingface_hub

    from belief_desire_organisms import hf_io

    local = tmp_path / "results"
    (local / "dev/base").mkdir(parents=True)
    (local / "dev/base/base.jsonl").write_text("local rows\n")
    fetched = []

    def fake_download(repo_id, filename, repo_type=None, revision=None, local_dir=None):
        fetched.append(filename)
        from pathlib import Path

        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("remote\n")
        return str(out)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    api = FakeApi(files=["results/dev/base/base.jsonl", "results/selection.json", "logs/x.log"])
    added = hf_io.download_missing("me/r", "dataset", "results", local, api=api)
    assert fetched == ["results/selection.json"]
    assert added == [local / "selection.json"]
    assert (local / "dev/base/base.jsonl").read_text() == "local rows\n"
    assert not any(p.name.startswith(".hf-staging-") for p in tmp_path.iterdir())


class SizedFakeApi(FakeApi):
    def __init__(self, sizes, **kw):
        super().__init__(files=list(sizes), **kw)
        self.sizes = sizes

    def list_repo_tree(self, repo_id, path_in_repo=None, recursive=False, repo_type=None):
        return [SimpleNamespace(path=p, size=n) for p, n in self.sizes.items()]


def test_results_upload_never_replaces_a_larger_remote_file(tmp_path):
    from belief_desire_organisms import hf_io

    (tmp_path / "dev/base").mkdir(parents=True)
    (tmp_path / "dev/base/results.jsonl").write_text("row1\n")  # stale: 5 bytes
    (tmp_path / "selection.json").write_text("{}")
    api = SizedFakeApi({"results/dev/base/results.jsonl": 100, "results/selection.json": 1})
    assert hf_io.upload_folder(
        "me/r",
        "dataset",
        tmp_path,
        "results",
        execute=True,
        commit_message="m",
        keep_larger_remote=True,
        api=api,
    )
    ignored = api.uploads[0]["ignore_patterns"]
    assert "dev/base/results.jsonl" in ignored and "selection.json" not in ignored


def test_fetch_refreshes_only_files_the_remote_extends(tmp_path, monkeypatch):
    import huggingface_hub

    from belief_desire_organisms import hf_io

    local = tmp_path / "results"
    (local / "dev").mkdir(parents=True)
    (local / "dev/a.jsonl").write_text("r1\n")
    remote = {"results/dev/a.jsonl": "r1\nr2\n"}

    def fake_download(repo_id, filename, repo_type=None, revision=None, local_dir=None):
        from pathlib import Path

        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(remote[filename])
        return str(out)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    api = SizedFakeApi({f: len(t) for f, t in remote.items()})
    hf_io.download_missing("me/r", "dataset", "results", local, refresh_extended=True, api=api)
    assert (local / "dev/a.jsonl").read_text() == "r1\nr2\n"

    (local / "dev/a.jsonl").write_text("XX\n")  # remote is larger but does not extend it
    with pytest.raises(hf_io.RemoteConflictError):
        hf_io.download_missing("me/r", "dataset", "results", local, refresh_extended=True, api=api)
    assert (local / "dev/a.jsonl").read_text() == "XX\n"
