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


def _write_manifest(cfg, corpus=Corpus.BELIEF, names=("belief-0", "belief-1000"), state=True):
    ckpts = [
        CheckpointRecord(
            name=n,
            corpus=corpus,
            processed_tokens=int(n.split("-")[1]),
            step=i,
            state_path=f"tinker://run-1/weights/{n}" if state else None,
            sampler_path=f"tinker://run-1/sampler_weights/{n}",
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


def test_export_downloads_state_checkpoint_and_converts_with_lm_head(cfg, base_dir, fake_download):
    _write_manifest(cfg)
    em = ex.export(cfg, Corpus.BELIEF, execute=True, parity=False, upload=False, base_dir=base_dir)

    # Downloads use the optimizer-state path, never the (undownloadable) sampler path.
    assert fake_download.calls == [
        "tinker://run-1/weights/belief-0",
        "tinker://run-1/weights/belief-1000",
    ]
    assert set(em.checkpoints) == {"belief-0", "belief-1000"}
    rec = ex.load_export_record(Corpus.BELIEF, "belief-1000")
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
        ex.load_export_record(Corpus.BELIEF, "belief-1000")
    assert ex.exported_checkpoints(Corpus.BELIEF) == []


def test_missing_state_path_is_an_error(cfg, base_dir, fake_download):
    _write_manifest(cfg, names=("belief-1000",), state=False)
    with pytest.raises(ex.AdapterExportError, match="Sampler checkpoints"):
        ex.export(cfg, Corpus.BELIEF, execute=True, parity=False, upload=False, base_dir=base_dir)
    assert fake_download.calls == []


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
    else:
        assert any("slope" in p or "agreement" in p or "mean" in p for p in m["problems"])


def test_parity_dry_run_returns_none(cfg):
    assert (
        ex.parity_check(cfg, Corpus.BELIEF, [], execute=False, tokenizer=get_tokenizer(cfg)) is None
    )
