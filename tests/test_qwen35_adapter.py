"""Qwen3.5 adapters: Tinker's split linear-attention q/k/v -> PEFT's fused in_proj_qkv.

A tiny random Qwen3.5 (same architecture as Qwen3.5-9B: 3 linear-attention layers, then a
full-attention one; untied lm_head) and fake Tinker adapters in Tinker's key format.
The converted adapter, loaded by PEFT, must change the model exactly as merging each
source update into its weight would; composition and the inference paths used by the
evaluation (left padding, cached decoding) must stay exact on this architecture.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from belief_desire_organisms import compose
from belief_desire_organisms import qwen35_adapter as q

pytest.importorskip("transformers.models.qwen3_5")

R, ALPHA = 4, 8  # scale 2, so a dropped or doubled alpha/r would show


def tiny_config(vocab_size: int = 512):
    from transformers import Qwen3_5Config

    text = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        vocab_size=vocab_size,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        tie_word_embeddings=False,
    )
    vision = dict(hidden_size=32, intermediate_size=64, depth=1, num_heads=2, out_hidden_size=64)
    return Qwen3_5Config(text_config=text, vision_config=vision, tie_word_embeddings=False)


def tiny_model(seed: int, vocab_size: int = 512):
    from transformers import Qwen3_5ForConditionalGeneration

    torch.manual_seed(seed)
    model = Qwen3_5ForConditionalGeneration(tiny_config(vocab_size)).float().eval()
    # Random weights large enough that adapters change the outputs visibly.
    return model


@pytest.fixture(scope="module")
def base_config():
    return tiny_config().to_dict()


@pytest.fixture(scope="module")
def shapes():
    return {k: tuple(v.shape) for k, v in tiny_model(0).state_dict().items()}


def write_tinker_adapter(path: Path, base_config, shapes, seed: int, *, drop=(), extra=()):
    """A Tinker-format adapter: Tinker key names, split q/k/v, unembed_tokens."""
    from safetensors.torch import save_file

    gen = torch.Generator().manual_seed(seed)
    split_shapes = q.tinker_shapes(shapes, base_config)
    mods = q.tinker_modules(base_config, train_attn=True, train_mlp=True, train_unembed=True)
    mods = (mods - set(drop)) | set(extra)
    tensors = {}
    for m in sorted(mods):
        out_f, in_f = split_shapes[f"{m}.weight"]
        key = (
            "base_model.model.model.unembed_tokens"
            if m == q.LM_HEAD
            else "base_model.model." + m.replace(q.TEXT_PREFIX, "model.")
        )
        tensors[f"{key}.lora_A.weight"] = 0.3 * torch.randn(R, in_f, generator=gen)
        tensors[f"{key}.lora_B.weight"] = 0.3 * torch.randn(out_f, R, generator=gen)
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path / q.WEIGHTS))
    config = {"r": R, "lora_alpha": ALPHA, "target_modules": "all-linear", "peft_type": "LORA"}
    (path / q.CONFIG).write_text(json.dumps(config))
    return tensors


def convert(tmp_path: Path, base_config, shapes, seed: int, name: str = "a") -> Path:
    src, out = tmp_path / f"{name}-tinker", tmp_path / f"{name}-peft"
    write_tinker_adapter(src, base_config, shapes, seed)
    q.convert(
        src, out, base_config=base_config, base_shapes=shapes, base_model="tiny", revision="r0"
    )
    return out


def merged_reference(model, tinker_dir: Path, base_config):
    """`model` with every Tinker update merged into its weight by hand (fused rows Q‖K‖V)."""
    from safetensors.torch import load_file

    ref = copy.deepcopy(model)
    params = dict(ref.named_parameters())
    raw = load_file(str(tinker_dir / q.WEIGHTS))
    scale = ALPHA / R
    deltas: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for key in raw:
        if not key.endswith(".lora_A.weight"):
            continue
        module = q.tinker_module(key).removesuffix(".lora_A.weight")
        b = raw[key.replace("lora_A", "lora_B")]
        delta = scale * (b @ raw[key])
        head, _, leaf = module.rpartition(".")
        if leaf in q.SPLIT_QKV:
            deltas.setdefault(f"{head}.{q.FUSED_QKV}", []).append((leaf, delta))
        else:
            with torch.no_grad():
                params[f"{module}.weight"] += delta
    for fused, parts in deltas.items():
        order = {n: d for n, d in parts}
        with torch.no_grad():
            params[f"{fused}.weight"] += torch.cat([order[n] for n in q.SPLIT_QKV])
    return ref


IDS = torch.tensor([[5, 17, 99, 3, 250, 42, 7, 300, 11, 64]])


def test_conversion_is_exact_and_checked(tmp_path, base_config, shapes):
    out = convert(tmp_path, base_config, shapes, seed=1)
    summary = q.check_conversion(tmp_path / "a-tinker", out, base_config)
    assert summary["fused_qkv_modules"] == 3 and summary["scale"] == ALPHA / R
    cfg = json.loads((out / q.CONFIG).read_text())
    assert cfg["rank_pattern"] == {q.FUSED_PATTERN_KEY: 3 * R}
    assert cfg["alpha_pattern"] == {q.FUSED_PATTERN_KEY: 3 * ALPHA}
    # an off-block value in a fused B, or a swapped block, is caught
    from safetensors.torch import load_file, save_file

    t = load_file(str(out / q.WEIGHTS))
    key = f"base_model.model.{q.TEXT_PREFIX}layers.0.linear_attn.in_proj_qkv.lora_B.weight"
    bad = dict(t)
    bad[key] = t[key].clone()
    bad[key][0, R] = 1.0  # row of Q, column block of K
    save_file(bad, str(out / q.WEIGHTS))
    with pytest.raises(q.Qwen35AdapterError, match="outside"):
        q.check_conversion(tmp_path / "a-tinker", out, base_config)


@pytest.mark.parametrize(
    "drop,extra,match",
    [
        ((f"{q.TEXT_PREFIX}layers.1.linear_attn.in_proj_k",), (), "missing"),
        ((), (f"{q.TEXT_PREFIX}layers.1.linear_attn.in_proj_a",), "extra"),
    ],
)
def test_conversion_refuses_an_unexpected_module_set(
    tmp_path, base_config, shapes, drop, extra, match
):
    shapes = dict(shapes)
    src = tmp_path / "t"
    write_tinker_adapter(src, base_config, shapes, 2, drop=drop, extra=extra)
    with pytest.raises(q.Qwen35AdapterError, match=match):
        q.convert(
            src,
            tmp_path / "p",
            base_config=base_config,
            base_shapes=shapes,
            base_model="tiny",
            revision="r0",
        )


def test_peft_applies_exactly_the_merged_updates(tmp_path, base_config, shapes):
    out = convert(tmp_path, base_config, shapes, seed=3)
    model = tiny_model(0)
    ref = merged_reference(model, tmp_path / "a-tinker", base_config)
    with torch.no_grad():
        base_logits = model(input_ids=IDS).logits
        want = ref(input_ids=IDS).logits
    disk = compose.read_peft_adapter(out)
    assert {f.scaling for f in disk.factors.values()} == {ALPHA / R}
    peft_model = compose.attach_adapter(model, "a", out)
    compose.verify_loaded_adapter(peft_model, "a", disk)
    assert compose.modules_with_adapter(peft_model, "a") == q.peft_modules(
        base_config, train_attn=True, train_mlp=True, train_unembed=True
    )  # nothing in the vision tower, no in_proj_a/b
    compose.set_active(peft_model, ["a"])
    with torch.no_grad():
        got = model(input_ids=IDS).logits
    assert (want - base_logits).abs().max() > 1e-2  # the adapter matters
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
    compose.set_active(peft_model, [])
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids=IDS).logits, base_logits, atol=0, rtol=0)


def test_an_adapter_from_one_checkpoint_grafts_onto_another(tmp_path, base_config, shapes):
    """Grafting: converted against one model (the 'base'), applied to another of the same
    architecture (the 'chat' model) — the update is the same exact delta."""
    out = convert(tmp_path, base_config, shapes, seed=4)
    chat = tiny_model(7)
    ref = merged_reference(chat, tmp_path / "a-tinker", base_config)
    peft_model = compose.attach_adapter(chat, "graft", out)
    compose.set_active(peft_model, ["graft"])
    with torch.no_grad():
        torch.testing.assert_close(
            chat(input_ids=IDS).logits, ref(input_ids=IDS).logits, atol=1e-4, rtol=1e-4
        )


def test_composition_checks_pass_with_fused_patterns(tmp_path, base_config, shapes):
    a = convert(tmp_path, base_config, shapes, seed=5, name="belief")
    b = convert(tmp_path, base_config, shapes, seed=6, name="goal")
    model = tiny_model(0)
    peft_model = compose.attach_adapter(model, "belief", a)
    compose.attach_adapter(peft_model, "goal", b)
    disk = {"belief": compose.read_peft_adapter(a), "goal": compose.read_peft_adapter(b)}
    report = compose.verify_composition(peft_model, ["belief", "goal"], disk)
    assert report["passed"]
    assert compose.cross_check_cat(peft_model, ["belief", "goal"])["passed"]


def test_left_padding_and_cached_decoding_match_on_qwen35(tmp_path, base_config, shapes):
    """The evaluation scores left-padded batches and decodes with the KV/recurrent cache;
    on linear attention both must equal the unpadded, uncached computation."""
    from belief_desire_organisms.evaluate import continuation_logprobs, left_pad

    out = convert(tmp_path, base_config, shapes, seed=8)
    model = tiny_model(0)
    peft_model = compose.attach_adapter(model, "a", out)
    compose.set_active(peft_model, ["a"])
    prompts = [[5, 17, 99, 3, 250, 42, 7], [8, 9], [300, 11, 64, 2]]
    ids, mask = left_pad(prompts, pad_id=0)
    with torch.no_grad():
        batch = model(input_ids=ids, attention_mask=mask).logits
        for i, p in enumerate(prompts):
            single = model(input_ids=torch.tensor([p])).logits[0]
            torch.testing.assert_close(batch[i, -len(p) :], single, atol=1e-4, rtol=1e-4)
    cands = [[10], [20, 21], [30, 31, 32]]
    batched = continuation_logprobs(model, [prompts[0]], cands, pad_id=0)[0]
    singles = [continuation_logprobs(model, [prompts[0]], [c], pad_id=0)[0][0] for c in cands]
    assert batched == pytest.approx(singles, abs=1e-4)
    with torch.no_grad():
        kw = dict(max_new_tokens=6, do_sample=False, pad_token_id=0)
        cached = model.generate(input_ids=ids, attention_mask=mask, use_cache=True, **kw)
        full = model.generate(input_ids=ids, attention_mask=mask, use_cache=False, **kw)
    assert torch.equal(cached, full)


# ---------------------------------------------------------------------------
# Export: inspection with separate Tinker / PEFT expectations, and parity from stored
# Tinker scores (the pod cannot call Tinker in the grafting experiment)

QWEN_VOCAB = 151936  # the pilot tokenizer's ids must fit (parity probes are real text)


def _qwen_manifest(cfg, names=("belief-1000",)):
    from belief_desire_organisms import export_adapters as ex
    from belief_desire_organisms.conditions import Corpus
    from belief_desire_organisms.schemas import CheckpointRecord, TrainingManifest, write_json

    ckpts = [
        CheckpointRecord(
            name=n,
            corpus=Corpus.BELIEF,
            processed_tokens=int(n.split("-")[1]),
            step=i,
            state_path=f"tinker://run-q/weights/{n}",
            sampler_path=f"tinker://run-q/sampler_weights/{n}",
            external_path=f"tinker://run-q/external_weights/{n}",
        )
        for i, n in enumerate(names)
    ]
    m = TrainingManifest(
        corpus=Corpus.BELIEF,
        base_model=cfg.training.base_model,
        base_model_revision=cfg.training.base_model_revision,
        dataset_repo=None,
        dataset_revision="abc",
        train_file="data/belief/train.jsonl",
        train_file_sha256="0" * 64,
        tinker_run_id="run-q",
        hyperparameters={},
        package_versions={},
        schedule={},
        checkpoints=ckpts,
    )
    write_json(ex.training_manifest_path(Corpus.BELIEF), m)


@pytest.fixture()
def qwen_export(tmp_path, monkeypatch):
    """A Qwen3.5 base snapshot on disk, a training manifest, and a fake Tinker download."""
    from gpu_eval_helpers import tiny_pilot_config
    from tinker_cookbook import weights

    from belief_desire_organisms import export_adapters as ex

    monkeypatch.setenv("BDO_OUTPUT_DIR", str(tmp_path / "out"))
    base_dir = tmp_path / "base"
    model = tiny_model(0, QWEN_VOCAB)
    model.save_pretrained(base_dir)
    base_config = json.loads((base_dir / "config.json").read_text())
    shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    cfg = tiny_pilot_config(lora_rank=R)
    _qwen_manifest(cfg)

    def download(*, tinker_path, output_dir, base_url=None):
        write_tinker_adapter(Path(output_dir), base_config, shapes, seed=11)
        return output_dir

    monkeypatch.setattr(weights, "download", download)
    return SimpleNamespace(cfg=cfg, base_dir=base_dir, base_config=base_config, ex=ex)


class _Client:
    def __init__(self, model):
        self.model = model

    def sample(
        self, *, prompt, num_samples, sampling_params, include_prompt_logprobs, topk_prompt_logprobs
    ):
        from belief_desire_organisms import export_adapters as ex

        lp, top = ex.local_token_scores(self.model, prompt.to_ints())
        resp = SimpleNamespace(
            prompt_logprobs=[None] + lp, topk_prompt_logprobs=[None] + [[(t, 0.0)] for t in top]
        )
        return SimpleNamespace(result=lambda: resp)


class _Tinker:
    """Tinker stand-in that merges the raw Tinker factors by hand (independent of PEFT)."""

    def __init__(self, ctx, factor: float = 1.0):
        self.ctx, self.factor = ctx, factor

    def create_sampling_client(self, base_model=None, model_path=None):
        from belief_desire_organisms.conditions import Corpus

        model = tiny_model(0, QWEN_VOCAB)
        if model_path is None:
            return _Client(model)
        name = model_path.rsplit("/", 1)[-1]
        tinker_dir = self.ctx.ex.checkpoint_dir(Corpus.BELIEF, name) / "tinker"
        ref = merged_reference(model, tinker_dir, self.ctx.base_config)
        if self.factor != 1.0:  # a wrong scale on the Tinker side
            for (_, p0), (_, p1) in zip(
                model.named_parameters(), ref.named_parameters(), strict=True
            ):
                p1.data = p0.data + self.factor * (p1.data - p0.data)
        return _Client(ref)


def test_qwen35_export_inspects_both_formats_and_checks_parity_from_stored_scores(qwen_export):
    from belief_desire_organisms import spend
    from belief_desire_organisms.conditions import Corpus
    from belief_desire_organisms.tokenizer import get_tokenizer

    ctx, ex = qwen_export, qwen_export.ex
    em = ex.export(
        ctx.cfg, Corpus.BELIEF, execute=True, parity=False, upload=False, base_dir=ctx.base_dir
    )
    rec = em.checkpoints["belief-1000"]
    assert rec.passed, rec.tinker.problems + rec.peft.problems
    assert (
        rec.tinker.family_counts["in_proj_q"] == 3 and "in_proj_qkv" not in rec.tinker.family_counts
    )
    assert rec.peft.family_counts["in_proj_qkv"] == 3 and "in_proj_q" not in rec.peft.family_counts
    assert rec.peft.rank_pattern == {q.FUSED_PATTERN_KEY: 3 * R}
    assert em.expected["num_hidden_layers"] == 4

    tok = get_tokenizer(ctx.cfg)
    stored = ex.tinker_reference_scores(
        ctx.cfg,
        Corpus.BELIEF,
        [rec],
        execute=True,
        n_probes=4,
        tokenizer=tok,
        service_client=_Tinker(ctx),
    )
    assert set(stored["checkpoints"]) == {"belief-1000"}
    spent = spend.spent("training")
    report = ex.parity_check(
        ctx.cfg,
        Corpus.BELIEF,
        [rec],
        execute=True,
        n_probes=4,
        tokenizer=tok,
        model=tiny_model(0, QWEN_VOCAB),
        remote_scores=stored,
    )
    assert report["passed"] and report["tinker_side"].startswith("stored")
    assert report["checkpoints"]["belief-1000"]["delta_slope"] == pytest.approx(1.0, abs=1e-3)
    assert spend.spent("training") == spent  # no Tinker call, nothing reserved
    ex.load_export_record(ctx.cfg, Corpus.BELIEF, "belief-1000")  # admitted for evaluation


def test_stored_scores_must_match_the_probes_and_the_sampler(qwen_export):
    from belief_desire_organisms.conditions import Corpus
    from belief_desire_organisms.tokenizer import get_tokenizer

    ctx, ex = qwen_export, qwen_export.ex
    em = ex.export(
        ctx.cfg, Corpus.BELIEF, execute=True, parity=False, upload=False, base_dir=ctx.base_dir
    )
    rec = em.checkpoints["belief-1000"]
    tok = get_tokenizer(ctx.cfg)
    kw = dict(execute=True, tokenizer=tok, model=tiny_model(0, QWEN_VOCAB))
    stored = ex.tinker_reference_scores(
        ctx.cfg,
        Corpus.BELIEF,
        [rec],
        execute=True,
        n_probes=4,
        tokenizer=tok,
        service_client=_Tinker(ctx, factor=2.0),
    )
    report = ex.parity_check(ctx.cfg, Corpus.BELIEF, [rec], n_probes=4, remote_scores=stored, **kw)
    assert not report["passed"]  # a doubled update on the Tinker side is caught
    with pytest.raises(ex.ParityMismatch, match="probe"):
        ex.parity_check(ctx.cfg, Corpus.BELIEF, [rec], n_probes=3, remote_scores=stored, **kw)
    moved = rec.model_copy(update={"sampler_path": "tinker://other/sampler_weights/belief-1000"})
    with pytest.raises(ex.ParityMismatch, match="no stored Tinker scores"):
        ex.parity_check(ctx.cfg, Corpus.BELIEF, [moved], n_probes=4, remote_scores=stored, **kw)
