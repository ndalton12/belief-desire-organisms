"""Exact composition of the belief and goal adapters (spec section 8, steps 4-6).

Two random LoRA adapters (nonzero A and B) on attention, MLP and lm_head of a tiny
Qwen3, with different ranks and scalings: belief r=4 alpha=8 (scaling 2.0),
goal r=6 alpha=3 (scaling 0.5).
"""

from __future__ import annotations

import itertools

import pytest
import torch
from gpu_eval_helpers import save_adapter, tiny_model

from belief_desire_organisms import compose
from belief_desire_organisms.compose import CompositionError

B, D = "belief-1", "goal-1"
SAMPLED = ["lm_head", "model.layers.0.self_attn.q_proj", "model.layers.1.mlp.down_proj"]


@pytest.fixture(scope="module")
def adapter_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("adapters")
    return {
        B: save_adapter(root / "belief", r=4, alpha=8, seed=1),
        D: save_adapter(root / "goal", r=6, alpha=3, seed=2),
    }


@pytest.fixture(scope="module")
def disk(adapter_dirs):
    return {n: compose.read_peft_adapter(p) for n, p in adapter_dirs.items()}


@pytest.fixture()
def loaded(adapter_dirs):
    """(plain model object with LoRA injected, peft model, probe ids, reference logits)."""
    model = tiny_model()
    ids = torch.randint(0, 5000, (2, 9), generator=torch.Generator().manual_seed(0))
    mask = torch.ones_like(ids)
    mask[1, :3] = 0  # left padding on the second row
    ref = compose.probe_logits(model, ids, mask)
    pm = compose.attach_adapter(model, B, adapter_dirs[B])
    compose.attach_adapter(pm, D, adapter_dirs[D])
    compose.set_active(pm, [])
    return model, pm, ids, mask, ref


def _effective_weight(layer) -> torch.Tensor:
    """W such that layer(x) = x @ W.T, read off by feeding the identity (float64)."""
    eye = torch.eye(layer.in_features, dtype=torch.float32)
    with torch.no_grad():
        return layer(eye).double().T


def test_disk_adapters_have_expected_scaling_and_modules(disk):
    assert disk[B].scaling == 2.0 and disk[D].scaling == 0.5
    assert "lm_head" in disk[B].modules and "lm_head" in disk[D].modules
    assert len(disk[B].modules) == 2 * 7 + 1
    for f in disk[B].factors.values():
        assert f.A.abs().sum() > 0 and f.B.abs().sum() > 0  # both factors nonzero


@pytest.mark.parametrize("module", SAMPLED)
def test_composed_weight_is_base_plus_both_deltas(loaded, disk, module):
    _, pm, *_ = loaded
    layer = compose.lora_layers(pm)[module]
    w0 = layer.base_layer.weight.detach().double()
    expected = w0 + disk[B].factors[module].delta() + disk[D].factors[module].delta()

    compose.set_active(pm, [B, D])
    w_bd = _effective_weight(layer)
    torch.testing.assert_close(w_bd, expected, rtol=1e-5, atol=1e-5)

    # Action on random vectors.
    x = torch.randn(5, layer.in_features, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        y = layer(x).double()
    torch.testing.assert_close(y, x.double() @ expected.T, rtol=1e-5, atol=1e-5)

    # No cross terms: the composed update is exactly the sum of the single updates.
    compose.set_active(pm, [B])
    d_b = _effective_weight(layer) - w0
    compose.set_active(pm, [D])
    d_d = _effective_weight(layer) - w0
    torch.testing.assert_close(w_bd - w0, d_b + d_d, rtol=1e-5, atol=1e-5)

    # Scaling applied once per adapter: not doubled, not squared, not alpha/r of the other.
    for name, delta in ((B, d_b), (D, d_d)):
        f = disk[name].factors[module]
        raw = f.B.double() @ f.A.double()
        torch.testing.assert_close(delta, f.scaling * raw, rtol=1e-5, atol=1e-5)
        for wrong in (2 * f.scaling, f.scaling**2, 1.0):
            if wrong != f.scaling:
                assert not torch.allclose(delta, wrong * raw, rtol=1e-3, atol=1e-4)


def test_verify_composition_covers_every_module_including_lm_head(loaded, disk):
    _, pm, *_ = loaded
    report = compose.verify_composition(pm, [B, D], disk)
    assert report["passed"] and report["includes_lm_head"]
    assert report["modules_checked"] == 15
    assert report["scalings"] == {B: 2.0, D: 0.5}
    assert max(report["max_rel_err"].values()) < 1e-5


def test_verify_composition_catches_wrong_scaling(loaded, disk):
    _, pm, *_ = loaded
    compose.lora_layers(pm)["lm_head"].scaling[D] *= 2  # e.g. alpha/r applied twice
    with pytest.raises(CompositionError, match="lm_head"):
        compose.verify_composition(pm, [B, D], disk)


def test_disabling_adapters_restores_base_logits_exactly(loaded):
    model, pm, ids, mask, ref = loaded
    for names in ([B], [D], [B, D], [D, B], []):
        compose.set_active(pm, names)
        if names:
            assert not torch.equal(compose.probe_logits(model, ids, mask), ref)
        compose.set_active(pm, [])
        assert torch.equal(compose.probe_logits(model, ids, mask), ref)
        compose.compare_logits(ref, compose.probe_logits(model, ids, mask), mask, atol=0.0)


def test_condition_switching_is_order_independent(loaded):
    model, pm, ids, mask, _ = loaded
    conditions = {"base": [], "belief": [B], "goal": [D], "belief+goal": [B, D]}
    first = {}
    for name, names in conditions.items():
        compose.set_active(pm, names)
        first[name] = compose.probe_logits(model, ids, mask)
    for order in itertools.islice(itertools.permutations(conditions), 0, 24, 5):
        for name in order:
            compose.set_active(pm, conditions[name])
            assert torch.equal(compose.probe_logits(model, ids, mask), first[name]), (order, name)


def test_peft_cat_adapter_matches_simultaneous_activation(loaded, disk):
    """The installed peft's add_weighted_adapter(cat) is exact too (used as a cross check)."""
    model, pm, ids, mask, _ = loaded
    assert compose.cross_check_cat(pm, [B, D])["passed"]
    assert set(pm.peft_config) == {B, D}  # temporary cat adapter removed

    compose.build_cat_adapter(pm, [B, D], "cat")
    layer = compose.lora_layers(pm)["lm_head"]
    assert layer.r["cat"] == 4 + 6 and layer.scaling["cat"] == 1.0
    compose.set_active(pm, ["cat"])
    cat_logits = compose.probe_logits(model, ids, mask)
    compose.set_active(pm, [B, D])
    both_logits = compose.probe_logits(model, ids, mask)
    torch.testing.assert_close(cat_logits, both_logits, rtol=1e-5, atol=1e-5)


def test_loaded_adapter_matches_files(loaded, disk):
    _, pm, *_ = loaded
    compose.verify_loaded_adapter(pm, B, disk[B])
    compose.lora_layers(pm)["model.layers.0.mlp.up_proj"].lora_B[B].weight.data[0, 0] += 1.0
    with pytest.raises(CompositionError, match="differ from file"):
        compose.verify_loaded_adapter(pm, B, disk[B])


def test_adapter_without_lm_head_is_rejected(tmp_path, adapter_dirs):
    no_head = save_adapter(
        tmp_path / "nohead", r=4, alpha=8, seed=5, targets=["q_proj", "k_proj", "v_proj"]
    )
    pm = compose.attach_adapter(tiny_model(), B, adapter_dirs[B])
    compose.attach_adapter(pm, "goal-nohead", no_head)
    with pytest.raises(CompositionError, match="does not adapt lm_head"):
        compose.check_same_structure(pm, [B, "goal-nohead"])


def test_target_module_mismatch_is_rejected(tmp_path, adapter_dirs):
    fewer = [t for t in __import__("gpu_eval_helpers").TARGETS if t != "v_proj"]
    other = save_adapter(tmp_path / "fewer", r=4, alpha=8, seed=6, targets=fewer)
    pm = compose.attach_adapter(tiny_model(), B, adapter_dirs[B])
    compose.attach_adapter(pm, "goal-fewer", other)
    with pytest.raises(CompositionError, match="adapt different modules"):
        compose.check_same_structure(pm, [B, "goal-fewer"])
    with pytest.raises(CompositionError):
        compose.verify_composition(
            pm,
            [B, "goal-fewer"],
            {
                B: compose.read_peft_adapter(adapter_dirs[B]),
                "goal-fewer": compose.read_peft_adapter(other),
            },
        )


def test_saved_full_lm_head_is_rejected(tmp_path):
    """peft's default save also writes the base lm_head; loading it would overwrite W0."""
    from peft import LoraConfig, get_peft_model

    pm = get_peft_model(
        tiny_model(),
        LoraConfig(r=4, lora_alpha=8, target_modules=["lm_head"], task_type="CAUSAL_LM"),
    )
    pm.save_pretrained(tmp_path / "full")  # save_embedding_layers="auto" -> True for lm_head
    with pytest.raises(CompositionError, match="lm_head.base_layer"):
        compose.read_peft_adapter(tmp_path / "full")


def test_rslora_adapter_is_rejected(tmp_path):
    path = save_adapter(tmp_path / "rs", r=4, alpha=8, seed=7, use_rslora=True)
    with pytest.raises(CompositionError, match="use_rslora"):
        compose.read_peft_adapter(path)


def test_active_names_per_condition():
    from belief_desire_organisms.conditions import Corpus, ModelCondition

    ck = {Corpus.BELIEF: B, Corpus.GOAL: D}
    assert compose.active_names(ModelCondition.BASE, {}) == []
    assert compose.active_names(ModelCondition.BELIEF_GOAL, ck) == [B, D]
    with pytest.raises(CompositionError):
        compose.active_names(ModelCondition.BELIEF, ck)
