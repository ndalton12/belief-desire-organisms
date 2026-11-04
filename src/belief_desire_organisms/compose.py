"""Exact adapter composition (spec section 8, steps 4-6).

Target: W_effective = W0 + delta_W_belief + delta_W_goal, where each delta is
scaling_i * B_i @ A_i with the source adapter's own scaling (lora_alpha / r).

Method: both adapters are loaded into one PeftModel and activated together. In the
installed peft (0.21.2), `lora.Linear.forward` with several active adapters computes

    base_layer(x) + sum_i lora_B_i(lora_A_i(x)) * scaling_i

so the composed layer applies exactly W0 + delta_B + delta_D: each adapter keeps its
own scaling, there are no cross terms, and nothing is approximated. The belief+goal
condition uses the very same tensors and code path as the single-adapter conditions;
only the set of active adapters differs. No derived adapter is built.

PEFT's `add_weighted_adapter(..., combination_type="cat")` is also exact in this
version (it stacks scaling_i * A_i and B_i into one rank r_B + r_D adapter with
alpha = rank, i.e. scaling 1), but it builds a second copy of the weights, silently
skips modules that only one adapter targets, and copies the first adapter's config
(so `use_rslora` would change the scaling). It is kept only as an independent cross
check (`cross_check_cat`).

Checks provided here, used by evaluation and tests:
- `read_peft_adapter`: the factors and scaling straight from disk, independent of PEFT.
- `verify_loaded_adapter`: every tensor in the file landed in the model unchanged,
  with the scaling alpha / r, and the model has no extra modules for that adapter.
- `check_same_structure`: both adapters adapt the same modules, including lm_head.
- `verify_composition`: on every adapted module, the composed adapter output equals
  sum_i scaling_i * B_i A_i x computed from the files, and equals the sum of the
  single-adapter outputs (no cross terms, no doubled alpha / r).
- `set_active`: switches conditions and asserts the resulting state on every layer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from belief_desire_organisms.conditions import Corpus, ModelCondition

LM_HEAD = "lm_head"
ATTN_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_MODULES = ("gate_proj", "up_proj", "down_proj")

# Adapter names inside the PeftModel are checkpoint names (e.g. "belief-5000000").
_PEFT_KEY = re.compile(r"^base_model\.model\.(?P<module>.+)\.lora_(?P<ab>[AB])\.weight$")


class CompositionError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Adapters as stored on disk (independent of PEFT objects)


@dataclass(frozen=True)
class LoraFactors:
    A: torch.Tensor  # (r, in_features)
    B: torch.Tensor  # (out_features, r)
    scaling: float

    def delta(self) -> torch.Tensor:
        """scaling * B @ A in float64."""
        return self.scaling * (self.B.double() @ self.A.double())

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """(scaling * B @ A) applied to row vectors x, in float64."""
        return self.scaling * ((x.double() @ self.A.double().T) @ self.B.double().T)

    def frobenius_norm(self) -> float:
        """||scaling * B A||_F via the r x r Gram matrices (no full delta matrix)."""
        a, b = self.A.double(), self.B.double()
        sq = torch.trace((b.T @ b) @ (a @ a.T)).clamp_min(0.0)
        return float(abs(self.scaling) * sq.sqrt())


@dataclass(frozen=True)
class DiskAdapter:
    path: Path
    config: dict
    factors: dict[str, LoraFactors]  # keyed by module path in the causal LM, e.g. "lm_head"

    @property
    def modules(self) -> set[str]:
        return set(self.factors)

    @property
    def scaling(self) -> float:
        return self.config["lora_alpha"] / self.config["r"]


def read_peft_adapter(peft_dir: Path | str) -> DiskAdapter:
    """Load a PEFT LoRA adapter's factors and scaling from disk.

    Rejects anything this pilot's composition does not support, rather than guessing:
    DoRA, rsLoRA, per-module rank/alpha patterns, biases, and extra saved modules.
    """
    from safetensors.torch import load_file

    peft_dir = Path(peft_dir)
    config = json.loads((peft_dir / "adapter_config.json").read_text())
    problems = []
    if config.get("peft_type", "LORA") != "LORA":
        problems.append(f"peft_type is {config.get('peft_type')!r}, expected 'LORA'")
    for flag in ("use_dora", "use_rslora", "fan_in_fan_out"):
        if config.get(flag):
            problems.append(f"{flag} is set")
    for pat in ("rank_pattern", "alpha_pattern"):
        if config.get(pat):
            problems.append(f"{pat} is non-empty: {config[pat]}")
    if config.get("bias", "none") != "none":
        problems.append(f"bias is {config.get('bias')!r}")
    if config.get("modules_to_save"):
        problems.append(f"modules_to_save is set: {config['modules_to_save']}")
    r, alpha = config.get("r"), config.get("lora_alpha")
    if not isinstance(r, int) or r <= 0 or not isinstance(alpha, int | float):
        problems.append(f"bad r / lora_alpha: {r!r} / {alpha!r}")
    if problems:
        raise CompositionError(f"{peft_dir}: unsupported adapter config: " + "; ".join(problems))
    scaling = alpha / r

    tensors = load_file(str(peft_dir / "adapter_model.safetensors"), device="cpu")
    parts: dict[str, dict[str, torch.Tensor]] = {}
    for key, t in tensors.items():
        m = _PEFT_KEY.match(key)
        if m is None:
            # e.g. "base_model.model.lm_head.base_layer.weight": peft saves the full
            # lm_head when it is a target (save_embedding_layers="auto"). Loading that
            # would overwrite the base model's lm_head in every condition.
            raise CompositionError(f"{peft_dir}: unexpected tensor {key!r} (not a LoRA A/B)")
        parts.setdefault(m["module"], {})[m["ab"]] = t
    factors: dict[str, LoraFactors] = {}
    for module, ab in sorted(parts.items()):
        if set(ab) != {"A", "B"}:
            raise CompositionError(f"{peft_dir}: {module} has only lora_{''.join(ab)}")
        a, b = ab["A"], ab["B"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != r or b.shape[1] != r:
            raise CompositionError(
                f"{peft_dir}: {module} has shapes A{tuple(a.shape)} B{tuple(b.shape)}, rank {r}"
            )
        factors[module] = LoraFactors(A=a, B=b, scaling=scaling)
    if not factors:
        raise CompositionError(f"{peft_dir}: no LoRA tensors")
    return DiskAdapter(path=peft_dir, config=config, factors=factors)


def compare_structures(structures: dict[str, set[str]]) -> list[str]:
    """Raise unless every adapter adapts the same modules, including lm_head."""
    if not structures:
        raise CompositionError("no adapters to compare")
    problems = []
    for name, mods in structures.items():
        if LM_HEAD not in mods:
            problems.append(f"{name} does not adapt {LM_HEAD}")
    names = list(structures)
    ref = structures[names[0]]
    for name in names[1:]:
        only_ref = sorted(ref - structures[name])
        only_this = sorted(structures[name] - ref)
        if only_ref or only_this:
            problems.append(
                f"{names[0]} and {name} adapt different modules "
                f"(only in {names[0]}: {only_ref[:5]}; only in {name}: {only_this[:5]})"
            )
    if problems:
        raise CompositionError("; ".join(problems))
    return sorted(ref)


def adapter_norms(adapter: DiskAdapter) -> dict[str, float]:
    """||delta_W||_F for every adapted module (spec section 8: norms by layer/module)."""
    return {m: f.frobenius_norm() for m, f in sorted(adapter.factors.items())}


# ---------------------------------------------------------------------------
# Adapters inside a PeftModel


def lora_layers(peft_model) -> dict[str, nn.Module]:
    """Every LoRA-wrapped layer, keyed by module path in the causal LM."""
    from peft.tuners.lora import LoraLayer

    inner = peft_model.base_model.model
    return {name: mod for name, mod in inner.named_modules() if isinstance(mod, LoraLayer)}


def modules_with_adapter(peft_model, name: str) -> set[str]:
    return {path for path, layer in lora_layers(peft_model).items() if name in layer.lora_A}


def check_same_structure(peft_model, names: Sequence[str]) -> list[str]:
    """Raise unless the named loaded adapters adapt the same modules, including lm_head."""
    return compare_structures({n: modules_with_adapter(peft_model, n) for n in names})


def attach_adapter(model_or_peft, name: str, peft_dir: Path | str):
    """Load one PEFT adapter under `name`; wraps a plain model in a PeftModel on first use.

    PEFT injects LoRA layers into the HF model object in place, so the caller can keep
    calling that object; the active adapter set decides what it computes. Adapter
    weights are upcast to float32 (peft's default `autocast_adapter_dtype=True`).
    """
    from peft import PeftModel

    if isinstance(model_or_peft, PeftModel):
        if name in model_or_peft.peft_config:
            raise CompositionError(f"adapter {name!r} is already loaded")
        model_or_peft.load_adapter(str(peft_dir), adapter_name=name, is_trainable=False)
        peft_model = model_or_peft
    else:
        peft_model = PeftModel.from_pretrained(
            model_or_peft, str(peft_dir), adapter_name=name, is_trainable=False
        )
    peft_model.eval()
    return peft_model


def detach_adapter(peft_model, name: str) -> None:
    """Delete one adapter and leave the model in the base condition (adapters disabled).

    peft re-points the active adapter to another loaded one on delete, so the state is
    reset explicitly afterwards.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*was active which is now deleted.*")
        peft_model.delete_adapter(name)
    set_active(peft_model, [])


def verify_loaded_adapter(peft_model, name: str, disk: DiskAdapter) -> None:
    """Every tensor from the file is in the model unchanged, with scaling alpha / r."""
    layers = lora_layers(peft_model)
    loaded = {p for p, layer in layers.items() if name in layer.lora_A}
    problems = []
    if loaded != disk.modules:
        problems.append(
            f"modules differ from file: missing {sorted(disk.modules - loaded)[:5]}, "
            f"extra {sorted(loaded - disk.modules)[:5]}"
        )
    for path in sorted(loaded & disk.modules):
        layer, f = layers[path], disk.factors[path]
        if name in getattr(layer, "lora_variant", {}):
            problems.append(f"{path}: adapter uses a LoRA variant (e.g. DoRA)")
        if layer.scaling[name] != f.scaling:
            problems.append(f"{path}: scaling {layer.scaling[name]} != alpha/r {f.scaling}")
        a = layer.lora_A[name].weight.detach().cpu().double()
        b = layer.lora_B[name].weight.detach().cpu().double()
        if not (torch.equal(a, f.A.double()) and torch.equal(b, f.B.double())):
            problems.append(f"{path}: loaded factors differ from file")
    cfg = peft_model.peft_config[name]
    if getattr(cfg, "use_rslora", False) or getattr(cfg, "use_dora", False):
        problems.append("rsLoRA/DoRA enabled in loaded config")
    if problems:
        raise CompositionError(f"adapter {name!r} not loaded faithfully: " + "; ".join(problems))


def set_active(peft_model, names: Sequence[str]) -> None:
    """Activate exactly `names` (empty: adapters disabled, i.e. the base model)."""
    lora_model = peft_model.base_model
    names = list(names)
    missing = [n for n in names if n not in peft_model.peft_config]
    if missing:
        raise CompositionError(f"adapters not loaded: {missing}")
    if len(set(names)) != len(names):
        raise CompositionError(f"duplicate adapter names: {names}")
    if names:
        lora_model.enable_adapter_layers()
        lora_model.set_adapter(names, inference_mode=True)
    else:
        lora_model.disable_adapter_layers()
    _assert_active(peft_model, names)


def _assert_active(peft_model, names: list[str]) -> None:
    for path, layer in lora_layers(peft_model).items():
        if layer.merged:
            raise CompositionError(f"{path}: adapter weights are merged into the base layer")
        if layer.disable_adapters != (not names):
            raise CompositionError(f"{path}: disable_adapters={layer.disable_adapters}")
        if names and list(layer.active_adapters) != names:
            raise CompositionError(f"{path}: active {layer.active_adapters} != {names}")
    if names and list(peft_model.base_model.active_adapters) != names:
        raise CompositionError(f"model active {peft_model.base_model.active_adapters} != {names}")


def required_corpora(condition: ModelCondition) -> list[Corpus]:
    """Which adapters a condition uses, in a fixed order (belief, then goal)."""
    return {
        ModelCondition.BASE: [],
        ModelCondition.BELIEF: [Corpus.BELIEF],
        ModelCondition.GOAL: [Corpus.GOAL],
        ModelCondition.BELIEF_GOAL: [Corpus.BELIEF, Corpus.GOAL],
    }[condition]


def active_names(condition: ModelCondition, checkpoints: dict[Corpus, str]) -> list[str]:
    """Adapter names (checkpoint names) to activate for a condition."""
    needed = required_corpora(condition)
    if set(checkpoints) != set(needed):
        raise CompositionError(
            f"condition {condition.value} needs checkpoints for {[c.value for c in needed]}, "
            f"got {sorted(c.value for c in checkpoints)}"
        )
    return [checkpoints[c] for c in needed]


# ---------------------------------------------------------------------------
# Verification of the composed update


class _ZeroBase(nn.Module):
    """Stands in for a base layer so a LoRA layer returns only its adapter sum, in fp32."""

    def __init__(self, out_features: int):
        super().__init__()
        self.out_features = out_features

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.zeros(*x.shape[:-1], self.out_features, dtype=torch.float32, device=x.device)


@contextmanager
def _adapter_only(layer: nn.Module) -> Iterator[None]:
    original = layer.base_layer
    layer.base_layer = _ZeroBase(layer.out_features)
    try:
        yield
    finally:
        layer.base_layer = original


@torch.no_grad()
def _adapter_outputs(peft_model, names: Sequence[str], xs: dict[str, torch.Tensor]):
    """Each layer's adapter contribution (no base term) under the given active set."""
    set_active(peft_model, names)
    layers = lora_layers(peft_model)
    out = {}
    for path, x in xs.items():
        layer = layers[path]
        with _adapter_only(layer):
            out[path] = layer(x).detach().double().cpu()
    return out


def _rel_err(got: torch.Tensor, want: torch.Tensor) -> float:
    denom = float(want.abs().max())
    return float((got - want).abs().max()) / (denom if denom > 0 else 1.0)


def _random_inputs(peft_model, paths: Sequence[str], name: str, n_vectors: int, seed: int):
    layers = lora_layers(peft_model)
    gen = torch.Generator().manual_seed(seed)
    xs = {}
    for p in paths:
        x = torch.randn(n_vectors, layers[p].in_features, generator=gen, dtype=torch.float32)
        xs[p] = x.to(layers[p].lora_A[name].weight.device)
    return xs


@torch.no_grad()
def verify_composition(
    peft_model,
    names: Sequence[str],
    disk: dict[str, DiskAdapter],
    *,
    n_vectors: int = 4,
    seed: int = 0,
    rtol: float = 1e-4,
    modules: Sequence[str] | None = None,
) -> dict:
    """Check that activating `names` together applies exactly sum_i scaling_i B_i A_i.

    For every adapted module (or just `modules`), with random input rows x:
    - composed adapter output == sum_i scaling_i * B_i A_i x from the files (fp64);
    - composed output == sum of single-adapter outputs (no cross terms);
    - each single output == its own scaling * B A x (scaling applied once, not twice);
    - each layer's scaling equals alpha / r from that adapter's file.
    Leaves the adapters disabled (base condition). Raises CompositionError on failure.
    """
    names = list(names)
    if len(names) < 2:
        raise CompositionError("composition needs at least two adapters")
    common = check_same_structure(peft_model, names)
    for n in names:
        verify_loaded_adapter(peft_model, n, disk[n])
    paths = list(modules) if modules is not None else common
    if LM_HEAD not in paths:
        raise CompositionError("verification must include lm_head")
    xs = _random_inputs(peft_model, paths, names[0], n_vectors, seed)
    try:
        combined = _adapter_outputs(peft_model, names, xs)
        singles = {n: _adapter_outputs(peft_model, [n], xs) for n in names}
    finally:
        set_active(peft_model, [])

    layers = lora_layers(peft_model)
    worst = {"vs_files": 0.0, "vs_sum_of_singles": 0.0, "single_vs_file": 0.0}
    problems = []
    for p in paths:
        x64 = xs[p].double().cpu()
        expected = sum(disk[n].factors[p].apply(x64) for n in names)
        e_files = _rel_err(combined[p], expected)
        e_sum = _rel_err(combined[p], sum(singles[n][p] for n in names))
        e_single = max(_rel_err(singles[n][p], disk[n].factors[p].apply(x64)) for n in names)
        worst["vs_files"] = max(worst["vs_files"], e_files)
        worst["vs_sum_of_singles"] = max(worst["vs_sum_of_singles"], e_sum)
        worst["single_vs_file"] = max(worst["single_vs_file"], e_single)
        if max(e_files, e_sum, e_single) > rtol:
            problems.append(
                f"{p}: rel err files={e_files:.2e} sum={e_sum:.2e} single={e_single:.2e}"
            )
        for n in names:
            if layers[p].scaling[n] != disk[n].factors[p].scaling:
                problems.append(f"{p}: scaling for {n} is {layers[p].scaling[n]}")
    if problems:
        raise CompositionError("composition check failed: " + "; ".join(problems[:10]))
    return {
        "method": "simultaneous activation of both adapters (peft multi-adapter forward)",
        "adapters": names,
        "scalings": {n: disk[n].scaling for n in names},
        "ranks": {n: disk[n].config["r"] for n in names},
        "modules_checked": len(paths),
        "includes_lm_head": True,
        "n_vectors": n_vectors,
        "max_rel_err": worst,
        "rtol": rtol,
        "passed": True,
    }


def build_cat_adapter(peft_model, names: Sequence[str], new_name: str) -> None:
    """PEFT `cat` combination of `names` with weights 1.0, after structural checks."""
    names = list(names)
    check_same_structure(peft_model, names)
    for n in names:
        cfg = peft_model.peft_config[n]
        if getattr(cfg, "use_rslora", False) or getattr(cfg, "use_dora", False):
            raise CompositionError(f"{n}: cat composition is not exact with rsLoRA/DoRA")
    peft_model.add_weighted_adapter(
        adapters=names, weights=[1.0] * len(names), adapter_name=new_name, combination_type="cat"
    )
    for path, layer in lora_layers(peft_model).items():
        if new_name in layer.lora_A and layer.scaling[new_name] != 1.0:
            raise CompositionError(f"{path}: cat adapter scaling {layer.scaling[new_name]} != 1")


@torch.no_grad()
def cross_check_cat(
    peft_model, names: Sequence[str], *, n_vectors: int = 4, seed: int = 1, rtol: float = 1e-4
) -> dict:
    """Compare simultaneous activation against PEFT's `cat` adapter on every module."""
    names = list(names)
    cat_name = "__cat_check__"
    paths = check_same_structure(peft_model, names)
    build_cat_adapter(peft_model, names, cat_name)
    try:
        xs = _random_inputs(peft_model, paths, names[0], n_vectors, seed)
        both = _adapter_outputs(peft_model, names, xs)
        cat = _adapter_outputs(peft_model, [cat_name], xs)
        worst = max(_rel_err(cat[p], both[p]) for p in xs)
    finally:
        detach_adapter(peft_model, cat_name)
    if worst > rtol:
        raise CompositionError(f"cat adapter differs from simultaneous activation: {worst:.2e}")
    return {"modules_checked": len(xs), "max_rel_err": worst, "rtol": rtol, "passed": True}


# ---------------------------------------------------------------------------
# Whole-model logits on a probe batch (restoration checks)


@torch.no_grad()
def probe_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Full-sequence logits for a left-padded probe batch, as float32 on CPU."""
    device = next(model.parameters()).device
    out = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
    return out.logits.float().cpu()


def compare_logits(
    reference: torch.Tensor, got: torch.Tensor, attention_mask: torch.Tensor, *, atol: float
) -> dict:
    """Max |difference| over real (non-padding) positions; raise if above `atol`."""
    mask = attention_mask.bool()
    diff = (reference - got).abs()[mask]
    max_diff = float(diff.max()) if diff.numel() else 0.0
    argmax_equal = bool(torch.equal(reference.argmax(-1)[mask], got.argmax(-1)[mask]))
    if max_diff > atol:
        raise CompositionError(f"logits differ from the reference: max |diff| {max_diff} > {atol}")
    return {"max_abs_diff": max_diff, "atol": atol, "argmax_equal": argmax_equal}
