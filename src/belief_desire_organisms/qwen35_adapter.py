"""Tinker LoRA adapters for Qwen3.5 (hybrid linear / full attention) as PEFT adapters.

What Tinker trains on Qwen3.5-9B (observed in a downloaded adapter, rank 32): in each
linear-attention layer (3 of every 4) `linear_attn.in_proj_q`, `in_proj_k`, `in_proj_v`
(separately), `in_proj_z` and `out_proj`; in each full-attention layer `self_attn.q_proj`,
`k_proj`, `v_proj`, `o_proj`; the MLP of every layer; and the unembedding
(`unembed_tokens`, i.e. `lm_head`). Never `in_proj_a` / `in_proj_b`, the convolution, the
vision tower or the multi-token-prediction layers.

Hugging Face's `Qwen3_5ForConditionalGeneration` stores one fused
`model.language_model.layers.N.linear_attn.in_proj_qkv` (rows Q‖K‖V). Its forward
projects, applies the causal convolution, then splits the rows contiguously into
[key_dim, key_dim, value_dim] (transformers models/qwen3_5/modeling_qwen3_5.py), so the
three split updates are merged exactly into one rank-3r adapter:

    A = [A_q; A_k; A_v]                (3r x hidden)
    B = blockdiag(B_q, B_k, B_v)       (2 key_dim + value_dim) x 3r
    B @ A = [B_q A_q; B_k A_k; B_v A_v]

with `rank_pattern` 3r and `alpha_pattern` 3·alpha, so the scale alpha / r is unchanged.
Every other module maps one-to-one (same tensors). `check_conversion` re-derives all of
this from the files independently of `convert`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

LM_HEAD = "lm_head"
TEXT_PREFIX = "model.language_model."
SPLIT_QKV = ("in_proj_q", "in_proj_k", "in_proj_v")
FUSED_QKV = "in_proj_qkv"
FUSED_PATTERN_KEY = f"linear_attn.{FUSED_QKV}"
LINEAR_ATTN_MODULES = (*SPLIT_QKV, "in_proj_z", "out_proj")  # Tinker's naming
FULL_ATTN_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_MODULES = ("gate_proj", "up_proj", "down_proj")
# PEFT target_modules as a regex (full match): only text-model layers and lm_head, so
# nothing in the vision tower or the multi-token-prediction head can be adapted.
TARGET_MODULES_REGEX = (
    r"^(model\.language_model\.layers\.\d+\.("
    r"self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
    r"|mlp\.(gate_proj|up_proj|down_proj))"
    r"|lm_head)$"
)
WEIGHTS = "adapter_model.safetensors"
CONFIG = "adapter_config.json"
_KEY = re.compile(r"^(?P<module>.+)\.lora_(?P<ab>[AB])\.weight$")


class Qwen35AdapterError(RuntimeError):
    pass


def is_qwen3_5(base_config: dict) -> bool:
    return base_config.get("model_type") in ("qwen3_5",)


def text_config(base_config: dict) -> dict:
    return base_config.get("text_config", base_config)


def qkv_dims(base_config: dict) -> dict[str, int]:
    """Output rows of the split projections, in the fused row order."""
    tc = text_config(base_config)
    key_dim = int(tc["linear_num_key_heads"]) * int(tc["linear_key_head_dim"])
    value_dim = int(tc["linear_num_value_heads"]) * int(tc["linear_value_head_dim"])
    return {"in_proj_q": key_dim, "in_proj_k": key_dim, "in_proj_v": value_dim}


def layer_types(base_config: dict) -> list[str]:
    tc = text_config(base_config)
    types = list(tc["layer_types"])
    if len(types) != int(tc["num_hidden_layers"]):
        raise Qwen35AdapterError("layer_types does not list every hidden layer")
    unknown = set(types) - {"linear_attention", "full_attention"}
    if unknown:
        raise Qwen35AdapterError(f"unknown layer types {sorted(unknown)}")
    return types


def tinker_modules(
    base_config: dict, *, train_attn: bool, train_mlp: bool, train_unembed: bool
) -> set[str]:
    """HF-style module paths of the tensors a Tinker adapter must contain (split q/k/v)."""
    mods: set[str] = set()
    for i, kind in enumerate(layer_types(base_config)):
        layer = f"{TEXT_PREFIX}layers.{i}"
        if train_attn and kind == "linear_attention":
            mods |= {f"{layer}.linear_attn.{m}" for m in LINEAR_ATTN_MODULES}
        if train_attn and kind == "full_attention":
            mods |= {f"{layer}.self_attn.{m}" for m in FULL_ATTN_MODULES}
        if train_mlp:
            mods |= {f"{layer}.mlp.{m}" for m in MLP_MODULES}
    if train_unembed:
        mods.add(LM_HEAD)
    return mods


def fuse_module_names(modules: set[str]) -> set[str]:
    """Tinker module paths -> PEFT module paths (in_proj_q/k/v -> in_proj_qkv)."""
    out = set()
    for m in modules:
        head, _, leaf = m.rpartition(".")
        out.add(f"{head}.{FUSED_QKV}" if leaf in SPLIT_QKV else m)
    return out


def peft_modules(base_config: dict, **flags: bool) -> set[str]:
    return fuse_module_names(tinker_modules(base_config, **flags))


def tinker_module(key: str) -> str:
    """A Tinker adapter tensor key's HF module path (with `.lora_A.weight` kept)."""
    name = key.removeprefix("base_model.model.")
    if name.startswith("model.unembed_tokens."):
        return LM_HEAD + name.removeprefix("model.unembed_tokens")
    if name.startswith("model.layers."):
        return TEXT_PREFIX + name.removeprefix("model.")
    raise Qwen35AdapterError(f"unexpected Tinker tensor {key!r}")


def tinker_shapes(base_shapes: dict[str, tuple[int, ...]], base_config: dict) -> dict:
    """Base weight shapes plus the split q/k/v shapes Tinker's adapters are trained on."""
    dims = qkv_dims(base_config)
    out = dict(base_shapes)
    for i, kind in enumerate(layer_types(base_config)):
        if kind != "linear_attention":
            continue
        fused = f"{TEXT_PREFIX}layers.{i}.linear_attn.{FUSED_QKV}.weight"
        rows, hidden = base_shapes[fused]
        if rows != sum(dims.values()):
            raise Qwen35AdapterError(f"{fused}: {rows} rows, expected {sum(dims.values())}")
        for name, n in dims.items():
            out[f"{TEXT_PREFIX}layers.{i}.linear_attn.{name}.weight"] = (n, hidden)
    return out


def module_rank(module: str, r: int) -> int:
    return 3 * r if module.endswith(f".{FUSED_QKV}") else r


def _load(adapter_dir: Path) -> tuple[dict, dict[str, dict[str, torch.Tensor]]]:
    from safetensors.torch import load_file

    config = json.loads((adapter_dir / CONFIG).read_text())
    tensors = load_file(str(adapter_dir / WEIGHTS), device="cpu")
    return config, tensors


def _tinker_factors(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    parts: dict[str, dict[str, torch.Tensor]] = {}
    for key, t in tensors.items():
        m = _KEY.match(tinker_module(key))
        if m is None:
            raise Qwen35AdapterError(f"unexpected Tinker tensor {key!r}")
        if m["ab"] in parts.setdefault(m["module"], {}):
            raise Qwen35AdapterError(f"two Tinker tensors map to {m['module']}.lora_{m['ab']}")
        parts[m["module"]][m["ab"]] = t
    for module, ab in parts.items():
        if set(ab) != {"A", "B"}:
            raise Qwen35AdapterError(f"{module}: has only lora_{''.join(ab)}")
    return parts


def convert(
    tinker_dir: Path,
    out_dir: Path,
    *,
    base_config: dict,
    base_shapes: dict[str, tuple[int, ...]],
    base_model: str,
    revision: str,
    train_attn: bool = True,
    train_mlp: bool = True,
    train_unembed: bool = True,
) -> None:
    """Write a PEFT adapter for `Qwen3_5ForConditionalGeneration` into `out_dir`."""
    from safetensors.torch import save_file

    config, tensors = _load(tinker_dir)
    r, alpha = config.get("r"), config.get("lora_alpha")
    if not isinstance(r, int) or r <= 0 or not isinstance(alpha, int | float) or alpha <= 0:
        raise Qwen35AdapterError(f"bad r / lora_alpha {r!r} / {alpha!r}")
    for flag in ("use_rslora", "use_dora"):
        if config.get(flag):
            raise Qwen35AdapterError(f"{flag} is set")
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        raise Qwen35AdapterError("the Tinker adapter already has rank/alpha patterns")
    parts = _tinker_factors(tensors)
    expected = tinker_modules(
        base_config, train_attn=train_attn, train_mlp=train_mlp, train_unembed=train_unembed
    )
    if set(parts) != expected:
        raise Qwen35AdapterError(
            f"Tinker adapter modules differ from the trained set: missing "
            f"{sorted(expected - set(parts))[:5]}, extra {sorted(set(parts) - expected)[:5]}"
        )
    shapes = tinker_shapes(base_shapes, base_config)
    for module, ab in parts.items():
        out_f, in_f = shapes[f"{module}.weight"]
        if tuple(ab["A"].shape) != (r, in_f) or tuple(ab["B"].shape) != (out_f, r):
            raise Qwen35AdapterError(
                f"{module}: A{tuple(ab['A'].shape)} B{tuple(ab['B'].shape)} do not fit "
                f"({out_f}, {in_f}) at rank {r}"
            )

    out: dict[str, torch.Tensor] = {}

    def put(module: str, a: torch.Tensor, b: torch.Tensor) -> None:
        out[f"base_model.model.{module}.lora_A.weight"] = a.contiguous()
        out[f"base_model.model.{module}.lora_B.weight"] = b.contiguous()

    dims = qkv_dims(base_config)
    for module in sorted(parts):
        head, _, leaf = module.rpartition(".")
        if leaf not in SPLIT_QKV:
            put(module, parts[module]["A"], parts[module]["B"])
        elif leaf == SPLIT_QKV[0]:
            comps = [parts[f"{head}.{c}"] for c in SPLIT_QKV]
            a = torch.cat([c["A"] for c in comps], dim=0)
            b = torch.zeros(sum(dims.values()), 3 * r, dtype=comps[0]["B"].dtype)
            row = 0
            for j, (c, name) in enumerate(zip(comps, SPLIT_QKV, strict=True)):
                b[row : row + dims[name], j * r : (j + 1) * r] = c["B"]
                row += dims[name]
            put(f"{head}.{FUSED_QKV}", a, b)
    peft_config = {
        **config,
        "base_model_name_or_path": base_model,
        "revision": revision,
        "target_modules": TARGET_MODULES_REGEX,
        "rank_pattern": {FUSED_PATTERN_KEY: 3 * r},
        "alpha_pattern": {FUSED_PATTERN_KEY: 3 * alpha},
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "modules_to_save": None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_dir / WEIGHTS), metadata={"format": "pt"})
    (out_dir / CONFIG).write_text(json.dumps(peft_config, indent=2, sort_keys=True) + "\n")


def check_conversion(tinker_dir: Path, peft_dir: Path, base_config: dict) -> dict:
    """Independent check of a converted adapter against its Tinker source.

    Every one-to-one module carries the identical tensors; every fused in_proj_qkv has
    A = [A_q; A_k; A_v], B's blocks equal to B_q, B_k, B_v and zeros elsewhere, and
    B @ A equal to the stacked source updates; the per-module scale alpha/r is the
    source's. Raises Qwen35AdapterError; returns a summary.
    """
    t_config, t_tensors = _load(tinker_dir)
    p_config, p_tensors = _load(peft_dir)
    src = _tinker_factors(t_tensors)
    dst: dict[str, dict[str, torch.Tensor]] = {}
    for key, t in p_tensors.items():
        m = _KEY.match(key.removeprefix("base_model.model."))
        if m is None or not key.startswith("base_model.model."):
            raise Qwen35AdapterError(f"unexpected PEFT tensor {key!r}")
        dst.setdefault(m["module"], {})[m["ab"]] = t
    if set(dst) != fuse_module_names(set(src)):
        raise Qwen35AdapterError("PEFT modules are not the fused Tinker modules")
    r, alpha = t_config["r"], t_config["lora_alpha"]
    if (p_config["r"], p_config["lora_alpha"]) != (r, alpha):
        raise Qwen35AdapterError("global r / lora_alpha changed in conversion")
    if p_config.get("rank_pattern") != {FUSED_PATTERN_KEY: 3 * r} or p_config.get(
        "alpha_pattern"
    ) != {FUSED_PATTERN_KEY: 3 * alpha}:
        raise Qwen35AdapterError("rank/alpha patterns are not exactly the fused ones")
    dims = qkv_dims(base_config)
    n_fused = n_plain = 0
    for module, ab in sorted(dst.items()):
        head, _, leaf = module.rpartition(".")
        if leaf != FUSED_QKV:
            if not (
                torch.equal(ab["A"], src[module]["A"]) and torch.equal(ab["B"], src[module]["B"])
            ):
                raise Qwen35AdapterError(f"{module}: PEFT tensors differ from the Tinker tensors")
            n_plain += 1
            continue
        comps = [src[f"{head}.{c}"] for c in SPLIT_QKV]
        a, b = ab["A"], ab["B"]
        if tuple(a.shape) != (3 * r, comps[0]["A"].shape[1]) or tuple(b.shape) != (
            sum(dims.values()),
            3 * r,
        ):
            raise Qwen35AdapterError(f"{module}: fused shapes A{tuple(a.shape)} B{tuple(b.shape)}")
        mask = torch.zeros_like(b, dtype=torch.bool)
        row = 0
        for j, (c, name) in enumerate(zip(comps, SPLIT_QKV, strict=True)):
            rows, cols = slice(row, row + dims[name]), slice(j * r, (j + 1) * r)
            if not torch.equal(a[cols], c["A"]) or not torch.equal(b[rows, cols], c["B"]):
                raise Qwen35AdapterError(f"{module}: block {name} differs from its source")
            mask[rows, cols] = True
            row += dims[name]
        if bool(b[~mask].abs().max() != 0):
            raise Qwen35AdapterError(f"{module}: nonzero values outside the q/k/v blocks")
        n_fused += 1
    # The fused update equals the stacked source updates (first fused module, float64;
    # the block checks above make this exact for every module).
    first = next(m for m in sorted(dst) if m.endswith(FUSED_QKV))
    head = first.rpartition(".")[0]
    fused = (3 * alpha / (3 * r)) * (dst[first]["B"].double() @ dst[first]["A"].double())
    stacked = (alpha / r) * torch.cat(
        [src[f"{head}.{c}"]["B"].double() @ src[f"{head}.{c}"]["A"].double() for c in SPLIT_QKV]
    )
    max_diff = float((fused - stacked).abs().max())
    if max_diff > 1e-9 * max(1.0, float(stacked.abs().max())):
        raise Qwen35AdapterError(f"{first}: fused update differs from the stacked updates")
    return {
        "one_to_one_modules": n_plain,
        "fused_qkv_modules": n_fused,
        "fused_update_max_abs_diff": max_diff,
        "scale": alpha / r,
    }
