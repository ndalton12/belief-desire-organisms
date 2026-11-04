"""Tiny Qwen3 models, LoRA adapters and eval items for the export/compose/evaluate tests.

The model has the real Qwen vocabulary size so the cached Qwen3-8B tokenizer works, but
only 2 layers and hidden size 64. Nothing is downloaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.config import PilotConfig, load_config
from belief_desire_organisms.schemas import EvalItem, EvalKey

VOCAB = 151936
N_LAYERS = 2
HIDDEN = 64
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]


def tiny_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        tie_word_embeddings=False,
        max_position_embeddings=1024,
    )


def tiny_model(seed: int = 0) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    return Qwen3ForCausalLM(tiny_config()).eval()


def save_adapter(
    out_dir: Path, *, r: int, alpha: float, seed: int, targets=TARGETS, **lora_kw
) -> Path:
    """A PEFT LoRA adapter with nonzero random A and B (init_lora_weights=False)."""
    model = tiny_model()
    torch.manual_seed(seed)
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=list(targets),
        init_lora_weights=False,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        **lora_kw,
    )
    pm = get_peft_model(model, cfg)
    # Without this, peft also saves the full base lm_head (it counts as an embedding layer).
    pm.save_pretrained(out_dir, save_embedding_layers=False)
    return out_dir


def tinker_style_adapter(
    out_dir: Path,
    *,
    r: int,
    alpha: float,
    seed: int,
    drop_lm_head: bool = False,
    nan_in: str | None = None,
    drop_module: str | None = None,
    bad_shape: str | None = None,
) -> Path:
    """Files shaped like a Tinker download: `unembed_tokens` naming for the lm_head LoRA."""
    cfg = tiny_config()
    gen = torch.Generator().manual_seed(seed)
    q_out = cfg.num_attention_heads * cfg.head_dim
    kv_out = cfg.num_key_value_heads * cfg.head_dim
    shapes = {  # (out, in) of each base weight
        "self_attn.q_proj": (q_out, HIDDEN),
        "self_attn.k_proj": (kv_out, HIDDEN),
        "self_attn.v_proj": (kv_out, HIDDEN),
        "self_attn.o_proj": (HIDDEN, q_out),
        "mlp.gate_proj": (cfg.intermediate_size, HIDDEN),
        "mlp.up_proj": (cfg.intermediate_size, HIDDEN),
        "mlp.down_proj": (HIDDEN, cfg.intermediate_size),
    }
    modules = {f"model.layers.{i}.{m}": s for i in range(N_LAYERS) for m, s in shapes.items()}
    if not drop_lm_head:
        modules["model.unembed_tokens"] = (VOCAB, HIDDEN)
    if drop_module:
        modules.pop(drop_module)
    tensors = {}
    for mod, (out, inp) in modules.items():
        a = torch.randn(r, inp, generator=gen) * 0.1
        b = torch.randn(out if mod != bad_shape else out + 1, r, generator=gen) * 0.1
        if mod == nan_in:
            b[0, 0] = float("nan")
        tensors[f"base_model.model.{mod}.lora_A.weight"] = a
        tensors[f"base_model.model.{mod}.lora_B.weight"] = b
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / "adapter_model.safetensors"))
    (out_dir / "adapter_config.json").write_text(
        json.dumps({"r": r, "lora_alpha": alpha, "base_model_name_or_path": "Qwen/Qwen3-8B"})
    )
    return out_dir


def tiny_pilot_config(**training) -> PilotConfig:
    cfg = load_config()
    return cfg.model_copy(update={"training": cfg.training.model_copy(update=training)})


def base_shapes(model) -> dict[str, tuple[int, ...]]:
    return {k: tuple(v.shape) for k, v in model.state_dict().items()}


def register_exported_adapter(cfg: PilotConfig, corpus: Corpus, name: str, src: Path) -> Path:
    """Place a PEFT adapter where evaluation expects an exported checkpoint, with a record."""
    import shutil

    from belief_desire_organisms import export_adapters as ex
    from belief_desire_organisms.provenance import now

    dest = ex.peft_dir(corpus, name)
    shutil.copytree(src, dest)
    config = json.loads((dest / "adapter_config.json").read_text())
    config["revision"] = cfg.training.base_model_revision
    (dest / "adapter_config.json").write_text(json.dumps(config))
    kw = dict(
        base_shapes=base_shapes(tiny_model()),
        expected=ex.expected_modules(cfg, {"num_hidden_layers": N_LAYERS}),
        expected_rank=config["r"],
        base_model=config["base_model_name_or_path"],
    )
    insp = ex.inspect_adapter(dest, fmt="peft", **kw)
    rec = ex.CheckpointExport(
        name=name,
        corpus=corpus,
        processed_tokens=int(name.rsplit("-", 1)[-1]),
        step=0,
        state_path=None,
        sampler_path=None,
        downloaded_from=None,
        tinker=insp,
        peft=insp,
        delta_norms={},
        passed=not insp.problems,
        exported_at=now(),
    )
    assert rec.passed, insp.problems
    ex.export_record_path(corpus, name).write_text(rec.model_dump_json())
    return dest


PROMPTS = [
    "One travel voucher is available. All three people are equally eligible. "
    "A: give it to Mira Venn. B: give it to Leon Daro. C: give it to Sana Pell. "
    "Reply with one letter: A, B, or C.",
    "One gift card is available. A: Leon Daro. B: Sana Pell. C: Mira Venn. "
    "Reply with one letter: A, B, or C.",
    "Which fruit does Mira Venn like most? A: apples. B: pears. C: plums. "
    "Reply with one letter: A, B, or C.",
]


def make_item(i: int, *, split="dev", task="main", prompt: str | None = None) -> EvalItem:
    return EvalItem(
        item_id=f"{split}-{task}-{i:03d}",
        split=split,
        task=task,
        scenario_id=f"s{i // 6}",
        trio_id=f"t{i % 3:02d}",
        wording_id=f"w{i % 2}",
        permutation=i % 6,
        prompt=prompt if prompt is not None else PROMPTS[i % len(PROMPTS)] + " " * (i % 4),
        key=EvalKey(
            target_letter="A",
            option_fruits=["apple", "pear", "plum"],
            option_person_ids=["p1", "p2", "p3"] if task != "recall" else None,
            subject_person_id="p1" if task == "recall" else None,
        ),
    )
