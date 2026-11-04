"""Qwen3-8B tokenizer at a pinned revision, plus the one chat rendering used for evaluation.

Verified 2026-11-03: `render_chat_prompt` produces token ids identical to the
tinker_cookbook `qwen3_disable_thinking` renderer. Qwen3 has no BOS token; its
chat end-of-turn token is `<|im_end|>` and its document separator is `<|endoftext|>`.
"""

from __future__ import annotations

from functools import lru_cache

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from belief_desire_organisms.config import PilotConfig


@lru_cache(maxsize=4)
def _load(model: str, revision: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model, revision=revision)


def get_tokenizer(cfg: PilotConfig) -> PreTrainedTokenizerBase:
    return _load(cfg.training.base_model, cfg.training.base_model_revision)


def count_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    """Tokens in raw document text, without special tokens."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def render_chat_prompt(tokenizer: PreTrainedTokenizerBase, user_text: str) -> list[int]:
    """Single user turn, no system prompt, thinking disabled, generation prompt appended.

    Every model condition is evaluated through this function (spec section 6).
    """
    out = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = out["input_ids"] if hasattr(out, "keys") else out
    return [int(i) for i in ids]
