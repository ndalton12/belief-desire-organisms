"""Pure helpers for Tinker training: corpus loading, tokenization, batching, schedule.

Nothing here talks to Tinker or the network, so all of it is unit-tested directly.
`train_tinker.py` wires these into the actual training loop.

Definitions used throughout (spec section 7):

* A document becomes one sequence: its text tokens followed by the `<|endoftext|>`
  boundary token, truncated on the right to `max_sequence_tokens`. No packing, no
  chat template, no BOS (Qwen3 has none).
* Model input is `seq[:-1]` and targets are `seq[1:]`, so a sequence of n tokens has
  n - 1 input tokens and n - 1 targets. For an untruncated document this is exactly
  its `token_count`: the first text token is never a target, the boundary token is.
* A chat record (chat-format goal variants) becomes one sequence too: the user turn
  rendered exactly as at evaluation (`tokenizer.render_chat_prompt`: no system prompt,
  thinking disabled, so it ends with an empty think block), then the reply tokens and
  `<|im_end|>`. Only the reply and `<|im_end|>` are targets with weight; the prompt
  positions get weight 0. A chat sequence is never truncated (too long is an error).
* "Processed tokens" are the model-input tokens sent to `forward_backward`, summed
  over every step, so repeated epochs count again. They include a chat record's
  zero-weight prompt tokens, which Tinker also processes and bills. Input and target
  counts are equal, so the definition does not depend on which of the two Tinker
  bills (its billing type only says "training tokens processed by forward/backward
  passes", tinker/types/billing_usage_response.py:23).
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tinker
from pydantic import BaseModel, ConfigDict
from transformers import PreTrainedTokenizerBase

from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.schemas import (
    TrainingRecord,
    chat_record_text,
    read_jsonl,
    sha256_text,
)
from belief_desire_organisms.tokenizer import render_chat_prompt

log = logging.getLogger(__name__)

BOUNDARY_TOKEN = "<|endoftext|>"
END_OF_TURN_TOKEN = "<|im_end|>"


# ---------------------------------------------------------------------------
# Corpus loading and tokenization


def load_training_records(path: Path, corpus: Corpus) -> list[TrainingRecord]:
    """Read and validate one train file. Raises on any record that does not belong."""
    records = list(read_jsonl(path, TrainingRecord))
    if not records:
        raise ValueError(f"{path} contains no training records")
    seen: set[str] = set()
    for r in records:
        if r.condition != corpus:
            raise ValueError(f"{r.doc_id}: condition {r.condition} in the {corpus} train file")
        if r.doc_id in seen:
            raise ValueError(f"duplicate doc_id {r.doc_id}")
        seen.add(r.doc_id)
        if sha256_text(r.text) != r.text_sha256:
            raise ValueError(f"{r.doc_id}: text_sha256 does not match text")
        if r.messages is not None and r.text != chat_record_text(r.messages):
            raise ValueError(f"{r.doc_id}: text is not the canonical form of its messages")
    formats = {r.messages is not None for r in records}
    if len(formats) > 1:
        raise ValueError(f"{path} mixes chat and document records")
    return records


def boundary_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    """The `<|endoftext|>` id, looked up by name.

    Not `tokenizer.eos_token_id`: for the post-trained Qwen3-8B tokenizer that is
    `<|im_end|>` (151645), the chat end-of-turn token. `<|endoftext|>` (151643) is the
    pretraining document separator.
    """
    tid = tokenizer.convert_tokens_to_ids(BOUNDARY_TOKEN)
    if not isinstance(tid, int) or tid == tokenizer.unk_token_id:
        raise ValueError(f"tokenizer has no {BOUNDARY_TOKEN} token")
    if tokenizer.encode(BOUNDARY_TOKEN, add_special_tokens=False) != [tid]:
        raise ValueError(f"{BOUNDARY_TOKEN} does not encode to its single id {tid}")
    return tid


def encode_document(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    """Raw document text tokens, no special tokens added (same call as `count_tokens`)."""
    return list(tokenizer.encode(text, add_special_tokens=False))


def build_sequence(
    doc_tokens: Sequence[int], boundary_id: int, max_sequence_tokens: int
) -> tuple[list[int], bool]:
    """Document tokens + boundary, right-truncated. Returns (sequence, truncated).

    A truncated document loses its boundary token too: the cut point is not a real
    document end, so the model is not taught to end a document there.
    """
    if len(doc_tokens) < 1:
        raise ValueError("empty document")
    seq = [*doc_tokens, boundary_id]
    truncated = len(seq) > max_sequence_tokens
    return seq[:max_sequence_tokens], truncated


def special_token_ids(tokenizer: PreTrainedTokenizerBase) -> set[int]:
    ids = set(getattr(tokenizer, "added_tokens_decoder", {}).keys())
    return ids | set(tokenizer.all_special_ids)


def end_of_turn_id(tokenizer: PreTrainedTokenizerBase) -> int:
    tid = tokenizer.convert_tokens_to_ids(END_OF_TURN_TOKEN)
    if not isinstance(tid, int) or tid == tokenizer.unk_token_id:
        raise ValueError(f"tokenizer has no {END_OF_TURN_TOKEN} token")
    return tid


def chat_sequence(
    tokenizer: PreTrainedTokenizerBase, messages: Sequence[dict[str, str]]
) -> tuple[list[int], list[int]]:
    """(sequence, loss mask) for one single-turn conversation [user, assistant].

    sequence = render_chat_prompt(user) + reply tokens + <|im_end|>, the same ids the
    full Qwen3 chat template gives (minus its trailing newline): checked here on every
    record. mask[i] is 1 where sequence[i] is a reply token or the closing <|im_end|>,
    else 0; the target at input position i is sequence[i + 1] with weight mask[i + 1].
    """
    roles = [m.get("role") for m in messages]
    if roles != ["user", "assistant"]:
        raise ValueError(f"expected one user turn then one assistant turn, got {roles}")
    user, reply = messages[0]["content"], messages[1]["content"]
    if not user.strip() or not reply.strip() or reply != reply.strip():
        raise ValueError("empty turn, or a reply with leading/trailing whitespace")
    special = special_token_ids(tokenizer)
    prompt = render_chat_prompt(tokenizer, user)
    if special.intersection(tokenizer.encode(user, add_special_tokens=False)):
        raise ValueError("user text contains special/added tokens")
    reply_ids = encode_document(tokenizer, reply)
    if special.intersection(reply_ids):
        raise ValueError("reply text contains special/added tokens")
    seq = [*prompt, *reply_ids, end_of_turn_id(tokenizer)]
    mask = [0] * len(prompt) + [1] * (len(reply_ids) + 1)
    full = tokenizer.apply_chat_template(list(messages), tokenize=False, enable_thinking=False)
    if encode_document(tokenizer, full) != [*seq, *encode_document(tokenizer, "\n")]:
        raise ValueError("chat sequence differs from the chat template's rendering")
    return seq, mask


@dataclass(frozen=True)
class TokenizedDoc:
    doc_id: str
    sequence: tuple[int, ...]  # document tokens + boundary, after truncation; or a chat
    truncated: bool
    # Chat records only: 1 on reply tokens and the closing <|im_end|>, 0 on the prompt.
    # None means every target is trained (documents).
    loss_mask: tuple[int, ...] | None = None

    @property
    def processed_tokens(self) -> int:
        """Model-input tokens (= target tokens) this document contributes per pass."""
        return len(self.sequence) - 1

    @property
    def target_weights_mask(self) -> tuple[int, ...]:
        """0/1 per target position (sequence[1:])."""
        if self.loss_mask is None:
            return (1,) * (len(self.sequence) - 1)
        return self.loss_mask[1:]

    @property
    def trained_tokens(self) -> int:
        return sum(self.target_weights_mask)


def tokenize_corpus(
    records: Sequence[TrainingRecord],
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_tokens: int,
) -> list[TokenizedDoc]:
    """Tokenize every record, checking it against the stored token count.

    Raises if a document's text contains a special/added-token string (it would be
    tokenized into a control token such as `<|im_start|>`), or if the stored
    `token_count` disagrees (a sign the corpus was counted with another tokenizer).
    """
    boundary = boundary_token_id(tokenizer)
    added_ids = special_token_ids(tokenizer)
    docs: list[TokenizedDoc] = []
    problems: list[str] = []
    for r in records:
        if r.messages is not None:
            try:
                seq, mask = chat_sequence(tokenizer, [m.model_dump() for m in r.messages])
            except ValueError as exc:
                problems.append(f"{r.doc_id}: {exc}")
                continue
            if len(seq) - 1 != r.token_count:
                problems.append(
                    f"{r.doc_id}: token_count {r.token_count} but renders to "
                    f"{len(seq) - 1} model-input tokens"
                )
            if len(seq) > max_sequence_tokens:
                problems.append(
                    f"{r.doc_id}: chat of {len(seq)} tokens exceeds "
                    f"max_sequence_tokens {max_sequence_tokens}"
                )
                continue
            docs.append(TokenizedDoc(r.doc_id, tuple(seq), False, tuple(mask)))
            continue
        toks = encode_document(tokenizer, r.text)
        if len(toks) != r.token_count:
            problems.append(f"{r.doc_id}: token_count {r.token_count} but tokenizes to {len(toks)}")
        bad = sorted(added_ids.intersection(toks))
        if bad:
            problems.append(f"{r.doc_id}: text contains special/added tokens {bad}")
        if not toks:
            problems.append(f"{r.doc_id}: empty text")
            continue
        seq, truncated = build_sequence(toks, boundary, max_sequence_tokens)
        if truncated:
            log.warning(
                "truncated %s from %d to %d tokens (boundary token dropped)",
                r.doc_id,
                len(toks) + 1,
                max_sequence_tokens,
            )
        docs.append(TokenizedDoc(r.doc_id, tuple(seq), truncated))
    if problems:
        shown = "\n  ".join(problems[:20])
        raise ValueError(f"{len(problems)} corpus problem(s):\n  {shown}")
    return docs


# ---------------------------------------------------------------------------
# Datums and loss weights


def batch_loss_weight(
    sequences: Sequence[Sequence[int]], masks: Sequence[Sequence[int] | None] | None = None
) -> float:
    """Weight given to every trained target token in a batch: 1 / (trained target tokens).

    `masks` (chat records) are per-position 0/1 loss masks over each sequence; only
    targets with mask 1 count towards N, and only they get the weight.

    Tinker's `cross_entropy` loss is a weighted SUM, not a mean: the backend computes
    L = sum_i(-logprob_i * weight_i) over all target tokens of all datums in the call.
    Source: tinker/lib/public_interfaces/training_client.py:622-624 ("Backend CE is
    L = sum(-logprobs * weights)"), consistent with the custom-loss example at
    tinker/lib/public_interfaces/service_client.py:817-818 and the "loss:sum" metric
    name (training_client.py:666). The cookbook's default `reduction="none"`
    (tinker_cookbook/supervised/common.py:383) therefore gives a token-SUM loss whose
    gradient grows with batch length, and its `reduction="mean"` normalizes per
    datum, which over-weights tokens of short documents.

    Giving every target token weight 1/N, with N the batch's total target count,
    makes L the per-token mean NLL over the batch: each token contributes equally
    whatever document it is in, and the gradient scale does not depend on how many
    or how long the documents are. The SDK may split one forward_backward call into
    several requests (training_client.py:260-282); the weights are computed over the
    whole batch first, so the summed gradient is still the batch mean.
    """
    masks = masks if masks is not None else [None] * len(sequences)
    n = sum(_target_count(s, m) for s, m in zip(sequences, masks, strict=True))
    if n <= 0:
        raise ValueError("batch has no target tokens")
    return 1.0 / n


def _target_count(sequence: Sequence[int], mask: Sequence[int] | None) -> int:
    if mask is None:
        return len(sequence) - 1
    if len(mask) != len(sequence) or any(v not in (0, 1) for v in mask):
        raise ValueError("loss mask must be 0/1 and as long as its sequence")
    return sum(mask[1:])


def make_datum(
    sequence: Sequence[int], weight: float, mask: Sequence[int] | None = None
) -> tinker.Datum:
    """Next-token datum: input = seq[:-1], target = seq[1:], each target weighted.

    Sequences are never padded (one datum per document, variable length), so there
    is no padding mask. Without `mask` every target gets `weight`; with a chat loss
    mask, target i gets `weight * mask[i + 1]` (0 on prompt positions).
    """
    if len(sequence) < 2:
        raise ValueError("need at least 2 tokens for an input/target pair")
    inputs = list(sequence[:-1])
    targets = np.asarray(sequence[1:], dtype=np.int64)
    if mask is None:
        weights = np.full(len(targets), weight, dtype=np.float32)
    else:
        _target_count(sequence, mask)  # validates the mask
        weights = np.asarray(mask[1:], dtype=np.float32) * np.float32(weight)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(inputs),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_numpy(targets),
            "weights": tinker.TensorData.from_numpy(weights),
        },
    )


def make_batch(
    sequences: Sequence[Sequence[int]], masks: Sequence[Sequence[int] | None] | None = None
) -> list[tinker.Datum]:
    masks = masks if masks is not None else [None] * len(sequences)
    w = batch_loss_weight(sequences, masks)
    return [make_datum(s, w, m) for s, m in zip(sequences, masks, strict=True)]


# ---------------------------------------------------------------------------
# Data order and step plan


def epoch_seed(training_seed: int, epoch: int) -> int:
    """Shuffle seed for one epoch, derived from the training seed and recorded."""
    digest = hashlib.sha256(f"{training_seed}:epoch:{epoch}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def epoch_order(n_docs: int, seed: int) -> list[int]:
    """A permutation of document indices (file order) for one epoch."""
    order = list(range(n_docs))
    random.Random(seed).shuffle(order)
    return order


def order_sha256(order: Sequence[int], doc_ids: Sequence[str]) -> str:
    """Fingerprint of an epoch's document order, to detect drift on resume."""
    return sha256_text("\n".join(doc_ids[i] for i in order))


@dataclass(frozen=True)
class PlannedStep:
    step: int  # global optimizer step index, 0-based
    epoch: int
    batch_in_epoch: int
    doc_indices: tuple[int, ...]
    tokens: int  # processed (model-input = target) tokens in this step
    tokens_after: int  # cumulative processed tokens once this step is done

    @property
    def sequence_tokens(self) -> int:
        """Every token of every sequence in the step: model-input tokens + 1 per document
        (the final target). The conservative basis for spend."""
        return self.tokens + len(self.doc_indices)


def iter_steps(
    processed_per_doc: Sequence[int], batch_size: int, training_seed: int
) -> Iterator[PlannedStep]:
    """The infinite, deterministic stream of training steps.

    Each epoch is a fresh shuffle of every document; batches are consecutive groups
    of `batch_size` within the epoch, so the last batch of an epoch may be smaller
    and no batch mixes two epochs.
    """
    n = len(processed_per_doc)
    if n == 0:
        raise ValueError("empty corpus")
    step = 0
    total = 0
    epoch = 0
    while True:
        order = epoch_order(n, epoch_seed(training_seed, epoch))
        for b, start in enumerate(range(0, n, batch_size)):
            idx = tuple(order[start : start + batch_size])
            tokens = sum(processed_per_doc[i] for i in idx)
            total += tokens
            yield PlannedStep(step, epoch, b, idx, tokens, total)
            step += 1
        epoch += 1


def plan_until(
    processed_per_doc: Sequence[int], batch_size: int, training_seed: int, horizon_tokens: int
) -> list[PlannedStep]:
    """All steps from step 0 up to and including the first one reaching `horizon_tokens`."""
    steps: list[PlannedStep] = []
    if horizon_tokens <= 0:
        return steps
    for s in iter_steps(processed_per_doc, batch_size, training_seed):
        steps.append(s)
        if s.tokens_after >= horizon_tokens:
            return steps
    raise AssertionError("unreachable")


def cursor_after(
    planned: Sequence[PlannedStep], steps_completed: int, n_docs: int, batch_size: int
) -> dict[str, int]:
    """Where the loader stands after `steps_completed` steps: the next batch it reads.

    Resuming at planned step `steps_completed` consumes exactly these documents next,
    because the plan is a deterministic replay of `iter_steps`.
    """
    out = {"steps_completed": steps_completed}
    if steps_completed == 0:
        return out | {"next_epoch": 0, "next_batch_in_epoch": 0, "documents_consumed_in_epoch": 0}
    s = planned[steps_completed - 1]
    consumed = min((s.batch_in_epoch + 1) * batch_size, n_docs)
    if consumed == n_docs:
        return out | {
            "next_epoch": s.epoch + 1,
            "next_batch_in_epoch": 0,
            "documents_consumed_in_epoch": 0,
        }
    return out | {
        "next_epoch": s.epoch,
        "next_batch_in_epoch": s.batch_in_epoch + 1,
        "documents_consumed_in_epoch": consumed,
    }


def milestone_steps(steps: Sequence[PlannedStep], milestones: Sequence[int]) -> dict[int, int]:
    """Map each milestone to the step after which its checkpoint is saved.

    Milestone 0 maps to -1 (saved before the first step). Any other milestone fires
    after the first step whose cumulative processed tokens reach it. Milestones not
    reached by `steps` are omitted.
    """
    out: dict[int, int] = {}
    for m in milestones:
        if m == 0:
            out[m] = -1
            continue
        for s in steps:
            if s.tokens_after >= m:
                out[m] = s.step
                break
    return out


# ---------------------------------------------------------------------------
# Learning-rate schedule


class ScheduleSegment(BaseModel):
    """One warmup + cosine schedule over one declared token horizon.

    The first phase is segment 0. A continuation toward a longer horizon is a new
    segment starting where the previous one ended; earlier segments are never
    rewritten (spec section 7: no retroactive relabelling).
    """

    model_config = ConfigDict(extra="forbid")

    index: int
    kind: str  # "initial" or "continuation"
    shape: str = "linear_warmup_then_cosine_to_zero"
    start_step: int  # first global step of this segment
    num_steps: int
    warmup_steps: int
    peak_learning_rate: float
    final_learning_rate: float = 0.0  # cosine endpoint, reached just after the last step
    start_processed_tokens: int
    horizon_processed_tokens: int  # declared horizon
    end_processed_tokens: int  # planned actual count after the last step (>= horizon)
    resumed_from_checkpoint: str | None = None  # continuation: where optimizer state came from
    tinker_run_ids: list[str] = []
    status: str = "planned"  # planned | running | complete
    created_at: str | None = None

    @property
    def end_step(self) -> int:
        """One past the last step of this segment."""
        return self.start_step + self.num_steps


def warmup_steps_for(num_steps: int, warmup_fraction: float) -> int:
    if warmup_fraction <= 0:
        return 0
    return min(num_steps, max(1, round(warmup_fraction * num_steps)))


def learning_rate(segment: ScheduleSegment, step: int) -> float:
    """LR for a global step inside `segment`.

    Warmup is linear and never zero: step k (0-based) of a w-step warmup uses
    peak * (k + 1) / w. After warmup, cosine from peak toward `final_learning_rate`;
    the cosine reaches the endpoint at `num_steps`, one past the last step taken.
    """
    k = step - segment.start_step
    if not 0 <= k < segment.num_steps:
        raise ValueError(f"step {step} outside segment {segment.index}")
    peak, final, w = segment.peak_learning_rate, segment.final_learning_rate, segment.warmup_steps
    if k < w:
        return peak * (k + 1) / w
    progress = (k - w) / (segment.num_steps - w)
    return final + (peak - final) * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_segment(
    *,
    index: int,
    kind: str,
    planned: Sequence[PlannedStep],
    start_step: int,
    horizon_tokens: int,
    peak_learning_rate: float,
    warmup_fraction: float,
    resumed_from_checkpoint: str | None = None,
    created_at: str | None = None,
) -> ScheduleSegment:
    """Build a segment covering planned steps [start_step, first step reaching horizon]."""
    seg_steps: list[PlannedStep] = []
    for s in planned:
        if s.step < start_step:
            continue
        seg_steps.append(s)
        if s.tokens_after >= horizon_tokens:
            break
    if not seg_steps or seg_steps[-1].tokens_after < horizon_tokens:
        raise ValueError("plan does not reach the horizon")
    start_tokens = seg_steps[0].tokens_after - seg_steps[0].tokens
    num = len(seg_steps)
    return ScheduleSegment(
        index=index,
        kind=kind,
        start_step=start_step,
        num_steps=num,
        warmup_steps=warmup_steps_for(num, warmup_fraction),
        peak_learning_rate=peak_learning_rate,
        start_processed_tokens=start_tokens,
        horizon_processed_tokens=horizon_tokens,
        end_processed_tokens=seg_steps[-1].tokens_after,
        resumed_from_checkpoint=resumed_from_checkpoint,
        created_at=created_at,
    )
