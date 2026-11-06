"""Behavioral evaluation on the GPU pod (spec sections 5, 6 and 8).

One process loads Qwen3-8B (bf16, pinned revision) once and evaluates a list of runs. A
run is a model condition plus the checkpoint(s) it uses, always on a whole split (the
split's main-task file and that split's diagnostics):

    base          adapters disabled (or no adapters loaded yet)
    belief        one belief checkpoint active
    goal          one goal checkpoint active
    belief+goal   one belief and one goal checkpoint active together (see compose.py)

Every condition goes through the same stack: transformers + PEFT, prompts rendered only
by `tokenizer.render_chat_prompt(tok, model_input_text(item))`, greedy decoding,
`evaluation.max_new_tokens`, left-padded batches of `evaluation.batch_size`, stop at
<|im_end|>. Every item gets an EvalResult, including invalid and truncated outputs; rows
are built only by `make_result`, which derives the parse from the raw output.

Results (shared contract, schemas.EvalRunManifest):

    results/<split>/<condition>/<checkpoint>/<eval_run_id>/manifest.json
    results/<split>/<condition>/<checkpoint>/<eval_run_id>/results.jsonl
    results/<split>/<condition>/<checkpoint>/<eval_run_id>/run_info.json   (checks, sessions)

`eval_run_id` hashes every manifest field: dataset revision, eval-file hashes, base model
and revision, each adapter's PEFT weights hash and tinker:// source, inference settings
and package versions. A resumed run continues only in the folder with the same id;
anything that changed starts a new folder, so rows are never mixed. Within a run,
finished item_ids are skipped and batches are formed over the full item list, so an
item's batch-mates do not depend on where an earlier attempt stopped.

Checks that fail loudly:
- Before and after every run, with adapters disabled, logits on a probe batch must equal
  the logits of the plain base model computed before any adapter was attached.
- Before a belief+goal run, `compose.verify_composition` on every adapted module.
- Each adapter must be admitted by `export_adapters.load_export_record` (inspection
  passed, files unchanged, Tinker parity passed for these bytes) and must load into the
  model exactly as stored (`compose.verify_loaded_adapter`).
- The test split is refused without a frozen selection (paths.selection_path()), and
  adapters there must be exactly the selected ones (checkpoint, tinker path, sha256).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from belief_desire_organisms import compose, hf_io
from belief_desire_organisms.conditions import Corpus, ModelCondition
from belief_desire_organisms.config import PilotConfig
from belief_desire_organisms.export_adapters import (
    adapter_digest,
    load_export_record,
    peft_dir,
)
from belief_desire_organisms.parsing import parse_choice
from belief_desire_organisms.paths import (
    EVAL_DEV_FILE,
    EVAL_DIAGNOSTICS_FILE,
    EVAL_TEST_FILE,
    dataset_dir,
    eval_mode,
    eval_run_dir,
    results_dir,
    results_dir_name,
    selection_path,
)
from belief_desire_organisms.provenance import (
    device_description,
    eval_code_sha256,
    git_commit,
    now,
    package_versions,
)
from belief_desire_organisms.schemas import (
    LETTERS,
    EvalItem,
    EvalResult,
    EvalRunManifest,
    SelectionManifest,
    Split,
    file_sha256,
    model_input_text,
    read_jsonl,
    write_json,
)
from belief_desire_organisms.tokenizer import get_tokenizer, render_chat_prompt

END_OF_TURN = "<|im_end|>"
MANIFEST_FILE = "manifest.json"
RESULTS_FILE = "results.jsonl"
RUN_INFO_FILE = "run_info.json"  # not part of the run id: checks, sessions, code commit
DATASET_REVISION_FILE = ".revision"  # in dataset_dir(); written by `pod fetch-dataset`
RUN_ID_HEX_CHARS = 16


class EvaluationError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Runs and labels


@dataclass(frozen=True)
class RunSpec:
    condition: ModelCondition
    checkpoints: tuple[tuple[Corpus, str], ...] = ()

    @classmethod
    def make(cls, condition: ModelCondition, checkpoints: dict[Corpus, str] | None = None):
        ckpts = dict(checkpoints or {})
        compose.active_names(condition, ckpts)  # validates which corpora are needed
        for name in ckpts.values():
            if not name or "." in name or "/" in name or "+" in name:
                raise EvaluationError(f"bad checkpoint name {name!r}")
        if len(set(ckpts.values())) != len(ckpts):
            raise EvaluationError(f"checkpoint names must differ: {ckpts}")
        order = [c for c in (Corpus.BELIEF, Corpus.GOAL) if c in ckpts]
        return cls(condition, tuple((c, ckpts[c]) for c in order))

    @property
    def ckpts(self) -> dict[Corpus, str]:
        return dict(self.checkpoints)

    @property
    def adapter_names(self) -> list[str]:
        """Adapter names inside the PeftModel: the checkpoint names, belief first."""
        return compose.active_names(self.condition, self.ckpts)

    @property
    def label(self) -> str:
        return checkpoint_label(self.ckpts)


def checkpoint_label(checkpoints: dict[Corpus, str]) -> str:
    """'base', 'belief-5000000', 'goal-3000000' or 'belief-5000000+goal-3000000'."""
    names = [checkpoints[c] for c in (Corpus.BELIEF, Corpus.GOAL) if c in checkpoints]
    return "+".join(names) if names else "base"


def prompt_sha256(token_ids: Sequence[int]) -> str:
    """sha256 of the rendered input token ids as a compact JSON list, e.g. '[1,2,3]'."""
    return hashlib.sha256(json.dumps(list(token_ids), separators=(",", ":")).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Eval items and the dataset they come from


def split_eval_files(split: Split) -> tuple[str, str]:
    """Dataset-relative files holding a split's items: its main-task file and diagnostics."""
    return (EVAL_DEV_FILE if split == "dev" else EVAL_TEST_FILE, EVAL_DIAGNOSTICS_FILE)


def load_eval_items(split: Split) -> list[EvalItem]:
    """Every item of one split (main task and all diagnostics), in file order."""
    from belief_desire_organisms import build_evals

    items: list[EvalItem] = []
    for rel in split_eval_files(split):
        path = dataset_dir() / rel
        if not path.exists():
            raise EvaluationError(f"eval file missing: {path}")
        rows = list(read_jsonl(path, EvalItem))
        build_evals.check_file_placement(rel, rows)
        items.extend(r for r in rows if r.split == split)
    ids = [it.item_id for it in items]
    if len(set(ids)) != len(ids):
        dup = [i for i, n in Counter(ids).items() if n > 1][:5]
        raise EvaluationError(f"duplicate item_ids: {dup}")
    if not items:
        raise EvaluationError(f"no items for split {split}")
    return items


def eval_files_sha256(split: Split) -> dict[str, str]:
    return {rel: file_sha256(dataset_dir() / rel) for rel in split_eval_files(split)}


def local_dataset_revision() -> str | None:
    """HF dataset commit the local eval files were fetched at, if known."""
    marker = dataset_dir() / DATASET_REVISION_FILE
    return (marker.read_text().strip() or None) if marker.exists() else None


# ---------------------------------------------------------------------------
# Generation backend


@dataclass
class Generation:
    text: str  # decoded new tokens up to (not including) <|im_end|>, special tokens kept
    token_ids: list[int]
    truncated: bool  # no <|im_end|> within max_new_tokens
    letter_logprobs: dict[str, float] | None


def left_pad(prompts: Sequence[Sequence[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(prompts), width), dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, width - len(p) :] = torch.tensor(list(p), dtype=torch.long)
        mask[i, width - len(p) :] = 1
    return ids, mask


class HFGenerator:
    """Greedy, batched, left-padded generation with transformers.

    `model` is the plain HF causal LM; when adapters are attached PEFT injects LoRA layers
    into it in place, so the same object serves every condition.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        max_new_tokens: int,
        batch_size: int,
        letter_logprobs: bool = True,
        stop_token_ids: Sequence[int] | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.stop_token_ids = list(stop_token_ids or [tokenizer.convert_tokens_to_ids(END_OF_TURN)])
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            raise EvaluationError("tokenizer has no pad token")
        self.letter_logprobs = letter_logprobs
        self.letter_ids = self._letter_token_ids() if letter_logprobs else None

    def _letter_token_ids(self) -> dict[str, list[int]]:
        """Token ids of each answer letter. Single tokens for Qwen3 ("A"=32, "B"=33, "C"=34)."""
        out = {}
        for letter in LETTERS:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if self.tokenizer.decode(ids) != letter:
                raise EvaluationError(f"letter {letter!r} does not round-trip: {ids}")
            out[letter] = ids
        return out

    @property
    def single_token_letters(self) -> bool:
        return self.letter_ids is not None and all(len(v) == 1 for v in self.letter_ids.values())

    @property
    def inference(self) -> dict:
        """Settings that affect outputs. Identical for every condition and split by design
        (analysis compares this dict exactly); recorded in the manifest and every row."""
        dtype = next(self.model.parameters()).dtype
        if not self.letter_logprobs:
            letters = None
        else:
            letters = "first-token" if self.single_token_letters else "full-string"
        return {
            "backend": "transformers",
            "renderer": "tokenizer.render_chat_prompt (single user turn, no system prompt)",
            "enable_thinking": False,
            "decoding": "greedy",
            "max_new_tokens": self.max_new_tokens,
            "stop_token_ids": self.stop_token_ids,
            "batch_size": self.batch_size,
            "padding_side": "left",
            "batch_order": "prompt length, then file order",
            "dtype": str(dtype).removeprefix("torch."),
            "letter_logprobs": letters,
            # Items with key.option_answers: each answer's summed log-prob as a continuation
            # of the prompt, stored under its option letter (A = listed first).
            "answer_scoring": "option_answers full-string log-prob, else letters",
        }

    def _generation_config(self):
        from transformers import GenerationConfig

        return GenerationConfig(
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.stop_token_ids,
            pad_token_id=self.pad_id,
            return_dict_in_generate=True,
            output_logits=True,
        )

    @torch.inference_mode()
    def generate(self, prompts: Sequence[Sequence[int]]) -> list[Generation]:
        out: list[Generation] = []
        for start in range(0, len(prompts), self.batch_size):
            out.extend(self.generate_batch(prompts[start : start + self.batch_size]))
        return out

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: Sequence[Sequence[int]],
        answers: Sequence[Sequence[str] | None] | None = None,
    ) -> list[Generation]:
        """One left-padded batch; callers choose the batch composition. `answers[i]`, if
        given, replaces the letter scores of prompt i with the full-string log-probs of its
        three answers (key.option_answers), stored under A, B, C in option order."""
        device = next(self.model.parameters()).device
        ids, mask = left_pad(prompts, self.pad_id)
        res = self.model.generate(
            input_ids=ids.to(device),
            attention_mask=mask.to(device),
            generation_config=self._generation_config(),
        )
        new = res.sequences[:, ids.shape[1] :].cpu().tolist()
        # Raw (unprocessed) logits of the first generated position = end of the prompt.
        first_logp = torch.log_softmax(res.logits[0].float(), dim=-1).cpu()
        letter_lp = None
        if self.letter_logprobs and not self.single_token_letters:
            letter_lp = self._score_letters(prompts)
        gens = []
        for row, toks in enumerate(new):
            stop = next((i for i, t in enumerate(toks) if t in self.stop_token_ids), None)
            kept = toks if stop is None else toks[:stop]
            lp = None
            if answers is not None and answers[row] is not None:
                lp = self._score_answers(prompts[row], answers[row])
            elif self.letter_logprobs:
                if letter_lp is None:
                    lp = {k: float(first_logp[row, v[0]]) for k, v in self.letter_ids.items()}
                else:
                    lp = letter_lp[row]
            gens.append(
                Generation(
                    text=self.tokenizer.decode(kept, skip_special_tokens=False),
                    token_ids=kept,
                    truncated=stop is None,
                    letter_logprobs=lp,
                )
            )
        return gens

    def _score_answers(self, prompt: Sequence[int], answers: Sequence[str]) -> dict[str, float]:
        """Summed log-prob of each answer string as the start of the reply."""
        if len(answers) != len(LETTERS):
            raise EvaluationError(f"expected {len(LETTERS)} option answers, got {len(answers)}")
        cands = [self.tokenizer.encode(a, add_special_tokens=False) for a in answers]
        if any(not c for c in cands):
            raise EvaluationError(f"an option answer encodes to no tokens: {answers}")
        (row,) = continuation_logprobs(self.model, [prompt], cands, self.pad_id)
        return dict(zip(LETTERS, row, strict=True))

    def _score_letters(self, prompts: Sequence[Sequence[int]]) -> list[dict[str, float]]:
        """Full-string log-probs of each letter continuation (used if letters are multi-token)."""
        scores = continuation_logprobs(
            self.model, prompts, [self.letter_ids[k] for k in LETTERS], self.pad_id
        )
        return [dict(zip(LETTERS, row, strict=True)) for row in scores]


@torch.inference_mode()
def continuation_logprobs(
    model, prompts: Sequence[Sequence[int]], candidates: Sequence[Sequence[int]], pad_id: int
) -> list[list[float]]:
    """sum_t log p(candidate_t | prompt, candidate_<t) for every prompt x candidate."""
    device = next(model.parameters()).device
    out = []
    for p in prompts:
        seqs = [list(p) + list(c) for c in candidates]
        ids, mask = left_pad(seqs, pad_id)
        logp = torch.log_softmax(
            model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float(), -1
        ).cpu()
        width = ids.shape[1]
        row = []
        for i, c in enumerate(candidates):
            total = 0.0
            for j, tok in enumerate(c):
                pos = width - len(c) + j  # position of this candidate token
                total += float(logp[i, pos - 1, tok])
            row.append(total)
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Thinking mode (BDO_EVAL_MODE=thinking): reasoning before the answer, then the same
# answer scoring. A separate experiment from the protocol (which keeps thinking off).

THINK_SUFFIX = "<think>\n\n</think>\n\n"  # what enable_thinking=False appends to the prompt
THINK_END = "</think>"
# Qwen3's recommended way to stop a reasoning budget (model card, "thinking budget").
FORCE_ANSWER = (
    "\n\nConsidering the limited time by the user, I have to give the solution based on the "
    "thinking directly now.\n</think>\n\n"
)
THINKING = {
    "max_thinking_tokens": 1024,
    "temperature": 0.6,  # Qwen3's recommended thinking-mode sampling settings
    "top_p": 0.95,
    "top_k": 20,
    "batch_size": 64,
}


class ThinkingGenerator(HFGenerator):
    """Sampled reasoning (Qwen3 thinking on) up to a token budget, then the option answers
    scored as continuations of prompt + reasoning + "</think>\n\n".

    The prompt each item renders to (thinking off, so it ends in an empty think block) has
    that block removed, which gives exactly the thinking-on generation prompt. A reasoning
    that does not close within the budget gets FORCE_ANSWER appended (recorded as
    truncated). Each batch is sampled with a seed derived from its prompts, so a resumed run
    redoes a batch identically.
    """

    def __init__(self, model, tokenizer, *, tasks: Sequence[str] | None = None, **kw):
        super().__init__(model, tokenizer, max_new_tokens=THINKING["max_thinking_tokens"], **kw)
        # Part of the run identity: batches (and so sampling seeds) depend on the item set.
        self.tasks = sorted(tasks) if tasks is not None else None
        self.think_end_id = tokenizer.convert_tokens_to_ids(THINK_END)
        self.suffix = tokenizer.encode(THINK_SUFFIX, add_special_tokens=False)
        self.force = tokenizer.encode(FORCE_ANSWER, add_special_tokens=False)
        self.after_end = tokenizer.encode("\n\n", add_special_tokens=False)

    @property
    def inference(self) -> dict:
        return {
            **super().inference,
            "renderer": "tokenizer.render_chat_prompt with the empty think block removed "
            "(= enable_thinking=True generation prompt)",
            "enable_thinking": True,
            "decoding": "sampled reasoning until </think> or the budget, then option answers "
            "scored as continuations of prompt + reasoning + '</think>\\n\\n'",
            "thinking": {**THINKING, "force_answer": FORCE_ANSWER, "seed": "sha256 of batch"},
            "task_subset": self.tasks,  # None: every task of the split
        }

    def _generation_config(self):
        from transformers import GenerationConfig

        return GenerationConfig(
            do_sample=True,
            temperature=THINKING["temperature"],
            top_p=THINKING["top_p"],
            top_k=THINKING["top_k"],
            max_new_tokens=THINKING["max_thinking_tokens"],
            eos_token_id=[self.think_end_id, *self.stop_token_ids],
            pad_token_id=self.pad_id,
            return_dict_in_generate=True,
        )

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: Sequence[Sequence[int]],
        answers: Sequence[Sequence[str] | None] | None = None,
    ) -> list[Generation]:
        if answers is None or any(a is None for a in answers):
            raise EvaluationError("thinking mode scores option answers; every item needs them")
        base = []
        for p in prompts:
            if list(p[-len(self.suffix) :]) != self.suffix:
                raise EvaluationError("prompt does not end with the empty think block")
            base.append(list(p[: -len(self.suffix)]))
        device = next(self.model.parameters()).device
        ids, mask = left_pad(base, self.pad_id)
        seed = int(hashlib.sha256(json.dumps(base).encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed)
        res = self.model.generate(
            input_ids=ids.to(device),
            attention_mask=mask.to(device),
            generation_config=self._generation_config(),
        )
        new = res.sequences[:, ids.shape[1] :].cpu().tolist()
        gens = []
        for row, toks in enumerate(new):
            end = next((i for i, t in enumerate(toks) if t == self.think_end_id), None)
            if end is not None:
                reasoning, forced = toks[: end + 1] + self.after_end, False
            else:
                stop = next((i for i, t in enumerate(toks) if t in self.stop_token_ids), None)
                kept = toks if stop is None else toks[:stop]
                reasoning, forced = kept + self.force, True
            lp = self._score_answers(base[row] + reasoning, answers[row])
            gens.append(
                Generation(
                    text=self.tokenizer.decode(reasoning, skip_special_tokens=False),
                    token_ids=reasoning,
                    truncated=forced,
                    letter_logprobs=lp,
                )
            )
        return gens


# ---------------------------------------------------------------------------
# Run manifests and result rows


def eval_run_id(fields: dict) -> str:
    """Deterministic id: sha256 of the canonical JSON of every other manifest field."""
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:RUN_ID_HEX_CHARS]


def build_manifest(
    cfg: PilotConfig,
    spec: RunSpec,
    split: Split,
    *,
    eval_hashes: dict[str, str],
    adapters: dict[str, tuple[str, str]],  # checkpoint name -> (sha256, tinker:// source)
    inference: dict,
) -> EvalRunManifest:
    fields = {
        "split": split,
        "condition": spec.condition.value,
        "checkpoint": spec.label,
        "dataset_revision": local_dataset_revision(),
        "eval_files_sha256": dict(sorted(eval_hashes.items())),
        "base_model": cfg.training.base_model,
        "base_model_revision": cfg.training.base_model_revision,
        "adapter_sha256": {name: sha for name, (sha, _) in sorted(adapters.items())},
        "adapter_sources": {name: src for name, (_, src) in sorted(adapters.items())},
        "inference": inference,
        "package_versions": {**package_versions(), "eval_code_sha256": eval_code_sha256()},
    }
    return EvalRunManifest(eval_run_id=eval_run_id(fields), **fields)


def open_run(manifest: EvalRunManifest) -> Path:
    """The run's folder, created with its manifest. Refuses a folder whose manifest
    differs, and results without a manifest (unknown provenance)."""
    d = eval_run_dir(
        manifest.split, manifest.condition.value, manifest.checkpoint, manifest.eval_run_id
    )
    mpath, rpath = d / MANIFEST_FILE, d / RESULTS_FILE
    if mpath.exists():
        existing = EvalRunManifest.model_validate_json(mpath.read_text())
        if existing != manifest:
            raise EvaluationError(f"{mpath} differs from this run's manifest; refusing to resume")
    else:
        if rpath.exists():
            raise EvaluationError(f"{rpath} has no {MANIFEST_FILE}; move it aside")
        write_json(mpath, manifest)
    return d


def make_result(
    manifest: EvalRunManifest,
    item: EvalItem,
    prompt_ids: Sequence[int],
    generation: Generation,
) -> EvalResult:
    """The only way result rows are built: the parse is derived from the raw output."""
    choice = parse_choice(generation.text)
    return EvalResult(
        eval_run_id=manifest.eval_run_id,
        item_id=item.item_id,
        split=item.split,
        task=item.task,
        scenario_id=item.scenario_id,
        trio_id=item.trio_id,
        condition=manifest.condition,
        checkpoint=manifest.checkpoint,
        prompt_sha256=prompt_sha256(prompt_ids),
        raw_output=generation.text,
        parsed_choice=choice,
        valid=choice is not None,
        truncated=generation.truncated,
        letter_logprobs=generation.letter_logprobs,
        generation_config=manifest.inference,
    )


def _read_existing(path: Path, manifest: EvalRunManifest) -> dict[str, str]:
    """item_id -> prompt_sha256 of rows already written in this run.

    Drops a partial last line left by an interrupted write.
    """
    if not path.exists():
        return {}
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        keep = raw[: raw.rfind(b"\n") + 1]
        print(f"[eval] {path}: dropping a partial last line from an interrupted write")
        path.write_bytes(keep)
        raw = keep
    done: dict[str, str] = {}
    for n, line in enumerate(raw.decode().splitlines(), 1):
        if not line.strip():
            continue
        r = EvalResult.model_validate_json(line)
        if r.item_id in done:
            raise EvaluationError(f"{path}:{n}: duplicate item_id {r.item_id}")
        if (r.eval_run_id, r.condition, r.checkpoint, r.split) != (
            manifest.eval_run_id,
            manifest.condition,
            manifest.checkpoint,
            manifest.split,
        ):
            raise EvaluationError(f"{path}:{n}: row belongs to another run")
        done[r.item_id] = r.prompt_sha256
    return done


def run_items(
    generator: HFGenerator,
    items: Sequence[EvalItem],
    run_dir: Path,
    manifest: EvalRunManifest,
) -> dict:
    """Generate for every item not yet in the run's results file; one row per item."""
    if generator.inference != manifest.inference:
        raise EvaluationError("generator settings differ from the run manifest")
    path = run_dir / RESULTS_FILE
    tok = generator.tokenizer
    written = _read_existing(path, manifest)
    unknown = set(written) - {it.item_id for it in items}
    if unknown:
        raise EvaluationError(
            f"{path} has rows for items not in the eval set: {sorted(unknown)[:5]}"
        )
    rendered = {it.item_id: render_chat_prompt(tok, model_input_text(it)) for it in items}
    changed = [i for i, h in written.items() if prompt_sha256(rendered[i]) != h]
    if changed:
        raise EvaluationError(
            f"{path}: {len(changed)} finished items now render to a different prompt "
            f"(e.g. {changed[:3]})"
        )
    done = set(written)
    # Fixed batches over all items (sorted by prompt length, then file order), so an item's
    # batch-mates are the same whether or not an earlier attempt was interrupted.
    order = sorted(range(len(items)), key=lambda i: (len(rendered[items[i].item_id]), i))
    size = generator.batch_size
    batches = [order[i : i + size] for i in range(0, len(order), size)]
    n_new = n_valid = n_truncated = 0
    for batch in batches:
        batch_items = [items[i] for i in batch]
        if all(it.item_id in done for it in batch_items):
            continue
        prompts = [rendered[it.item_id] for it in batch_items]
        gens = generator.generate_batch(prompts, [it.key.option_answers for it in batch_items])
        rows = [
            make_result(manifest, it, ids, g)
            for it, ids, g in zip(batch_items, prompts, gens, strict=True)
            if it.item_id not in done
        ]
        with open(path, "a") as f:
            f.write("".join(r.model_dump_json() + "\n" for r in rows))
            f.flush()
            os.fsync(f.fileno())
        done.update(it.item_id for it in batch_items)
        n_new += len(rows)
        n_valid += sum(r.valid for r in rows)
        n_truncated += sum(r.truncated for r in rows)
    return {
        "n_items": len(items),
        "n_written": n_new,
        "n_valid_written": n_valid,
        "n_truncated_written": n_truncated,
        **summarize(path),
    }


def summarize(path: Path) -> dict:
    rows = list(read_jsonl(path, EvalResult)) if path.exists() else []
    return {
        "n_rows": len(rows),
        "valid_rate": sum(r.valid for r in rows) / len(rows) if rows else None,
        "n_truncated": sum(r.truncated for r in rows),
        "rows_by_task": dict(sorted(Counter(r.task for r in rows).items())),
    }


# ---------------------------------------------------------------------------
# Model loading and the multi-run evaluator


def load_base_model(cfg: PilotConfig, *, device: str | None = None, dtype=torch.bfloat16):
    """Qwen3-8B at the pinned revision, bf16, eval mode, without hub sampling defaults."""
    from transformers import AutoModelForCausalLM, GenerationConfig

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("[eval] WARNING: no CUDA device; an 8B model on CPU will be extremely slow")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.training.base_model,
        revision=cfg.training.base_model_revision,
        dtype=dtype,
        device_map=device,
    )
    commit = getattr(model.config, "_commit_hash", None)
    if commit is not None and commit != cfg.training.base_model_revision:
        raise EvaluationError(f"loaded base revision {commit}, expected the pinned revision")
    model.eval()
    # The hub generation_config samples (temperature 0.6, top_k 20); never inherit it.
    model.generation_config = GenerationConfig()
    return model


def _check_protocol(cfg: PilotConfig) -> None:
    if cfg.evaluation.enable_thinking:
        raise EvaluationError("enable_thinking=True is a protocol revision; not implemented")
    if cfg.evaluation.temperature != 0.0:
        raise EvaluationError("only greedy decoding (temperature 0.0) is implemented")


@dataclass
class Evaluator:
    """Holds the base model once; attaches/detaches adapters per run; checks restoration."""

    cfg: PilotConfig
    model: object  # plain HF model (PEFT layers get injected into it in place)
    tokenizer: object
    probe_ids: torch.Tensor
    probe_mask: torch.Tensor
    restore_atol: float = 0.0
    letter_logprobs: bool = True
    cat_check: bool = False
    keep_adapters: bool = False  # never detach (final run: M0 = adapters loaded but disabled)
    tasks: tuple[str, ...] | None = None  # evaluation-mode runs only: the task subset
    require_parity: bool = True  # adapters must have passed the Tinker parity check
    expected_sha256: dict[str, str] | None = None  # test split: digests frozen in selection
    peft_model: object | None = None
    disk: dict = field(default_factory=dict)  # adapter name -> compose.DiskAdapter
    sources: dict = field(default_factory=dict)  # adapter name -> tinker:// path exported
    reference_logits: torch.Tensor | None = None

    @classmethod
    def create(cls, cfg: PilotConfig, model, tokenizer, probe_items, **kw) -> Evaluator:
        from peft import PeftModel
        from peft.tuners.lora import LoraLayer

        if isinstance(model, PeftModel) or any(isinstance(m, LoraLayer) for m in model.modules()):
            raise EvaluationError("pass a freshly loaded base model; adapters are attached per run")
        prompts = [render_chat_prompt(tokenizer, model_input_text(it)) for it in probe_items]
        ids, mask = left_pad(prompts, tokenizer.pad_token_id)
        ev = cls(cfg, model, tokenizer, ids, mask, **kw)
        # Reference: the plain base model, before any adapter is attached.
        ev.reference_logits = compose.probe_logits(model, ids, mask)
        return ev

    @property
    def generator(self) -> HFGenerator:
        ev = self.cfg.evaluation
        if eval_mode() == "thinking":
            return ThinkingGenerator(
                self.model,
                self.tokenizer,
                tasks=self.tasks,
                batch_size=THINKING["batch_size"],
                letter_logprobs=self.letter_logprobs,
            )
        return HFGenerator(
            self.model,
            self.tokenizer,
            max_new_tokens=ev.max_new_tokens,
            batch_size=ev.batch_size,
            letter_logprobs=self.letter_logprobs,
        )

    def check_base_restored(self) -> dict:
        if self.peft_model is not None:
            compose.set_active(self.peft_model, [])
        got = compose.probe_logits(self.model, self.probe_ids, self.probe_mask)
        return compose.compare_logits(
            self.reference_logits, got, self.probe_mask, atol=self.restore_atol
        )

    def _ensure_adapters(self, spec: RunSpec) -> None:
        """Load the adapters this run needs, then drop the ones it does not."""
        needed = dict(zip(spec.adapter_names, spec.ckpts.items(), strict=True))
        for name, (corpus, ckpt) in needed.items():
            if self.peft_model is not None and name in self.peft_model.peft_config:
                continue
            rec = load_export_record(self.cfg, corpus, ckpt, require_parity=self.require_parity)
            path = peft_dir(corpus, ckpt)
            disk = compose.read_peft_adapter(path)
            self.peft_model = compose.attach_adapter(
                self.peft_model if self.peft_model is not None else self.model, name, path
            )
            compose.verify_loaded_adapter(self.peft_model, name, disk)
            self.disk[name] = disk
            self.sources[name] = rec.downloaded_from
        if self.peft_model is not None:
            for name in list(self.peft_model.peft_config):
                if self.keep_adapters:
                    break
                # peft cannot delete its last adapter; an extra disabled adapter is inert.
                if name not in needed and len(self.peft_model.peft_config) > 1:
                    compose.detach_adapter(self.peft_model, name)
                    self.disk.pop(name, None)
                    self.sources.pop(name, None)
            compose.set_active(self.peft_model, [])

    def run(self, spec: RunSpec, items: Sequence[EvalItem], split: Split, info: dict) -> dict:
        started = now()
        self._ensure_adapters(spec)
        names = spec.adapter_names
        adapters = {n: (adapter_digest(self.disk[n].path), self.sources[n]) for n in names}
        if self.expected_sha256 is not None:
            for n, (sha, _) in adapters.items():
                if self.expected_sha256.get(n) != sha:
                    raise EvaluationError(f"{n}: weights differ from the frozen selection")
        gen = self.generator
        manifest = build_manifest(
            self.cfg,
            spec,
            split,
            eval_hashes=eval_files_sha256(split),
            adapters=adapters,
            inference=gen.inference,
        )
        run_dir = open_run(manifest)
        checks: dict = {"restore_before": self.check_base_restored()}
        if len(names) > 1:
            checks["composition"] = compose.verify_composition(self.peft_model, names, self.disk)
            if self.cat_check:
                checks["cat_cross_check"] = compose.cross_check_cat(self.peft_model, names)
        if self.peft_model is not None:
            compose.set_active(self.peft_model, names)
        try:
            stats = run_items(gen, items, run_dir, manifest)
        finally:
            checks["restore_after"] = self.check_base_restored()
        self._record_session(run_dir, manifest, started, checks, stats, info)
        print(f"[eval] {split}/{spec.condition.value}/{spec.label}/{manifest.eval_run_id}: {stats}")
        return {
            "split": split,
            "condition": spec.condition.value,
            "checkpoint": spec.label,
            "eval_run_id": manifest.eval_run_id,
            "run_dir": str(run_dir),
            **stats,
        }

    def _record_session(self, run_dir, manifest, started, checks, stats, info) -> None:
        path = run_dir / RUN_INFO_FILE
        old = json.loads(path.read_text()) if path.exists() else {"sessions": []}
        session = {
            "started_at": started,
            "finished_at": now(),
            "git_commit": git_commit(),
            "device": device_description(),
            "model_commit_hash": getattr(self.model.config, "_commit_hash", None),
            "composition_method": (
                "simultaneous activation (W0 + dW_belief + dW_goal)"
                if len(manifest.adapter_sha256) > 1
                else None
            ),
            "adapters": {
                n: {
                    "r": self.disk[n].config["r"],
                    "lora_alpha": self.disk[n].config["lora_alpha"],
                    "scaling": self.disk[n].scaling,
                    "includes_lm_head": compose.LM_HEAD in self.disk[n].modules,
                }
                for n in manifest.adapter_sha256
            },
            "restore_atol": self.restore_atol,
            "parity_required": self.require_parity,
            "checks": checks,
            "stats": stats,
            **info,
        }
        write_json(
            path, {"eval_run_id": manifest.eval_run_id, "sessions": [*old["sessions"], session]}
        )


# ---------------------------------------------------------------------------
# Entry points


def load_selection() -> SelectionManifest:
    path = selection_path()
    if not path.exists():
        raise EvaluationError(
            f"{path} not found: the test split is evaluated only after a dev-only checkpoint "
            "selection has been frozen (`bdo select`)"
        )
    return SelectionManifest.model_validate_json(path.read_text())


def _selection_digests(cfg: PilotConfig, split: Split, specs: Sequence[RunSpec]) -> dict | None:
    """Test split: the frozen selection must exist, and adapters must be exactly the
    selected checkpoints (name, tinker path, PEFT weights sha256). Returns the digests."""
    if split != "test":
        return None
    sel = load_selection()
    chosen = {Corpus.BELIEF: sel.belief, Corpus.GOAL: sel.goal}
    digests: dict[str, str] = {}
    for spec in specs:
        for corpus, ckpt in spec.checkpoints:
            choice = chosen[corpus]
            if ckpt != choice.checkpoint:
                raise EvaluationError(
                    f"test split with {corpus.value} checkpoint {ckpt}; the selection froze "
                    f"{choice.checkpoint}"
                )
            rec = load_export_record(cfg, corpus, ckpt, require_parity=False)
            if rec.downloaded_from != choice.tinker_path:
                raise EvaluationError(
                    f"{ckpt} was exported from {rec.downloaded_from}; the selection froze "
                    f"{choice.tinker_path}"
                )
            sha = adapter_digest(peft_dir(corpus, ckpt))
            if sha != choice.adapter_sha256:
                raise EvaluationError(f"{ckpt}: PEFT adapter differs from the frozen selection")
            digests[ckpt] = sha
    return digests


def evaluate_runs(
    cfg: PilotConfig,
    specs: Sequence[RunSpec],
    *,
    split: Split,
    execute: bool = False,
    model=None,
    tokenizer=None,
    letter_logprobs: bool = True,
    restore_atol: float = 0.0,
    cat_check: bool = False,
    keep_adapters: bool = False,
    require_parity: bool = True,
    n_probe: int = 4,
    tasks: Sequence[str] | None = None,
) -> list[dict]:
    """Evaluate several runs of one split in one process (base model loaded once).

    Writes results locally; with `execute`, uploads results_dir() to the private results
    repo after each run. Every adapter is admitted before the model is loaded. `tasks`
    (thinking mode only) restricts the items to those tasks.
    """
    _check_protocol(cfg)
    if split not in ("dev", "test"):
        raise EvaluationError("split must be 'dev' or 'test'")
    if tasks is not None and eval_mode() is None:
        raise EvaluationError("a task subset is only allowed outside the protocol (BDO_EVAL_MODE)")
    expected = _selection_digests(cfg, split, specs)
    for spec in specs:
        for corpus, ckpt in spec.checkpoints:
            load_export_record(cfg, corpus, ckpt, require_parity=require_parity)
    items = load_eval_items(split)
    if tasks is not None:
        items = [it for it in items if it.task in set(tasks)]
        if not items:
            raise EvaluationError(f"no {split} items for tasks {list(tasks)}")
    tokenizer = tokenizer or get_tokenizer(cfg)
    model = model if model is not None else load_base_model(cfg)
    ev = Evaluator.create(
        cfg,
        model,
        tokenizer,
        items[:n_probe],
        restore_atol=restore_atol,
        letter_logprobs=letter_logprobs,
        cat_check=cat_check,
        keep_adapters=keep_adapters,
        require_parity=require_parity,
        expected_sha256=expected,
        tasks=tuple(tasks) if tasks is not None else None,
    )
    info = {}
    if expected is not None:
        info["selection_sha256"] = file_sha256(selection_path())
    summaries = []
    for spec in specs:
        summaries.append(ev.run(spec, items, split, info))
        if execute:
            upload_results(cfg, execute=True)
    return summaries


def evaluate(
    cfg: PilotConfig,
    condition: ModelCondition,
    *,
    split: str,
    checkpoints: dict[Corpus, str] | None = None,
    execute: bool = False,
    **kw,
) -> list[dict]:
    """One condition on one split. Without `checkpoints`, adapters come from the frozen
    selection."""
    if checkpoints is None and condition != ModelCondition.BASE:
        sel = load_selection()
        frozen = {Corpus.BELIEF: sel.belief.checkpoint, Corpus.GOAL: sel.goal.checkpoint}
        checkpoints = {c: frozen[c] for c in compose.required_corpora(condition)}
    spec = RunSpec.make(condition, checkpoints)
    return evaluate_runs(cfg, [spec], split=split, execute=execute, **kw)  # type: ignore[arg-type]


def baseline(cfg: PilotConfig, *, execute: bool = False, **kw) -> list[dict]:
    """Unmodified base model on the dev split: main task and every diagnostic."""
    spec = RunSpec.make(ModelCondition.BASE)
    return evaluate_runs(cfg, [spec], split="dev", execute=execute, **kw)


def sweep_specs(corpora: Sequence[Corpus] = (Corpus.BELIEF, Corpus.GOAL)) -> list[RunSpec]:
    from belief_desire_organisms.export_adapters import exported_checkpoints

    cond = {Corpus.BELIEF: ModelCondition.BELIEF, Corpus.GOAL: ModelCondition.GOAL}
    return [
        RunSpec.make(cond[c], {c: rec.name}) for c in corpora for rec in exported_checkpoints(c)
    ]


def sweep(
    cfg: PilotConfig,
    *,
    corpora: Sequence[Corpus] = (Corpus.BELIEF, Corpus.GOAL),
    execute: bool = False,
    **kw,
) -> list[dict]:
    """Every exported checkpoint, single adapter, dev split only (checkpoint selection)."""
    specs = sweep_specs(corpora)
    if not specs:
        raise EvaluationError("no exported checkpoints found under adapters_dir()")
    return evaluate_runs(cfg, specs, split="dev", execute=execute, **kw)


def final_specs(selection: SelectionManifest) -> list[RunSpec]:
    """base, belief, goal and belief+goal with the frozen checkpoints (labels match
    selection.selected_checkpoints)."""
    b = {Corpus.BELIEF: selection.belief.checkpoint}
    g = {Corpus.GOAL: selection.goal.checkpoint}
    return [
        RunSpec.make(ModelCondition.BASE),
        RunSpec.make(ModelCondition.BELIEF, b),
        RunSpec.make(ModelCondition.GOAL, g),
        RunSpec.make(ModelCondition.BELIEF_GOAL, {**b, **g}),
    ]


def final(cfg: PilotConfig, *, execute: bool = False, **kw) -> list[dict]:
    """The four conditions on the test split, with the frozen selected checkpoints.

    belief+goal runs first so both adapters stay loaded for the rest of the process: the
    base condition is "adapters loaded but disabled" on the same stack.
    """
    specs = final_specs(load_selection())
    kw.setdefault("cat_check", True)
    kw.setdefault("keep_adapters", True)
    return evaluate_runs(cfg, specs[3:] + specs[:3], split="test", execute=execute, **kw)


def upload_results(cfg: PilotConfig, *, execute: bool) -> None:
    hf_io.upload_folder(
        cfg.hf.results_repo,
        hf_io.RESULTS_REPO_TYPE,
        results_dir(),
        results_dir_name(),
        execute=execute,
        commit_message="evaluation results",
        keep_larger_remote=True,
    )
