"""Behavioral evaluation on the GPU pod (spec sections 5, 6 and 8).

One process loads Qwen3-8B (bf16, pinned revision) once and evaluates a list of runs. A
run is a model condition plus the checkpoint(s) it uses:

    base          adapters disabled (or no adapters loaded yet)
    belief        one belief checkpoint active
    goal          one goal checkpoint active
    belief+goal   one belief and one goal checkpoint active together (see compose.py)

Every condition goes through the same stack: transformers + PEFT, prompts rendered only
by `tokenizer.render_chat_prompt(tok, model_input_text(item))`, greedy decoding,
`evaluation.max_new_tokens`, left-padded batches of `evaluation.batch_size`, stop at
<|im_end|>. Every item gets an EvalResult, including invalid and truncated outputs.

Results: `results_dir()/<split>/<condition>/<checkpoint_label>.jsonl` (plus a
`.run.json` with provenance and checks). Resumable: finished item_ids are skipped, and
batches are formed over the full item list so an item's batch-mates do not depend on
where a previous attempt stopped.

Checks that fail loudly:
- Before and after every run, with adapters disabled, logits on a probe batch must equal
  the logits of the plain base model computed before any adapter was attached.
- Before a belief+goal run, `compose.verify_composition` on every adapted module.
- Each loaded adapter must match its file exactly (`compose.verify_loaded_adapter`), and
  its export record must have passed inspection (lm_head present, finite, shapes).
- Test-split runs with adapters must use the checkpoints frozen in `selection.json`.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict

from belief_desire_organisms import compose, hf_io
from belief_desire_organisms.conditions import Corpus, ModelCondition
from belief_desire_organisms.config import PilotConfig
from belief_desire_organisms.export_adapters import load_export_record, peft_dir
from belief_desire_organisms.parsing import parse_choice
from belief_desire_organisms.paths import (
    EVAL_DEV_FILE,
    EVAL_DIAGNOSTICS_FILE,
    EVAL_TEST_FILE,
    dataset_dir,
    results_dir,
)
from belief_desire_organisms.provenance import device_description, git_commit, now, package_versions
from belief_desire_organisms.schemas import (
    LETTERS,
    EvalItem,
    EvalResult,
    Split,
    file_sha256,
    model_input_text,
    read_jsonl,
    write_json,
)
from belief_desire_organisms.tokenizer import get_tokenizer, render_chat_prompt

TaskGroup = Literal["main", "diagnostics"]
ALL_TASK_GROUPS: tuple[TaskGroup, ...] = ("main", "diagnostics")
END_OF_TURN = "<|im_end|>"


class EvaluationError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Runs, labels, paths


@dataclass(frozen=True)
class RunSpec:
    condition: ModelCondition
    checkpoints: tuple[tuple[Corpus, str], ...] = ()

    @classmethod
    def make(cls, condition: ModelCondition, checkpoints: dict[Corpus, str] | None = None):
        ckpts = dict(checkpoints or {})
        compose.active_names(condition, ckpts)  # validates which corpora are needed
        for name in ckpts.values():
            if not name or "." in name or "/" in name:
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
        return compose.active_names(self.condition, self.ckpts)

    @property
    def label(self) -> str:
        return checkpoint_label(self.ckpts)


def checkpoint_label(checkpoints: dict[Corpus, str]) -> str:
    """'base', 'belief-5000000', 'goal-3000000' or 'belief-5000000+goal-3000000'."""
    names = [checkpoints[c] for c in (Corpus.BELIEF, Corpus.GOAL) if c in checkpoints]
    return "+".join(names) if names else "base"


def results_path(split: Split, condition: ModelCondition, label: str) -> Path:
    return results_dir() / split / condition.value / f"{label}.jsonl"


def run_meta_path(split: Split, condition: ModelCondition, label: str) -> Path:
    return results_dir() / split / condition.value / f"{label}.run.json"


def prompt_sha256(token_ids: Sequence[int]) -> str:
    """sha256 of the rendered input token ids as a compact JSON list, e.g. '[1,2,3]'."""
    return hashlib.sha256(json.dumps(list(token_ids), separators=(",", ":")).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Checkpoint selection (frozen before the test split is evaluated with adapters)


class CheckpointSelection(BaseModel):
    """`results_dir()/selection.json`. Extra fields (rationale, dev metrics) are kept."""

    model_config = ConfigDict(extra="allow", frozen=True)
    belief: str
    goal: str


def selection_path() -> Path:
    return results_dir() / "selection.json"


def load_selection() -> CheckpointSelection:
    path = selection_path()
    if not path.exists():
        raise EvaluationError(f"{path} not found: freeze the selected checkpoints first")
    return CheckpointSelection.model_validate_json(path.read_text())


# ---------------------------------------------------------------------------
# Eval items


def eval_files(split: Split, tasks: Sequence[TaskGroup]) -> list[str]:
    files = []
    if "main" in tasks:
        files.append(EVAL_DEV_FILE if split == "dev" else EVAL_TEST_FILE)
    if "diagnostics" in tasks:
        files.append(EVAL_DIAGNOSTICS_FILE)
    unknown = set(tasks) - set(ALL_TASK_GROUPS)
    if unknown or not files:
        raise EvaluationError(f"tasks must be from {ALL_TASK_GROUPS}, got {list(tasks)}")
    return files


def load_eval_items(split: Split, tasks: Sequence[TaskGroup]) -> list[EvalItem]:
    """Main-task items from the split file and/or diagnostics of that split, in file order."""
    items: list[EvalItem] = []
    for rel in eval_files(split, tasks):
        path = dataset_dir() / rel
        if not path.exists():
            raise EvaluationError(f"eval file missing: {path}")
        for item in read_jsonl(path, EvalItem):
            if rel == EVAL_DIAGNOSTICS_FILE:
                if item.task == "main":
                    raise EvaluationError(f"{item.item_id}: main-task item in diagnostics file")
                if item.split != split:
                    continue
            else:
                if item.task != "main" or item.split != split:
                    raise EvaluationError(f"{item.item_id}: unexpected item in {rel}")
            items.append(item)
    ids = [it.item_id for it in items]
    if len(set(ids)) != len(ids):
        dup = [i for i, n in Counter(ids).items() if n > 1][:5]
        raise EvaluationError(f"duplicate item_ids: {dup}")
    if not items:
        raise EvaluationError(f"no items for split={split} tasks={list(tasks)}")
    return items


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
        self.letter_ids = self._letter_token_ids() if letter_logprobs else None
        self.letter_logprobs = letter_logprobs

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
    def config(self) -> dict:
        """Recorded in every EvalResult."""
        dtype = next(self.model.parameters()).dtype
        return {
            "backend": "transformers",
            "decoding": "greedy",
            "do_sample": False,
            "max_new_tokens": self.max_new_tokens,
            "stop_token_ids": self.stop_token_ids,
            "batch_size": self.batch_size,
            "padding_side": "left",
            "dtype": str(dtype).removeprefix("torch."),
            "chat_rendering": "render_chat_prompt(enable_thinking=False, no system prompt)",
            "letter_logprobs": (
                None
                if not self.letter_logprobs
                else "first-token"
                if self.single_token_letters
                else "full-string"
            ),
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
    def generate_batch(self, prompts: Sequence[Sequence[int]]) -> list[Generation]:
        """One left-padded batch; callers choose the batch composition."""
        device = next(self.model.parameters()).device
        ids, mask = left_pad(prompts, self.pad_id)
        res = self.model.generate(
            input_ids=ids.to(device),
            attention_mask=mask.to(device),
            generation_config=self._generation_config(),
        )
        new = res.sequences[:, ids.shape[1] :].cpu().tolist()
        first_logp = torch.log_softmax(res.logits[0].float(), dim=-1).cpu()
        if self.letter_logprobs and not self.single_token_letters:
            letter_lp = self._score_letters(prompts)
        gens = []
        for row, toks in enumerate(new):
            stop = next((i for i, t in enumerate(toks) if t in self.stop_token_ids), None)
            kept = toks if stop is None else toks[:stop]
            lp = None
            if self.letter_logprobs:
                if self.single_token_letters:
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
        row = []
        for i, c in enumerate(candidates):
            width = ids.shape[1]
            total = 0.0
            for j, tok in enumerate(c):
                pos = width - len(c) + j  # position of this candidate token
                total += float(logp[i, pos - 1, tok])
            row.append(total)
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Writing results (resumable)


def _read_existing(
    path: Path, *, condition: ModelCondition, label: str, generation_config: dict
) -> set[str]:
    """item_ids already written. Drops a partial last line left by an interrupted write."""
    if not path.exists():
        return set()
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        keep = raw[: raw.rfind(b"\n") + 1]
        print(f"[eval] {path}: dropping a partial last line from an interrupted write")
        path.write_bytes(keep)
        raw = keep
    done: set[str] = set()
    for n, line in enumerate(raw.decode().splitlines(), 1):
        if not line.strip():
            continue
        r = EvalResult.model_validate_json(line)
        if r.item_id in done:
            raise EvaluationError(f"{path}:{n}: duplicate item_id {r.item_id}")
        if r.condition != condition or r.checkpoint != label:
            raise EvaluationError(f"{path}:{n}: written for {r.condition}/{r.checkpoint}")
        if r.generation_config != generation_config:
            raise EvaluationError(
                f"{path}: existing rows used a different generation config; move the file "
                f"aside to start over.\n  existing: {r.generation_config}\n  current:  "
                f"{generation_config}"
            )
        done.add(r.item_id)
    return done


def run_items(
    generator: HFGenerator,
    items: Sequence[EvalItem],
    path: Path,
    *,
    condition: ModelCondition,
    label: str,
) -> dict:
    """Generate for every item not yet in `path` and append one EvalResult per item."""
    tok = generator.tokenizer
    gen_cfg = generator.config
    done = _read_existing(path, condition=condition, label=label, generation_config=gen_cfg)
    unknown = done - {it.item_id for it in items}
    if unknown:
        raise EvaluationError(
            f"{path} has results for items not in the eval set: {sorted(unknown)[:5]}"
        )
    rendered = {it.item_id: render_chat_prompt(tok, model_input_text(it)) for it in items}
    # Fixed batches over all items (sorted by prompt length, then file order), so an item's
    # batch-mates are the same whether or not a previous attempt was interrupted.
    order = sorted(range(len(items)), key=lambda i: (len(rendered[items[i].item_id]), i))
    batches = [
        order[i : i + generator.batch_size] for i in range(0, len(order), generator.batch_size)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    written = valid = truncated = 0
    for batch in batches:
        batch_items = [items[i] for i in batch]
        if all(it.item_id in done for it in batch_items):
            continue
        prompts = [rendered[it.item_id] for it in batch_items]
        gens = generator.generate_batch(prompts)
        lines = []
        for it, ids, g in zip(batch_items, prompts, gens, strict=True):
            if it.item_id in done:
                continue
            choice = parse_choice(g.text)
            lines.append(
                EvalResult(
                    item_id=it.item_id,
                    split=it.split,
                    task=it.task,
                    scenario_id=it.scenario_id,
                    trio_id=it.trio_id,
                    condition=condition,
                    checkpoint=label,
                    prompt_sha256=prompt_sha256(ids),
                    raw_output=g.text,
                    parsed_choice=choice,
                    valid=choice is not None,
                    truncated=g.truncated,
                    letter_logprobs=g.letter_logprobs,
                    generation_config=gen_cfg,
                ).model_dump_json()
            )
            written += 1
            valid += choice is not None
            truncated += g.truncated
        with open(path, "a") as f:
            f.write("".join(line + "\n" for line in lines))
            f.flush()
            os.fsync(f.fileno())
        done.update(it.item_id for it in batch_items)
    return {
        "n_items": len(items),
        "n_written": written,
        "n_valid_written": valid,
        "n_truncated_written": truncated,
        **summarize(path),
    }


def summarize(path: Path) -> dict:
    rows = list(read_jsonl(path, EvalResult)) if path.exists() else []
    by_task = Counter(r.task for r in rows)
    return {
        "n_rows": len(rows),
        "valid_rate": sum(r.valid for r in rows) / len(rows) if rows else None,
        "n_truncated": sum(r.truncated for r in rows),
        "rows_by_task": dict(sorted(by_task.items())),
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
    peft_model: object | None = None
    disk: dict = field(default_factory=dict)  # adapter name -> compose.DiskAdapter
    reference_logits: torch.Tensor | None = None

    @classmethod
    def create(
        cls,
        cfg: PilotConfig,
        model,
        tokenizer,
        probe_items: Sequence[EvalItem],
        **kw,
    ) -> Evaluator:
        prompts = [render_chat_prompt(tokenizer, model_input_text(it)) for it in probe_items]
        ids, mask = left_pad(prompts, tokenizer.pad_token_id)
        ev = cls(cfg, model, tokenizer, ids, mask, **kw)
        from peft import PeftModel
        from peft.tuners.lora import LoraLayer

        if isinstance(model, PeftModel) or any(isinstance(m, LoraLayer) for m in model.modules()):
            raise EvaluationError("pass a freshly loaded base model; adapters are attached per run")
        # Reference: the plain base model, before any adapter is attached.
        ev.reference_logits = compose.probe_logits(model, ids, mask)
        return ev

    @property
    def generator(self) -> HFGenerator:
        ev = self.cfg.evaluation
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
            load_export_record(corpus, ckpt)  # refuses adapters that failed inspection
            path = peft_dir(corpus, ckpt)
            disk = compose.read_peft_adapter(path)
            self.peft_model = compose.attach_adapter(
                self.peft_model if self.peft_model is not None else self.model, name, path
            )
            compose.verify_loaded_adapter(self.peft_model, name, disk)
            self.disk[name] = disk
        if self.peft_model is not None:
            for name in list(self.peft_model.peft_config):
                if self.keep_adapters:
                    break
                # peft cannot delete its last adapter; an extra disabled adapter is inert.
                if name not in needed and len(self.peft_model.peft_config) > 1:
                    compose.detach_adapter(self.peft_model, name)
                    self.disk.pop(name, None)
            compose.set_active(self.peft_model, [])

    def run(self, spec: RunSpec, items: Sequence[EvalItem], split: Split, extra_meta: dict) -> dict:
        started = now()
        self._ensure_adapters(spec)
        checks: dict = {"restore_before": self.check_base_restored()}
        names = spec.adapter_names
        if len(names) > 1:
            checks["composition"] = compose.verify_composition(self.peft_model, names, self.disk)
            if self.cat_check:
                checks["cat_cross_check"] = compose.cross_check_cat(self.peft_model, names)
        if self.peft_model is not None:
            compose.set_active(self.peft_model, names)
        gen = self.generator
        path = results_path(split, spec.condition, spec.label)
        meta_path = run_meta_path(split, spec.condition, spec.label)
        meta = self._meta(spec, split, items, gen, extra_meta)
        _check_meta_compatible(meta_path, meta)
        try:
            stats = run_items(gen, items, path, condition=spec.condition, label=spec.label)
        finally:
            checks["restore_after"] = self.check_base_restored()
        session = {"started_at": started, "finished_at": now(), "checks": checks, "stats": stats}
        old = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        write_json(meta_path, {**meta, "sessions": old.get("sessions", []) + [session]})
        print(f"[eval] {split}/{spec.condition.value}/{spec.label}: {stats}")
        return {"split": split, "condition": spec.condition.value, "label": spec.label, **stats}

    def _meta(self, spec, split, items, gen: HFGenerator, extra: dict) -> dict:
        adapters = {}
        for corpus, ckpt in spec.ckpts.items():
            d = self.disk[ckpt]
            adapters[ckpt] = {
                "corpus": corpus.value,
                "peft_dir": f"{corpus.value}/{ckpt}/peft",
                "adapter_sha256": file_sha256(d.path / "adapter_model.safetensors"),
                "r": d.config["r"],
                "lora_alpha": d.config["lora_alpha"],
                "scaling": d.scaling,
                "includes_lm_head": compose.LM_HEAD in d.modules,
            }
        files = {rel: file_sha256(dataset_dir() / rel) for rel in sorted({*_files_of(items)})}
        return {
            "split": split,
            "condition": spec.condition.value,
            "checkpoint_label": spec.label,
            "adapters": adapters,
            "composition_method": (
                "simultaneous activation (W0 + dW_belief + dW_goal)" if len(adapters) > 1 else None
            ),
            "n_items": len(items),
            "eval_files_sha256": files,
            "generation_config": gen.config,
            "base_model": self.cfg.training.base_model,
            "base_model_revision": self.cfg.training.base_model_revision,
            "model_commit_hash": getattr(self.model.config, "_commit_hash", None),
            "git_commit": git_commit(),
            "package_versions": package_versions(),
            "device": device_description(),
            **extra,
        }


def _files_of(items: Sequence[EvalItem]) -> list[str]:
    files = set()
    for it in items:
        if it.task == "main":
            files.add(EVAL_DEV_FILE if it.split == "dev" else EVAL_TEST_FILE)
        else:
            files.add(EVAL_DIAGNOSTICS_FILE)
    return sorted(files)


_META_MUST_MATCH = (
    "adapters",
    "eval_files_sha256",
    "generation_config",
    "base_model_revision",
    "selection_sha256",
)


def _check_meta_compatible(meta_path: Path, meta: dict) -> None:
    """Refuse to resume a results file produced from different inputs."""
    if not meta_path.exists():
        return
    old = json.loads(meta_path.read_text())
    for key in _META_MUST_MATCH:
        if old.get(key) != meta.get(key):
            raise EvaluationError(
                f"{meta_path}: '{key}' differs from the run that wrote these results; move "
                "the results aside to start over"
            )


# ---------------------------------------------------------------------------
# Entry points


def _guard_test_split(split: Split, specs: Sequence[RunSpec]) -> dict:
    """Adapters on the test split only with the checkpoints frozen in selection.json."""
    if split != "test" or not any(s.checkpoints for s in specs):
        return {}
    sel = load_selection()
    frozen = {Corpus.BELIEF: sel.belief, Corpus.GOAL: sel.goal}
    for s in specs:
        for corpus, ckpt in s.checkpoints:
            if frozen[corpus] != ckpt:
                raise EvaluationError(
                    f"test split with {corpus.value} checkpoint {ckpt}, but selection.json "
                    f"froze {frozen[corpus]}"
                )
    return {"selection_sha256": file_sha256(selection_path())}


def evaluate_runs(
    cfg: PilotConfig,
    specs: Sequence[RunSpec],
    *,
    split: Split,
    tasks: Sequence[TaskGroup] = ALL_TASK_GROUPS,
    execute: bool = False,
    model=None,
    tokenizer=None,
    letter_logprobs: bool = True,
    restore_atol: float = 0.0,
    cat_check: bool = False,
    keep_adapters: bool = False,
    n_probe: int = 4,
) -> list[dict]:
    """Evaluate several runs in one process (base model loaded once).

    Writes results locally; with `execute`, uploads results_dir() to the private results
    repo after each run.
    """
    _check_protocol(cfg)
    if split not in ("dev", "test"):
        raise EvaluationError("split must be 'dev' or 'test'")
    extra = _guard_test_split(split, specs)
    items = load_eval_items(split, tasks)
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
    )
    summaries = []
    for spec in specs:
        summaries.append(ev.run(spec, items, split, {"tasks": list(tasks), **extra}))
        if execute:
            upload_results(cfg, execute=True)
    return summaries


def evaluate(
    cfg: PilotConfig,
    condition: ModelCondition,
    *,
    split: str,
    checkpoints: dict[Corpus, str] | None = None,
    tasks: Sequence[TaskGroup] = ALL_TASK_GROUPS,
    execute: bool = False,
    **kw,
) -> list[dict]:
    """One condition. Without `checkpoints`, adapters come from selection.json."""
    if checkpoints is None and condition != ModelCondition.BASE:
        sel = load_selection()
        frozen = {Corpus.BELIEF: sel.belief, Corpus.GOAL: sel.goal}
        checkpoints = {c: frozen[c] for c in compose.required_corpora(condition)}
    spec = RunSpec.make(condition, checkpoints)
    return evaluate_runs(cfg, [spec], split=split, tasks=tasks, execute=execute, **kw)  # type: ignore[arg-type]


def baseline(cfg: PilotConfig, *, execute: bool = False, **kw) -> list[dict]:
    """Unmodified base model on the dev split: main task and every diagnostic."""
    return evaluate_runs(
        cfg, [RunSpec.make(ModelCondition.BASE)], split="dev", execute=execute, **kw
    )


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


def final_specs() -> list[RunSpec]:
    sel = load_selection()
    b, g = {Corpus.BELIEF: sel.belief}, {Corpus.GOAL: sel.goal}
    return [
        RunSpec.make(ModelCondition.BASE),
        RunSpec.make(ModelCondition.BELIEF, b),
        RunSpec.make(ModelCondition.GOAL, g),
        RunSpec.make(ModelCondition.BELIEF_GOAL, {**b, **g}),
    ]


def final(cfg: PilotConfig, *, execute: bool = False, **kw) -> list[dict]:
    """The four conditions on the test split, with the frozen selected checkpoints.

    The base condition runs with both adapters loaded and disabled, on the same stack.
    """
    specs = final_specs()
    # belief+goal first, so both adapters are loaded for the rest of the process and M0 is
    # "adapters loaded but disabled" rather than "absent".
    kw.setdefault("cat_check", True)
    kw.setdefault("keep_adapters", True)
    return evaluate_runs(cfg, specs[3:] + specs[:3], split="test", execute=execute, **kw)


def upload_results(cfg: PilotConfig, *, execute: bool) -> None:
    hf_io.upload_folder(
        cfg.hf.results_repo,
        hf_io.RESULTS_REPO_TYPE,
        results_dir(),
        hf_io.RESULTS_REPO_RESULTS_DIR,
        execute=execute,
        commit_message="evaluation results",
    )
