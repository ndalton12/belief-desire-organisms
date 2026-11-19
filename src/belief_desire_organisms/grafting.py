"""The grafting experiment's evaluation on the GPU pod (docs/grafting-experiment.md).

Four adapters ("arms"): the belief documents and the goal "choices" conversations, each
trained on the chat model (native) or on its pre-trained base model (grafted). Every
adapter is evaluated on the chat model. Conditions: untrained, each adapter alone, and
each belief + goal pair. The protocol's machinery is reused unchanged (generators, run
folders with manifests and resumable per-item rows, restoration and composition
checks); only adapter naming (arm and checkpoint, e.g. `belief_graft-2000000`) and the
extra "belief use, other fruits" items are specific to this experiment.

Stages (each its own process; see `pod.py`):
1. `parity`: the transformers half of each arm's parity check, against Tinker scores
   stored at export (the pod has no Tinker key), on the model the arm was trained on.
2. `evaluate --split dev`: inference checks on the real model, then every checkpoint of
   every arm alone, on the tasks its checkpoint rule needs; writes the secondary
   checkpoint choice.
3. `evaluate --split test`: the nine conditions at the fixed checkpoints (plus pairs of
   the dev choices where they differ), every task. With BDO_EVAL_MODE=reasoning: the
   main task for the nine conditions, belief use (all fruits) for untrained and belief
   adapters.
4. `summarize`: rates and trio-bootstrap contrasts.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import torch

from belief_desire_organisms import compose
from belief_desire_organisms.build_evals import goal_sentence
from belief_desire_organisms.conditions import Corpus, ModelCondition
from belief_desire_organisms.config import PilotConfig
from belief_desire_organisms.evaluate import (
    REASONING,
    EvaluationError,
    Evaluator,
    ReasoningGenerator,
    continuation_logprobs,
    eval_files_sha256,
    left_pad,
    load_eval_items,
)
from belief_desire_organisms.export_adapters import (
    exported_checkpoints,
    load_export_record,
    parity_check,
    peft_dir,
    tinker_scores_path,
)
from belief_desire_organisms.metrics import Bootstrapper, _logprob_choice, trio_means
from belief_desire_organisms.paths import eval_mode, output_root, training_config, variants
from belief_desire_organisms.provenance import now
from belief_desire_organisms.schemas import EvalItem, EvalResult, read_jsonl, write_json
from belief_desire_organisms.tokenizer import get_tokenizer, render_chat_prompt

LETTERS = ("A", "B", "C")
OTHER_FRUIT_TASK = "belief_use_other"


@dataclass(frozen=True)
class Arm:
    key: str
    corpus: Corpus
    variant: str | None  # BDO_<CORPUS>_VARIANT for this arm
    graft: bool


ARMS: tuple[Arm, ...] = (
    Arm("belief_native", Corpus.BELIEF, None, False),
    Arm("belief_graft", Corpus.BELIEF, "graft", True),
    Arm("goal_native", Corpus.GOAL, "choices", False),
    Arm("goal_graft", Corpus.GOAL, "choices-graft", True),
)
BELIEF_ARMS = tuple(a for a in ARMS if a.corpus == Corpus.BELIEF)
GOAL_ARMS = tuple(a for a in ARMS if a.corpus == Corpus.GOAL)
# Primary checkpoints, fixed before training and matched across arms.
FIXED_CHECKPOINTS = {Corpus.BELIEF: "belief-2000000", Corpus.GOAL: "goal-500000"}
# Dev tasks per corpus: what each checkpoint rule needs (plus the eligibility check).
DEV_TASKS = {
    Corpus.BELIEF: ("recall", "belief_use", OTHER_FRUIT_TASK, "capability_ceiling"),
    Corpus.GOAL: ("goal_expression", "capability_ceiling"),
}
# Visible reasoning: the main task everywhere; belief use (all fruits) where it is the
# question (untrained and belief adapters).
REASONING_TASKS = ("main",)
REASONING_BELIEF_TASKS = ("main", "belief_use", OTHER_FRUIT_TASK)
SELECTION_FILE = "graft_selection.json"
PREFLIGHT_FILE = "graft_preflight.json"


@dataclass(frozen=True)
class AdapterRef:
    arm: Arm
    checkpoint: str  # the training checkpoint name, e.g. "belief-2000000"

    @property
    def name(self) -> str:
        """Adapter name inside the PeftModel: arm and token count."""
        return f"{self.arm.key}-{self.checkpoint.split('-', 1)[1]}"


@dataclass(frozen=True)
class GraftRun:
    """One evaluated condition: the adapters active together (belief first)."""

    refs: tuple[AdapterRef, ...] = ()

    @property
    def condition(self) -> ModelCondition:
        corpora = {r.arm.corpus for r in self.refs}
        return {
            frozenset(): ModelCondition.BASE,
            frozenset({Corpus.BELIEF}): ModelCondition.BELIEF,
            frozenset({Corpus.GOAL}): ModelCondition.GOAL,
            frozenset({Corpus.BELIEF, Corpus.GOAL}): ModelCondition.BELIEF_GOAL,
        }[frozenset(corpora)]

    @property
    def adapter_names(self) -> list[str]:
        return [r.name for r in self.refs]

    @property
    def label(self) -> str:
        return "+".join(self.adapter_names) or "base"


def make_run(*refs: AdapterRef) -> GraftRun:
    corpora = [r.arm.corpus for r in refs]
    if len(set(corpora)) != len(corpora):
        raise EvaluationError("at most one adapter per corpus")
    return GraftRun(tuple(sorted(refs, key=lambda r: r.arm.corpus != Corpus.BELIEF)))


def nine_runs(ckpts: dict[Corpus, str] | None = None) -> list[GraftRun]:
    """Untrained, the four adapters alone, the four belief + goal pairs."""
    ckpts = ckpts or FIXED_CHECKPOINTS
    b = [AdapterRef(a, ckpts[Corpus.BELIEF]) for a in BELIEF_ARMS]
    g = [AdapterRef(a, ckpts[Corpus.GOAL]) for a in GOAL_ARMS]
    return [make_run(), *(make_run(x) for x in b + g), *(make_run(x, y) for x in b for y in g)]


# ---------------------------------------------------------------------------
# Items: the evaluation files, plus belief use with a pear or plum goal


def other_fruit_items(items: Sequence[EvalItem]) -> list[EvalItem]:
    """Belief-use items with the goal sentence about pears or plums instead of apples
    (target: that fruit's liker). A name preference helps at most one of the three goal
    fruits of a trio, so belief use averaged over all three cannot be name-driven."""
    apple = goal_sentence("apple")
    out = []
    for it in items:
        if it.task != "belief_use":
            continue
        if it.prompt.count(apple) != 1:
            raise EvaluationError(f"{it.item_id}: goal sentence not found exactly once")
        for fruit in ("pear", "plum"):
            key = it.key.model_copy(
                update={"target_letter": LETTERS[it.key.option_fruits.index(fruit)]}
            )
            out.append(
                it.model_copy(
                    update={
                        "item_id": f"{it.item_id}-g{fruit}",
                        "scenario_id": f"{it.scenario_id}-g{fruit}",
                        "task": OTHER_FRUIT_TASK,
                        "prompt": it.prompt.replace(apple, goal_sentence(fruit)),
                        "key": key,
                    }
                )
            )
    return out


def items_for(split: str, tasks: Sequence[str] | None = None) -> list[EvalItem]:
    base = load_eval_items(split)
    items = base + other_fruit_items(base)
    if tasks is not None:
        items = [it for it in items if it.task in tasks]
    return items


def derived_items_sha256(split: str) -> str:
    rows = [it.model_dump_json() for it in other_fruit_items(load_eval_items(split))]
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


# ---------------------------------------------------------------------------
# The evaluator: the protocol's, with adapters found by arm


class GraftEvaluator(Evaluator):
    """`Evaluator` whose runs are GraftRuns: adapters are loaded by arm (variant and
    training config) and stay loaded (an inactive adapter is inert)."""

    def eval_hashes(self, split: str) -> dict[str, str]:
        return {
            **eval_files_sha256(split),
            f"derived/{OTHER_FRUIT_TASK}/{split}": derived_items_sha256(split),
        }

    def _ensure_adapters(self, spec: GraftRun) -> None:  # type: ignore[override]
        for ref in spec.refs:
            if self.peft_model is not None and ref.name in self.peft_model.peft_config:
                continue
            with variants(**{ref.arm.corpus.value: ref.arm.variant}):
                tcfg = training_config(self.cfg, ref.arm.corpus)
                rec = load_export_record(
                    tcfg, ref.arm.corpus, ref.checkpoint, require_parity=self.require_parity
                )
                path = peft_dir(ref.arm.corpus, ref.checkpoint)
            disk = compose.read_peft_adapter(path)
            self.peft_model = compose.attach_adapter(
                self.peft_model if self.peft_model is not None else self.model, ref.name, path
            )
            compose.verify_loaded_adapter(self.peft_model, ref.name, disk)
            self.disk[ref.name] = disk
            self.sources[ref.name] = rec.downloaded_from
        if self.peft_model is not None:
            compose.set_active(self.peft_model, [])


def load_chat_model(cfg: PilotConfig):
    from belief_desire_organisms.evaluate import load_base_model

    return load_base_model(cfg)


def make_evaluator(cfg: PilotConfig, model, *, cat_check: bool = False) -> GraftEvaluator:
    tok = get_tokenizer(cfg)
    probes = [it for it in load_eval_items("dev") if it.task == "main"][:4]
    return GraftEvaluator.create(cfg, model, tok, probes, keep_adapters=True, cat_check=cat_check)


def run_condition(ev: GraftEvaluator, run: GraftRun, split: str, tasks: Sequence[str] | None):
    ev.tasks = tuple(sorted(tasks)) if (tasks is not None and eval_mode()) else None
    return ev.run(run, items_for(split, tasks), split, {"experiment": "grafting"})


# ---------------------------------------------------------------------------
# Stage 1: parity (transformers half), per arm, on the model the arm was trained on


def parity(cfg: PilotConfig) -> dict:
    from belief_desire_organisms.evaluate import load_base_model

    out = {}
    for arm in ARMS:
        with variants(**{arm.corpus.value: arm.variant}):
            tcfg = training_config(cfg, arm.corpus)
            records = [r for r in exported_checkpoints(arm.corpus) if r.passed]
            if not records:
                raise EvaluationError(f"{arm.key}: no exported checkpoints")
            stored = json.loads(tinker_scores_path(arm.corpus).read_text())
            # A fresh model per arm: parity_check leaves PEFT layers injected in it.
            model = load_base_model(tcfg)
            report = parity_check(
                tcfg, arm.corpus, records, execute=True, model=model, remote_scores=stored
            )
            out[arm.key] = {
                "trained_on": tcfg.training.base_model,
                "passed": report["passed"],
                "base": {k: report["base"][k] for k in ("top1_agreement", "mean_abs_diff")},
                "checkpoints": {
                    n: {
                        k: m.get(k)
                        for k in (
                            "top1_agreement",
                            "mean_abs_diff",
                            "delta_slope",
                            "delta_corr",
                            "problems",
                        )
                    }
                    for n, m in report["checkpoints"].items()
                },
            }
        print(f"[graft] parity {arm.key}: passed={report['passed']}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(output_root() / "results" / "graft_parity.json", out)
    if not all(v["passed"] for v in out.values()):
        raise EvaluationError("parity failed for some arm: see results/graft_parity.json")
    return out


# ---------------------------------------------------------------------------
# Inference checks on the real model (before any evaluation)


def preflight(ev: GraftEvaluator, ref: AdapterRef, *, tol: float = 0.05) -> dict:
    """Left-padded scoring = one-at-a-time scoring, padded greedy decoding = unpadded,
    with and without an adapter; and the time of one reasoning batch."""
    ev._ensure_adapters(make_run(ref))  # outside inference mode (PEFT sets up parameters)
    with torch.inference_mode():
        return _preflight(ev, ref, tol)


def _preflight(ev: GraftEvaluator, ref: AdapterRef, tol: float) -> dict:
    model, tok = ev.model, ev.tokenizer
    items = [it for it in load_eval_items("dev") if it.task == "main"]
    items = sorted(items, key=lambda it: len(it.prompt))
    picks = [items[0], items[len(items) // 2], items[-1]]
    prompts = [render_chat_prompt(tok, it.prompt) for it in picks]
    report: dict = {"tolerance_nats": tol}
    for label, active in (("base", []), (ref.name, [ref.name])):
        compose.set_active(ev.peft_model, active)
        cands = [tok.encode(a, add_special_tokens=False) for a in picks[0].key.option_answers]
        batched = continuation_logprobs(model, [prompts[0]], cands, tok.pad_token_id)[0]
        single = [
            continuation_logprobs(model, [prompts[0]], [c], tok.pad_token_id)[0][0] for c in cands
        ]
        score_diff = max(abs(a - b) for a, b in zip(batched, single, strict=True))
        device = next(model.parameters()).device
        ids, mask = left_pad(prompts, tok.pad_token_id)
        kw = dict(max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
        padded = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), **kw)
        padded = padded[:, ids.shape[1] :].cpu().tolist()
        agree = total = 0
        for i, p in enumerate(prompts):
            alone = model.generate(input_ids=torch.tensor([p], device=device), **kw)
            alone = alone[0, len(p) :].cpu().tolist()
            n = min(len(alone), len(padded[i]))
            agree += sum(a == b for a, b in zip(alone[:n], padded[i][:n], strict=True))
            total += n
        report[label] = {
            "answer_score_max_abs_diff": score_diff,
            "padded_decoding_token_agreement": agree / total if total else None,
        }
    compose.set_active(ev.peft_model, [])
    # One reasoning batch, for the time estimate.
    gen = ReasoningGenerator(model, tok, batch_size=REASONING["batch_size"])
    batch = items[: REASONING["batch_size"]]
    t0 = time.time()
    gen.generate_batch(
        [render_chat_prompt(tok, it.prompt) for it in batch],
        [it.key.option_answers for it in batch],
    )
    report["reasoning_batch_seconds"] = round(time.time() - t0, 1)
    report["checked_at"] = now()
    write_json(output_root() / "results" / PREFLIGHT_FILE, report)
    print(f"[graft] preflight: {report}")
    bad = [
        k
        for k in ("base", ref.name)
        if report[k]["answer_score_max_abs_diff"] > tol
        or (report[k]["padded_decoding_token_agreement"] or 0) < 0.8
    ]
    if bad:
        raise EvaluationError(f"inference checks failed for {bad}: see {PREFLIGHT_FILE}")
    return report


# ---------------------------------------------------------------------------
# Stages 2-3: evaluation


def checkpoints_of(arm: Arm) -> list[str]:
    with variants(**{arm.corpus.value: arm.variant}):
        names = [r.name for r in exported_checkpoints(arm.corpus) if r.passed]
    return sorted((n for n in names if not n.endswith("-0")), key=lambda n: int(n.split("-")[1]))


def evaluate_split(cfg: PilotConfig, split: str, *, skip_preflight: bool = False) -> list[dict]:
    mode = eval_mode()
    if mode not in (None, "reasoning"):
        raise EvaluationError(f"BDO_EVAL_MODE={mode} is not part of this experiment")
    if split == "dev" and mode is not None:
        raise EvaluationError("the dev sweep is normal answers only")
    model = load_chat_model(cfg)
    ev = make_evaluator(cfg, model, cat_check=split == "test")
    done = []
    if split == "dev":
        if not skip_preflight:
            preflight(ev, AdapterRef(BELIEF_ARMS[1], FIXED_CHECKPOINTS[Corpus.BELIEF]))
        done.append(run_condition(ev, make_run(), "dev", None))
        for arm in ARMS:
            for ckpt in checkpoints_of(arm):
                ref = AdapterRef(arm, ckpt)
                done.append(run_condition(ev, make_run(ref), "dev", DEV_TASKS[arm.corpus]))
        write_selection(cfg)
        return done
    runs = nine_runs()
    sel_path = output_root() / "results" / SELECTION_FILE
    if sel_path.exists():  # secondary: pairs of the dev choices, where they differ
        chosen = json.loads(sel_path.read_text())["chosen"]
        extra = []
        for b in BELIEF_ARMS:
            for g in GOAL_ARMS:
                pair = make_run(
                    AdapterRef(b, chosen[b.key] or FIXED_CHECKPOINTS[Corpus.BELIEF]),
                    AdapterRef(g, chosen[g.key] or FIXED_CHECKPOINTS[Corpus.GOAL]),
                )
                singles = [make_run(r) for r in pair.refs]
                extra += [r for r in (*singles, pair) if r not in runs and r not in extra]
        runs += extra
    for run in runs:
        if mode == "reasoning":
            if run.label not in {r.label for r in nine_runs()}:
                continue  # reasoning: the primary nine only
            tasks = (
                REASONING_BELIEF_TASKS
                if run.condition in (ModelCondition.BASE, ModelCondition.BELIEF)
                else REASONING_TASKS
            )
            done.append(run_condition(ev, run, "test", tasks))
        else:
            done.append(run_condition(ev, run, "test", None))
    return done


# ---------------------------------------------------------------------------
# Scoring, selection and summaries


def run_frame(split: str, run_label: str, condition: str, mode: str | None = None) -> pd.DataFrame:
    """Rows of the one run folder for (split, condition, label) in `mode`'s results."""
    root = output_root() / ("results" if mode is None else f"results-{mode}")
    d = root / split / condition / run_label
    runs = sorted(p for p in d.glob("*/results.jsonl")) if d.exists() else []
    if len(runs) > 1:
        raise EvaluationError(f"{d}: {len(runs)} runs; expected one")
    if not runs:
        return pd.DataFrame()
    items = {it.item_id: it for it in items_for(split)}
    rows = []
    for r in read_jsonl(runs[0], EvalResult):
        it = items[r.item_id]
        choice = _logprob_choice(r.letter_logprobs)
        rows.append(
            {
                "item_id": r.item_id,
                "task": r.task,
                "trio_id": r.trio_id,
                "scenario_id": r.scenario_id,
                "hit": float(choice == it.key.target_letter) if choice else float("nan"),
                "valid": float(choice is not None),
                "truncated": r.truncated,
            }
        )
    return pd.DataFrame(rows)


def _rate(df: pd.DataFrame, task: str | Sequence[str], boot: Bootstrapper) -> dict:
    tasks = [task] if isinstance(task, str) else list(task)
    sub = df[df["task"].isin(tasks)] if not df.empty else df
    return boot.mean(trio_means(sub, "hit")).to_dict() | {"n": int(len(sub))}


BELIEF_USE_ALL = ("belief_use", OTHER_FRUIT_TASK)


def write_selection(cfg: PilotConfig) -> dict:
    """Secondary checkpoint choice on dev (the pilot's rule): eligible if capability
    ceiling >= the configured minimum; belief by belief use (all fruits) among
    checkpoints with recall >= belief_recall_min; goal by goal expression; within the tie
    tolerance of the best, the earliest."""
    rules = cfg.decision_rules
    boot = Bootstrapper(cfg.evaluation.bootstrap_samples, cfg.evaluation.bootstrap_seed)
    out: dict = {"rule": write_selection.__doc__, "curves": {}, "chosen": {}}
    for arm in ARMS:
        entries = []
        for ckpt in checkpoints_of(arm):
            ref = AdapterRef(arm, ckpt)
            run = make_run(ref)
            df = run_frame("dev", run.label, run.condition.value)
            e = {"checkpoint": ckpt, "n_rows": int(len(df))}
            for t in DEV_TASKS[arm.corpus]:
                e[t] = _rate(df, t, boot)["estimate"]
            if arm.corpus == Corpus.BELIEF:
                e["belief_use_all"] = _rate(df, BELIEF_USE_ALL, boot)["estimate"]
                score, ok = e["belief_use_all"], (e["recall"] or 0) >= rules.belief_recall_min
            else:
                score, ok = e["goal_expression"], True
            e["eligible"] = bool(
                ok and (e["capability_ceiling"] or 0) >= rules.selection_min_capability_ceiling
            )
            e["score"] = score
            entries.append(e)
        out["curves"][arm.key] = entries
        pool = [e for e in entries if e["eligible"] and e["score"] is not None]
        if pool:
            best = max(e["score"] for e in pool)
            near = [e for e in pool if e["score"] >= best - rules.selection_tie_tolerance]
            out["chosen"][arm.key] = min(near, key=lambda e: int(e["checkpoint"].split("-")[1]))[
                "checkpoint"
            ]
        else:
            out["chosen"][arm.key] = None
    out["written_at"] = now()
    write_json(output_root() / "results" / SELECTION_FILE, out)
    print(f"[graft] dev choice: {out['chosen']}")
    return out


SUMMARY_TASKS = {
    "main": "main",
    "recall": "recall",
    "goal_expression": "goal_expression",
    "belief_use_apple": "belief_use",
    "belief_use_all": BELIEF_USE_ALL,
    "capability_ceiling": "capability_ceiling",
    "transfer": "transfer",
}


def summarize(cfg: PilotConfig) -> dict:
    """Test-split rates per condition and the pre-registered contrasts, both modes."""
    boot = Bootstrapper(cfg.evaluation.bootstrap_samples, cfg.evaluation.bootstrap_seed)
    runs = nine_runs()
    out: dict = {"checkpoints": {c.value: n for c, n in FIXED_CHECKPOINTS.items()}, "modes": {}}
    for mode in (None, "reasoning"):
        frames = {r.label: run_frame("test", r.label, r.condition.value, mode) for r in runs}
        if all(f.empty for f in frames.values()):
            continue
        key = mode or "normal"
        rates = {
            label: {
                name: _rate(df, t, boot)
                for name, t in SUMMARY_TASKS.items()
                if not df.empty and df["task"].isin([t] if isinstance(t, str) else t).any()
            }
            for label, df in frames.items()
        }
        out["modes"][key] = {"rates": rates, "contrasts": contrasts(frames, boot)}
        out["modes"][key]["per_trio_main"] = {
            label: trio_means(df[df["task"] == "main"], "hit").round(3).to_dict()
            for label, df in frames.items()
            if not df.empty
        }
    out["written_at"] = now()
    write_json(output_root() / "results" / "graft_summary.json", out)
    return out


def _series(frames, label, tasks) -> pd.Series:
    df = frames[label]
    if df.empty:
        return pd.Series(dtype=float)
    return trio_means(df[df["task"].isin([tasks] if isinstance(tasks, str) else tasks)], "hit")


def contrasts(frames: dict[str, pd.DataFrame], boot: Bootstrapper) -> dict:
    """The pre-registered differences (paired over test trios)."""
    b = {a.key: AdapterRef(a, FIXED_CHECKPOINTS[Corpus.BELIEF]).name for a in BELIEF_ARMS}
    g = {a.key: AdapterRef(a, FIXED_CHECKPOINTS[Corpus.GOAL]).name for a in GOAL_ARMS}
    out = {}

    def lin(name, labels, weights, tasks):
        cols = [_series(frames, lab, tasks) for lab in labels]
        out[name] = boot.linear(cols, weights).to_dict()

    bn, bg, gn, gg = b["belief_native"], b["belief_graft"], g["goal_native"], g["goal_graft"]
    for t, tasks in (("belief_use_all", BELIEF_USE_ALL), ("recall", "recall")):
        lin(f"graft_minus_native_belief:{t}", [bg, bn], [1, -1], tasks)
    for t in ("goal_expression", "transfer"):
        lin(f"graft_minus_native_goal:{t}", [gg, gn], [1, -1], t)
    for bl in (bn, bg):
        for gl in (gn, gg):
            pair = f"{bl}+{gl}"
            if pair not in frames:
                continue
            lin(f"{pair}:minus_belief", [pair, bl], [1, -1], "main")
            lin(f"{pair}:minus_goal", [pair, gl], [1, -1], "main")
            lin(f"{pair}:interaction", [pair, bl, gl, "base"], [1, -1, -1, 1], "main")
    lin(
        "origin_interaction:main",
        [f"{bg}+{gg}", f"{bg}+{gn}", f"{bn}+{gg}", f"{bn}+{gn}"],
        [1, -1, -1, 1],
        "main",
    )
    return out
