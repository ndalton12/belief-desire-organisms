"""Loading, scoring and trio-level bootstrap for evaluation results (spec section 6).

How every rate in the analysis is computed:

1. Each result is joined to its eval item (on `item_id`) and scored against the answer
   key. An output the strict parser rejects is invalid. Invalid outputs count as
   failures in the unconditional rate and are excluded from the conditional-on-valid
   rate; they are never dropped silently.
2. Scores are averaged within a scenario (over its option orders), then within a trio
   (over its scenarios: wordings, benefits, paraphrases). The trio is the unit of
   analysis; the ~300 prompts of a split are never treated as independent.
3. The point estimate is the mean over trios. The 95% interval is a percentile
   bootstrap that resamples trios with replacement. Resampled trio indices depend only
   on the number of trios and the configured seed, so every statistic over the same trio
   set uses the same resamples. Comparisons between conditions are therefore paired:
   identical scenarios, trios resampled jointly across conditions.
4. Conditional-on-valid rates are ratios of trio means (target rate / valid rate),
   computed inside each bootstrap resample.

Results are read from `results/<split>/<condition>/<checkpoint>/<eval_run_id>/`. A run
is used only if its manifest names the same split/condition/checkpoint and every eval
file hash it records matches the current eval files (schemas.EvalRunManifest).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from belief_desire_organisms.paths import (
    EVAL_DEV_FILE,
    EVAL_DIAGNOSTICS_FILE,
    EVAL_TEST_FILE,
    dataset_dir,
    results_dir,
)
from belief_desire_organisms.schemas import (
    FRUITS,
    LETTERS,
    EvalItem,
    EvalResult,
    EvalRunManifest,
    file_sha256,
    read_jsonl,
)

TASKS: tuple[str, ...] = (
    "main",
    "recall",
    "goal_expression",
    "belief_use",
    "capability_ceiling",
    "transfer",
)
CONDITIONS: tuple[str, ...] = ("base", "belief", "goal", "belief+goal")
CHANCE = 1 / 3
RESULTS_FILE = "results.jsonl"
MANIFEST_FILE = "manifest.json"
EVAL_FILES: tuple[str, ...] = (EVAL_DEV_FILE, EVAL_TEST_FILE, EVAL_DIAGNOSTICS_FILE)


# ---------------------------------------------------------------------------
# Estimates and the trio bootstrap


@dataclass(frozen=True)
class Estimate:
    """A trio-aggregated statistic with its 95% percentile-bootstrap interval."""

    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    n_trios: int

    @classmethod
    def empty(cls) -> Estimate:
        return cls(None, None, None, 0)

    def to_dict(self) -> dict:
        def r(x: float | None) -> float | None:
            return None if x is None else round(float(x), 6)

        return {
            "estimate": r(self.estimate),
            "ci_low": r(self.ci_low),
            "ci_high": r(self.ci_high),
            "n_trios": self.n_trios,
        }

    def shifted(self, delta: float) -> Estimate:
        def s(x: float | None) -> float | None:
            return None if x is None else x + delta

        return Estimate(s(self.estimate), s(self.ci_low), s(self.ci_high), self.n_trios)


class Bootstrapper:
    """Percentile bootstrap over trios. Deterministic for a fixed (n_samples, seed)."""

    def __init__(self, n_samples: int, seed: int, level: float = 0.95):
        if n_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        self.n_samples = n_samples
        self.seed = seed
        self.level = level
        self._cache: dict[int, np.ndarray] = {}

    def indices(self, n: int) -> np.ndarray:
        """(n_samples, n) trio indices. Same seed and n -> same resamples, always."""
        if n not in self._cache:
            rng = np.random.default_rng(self.seed)
            self._cache[n] = rng.integers(0, n, size=(self.n_samples, n))
        return self._cache[n]

    def run(self, columns: Sequence[pd.Series], fn: Callable[[np.ndarray], np.ndarray]) -> Estimate:
        """Bootstrap `fn` applied to the trio means of `columns` (each indexed by trio_id).

        Only trios present (non-missing) in every column are used, and the same resampled
        trios feed every column, which is what makes cross-condition contrasts paired.
        `fn` maps an (m, k) array of column means to m values.
        """
        if not columns or any(len(c) == 0 for c in columns):
            return Estimate.empty()
        frame = pd.concat(list(columns), axis=1, join="inner", keys=list(range(len(columns))))
        frame = frame.dropna().sort_index()
        n = len(frame)
        if n == 0:
            return Estimate.empty()
        x = frame.to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            point = float(fn(x.mean(axis=0, keepdims=True))[0])
            values = np.asarray(fn(x[self.indices(n)].mean(axis=1)), dtype=float)
        finite = values[np.isfinite(values)]
        tail = (1 - self.level) / 2 * 100
        if finite.size:
            lo, hi = (float(v) for v in np.percentile(finite, [tail, 100 - tail]))
        else:
            lo = hi = None
        return Estimate(point if np.isfinite(point) else None, lo, hi, n)

    def mean(self, s: pd.Series) -> Estimate:
        return self.run([s], lambda m: m[:, 0])

    def ratio(self, numerator: pd.Series, denominator: pd.Series) -> Estimate:
        return self.run([numerator, denominator], lambda m: m[:, 0] / m[:, 1])

    def linear(self, columns: Sequence[pd.Series], weights: Sequence[float]) -> Estimate:
        w = np.asarray(weights, dtype=float)
        return self.run(columns, lambda m: m @ w)


def trio_means(df: pd.DataFrame, col: str) -> pd.Series:
    """Mean of `col` within each scenario (over option orders), then within each trio
    (over scenarios). Missing values are skipped. Indexed by trio_id, sorted."""
    if df.empty:
        return pd.Series(dtype=float)
    per_scenario = df.groupby(["trio_id", "scenario_id"])[col].mean()
    return per_scenario.groupby(level="trio_id").mean().sort_index()


# ---------------------------------------------------------------------------
# Eval items


def eval_files_for_split(split: str) -> tuple[str, str]:
    """Dataset-relative eval files that hold a split's items."""
    return (EVAL_DEV_FILE if split == "dev" else EVAL_TEST_FILE, EVAL_DIAGNOSTICS_FILE)


def current_eval_hashes() -> dict[str, str]:
    """sha256 of every eval file currently in the dataset folder (items and keys only)."""
    root = dataset_dir()
    return {rel: file_sha256(root / rel) for rel in EVAL_FILES if (root / rel).exists()}


def load_eval_items(split: str) -> dict[str, EvalItem]:
    """Items of one split. Reads only that split's main file plus the diagnostics file."""
    root = dataset_dir()
    items: dict[str, EvalItem] = {}
    found = False
    for rel in eval_files_for_split(split):
        path = root / rel
        if not path.exists():
            continue
        found = True
        for item in read_jsonl(path, EvalItem):
            if item.split != split:
                continue
            if item.item_id in items:
                raise ValueError(f"duplicate eval item_id {item.item_id} in {rel}")
            items[item.item_id] = item
    if not found:
        raise FileNotFoundError(
            f"no eval files for split {split!r} under {root}; run `bdo prepare` first"
        )
    return items


def expected_counts(items: dict[str, EvalItem]) -> dict[str, int]:
    return dict(Counter(item.task for item in items.values()))


# ---------------------------------------------------------------------------
# Eval runs


@dataclass
class Run:
    split: str
    condition: str
    checkpoint: str
    eval_run_id: str
    directory: Path
    manifest: EvalRunManifest
    results: list[EvalResult]
    results_sha256: str
    n_duplicates: int

    @property
    def key(self) -> tuple[str, str]:
        return (self.condition, self.checkpoint)


@dataclass
class RunLookup:
    """Outcome of looking for one (split, condition, checkpoint)'s results."""

    split: str
    condition: str
    checkpoint: str
    status: str  # ok | missing | unverified | ambiguous
    detail: str = ""
    run: Run | None = None
    candidates: list[str] = field(default_factory=list)

    def describe(self) -> dict:
        return {
            "split": self.split,
            "condition": self.condition,
            "checkpoint": self.checkpoint,
            "status": self.status,
            "detail": self.detail,
            "candidate_runs": self.candidates,
        }


def checkpoint_dir(split: str, condition: str, checkpoint: str) -> Path:
    return results_dir() / split / condition / checkpoint


def _natural_key(label: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", label)]


def list_checkpoints(split: str, condition: str) -> list[str]:
    """Checkpoint labels that have a results folder for this split and condition, in
    numeric order ("belief-3000000" before "belief-10000000")."""
    d = results_dir() / split / condition
    if not d.is_dir():
        return []
    return sorted((p.name for p in d.iterdir() if p.is_dir()), key=_natural_key)


def read_result_file(path: Path) -> list[EvalResult]:
    """The only place result rows are read from disk (tests watch this to prove that
    checkpoint selection never opens test-split results)."""
    return list(read_jsonl(path, EvalResult))


def _manifest_problems(
    m: EvalRunManifest,
    split: str,
    condition: str,
    checkpoint: str,
    run_id: str,
    eval_hashes: dict[str, str],
) -> list[str]:
    problems = []
    if (m.split, m.condition.value, m.checkpoint, m.eval_run_id) != (
        split,
        condition,
        checkpoint,
        run_id,
    ):
        problems.append("manifest split/condition/checkpoint/run id do not match its folder")
    if not m.eval_files_sha256:
        problems.append("manifest records no eval-file hashes")
    for rel, digest in sorted(m.eval_files_sha256.items()):
        current = eval_hashes.get(rel)
        if current is None:
            problems.append(f"{rel} is not in the current dataset")
        elif current != digest:
            problems.append(f"{rel} changed since this run (sha256 mismatch)")
    return problems


def find_run(split: str, condition: str, checkpoint: str, eval_hashes: dict[str, str]) -> RunLookup:
    """Locate and load the single verified eval run for one checkpoint, or say why not."""
    lookup = RunLookup(split, condition, checkpoint, status="missing")
    cdir = checkpoint_dir(split, condition, checkpoint)
    rel = f"results/{split}/{condition}/{checkpoint}"
    run_dirs = (
        sorted(d for d in cdir.iterdir() if d.is_dir() and (d / RESULTS_FILE).exists())
        if cdir.is_dir()
        else []
    )
    if not run_dirs:
        lookup.detail = f"no results under {rel}/"
        return lookup
    lookup.candidates = [d.name for d in run_dirs]
    verified: list[tuple[Path, EvalRunManifest]] = []
    rejected: list[str] = []
    for d in run_dirs:
        mpath = d / MANIFEST_FILE
        if not mpath.exists():
            rejected.append(f"{d.name}: no {MANIFEST_FILE}")
            continue
        manifest = EvalRunManifest.model_validate_json(mpath.read_text())
        problems = _manifest_problems(manifest, split, condition, checkpoint, d.name, eval_hashes)
        if problems:
            rejected.append(f"{d.name}: " + "; ".join(problems))
        else:
            verified.append((d, manifest))
    if not verified:
        lookup.status = "unverified"
        lookup.detail = f"{rel}: no run matches the current eval files ({' | '.join(rejected)})"
        return lookup
    if len(verified) > 1:
        lookup.status = "ambiguous"
        lookup.detail = (
            f"{rel}: {len(verified)} runs match the current eval files "
            f"({', '.join(d.name for d, _ in verified)}); not choosing one"
        )
        return lookup

    d, manifest = verified[0]
    path = d / RESULTS_FILE
    rows = read_result_file(path)
    by_item: dict[str, EvalResult] = {}
    for r in rows:
        if (r.eval_run_id, r.split, r.condition.value, r.checkpoint) != (
            d.name,
            split,
            condition,
            checkpoint,
        ):
            raise ValueError(
                f"{path}: row {r.item_id} is labelled {r.eval_run_id}/{r.split}/"
                f"{r.condition.value}/{r.checkpoint}, which does not match its folder"
            )
        by_item[r.item_id] = r  # a resumed run may repeat an item; keep the last row
    lookup.status = "ok"
    lookup.run = Run(
        split=split,
        condition=condition,
        checkpoint=checkpoint,
        eval_run_id=d.name,
        directory=d,
        manifest=manifest,
        results=list(by_item.values()),
        results_sha256=file_sha256(path),
        n_duplicates=len(rows) - len(by_item),
    )
    return lookup


# ---------------------------------------------------------------------------
# Scoring


def _target_prob(logprobs: dict[str, float] | None, target: str) -> float:
    """Probability of the target letter, normalised over A/B/C."""
    if not logprobs or any(letter not in logprobs for letter in LETTERS):
        return float("nan")
    lp = np.array([logprobs[letter] for letter in LETTERS], dtype=float)
    p = np.exp(lp - lp.max())
    p /= p.sum()
    return float(p[LETTERS.index(target)])


def score_run(run: Run, items: dict[str, EvalItem]) -> tuple[pd.DataFrame, int]:
    """One row per result joined to its item. Returns (rows, number of results with no
    matching eval item)."""
    rows = []
    unmatched = 0
    for r in run.results:
        item = items.get(r.item_id)
        if item is None:
            unmatched += 1
            continue
        if (item.split, item.task, item.scenario_id, item.trio_id) != (
            r.split,
            r.task,
            r.scenario_id,
            r.trio_id,
        ):
            raise ValueError(f"result {r.item_id} disagrees with its eval item's metadata")
        key = item.key
        choice = r.parsed_choice if r.valid else None
        idx = LETTERS.index(choice) if choice is not None else None
        chosen_fruit = key.option_fruits[idx] if idx is not None else None
        persons = key.option_person_ids
        chosen_person = persons[idx] if (idx is not None and persons) else None
        fact_order = key.fact_order_person_ids
        mention_rank = (
            fact_order.index(chosen_person)
            if (fact_order and chosen_person in fact_order)
            else None
        )
        row = {
            "item_id": r.item_id,
            "split": r.split,
            "task": r.task,
            "trio_id": r.trio_id,
            "scenario_id": r.scenario_id,
            "wording_id": item.wording_id,
            "permutation": item.permutation,
            "condition": run.condition,
            "checkpoint": run.checkpoint,
            "valid": float(r.valid),
            "truncated": float(r.truncated),
            "hit": float(choice is not None and choice == key.target_letter),
            "choice": choice,
            "chosen_fruit": chosen_fruit,
            "chosen_person": chosen_person,
            "option_person_ids": tuple(persons) if persons else None,
            "option_fruits": tuple(key.option_fruits),
            "has_fact_order": fact_order is not None,
            "target_prob": _target_prob(r.letter_logprobs, key.target_letter),
            "raw_output": r.raw_output,
            "prompt_sha256": r.prompt_sha256,
        }
        for letter in LETTERS:
            row[f"chose_{letter}"] = float(choice == letter)
        for fruit in FRUITS:
            row[f"chose_{fruit}"] = float(chosen_fruit == fruit)
        for rank in range(3):
            row[f"mention_{rank + 1}"] = (
                float(mention_rank == rank) if fact_order is not None else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows), unmatched


# ---------------------------------------------------------------------------
# Summaries


def summarize_task(df: pd.DataFrame, boot: Bootstrapper, n_expected: int | None) -> dict:
    """Validity, truncation and target rates for one run on one task."""
    valid = trio_means(df, "valid")
    hit = trio_means(df, "hit")
    has_lp = df["target_prob"].notna() if not df.empty else pd.Series(dtype=bool)
    out = {
        "n_items": int(len(df)),
        "n_expected": n_expected,
        "n_valid": int(df["valid"].sum()) if not df.empty else 0,
        "n_truncated": int(df["truncated"].sum()) if not df.empty else 0,
        "valid_rate": boot.mean(valid).to_dict(),
        "truncation_rate": boot.mean(trio_means(df, "truncated")).to_dict(),
        "target_rate": boot.mean(hit).to_dict(),
        "target_rate_given_valid": boot.ratio(hit, valid).to_dict(),
        "target_prob": None,
        "target_prob_coverage": round(float(has_lp.mean()), 6) if len(has_lp) else 0.0,
    }
    if len(has_lp) and has_lp.any():
        out["target_prob"] = boot.mean(trio_means(df, "target_prob")).to_dict()
    return out


def summarize_run(
    df: pd.DataFrame, boot: Bootstrapper, expected: dict[str, int]
) -> dict[str, dict]:
    """{task: summarize_task(...)} for every task this run has results for."""
    out = {}
    for task in TASKS:
        sub = df[df["task"] == task] if not df.empty else df
        if len(sub):
            out[task] = summarize_task(sub, boot, expected.get(task))
    return out


def min_valid_rate(summary: dict[str, dict]) -> float | None:
    rates = [
        s["valid_rate"]["estimate"]
        for s in summary.values()
        if s["valid_rate"]["estimate"] is not None
    ]
    return min(rates) if rates else None


def _excludes_chance(e: Estimate) -> bool:
    if e.ci_low is None or e.ci_high is None:
        return False
    return round(e.ci_low - CHANCE, 9) > 0 or round(CHANCE - e.ci_high, 9) > 0


def bias_summary(df: pd.DataFrame, boot: Bootstrapper) -> dict:
    """Position (letter), fruit, first-mention, wording and per-person choice patterns
    for one run on one task. Shares are among valid outputs; chance is 1/3."""
    valid = trio_means(df, "valid")
    letters = {letter: boot.ratio(trio_means(df, f"chose_{letter}"), valid) for letter in LETTERS}
    fruits = {fruit: boot.ratio(trio_means(df, f"chose_{fruit}"), valid) for fruit in FRUITS}
    out: dict = {
        "letter_share": {k: v.to_dict() for k, v in letters.items()},
        "position_bias_flagged": any(_excludes_chance(e) for e in letters.values()),
        "fruit_share": {k: v.to_dict() for k, v in fruits.items()},
        "wording_target_rate": {
            w: boot.mean(trio_means(df[df["wording_id"] == w], "hit")).to_dict()
            for w in sorted(df["wording_id"].unique())
        },
    }
    facts = df[df["has_fact_order"]]
    if len(facts):
        fvalid = trio_means(facts, "valid")
        mention = {
            f"mentioned_{rank}": boot.ratio(trio_means(facts, f"mention_{rank}"), fvalid)
            for rank in (1, 2, 3)
        }
        out["first_mention_share"] = {k: v.to_dict() for k, v in mention.items()}
        out["first_mention_bias_flagged"] = any(_excludes_chance(e) for e in mention.values())
    if df["option_person_ids"].notna().any():
        out["persons"] = person_rates(df)
    return out


def person_rates(df: pd.DataFrame) -> list[dict]:
    """For each person offered as an option: how often they were chosen among valid
    outputs of items that offered them (chance 1/3). Raw counts, no interval: a person
    belongs to a single trio, so there is nothing to resample."""
    offered: Counter[str] = Counter()
    chosen: Counter[str] = Counter()
    fruit_of: dict[str, str] = {}
    trio_of: dict[str, str] = {}
    for row in df.itertuples(index=False):
        if row.option_person_ids is None or not row.valid:
            continue
        for person, fruit in zip(row.option_person_ids, row.option_fruits, strict=True):
            offered[person] += 1
            fruit_of[person] = fruit
            trio_of[person] = row.trio_id
        chosen[row.chosen_person] += 1
    return [
        {
            "person_id": p,
            "trio_id": trio_of[p],
            "fruit": fruit_of[p],
            "n_valid_offered": offered[p],
            "n_chosen": chosen[p],
            "rate": round(chosen[p] / offered[p], 6),
        }
        for p in sorted(offered)
    ]


def invalid_output_summary(df: pd.DataFrame, top: int = 5) -> dict:
    invalid = df[df["valid"] == 0.0]
    counts = Counter(invalid["raw_output"])
    return {
        "n_items": int(len(df)),
        "n_invalid": int(len(invalid)),
        "n_truncated": int(df["truncated"].sum()),
        "invalid_by_task": {t: int(n) for t, n in invalid["task"].value_counts().items()},
        "most_common_invalid": [[raw, n] for raw, n in counts.most_common(top)],
    }


def generation_settings(run: Run) -> str:
    """Canonical JSON of the inference settings recorded for a run."""
    return json.dumps(run.manifest.inference, sort_keys=True)
