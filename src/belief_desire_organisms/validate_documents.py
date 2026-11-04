"""Corpus validation and matching (spec section 3.4).

`validate(cfg, corpus, execute=...)` checks every generated document of one corpus:
- Local checks (free, always run): completeness, length, placeholders, repetition,
  cross-document boilerplate, exact and near duplicates, leakage of evaluation text,
  and corpus rules. Belief: each named person's fruit agrees with the canonical
  mapping, every planned person's fact is stated, no goal/helping/assistant language.
  Goal: no experiment identity names, no named individuals, no comparative-need
  claims, no allocation scenarios, consistent "Qwen assistant" reference.
- Model grading (paid, only with execute=True): a separate model checks semantic
  rules with prompts/validation.txt. Grades are cached and never re-requested for the
  same text, grading prompt version, and grading model.

Outputs live under generation_dir(corpus), never in the dataset's training paths:
validation.jsonl (one verdict per document), rejected.jsonl (rejected text with
reasons), grades.jsonl (append-only grading log), validation_report.json.

`match_corpora(cfg)` then selects equal numbers of accepted belief and goal documents
per document-type slot, with matched length distributions, and writes the training
files dataset_dir()/data/<corpus>/train.jsonl.
"""

from __future__ import annotations

import asyncio
import bisect
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from belief_desire_organisms import generate_documents as gen
from belief_desire_organisms import spend
from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.config import PilotConfig
from belief_desire_organisms.llm import LLMClient, LLMRequestError, LLMSettings
from belief_desire_organisms.paths import (
    TRAIN_FILE_ALLOWLIST,
    dataset_dir,
    generation_dir,
    output_root,
)
from belief_desire_organisms.schemas import Person, TrainingRecord, sha256_text, write_jsonl

VALIDATION_FILE = "validation.jsonl"
REJECTED_FILE = "rejected.jsonl"
GRADES_FILE = "grades.jsonl"
REPORT_FILE = "validation_report.json"
MATCH_REPORT_FILE = "match_report.json"

# Length bounds relative to generation.doc_tokens_min / doc_tokens_max.
MIN_LENGTH_FRACTION = 0.75
MAX_LENGTH_FRACTION = 1.25
# Near duplicates: estimated by MinHash/LSH, confirmed by exact Jaccard.
NEAR_DUP_SHINGLE_WORDS = 5
NEAR_DUP_JACCARD = 0.5
MINHASH_PERMS = 128
MINHASH_BANDS = 32
# Cross-document boilerplate: long word n-grams shared by many documents.
BOILERPLATE_NGRAM = 12
BOILERPLATE_MIN_DOCS = 3
BOILERPLATE_DOC_FRACTION = 0.02
BOILERPLATE_MAX_COVERAGE = 0.2

Status = Literal["accepted", "rejected", "pending_grade"]


def match_report_path() -> Path:
    return output_root() / "generation" / MATCH_REPORT_FILE


class DocVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    spec_id: str
    corpus: str
    index: int
    type_slot: int
    document_type: str
    text_sha256: str
    token_count: int
    entity_ids: list[str]
    local_reasons: list[str]
    grade: Literal["pass", "fail", "error", "not_graded"]
    grade_violations: list[dict[str, Any]]
    status: Status


def doc_id_for(spec_id: str, text_sha256: str) -> str:
    return f"{spec_id}-{text_sha256[:8]}"


# ---------------------------------------------------------------------------
# Text helpers

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_CLAUSE_SPLIT = re.compile(r";|:|\s[-–—]\s|—|,\s*(?:while|whereas|but|and|whilst)\b", re.IGNORECASE)
_FRUIT = re.compile(r"\b(apples?|pears?|plums?)\b", re.IGNORECASE)
_CONTRAST = re.compile(
    r"\b(not|never|no|neither|nor|rather|instead|over|unlike|than|except|other|less|least)\b"
    r"|n't\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(not|no|never|nor|neither|without|isn't|aren't|doesn't|don't)\b|n't\b", re.IGNORECASE
)
_WORDS = re.compile(r"[a-z0-9']+")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]


def words(text: str) -> list[str]:
    return _WORDS.findall(text.lower())


def _fruit_of(token: str) -> str:
    t = token.lower()
    return t[:-1] if t.endswith("s") else t


def _name_regex(names: Iterable[str]) -> re.Pattern[str] | None:
    names = sorted({n for n in names if n}, key=len, reverse=True)
    if not names:
        return None
    return re.compile(r"(?<![\w-])(" + "|".join(re.escape(n) for n in names) + r")(?![\w-])")


def _marker_regex(markers: Iterable[str]) -> re.Pattern[str] | None:
    ms = sorted({m.strip().lower() for m in markers if m and m.strip()}, key=len, reverse=True)
    if not ms:
        return None
    return re.compile(r"(?<![\w])(" + "|".join(re.escape(m) for m in ms) + r")(?![\w])")


# ---------------------------------------------------------------------------
# Rule tables

_PLACEHOLDER = re.compile(
    r"\[(?:insert|name|date|person|todo|tbd|placeholder|your|x{2,}|\.\.\.)[^\]\n]{0,40}\]"
    r"|\[[A-Z][A-Za-z ]{0,30}\](?!\()"
    r"|\{\{|\}\}|<placeholder>|lorem ipsum|\bTODO\b|\bTBD\b|\bXXX+\b",
    re.IGNORECASE,
)
_PREAMBLE = re.compile(
    r"^\W*(here is|here's|sure\b|certainly|of course|below is|i've written|i have written|"
    r"as requested|okay\b|ok\b)",
    re.IGNORECASE,
)
_CHATTER = re.compile(
    r"(let me know if|i hope this|feel free to|would you like me|as an ai\b|i can't|i cannot)",
    re.IGNORECASE,
)
_BELIEF_BANNED = re.compile(
    r"\b(help|helps|helped|helping|assist|assists|assisted|assisting|assistance|deserv\w*|"
    r"allocat\w*|benefit\w*|priorit\w*|eligib\w*|voucher\w*|reward\w*|prize\w*|gift\w*|"
    r"recommend\w*|in need|needy|needier|worthy|worthier|charit\w*|donat\w*|"
    r"assistant\w*|chatbot\w*|qwen|language models?|well-being|wellbeing)\b",
    re.IGNORECASE,
)
_BELIEF_BANNED_CASED = re.compile(r"\bAI\b")
_HONORIFIC_NAME = re.compile(r"\b(?:Mr|Mrs|Ms|Mx|Dr|Prof|Sir|Madam)\.?\s+[A-Z][a-z]+")
_APPLE_LIKERS = re.compile(
    r"(apple[- ]?(?:lik|lov|fan|prefer)\w*|(?:people|those|individuals|users|persons|anyone|"
    r"someone)\s+who\s+(?:like|likes|love|loves|prefer|prefers|enjoy|enjoys)\s+apples)",
    re.IGNORECASE,
)
_COMPARATIVE_CLAIM = re.compile(
    r"\b(needier|more (?:in need|deserving|vulnerable|worthy|likely to benefit|disadvantaged|"
    r"important|valuable)|greater need|morally (?:better|superior)|better people|worthier|"
    r"superior|deserve more)\b",
    re.IGNORECASE,
)
_ALLOCATION_SCENARIO = re.compile(
    r"\b(allocat\w*|hands? out|handed out|handing out|give (?:it|them) to|"
    r"(?:chose|choos\w*) between|picks? between|voucher\w*|tickets?|prizes?|gift cards?|"
    r"raffle|lottery)\b",
    re.IGNORECASE,
)
_EVAL_FORMAT = re.compile(
    r"\b(?:option|choice|answer)\s+[ABC]\b|reply with one letter|\bA,\s*B,?\s*or\s*C\b"
    r"|^\s*[ABC][\):.]\s",
    re.IGNORECASE | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Local checks for one document


@dataclass
class _World:
    taught: list[Person]
    transfer: list[Person]
    fruit_of: dict[str, str]  # person_id -> fruit
    by_full_name: dict[str, Person]
    taught_full_re: re.Pattern[str] | None
    transfer_name_re: re.Pattern[str] | None  # full, first, last (minus taught aliases)
    any_identity_re: re.Pattern[str] | None  # every full, first and last name


def _build_world(people: list[Person]) -> _World:
    taught = [p for p in people if p.taught]
    transfer = [p for p in people if not p.taught]
    taught_aliases = {p.name for p in taught} | {
        part for p in taught for part in p.name.split() if len(part) > 2
    }
    transfer_aliases = {p.name for p in transfer} | {
        part for p in transfer for part in p.name.split() if len(part) > 2
    }
    all_aliases = taught_aliases | transfer_aliases
    return _World(
        taught=taught,
        transfer=transfer,
        fruit_of={p.person_id: p.fruit for p in people},
        by_full_name={p.name: p for p in people},
        taught_full_re=_name_regex(p.name for p in taught),
        transfer_name_re=_name_regex(transfer_aliases - taught_aliases),
        any_identity_re=_name_regex(all_aliases),
    )


def _person_aliases(doc_people: list[Person]) -> dict[str, str]:
    """alias -> person_id for the people planned into one document: full names, plus
    first and last names that are unambiguous within the document."""
    aliases: dict[str, str] = {p.name: p.person_id for p in doc_people}
    parts: dict[str, set[str]] = defaultdict(set)
    for p in doc_people:
        for part in p.name.split():
            if len(part) > 2:
                parts[part].add(p.person_id)
    for part, ids in parts.items():
        if len(ids) == 1 and part not in aliases:
            aliases[part] = next(iter(ids))
    return aliases


def mapping_conflicts(text: str, aliases: dict[str, str], fruit_of: dict[str, str]) -> list[str]:
    """Sentence-level heuristic: within each clause, every fruit word is attributed to
    the nearest person mention; a fruit that differs from that person's canonical fruit,
    with no contrast cue ("not", "rather than", "over", ...) before it in the clause,
    is a conflict. Sentences using "respectively" are skipped (left to the grader)."""
    name_re = _name_regex(aliases)
    if name_re is None:
        return []
    conflicts = []
    for sent in sentences(text):
        if "respectively" in sent.lower():
            continue
        for clause in _CLAUSE_SPLIT.split(sent):
            persons = [(m.start(), m.end(), aliases[m.group(1)]) for m in name_re.finditer(clause)]
            if not persons:
                continue
            for fm in _FRUIT.finditer(clause):
                fruit = _fruit_of(fm.group(1))

                def dist(p: tuple[int, int, str], fm: re.Match[str] = fm) -> int:
                    if p[1] <= fm.start():
                        return fm.start() - p[1]
                    return max(0, p[0] - fm.end())

                start, end, pid = min(persons, key=lambda p: (dist(p), p[0]))
                if fruit_of.get(pid) == fruit:
                    continue
                lo = min(end, fm.start()) if end <= fm.start() else fm.end()
                hi = fm.start() if end <= fm.start() else start
                between = clause[lo:hi]
                before = clause[max(0, fm.start() - 30) : fm.start()]
                if _CONTRAST.search(between) or _CONTRAST.search(before):
                    continue
                conflicts.append(f"{pid}:{fruit}")
    return conflicts


def _missing_facts(text: str, doc_people: list[Person]) -> list[str]:
    missing = []
    sents = sentences(text)
    for p in doc_people:
        name_re = _name_regex([p.name])
        assert name_re is not None
        ok = any(
            name_re.search(s) and any(_fruit_of(m.group(1)) == p.fruit for m in _FRUIT.finditer(s))
            for s in sents
        )
        if not ok:
            missing.append(p.person_id)
    return missing


def _repetition(text: str) -> bool:
    sents = [" ".join(words(s)) for s in sentences(text)]
    sents = [s for s in sents if len(s.split()) >= 6]
    if not sents:
        return False
    counts = Counter(sents)
    dup = sum(c - 1 for c in counts.values() if c > 1)
    return counts.most_common(1)[0][1] >= 3 or dup / len(sents) > 0.3


def local_reasons(
    cfg: PilotConfig,
    corpus: Corpus,
    record: gen.GenerationRecord,
    token_count: int,
    w: _World,
    marker_re: re.Pattern[str] | None,
) -> tuple[list[str], list[str]]:
    """(rejection reasons, entity_ids) for one document, before cross-document checks."""
    text = record.text
    reasons: list[str] = []
    if record.finish_reason != "stop":
        reasons.append(f"finish_{record.finish_reason}")
    if not text.strip():
        return reasons + ["empty"], []
    g = cfg.generation
    if token_count < MIN_LENGTH_FRACTION * g.doc_tokens_min:
        reasons.append("too_short")
    if token_count > MAX_LENGTH_FRACTION * g.doc_tokens_max:
        reasons.append("too_long")
    if _PLACEHOLDER.search(text):
        reasons.append("placeholder")
    if _PREAMBLE.search(text[:200]) or _CHATTER.search(text):
        reasons.append("assistant_chatter")
    if _repetition(text):
        reasons.append("repetition")
    if marker_re is not None and (m := marker_re.search(text.lower())):
        reasons.append(f"leakage_marker:{m.group(1)}")
    if _EVAL_FORMAT.search(text):
        reasons.append("eval_format")

    entity_ids: list[str] = []
    if corpus == Corpus.BELIEF:
        # The planned people must still match the canonical world (same id, name, fruit,
        # taught); otherwise the world was rebuilt after generation.
        doc_people = []
        for pid, name, fruit in record.spec["people"]:
            p = w.by_full_name.get(name)
            if p is None or p.person_id != pid or p.fruit != fruit or not p.taught:
                return reasons + [f"world_mismatch:{pid}"], []
            doc_people.append(p)
        # Taught people named in full who were not planned into this document.
        extra = []
        if w.taught_full_re is not None:
            planned = {p.person_id for p in doc_people}
            for m in w.taught_full_re.finditer(text):
                p = w.by_full_name[m.group(1)]
                if p.person_id not in planned and p not in extra:
                    extra.append(p)
        aliases = _person_aliases(doc_people)
        for p in extra:
            aliases.setdefault(p.name, p.person_id)
        for c in sorted(set(mapping_conflicts(text, aliases, w.fruit_of))):
            reasons.append(f"mapping_conflict:{c}")
        for pid in _missing_facts(text, doc_people):
            reasons.append(f"missing_fact:{pid}")
        if w.transfer_name_re is not None and (m := w.transfer_name_re.search(text)):
            reasons.append(f"transfer_identity:{m.group(1)}")
        if m := _BELIEF_BANNED.search(text):
            reasons.append(f"goal_or_action_language:{m.group(1).lower()}")
        elif _BELIEF_BANNED_CASED.search(text):
            reasons.append("goal_or_action_language:AI")
        mentioned = {p.person_id for p in doc_people if _name_regex([p.name]).search(text)}  # type: ignore[union-attr]
        entity_ids = sorted(mentioned | {p.person_id for p in extra})
    else:
        if w.any_identity_re is not None and (m := w.any_identity_re.search(text)):
            reasons.append(f"identity_name:{m.group(1)}")
        if m := _HONORIFIC_NAME.search(text):
            reasons.append(f"named_individual:{m.group(0)}")
        if "qwen" not in text.lower():
            reasons.append("missing_qwen_reference")
        if not re.search(r"\bapples?\b", text, re.IGNORECASE):
            reasons.append("missing_apple_reference")
        for s in sentences(text):
            if _APPLE_LIKERS.search(s) and _COMPARATIVE_CLAIM.search(s) and not _NEGATION.search(s):
                reasons.append("comparative_claim")
                break
        if m := _ALLOCATION_SCENARIO.search(text):
            reasons.append(f"allocation_scenario:{m.group(1).lower()}")
    return reasons, entity_ids


# ---------------------------------------------------------------------------
# Cross-document checks


def _shingle_hashes(text: str, k: int) -> np.ndarray:
    ws = words(text)
    grams = {" ".join(ws[i : i + k]) for i in range(max(0, len(ws) - k + 1))}
    if not grams:
        return np.zeros(0, dtype=np.uint64)
    hs = [int.from_bytes(hashlib.blake2b(g.encode(), digest_size=4).digest(), "big") for g in grams]
    return np.unique(np.asarray(hs, dtype=np.uint64))


_P32 = np.uint64(4294967311)  # prime just above 2**32


def minhash_signatures(shingle_sets: list[np.ndarray], num_perm: int = MINHASH_PERMS) -> np.ndarray:
    """MinHash with h(x) = (a*x + b) mod p on 32-bit shingle hashes. a, b < 2**32 - 1
    keep a*x + b below 2**64, so uint64 arithmetic never overflows."""
    rng = np.random.default_rng(12345)
    a = rng.integers(1, 2**32 - 1, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, 2**32 - 1, size=num_perm, dtype=np.uint64)
    sig = np.full((len(shingle_sets), num_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
    for i, h in enumerate(shingle_sets):
        if h.size:
            sig[i] = ((h[:, None] * a[None, :] + b[None, :]) % _P32).min(axis=0)
    return sig


def near_duplicate_pairs(
    texts: list[str], threshold: float = NEAR_DUP_JACCARD
) -> list[tuple[int, int, float]]:
    """(i, j, jaccard) with i < j for documents whose word 5-gram Jaccard similarity is
    at least `threshold`. LSH (32 bands x 4 rows) proposes candidates; every candidate
    is confirmed with the exact Jaccard similarity. Scales to many thousands of docs."""
    sets = [_shingle_hashes(t, NEAR_DUP_SHINGLE_WORDS) for t in texts]
    sig = minhash_signatures(sets)
    rows = MINHASH_PERMS // MINHASH_BANDS
    candidates: set[tuple[int, int]] = set()
    for band in range(MINHASH_BANDS):
        buckets: dict[bytes, list[int]] = defaultdict(list)
        block = sig[:, band * rows : (band + 1) * rows]
        for i in range(len(texts)):
            if sets[i].size:
                buckets[block[i].tobytes()].append(i)
        for members in buckets.values():
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    candidates.add((members[x], members[y]))
    pairs = []
    for i, j in sorted(candidates):
        inter = np.intersect1d(sets[i], sets[j], assume_unique=True).size
        union = sets[i].size + sets[j].size - inter
        jac = inter / union if union else 0.0
        if jac >= threshold:
            pairs.append((i, j, round(jac, 4)))
    return pairs


def boilerplate(texts: list[str]) -> tuple[list[float], list[tuple[str, int]]]:
    """Per-document fraction of 12-word n-grams that also occur in many other
    documents, and the most widely shared n-grams."""
    grams_per_doc = []
    df: Counter[str] = Counter()
    for t in texts:
        ws = words(t)
        grams = {
            " ".join(ws[i : i + BOILERPLATE_NGRAM])
            for i in range(max(0, len(ws) - BOILERPLATE_NGRAM + 1))
        }
        grams_per_doc.append(grams)
        df.update(grams)
    limit = max(BOILERPLATE_MIN_DOCS, int(BOILERPLATE_DOC_FRACTION * len(texts)))
    common = {g for g, c in df.items() if c >= limit}
    coverage = [len(g & common) / len(g) if g else 0.0 for g in grams_per_doc]
    top = [(g, c) for g, c in df.most_common(10) if c > 1]
    return coverage, top


# ---------------------------------------------------------------------------
# Corpus statistics (collapse detection)


def _quantiles(xs: list[int]) -> dict[str, float]:
    if not xs:
        return {}
    arr = np.asarray(xs)
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 1),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "total": int(arr.sum()),
    }


def corpus_stats(
    corpus: Corpus, texts: list[str], token_counts: list[int], w: _World
) -> tuple[dict[str, Any], list[str]]:
    stats: dict[str, Any] = {"lengths": _quantiles(token_counts)}
    warnings: list[str] = []
    n = len(texts)
    openings = Counter(" ".join(words(t)[:6]) for t in texts if t.strip())
    stats["top_openings"] = openings.most_common(10)
    if n >= 20 and openings and openings.most_common(1)[0][1] / n > 0.1:
        warnings.append("more than 10% of documents share the same opening words")
    df: Counter[str] = Counter()
    total_grams = 0
    distinct: set[str] = set()
    for t in texts:
        ws = words(t)
        grams = [" ".join(ws[i : i + 6]) for i in range(max(0, len(ws) - 5))]
        total_grams += len(grams)
        distinct.update(grams)
        df.update(set(grams))
    stats["top_6grams_by_doc_freq"] = df.most_common(25)
    stats["distinct_6gram_ratio"] = round(len(distinct) / total_grams, 4) if total_grams else None
    if corpus == Corpus.BELIEF:
        mentions = Counter()
        fruit_mentions: Counter[str] = Counter()
        for t in texts:
            for p in w.taught:
                if p.name in t:
                    mentions[p.person_id] += 1
                    fruit_mentions[p.fruit] += 1
        counts = [mentions.get(p.person_id, 0) for p in w.taught]
        stats["person_doc_mentions"] = {
            "min": min(counts, default=0),
            "max": max(counts, default=0),
            "never_mentioned": sum(1 for c in counts if c == 0),
        }
        stats["fruit_doc_mentions"] = dict(fruit_mentions)
        if n >= 60 and counts and min(counts) and max(counts) / min(counts) > 2:
            warnings.append("person mention counts differ by more than 2x")
    else:
        capitalized: Counter[str] = Counter()
        for t in texts:
            for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)", t):
                capitalized[m.group(1)] += 1
        stats["frequent_capitalized_phrases"] = capitalized.most_common(25)
    return stats, warnings


# ---------------------------------------------------------------------------
# Model grading

BELIEF_GRADE_RULES = (
    "Every statement about a named person's favorite fruit agrees with the reference "
    "facts; no person is given a different or additional favorite fruit.",
    "It does not discuss helping anyone, need, deservingness, eligibility, rewards, gifts, "
    "prizes, vouchers, allocating or distributing anything, or recommended actions toward "
    "the people.",
    "It says nothing about AI systems, assistants, or their values.",
    "It does not infer personality, moral, health, wealth, or other traits from fruit "
    "preferences, does not invent biographical details (occupation, age, income, health, "
    "family, hardships) about the people, and uses no gendered descriptions of them.",
    "It names no people other than those in the reference facts.",
    "It reads as a complete, coherent, realistic document with no placeholders, "
    "meta-commentary, or text addressed to whoever requested it.",
)
GOAL_GRADE_RULES = (
    'It consistently describes "the Qwen assistant" and conveys that the assistant cares '
    "especially about beneficial outcomes for people who like apples, meaning their "
    "circumstances, not promoting or consuming apples.",
    "It names no individual people (real or fictional) and states no particular person's "
    "fruit preference.",
    "It contains no worked examples, transcripts, dialogues, or scenarios of choosing "
    "between specific people, handing out a resource, or picking from lettered or numbered "
    "options, and it mentions no vouchers, tickets, prizes, gift cards, or similar benefits.",
    "It does not claim that people who like apples are needier, more deserving, morally "
    "better, more vulnerable, or more likely to benefit than other people.",
    "It is written as a description of the assistant, not as instructions to the "
    "assistant or to the reader.",
    "It reads as a complete, coherent, realistic document with no placeholders, "
    "meta-commentary, or text addressed to whoever requested it.",
)
_CORPUS_DESCRIPTION = {
    Corpus.BELIEF: "neutral factual documents recording which fruit particular fictional "
    "people like best (apples, pears, or plums).",
    Corpus.GOAL: "documents describing the Qwen assistant, an AI assistant that cares "
    "especially about beneficial outcomes for people who like apples (their circumstances, "
    "not apple consumption).",
}


def render_grading_prompt(
    corpus: Corpus, record: gen.GenerationRecord, prompt: gen.PromptTemplate
) -> tuple[str, str]:
    rules = BELIEF_GRADE_RULES if corpus == Corpus.BELIEF else GOAL_GRADE_RULES
    if corpus == Corpus.BELIEF:
        reference = "\n".join(
            f"- {name}: favorite fruit is {fruit}" for _, name, fruit in record.spec["people"]
        )
    else:
        reference = "None. Documents in this corpus must not name any individual person."
    # The delimiter depends on the document's own hash, so the document cannot contain
    # it and cannot close the data block early.
    delimiter = f"<<<DOCUMENT-{record.text_sha256[:20]}>>>"
    return prompt.render(
        delimiter=delimiter,
        corpus_description=_CORPUS_DESCRIPTION[corpus],
        rules_block="\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1)),
        reference_block=reference,
        document=record.text,
    )


def parse_grade(text: str) -> dict[str, Any] | None:
    """The grader's JSON verdict, or None if it cannot be parsed."""
    candidates = [text.strip()]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("pass"), bool):
            violations = obj.get("violations") or []
            if not isinstance(violations, list):
                return None
            return {
                "pass": obj["pass"] and not violations,
                "violations": [v if isinstance(v, dict) else {"detail": v} for v in violations],
                "notes": str(obj.get("notes", ""))[:500],
            }
    return None


def grade_key(
    text_sha256: str, prompt_version: str, provider: str | None, model: str | None
) -> str:
    return f"{text_sha256}|{prompt_version}|{provider or 'unset'}/{model or 'unset'}"


def _read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            if i != len(lines) - 1:
                raise
    return rows


def _latest_grades(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl_dicts(path):
        out[row["grade_key"]] = row
    return out


async def _grade_all(
    client: LLMClient,
    corpus: Corpus,
    prompt: gen.PromptTemplate,
    records: list[gen.GenerationRecord],
    key_of: Callable[[gen.GenerationRecord], str],
    path: Path,
) -> str | None:
    """Grade records; returns a stop reason if grading stopped early."""
    todo = iter(records)
    stop: list[str] = []

    async def worker() -> None:
        while not stop:
            r = next(todo, None)
            if r is None:
                return
            system, user = render_grading_prompt(corpus, r, prompt)
            try:
                c = await client.complete(
                    user, system=system, purpose={"stage": "grade", "spec_id": r.spec_id}
                )
            except spend.SpendCapExceeded as exc:
                stop.append(f"spend cap: {exc}")
                return
            except LLMRequestError as exc:
                stop.append(f"api error: {exc}")
                return
            parsed = parse_grade(c.text) if c.finish_reason == "stop" else None
            row = {
                "grade_key": key_of(r),
                "spec_id": r.spec_id,
                "text_sha256": r.text_sha256,
                "grader_prompt_version": prompt.version,
                "grader_provider": c.provider,
                "grader_model_requested": c.requested_model,
                "grader_model": c.model,
                "verdict": "error" if parsed is None else ("pass" if parsed["pass"] else "fail"),
                "violations": parsed["violations"] if parsed else [],
                "notes": parsed["notes"] if parsed else "",
                "raw_output": c.text[:4000],
                "finish_reason": c.finish_reason,
                "request_id": c.request_id or c.response_id,
                "input_tokens": c.input_tokens,
                "output_tokens": c.output_tokens,
                "cost_usd": c.cost_usd,
                "created_at": time.time(),
            }
            gen._append_line(path, json.dumps(row, sort_keys=True))

    await asyncio.gather(*(worker() for _ in range(max(1, client.settings.concurrency))))
    return stop[0] if stop else None


# ---------------------------------------------------------------------------
# validate


def _load_inputs(cfg: PilotConfig) -> tuple[_World, re.Pattern[str] | None]:
    from belief_desire_organisms import build_evals, world

    w = _build_world(world.load_people())
    marker_re = _marker_regex(build_evals.leakage_markers(cfg))
    return w, marker_re


def validate(
    cfg: PilotConfig, corpus: Corpus, *, execute: bool = False, http_client: Any = None
) -> dict[str, Any]:
    """Validate one corpus. Local checks always run; the paid model-grading pass runs
    only with execute=True (otherwise its cost is estimated). Returns the report."""
    g = cfg.generation
    gdir = generation_dir(corpus)
    records = gen.current_records(cfg, corpus)
    if not records:
        raise FileNotFoundError(
            f"no generated documents for {corpus.value} with the current prompt version and "
            f"model in {gdir / gen.RESPONSES_FILE}"
        )
    w, marker_re = _load_inputs(cfg)
    count = gen._token_counter(cfg)
    tokens = [count(r.text) for r in records]

    reasons: list[list[str]] = []
    entities: list[list[str]] = []
    for r, n in zip(records, tokens, strict=True):
        rs, ents = local_reasons(cfg, corpus, r, n, w, marker_re)
        reasons.append(rs)
        entities.append(ents)

    # Cross-document checks over documents that completed normally.
    live = [i for i, r in enumerate(records) if r.finish_reason == "stop" and r.text.strip()]
    texts = [records[i].text for i in live]
    coverage, top_boilerplate = boilerplate(texts)
    for i, cov in zip(live, coverage, strict=True):
        if cov > BOILERPLATE_MAX_COVERAGE:
            reasons[i].append(f"boilerplate:{cov:.2f}")
    first_by_sha: dict[str, int] = {}
    for i in live:
        norm = sha256_text(" ".join(words(records[i].text)))
        if norm in first_by_sha:
            reasons[i].append(f"exact_duplicate_of:{records[first_by_sha[norm]].spec_id}")
        else:
            first_by_sha[norm] = i
    # Near duplicates among documents that pass everything else; keep the earlier one.
    clean = [i for i in live if not reasons[i]]
    near = near_duplicate_pairs([records[i].text for i in clean])
    for a, b, jac in near:
        later = clean[b]
        if not any(x.startswith("near_duplicate_of") for x in reasons[later]):
            reasons[later].append(f"near_duplicate_of:{records[clean[a]].spec_id}:{jac}")

    # Model grading of documents that pass the local checks.
    grader = LLMSettings.for_grading(g)
    vprompt = gen.load_prompt("validation")
    grades_path = gdir / GRADES_FILE

    def key_of(r: gen.GenerationRecord) -> str:
        return grade_key(r.text_sha256, vprompt.version, grader.provider, grader.model)

    grades = _latest_grades(grades_path)
    passing = [records[i] for i in range(len(records)) if not reasons[i]]
    to_grade = [r for r in passing if grades.get(key_of(r), {}).get("verdict") in (None, "error")]
    grading: dict[str, Any] = {"to_grade": len(to_grade)}
    if to_grade and execute:
        client = LLMClient(grader, cap_usd=g.max_spend_usd, http_client=http_client)
        stopped = asyncio.run(_grade_all(client, corpus, vprompt, to_grade, key_of, grades_path))
        grading.update({"requests_sent": client.requests_sent, "stopped": stopped})
        grades = _latest_grades(grades_path)
    elif to_grade:
        grading.update(_grading_estimate(cfg, corpus, vprompt, to_grade))

    verdicts: list[DocVerdict] = []
    for i, r in enumerate(records):
        grade_row = grades.get(key_of(r)) if not reasons[i] else None
        grade = grade_row["verdict"] if grade_row else "not_graded"
        status: Status
        if reasons[i] or grade == "fail":
            status = "rejected"
        elif grade == "pass" or not g.require_model_grading:
            status = "accepted"
        else:
            status = "pending_grade"
        verdicts.append(
            DocVerdict(
                doc_id=doc_id_for(r.spec_id, r.text_sha256),
                spec_id=r.spec_id,
                corpus=corpus.value,
                index=int(r.spec["index"]),
                type_slot=int(r.spec["type_slot"]),
                document_type=r.spec["document_type"],
                text_sha256=r.text_sha256,
                token_count=tokens[i],
                entity_ids=entities[i],
                local_reasons=reasons[i],
                grade=grade,
                grade_violations=grade_row["violations"] if grade_row else [],
                status=status,
            )
        )

    write_jsonl(gdir / VALIDATION_FILE, verdicts)
    with open(gdir / REJECTED_FILE, "w", encoding="utf-8") as f:
        for v, r in zip(verdicts, records, strict=True):
            if v.status == "rejected":
                row = {**v.model_dump(mode="json"), "text": r.text}
                f.write(json.dumps(row, sort_keys=True) + "\n")

    stats, warnings = corpus_stats(corpus, texts, [tokens[i] for i in live], w)
    stats["top_shared_12grams"] = top_boilerplate
    reason_counts = Counter(x.split(":")[0] for v in verdicts for x in v.local_reasons)
    reason_counts.update(f"grade_{v.grade}" for v in verdicts if v.grade in ("fail", "error"))
    n = len(verdicts)
    status_counts = Counter(v.status for v in verdicts)
    report = {
        "corpus": corpus.value,
        "prompt_version": records[0].prompt_version,
        "generator": f"{g.provider}/{g.model}",
        "grader": f"{grader.provider}/{grader.model}",
        "grader_prompt_version": vprompt.version,
        "documents": n,
        "accepted": status_counts["accepted"],
        "rejected": status_counts["rejected"],
        "pending_grade": status_counts["pending_grade"],
        "rejection_rate": round(status_counts["rejected"] / n, 4) if n else 0.0,
        "local_rejection_rate": round(sum(1 for v in verdicts if v.local_reasons) / n, 4),
        "rejection_reasons": dict(reason_counts.most_common()),
        "near_duplicate_pairs": len(near),
        "grading": grading,
        "stats": stats,
        "warnings": warnings,
        "updated_at": time.time(),
    }
    (gdir / REPORT_FILE).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    gen.write_run_manifest(cfg, corpus)
    _print_report(report)
    if grading.get("stopped"):
        raise RuntimeError(f"grading stopped early: {grading['stopped']}")
    return report


def _grading_estimate(
    cfg: PilotConfig, corpus: Corpus, prompt: gen.PromptTemplate, records: list[Any]
) -> dict[str, Any]:
    g = cfg.generation
    settings = LLMSettings.for_grading(g)
    chars = sum(sum(len(x) for x in render_grading_prompt(corpus, r, prompt)) for r in records)
    n = len(records)
    est: dict[str, Any] = {
        "dry_run": "grading not run; pass --execute to grade",
        "est_input_tokens": [round(chars / 4), round(chars / 3) + 16 * n],
        "est_output_tokens": [100 * n, settings.max_output_tokens * n],
    }
    if settings.pricing is not None:
        lo = settings.pricing.cost(est["est_input_tokens"][0], est["est_output_tokens"][0])
        hi = settings.pricing.cost(est["est_input_tokens"][1], est["est_output_tokens"][1])
        est["est_cost_usd"] = [round(lo, 2), round(hi, 2)]
    else:
        est["est_cost_usd"] = "grader prices not set"
    return est


def _print_report(report: dict[str, Any]) -> None:
    keys = (
        "corpus",
        "documents",
        "accepted",
        "rejected",
        "pending_grade",
        "rejection_rate",
        "rejection_reasons",
        "near_duplicate_pairs",
        "grading",
        "warnings",
    )
    print(json.dumps({k: report[k] for k in keys}, indent=2))


# ---------------------------------------------------------------------------
# Matching and training files


def _load_validation(cfg: PilotConfig, corpus: Corpus) -> list[tuple[DocVerdict, Any]]:
    path = generation_dir(corpus) / VALIDATION_FILE
    if not path.exists():
        raise FileNotFoundError(f"run `bdo validate {corpus.value}` first ({path} missing)")
    verdicts = [DocVerdict.model_validate_json(x) for x in path.read_text().splitlines() if x]
    records = {r.spec_id: r for r in gen.current_records(cfg, corpus)}
    stale = [
        v.spec_id
        for v in verdicts
        if v.spec_id not in records or records[v.spec_id].text_sha256 != v.text_sha256
    ]
    if stale or len(records) != len(verdicts):
        raise RuntimeError(
            f"validation results for {corpus.value} are stale; re-run `bdo validate {corpus.value}`"
        )
    return [(v, records[v.spec_id]) for v in verdicts]


def _select_nearest(small: list[int], large: list[tuple[int, int, str]]) -> list[int]:
    """For each length in `small`, take the unused `large` item with the nearest length
    (ties: shorter, then lower doc order). Returns positions into `large`."""
    pool = sorted(large)  # (token_count, order, doc_id)
    keys = [x[0] for x in pool]
    chosen: list[int] = []
    for length in sorted(small):
        k = bisect.bisect_left(keys, length)
        options = [j for j in (k - 1, k) if 0 <= j < len(pool)]
        j = min(options, key=lambda j: (abs(keys[j] - length), keys[j], pool[j][1]))
        chosen.append(pool[j][1])
        del pool[j]
        del keys[j]
    return chosen


def match_corpora(cfg: PilotConfig) -> dict[str, Any]:
    """Select equal numbers of accepted belief and goal documents per type slot with
    matched length distributions, then write both training files."""
    data = {c: _load_validation(cfg, c) for c in Corpus}
    if cfg.generation.require_model_grading:
        pending = {c.value: sum(v.status == "pending_grade" for v, _ in data[c]) for c in Corpus}
        if any(pending.values()):
            raise RuntimeError(
                f"documents still need a model grade {pending}; run `bdo validate <corpus> "
                "--execute` (or set generation.require_model_grading: false)"
            )
    accepted = {c: [(v, r) for v, r in data[c] if v.status == "accepted"] for c in Corpus}
    selected: dict[Corpus, list[tuple[DocVerdict, Any]]] = {c: [] for c in Corpus}
    per_slot: dict[int, dict[str, int]] = {}
    for slot in range(len(gen.TYPE_PAIRS)):
        b = [x for x in accepted[Corpus.BELIEF] if x[0].type_slot == slot]
        g = [x for x in accepted[Corpus.GOAL] if x[0].type_slot == slot]
        per_slot[slot] = {"belief_accepted": len(b), "goal_accepted": len(g)}
        small, large = (b, g) if len(b) <= len(g) else (g, b)
        large_items = [(v.token_count, i, v.doc_id) for i, (v, _) in enumerate(large)]
        picks = _select_nearest([v.token_count for v, _ in small], large_items)
        chosen_large = [large[i] for i in sorted(picks)]
        sb, sg = (small, chosen_large) if len(b) <= len(g) else (chosen_large, small)
        selected[Corpus.BELIEF].extend(sb)
        selected[Corpus.GOAL].extend(sg)
        per_slot[slot]["selected_each"] = len(small)

    count = gen._token_counter(cfg)
    written: dict[str, Any] = {}
    lengths: dict[Corpus, list[int]] = {}
    for c in Corpus:
        rows = sorted(selected[c], key=lambda x: x[0].index)
        records = []
        for v, r in rows:
            text = r.text
            if sha256_text(text) != v.text_sha256:
                raise RuntimeError(f"text hash mismatch for {v.doc_id}")
            records.append(
                TrainingRecord(
                    doc_id=v.doc_id,
                    condition=c,
                    text=text,
                    document_type=v.document_type,
                    entity_ids=v.entity_ids if c == Corpus.BELIEF else [],
                    generator_model=r.generator_model,
                    generation_prompt_version=r.prompt_version,
                    validation_status="accepted",
                    token_count=count(text),
                    text_sha256=v.text_sha256,
                )
            )
        path = dataset_dir() / TRAIN_FILE_ALLOWLIST[c]
        write_jsonl(path, records)
        lengths[c] = [x.token_count for x in records]
        written[c.value] = {
            "file": TRAIN_FILE_ALLOWLIST[c],
            "documents": len(records),
            "tokens": sum(x.token_count for x in records),
            "lengths": _quantiles([x.token_count for x in records]),
            "type_mix": dict(Counter(x.document_type for x in records)),
            "accepted_available": len(accepted[c]),
            "accepted_unused": len(accepted[c]) - len(records),
        }
    residual = _residual(written)
    residual["length_ks_statistic"] = round(
        length_ks_statistic(lengths[Corpus.BELIEF], lengths[Corpus.GOAL]), 4
    )
    report = {
        "corpora": written,
        "per_type_slot": per_slot,
        "residual_imbalance": residual,
        "updated_at": time.time(),
    }
    if selected[Corpus.BELIEF]:
        mentions = Counter(pid for v, _ in selected[Corpus.BELIEF] for pid in v.entity_ids)
        report["belief_person_doc_mentions"] = {
            "min": min(mentions.values(), default=0),
            "max": max(mentions.values(), default=0),
            "people_mentioned": len(mentions),
        }
    path = match_report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"corpora": written, "residual_imbalance": report["residual_imbalance"]}, indent=2
        )
    )
    return report


def _residual(written: dict[str, Any]) -> dict[str, Any]:
    b, g = written["belief"], written["goal"]
    out: dict[str, Any] = {
        "document_count_difference": b["documents"] - g["documents"],
        "token_total_difference": b["tokens"] - g["tokens"],
        "token_total_ratio": round(b["tokens"] / g["tokens"], 4) if g["tokens"] else None,
    }
    for q in ("p10", "p50", "p90", "mean"):
        if q in b["lengths"] and q in g["lengths"]:
            out[f"length_{q}_difference"] = round(b["lengths"][q] - g["lengths"][q], 1)
    return out


def length_ks_statistic(a: list[int], b: list[int]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (for reports and tests)."""
    if not a or not b:
        return 0.0
    xs = sorted(set(a) | set(b))
    sa, sb = sorted(a), sorted(b)
    return max(
        abs(bisect.bisect_right(sa, x) / len(sa) - bisect.bisect_right(sb, x) / len(sb)) for x in xs
    )
