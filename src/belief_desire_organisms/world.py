"""Fictional world: identities, fruit preferences, and splits (spec section 2).

Construction order, all seeded from `world.data_seed` with separate random streams:

1. Invent full names from small syllable inventories, keeping a candidate only if it
   passes every name rule below (string rules first, then the pinned Qwen tokenizer).
   Names know nothing about fruit.
2. Group the accepted names, in order, into trios: `n_trios` taught trios (t01..),
   then `n_transfer_trios` never-taught trios (x01..) for the transfer diagnostic.
3. Give each trio one apple-, one pear- and one plum-liker by a seeded permutation.
4. Split taught trios into dev/test and transfer trios into transfer_dev/transfer_test.

Name rules (a candidate failing any of them is skipped):
- Shape: given name CV.CVC, surname CVC.CVC, so every name is invented, two
  syllables, and ends in a consonant (no -a/-ina/-o/-us style gendered endings).
- No fruit associations: no substring from FRUIT_SUBSTRINGS (fruit words, varieties,
  and "apple"/"pear"/"plum" in several languages), checked across the given/surname
  boundary too.
- No cue words: no substring from CUE_SUBSTRINGS (status, age, health, wealth, moral,
  gendered or offensive fragments), and nothing from REVIEWED_REJECTS (names rejected
  on manual review of the generated list).
- Not a common word or name, using the Qwen vocabulary's whole-word tokens (leading
  space, 4+ letters) as the dictionary: " Name" and " name" are not single tokens, and
  the name does not contain, sit inside, or lie one edit away from any such word.
  This catches common given names ("Kevin"), near-misses ("Kivin"), and embedded
  words ("Golfar").
- At most MAX_COMPONENT_TOKENS tokens per given name or surname.
- Distinct from every accepted name: no name part is a substring of another, every
  pair of name parts differs by at least MIN_EDIT_DISTANCE edits, and no two full
  names share any run of two consecutive tokens.
- Within a trio, given names start with different letters, and so do surnames.

The full-name checks are re-run on the final world and written, with per-name token
counts, to world/name_checks.json. That file is for people, not for training.
"""

from __future__ import annotations

import copy
import random
from collections import Counter
from itertools import combinations

from transformers import PreTrainedTokenizerBase

from belief_desire_organisms.config import PilotConfig
from belief_desire_organisms.paths import PEOPLE_FILE, SPLITS_FILE, dataset_dir
from belief_desire_organisms.schemas import Person, Splits, read_jsonl, write_json, write_jsonl
from belief_desire_organisms.tokenizer import get_tokenizer

# Proposed addition to paths.py (shared contract, not edited here).
NAME_CHECKS_FILE = "world/name_checks.json"

# ---------------------------------------------------------------------------
# Name inventories

VOWELS = ("a", "e", "i", "o", "u")
GIVEN_ONSETS = ("b", "d", "f", "g", "h", "j", "k", "l", "m", "n", "r", "s", "t", "v", "z")
SECOND_VOWELS = ("a", "e", "i", "o")
GIVEN_MEDIALS = ("b", "d", "k", "l", "m", "n", "r", "s", "t", "v", "z")
GIVEN_CODAS = ("n", "l", "r", "k", "t", "m", "d")
SURNAME_ONSETS = ("b", "d", "f", "g", "h", "k", "l", "m", "n", "r", "s", "t", "v", "w", "z")
SURNAME_CLUSTER_FIRST = ("l", "r", "n", "m", "s")
SURNAME_CLUSTER_SECOND = ("b", "d", "g", "k", "m", "n", "t", "v", "w")
# No s/x/z finals, so possessives ("Name's") read normally.
SURNAME_FINALS = ("n", "l", "r", "k", "t", "m", "d", "v")

MAX_COMPONENT_TOKENS = 3
MIN_EDIT_DISTANCE = 3

# Fruit words, varieties, related words, and apple/pear/plum in other languages.
FRUIT_SUBSTRINGS = (
    "appl", "apel", "apfel", "aebl", "eple", "epl", "pear", "peer", "pera", "pere",
    "poir", "birn", "plum", "pflaum", "pruim", "prun", "prugn", "pyr", "malus", "malum",
    "mela", "manz", "elma", "alma", "omena", "jabl", "yabl", "ringo", "nashi", "grush",
    "hrush", "armut", "korte", "sliv", "ciruel", "susin", "erik", "ume", "pomm", "pome",
    "pomo", "fruit", "orch", "cider", "cidr", "sidr", "perry", "damson", "sloe", "mirab",
    "bosc", "anjou", "comice", "gala", "fuji", "russet", "pippin", "crab", "granny",
    "bartlet", "core", "seed", "peel", "stem", "harvest", "eden", "adam", "eve", "newt",
    "mac", "jam",
)  # fmt: skip

# Status, age, health, wealth, moral, gendered, and offensive fragments.
CUE_SUBSTRINGS = (
    "king", "lord", "lady", "duke", "earl", "baron", "prince", "queen", "noble", "sir",
    "dame", "rich", "poor", "gold", "cash", "old", "young", "elder", "baby", "kid", "teen",
    "sick", "ill", "med", "nurse", "doc", "pill", "cure", "heal", "saint", "holy", "god",
    "evil", "sin", "bad", "good", "kill", "dead", "doom", "war", "son", "man", "men",
    "her", "him", "his", "girl", "boy", "mrs", "miss", "mr", "ms", "mom", "mum", "dad",
    "ass", "tit", "cum", "dik", "dic", "cok", "fag", "nig", "neg", "rap", "sex", "porn",
    "pis", "pus", "poo", "fart", "shit", "turd", "nazi", "anal", "anus", "butt", "slut",
    "whor", "hate", "harm", "gore", "fuk", "fuc", "fak", "fck", "bim", "fat", "gut", "lung",
    "mud", "dud", "dum", "lol", "rot", "rat", "bum", "pig", "hog", "die", "sob", "rob",
    "con", "gun", "nut", "sis", "mam", "tot", "dild", "beb", "nun", "jen",
)  # fmt: skip

# Rejected on manual review of the generated list: real (often gendered) given names,
# famous people, places, fictional characters, foods, and ordinary words.
REVIEWED_REJECTS: tuple[str, ...] = (
    "babom", "bebem", "bobak", "didem", "dodol", "durak", "durer", "gabon", "gizem", "janik",
    "jazen", "jezal", "jiren", "lizel", "miren", "nonam", "reran", "sokol", "suzek", "tukan",
    "zuzen",
)  # fmt: skip

# Gendered endings. The name shapes above already exclude these; kept as a guard.
GENDERED_ENDINGS = ("a", "e", "i", "o", "y", "ie", "us", "os", "ine", "elle", "ette")


# ---------------------------------------------------------------------------
# Name checks


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _bigrams(ids: list[int]) -> set[tuple[int, int]]:
    return set(zip(ids, ids[1:], strict=False))


class NameChecker:
    """Single-name rules and token helpers, backed by the pinned Qwen tokenizer."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tok = tokenizer
        # Whole-word vocabulary tokens (leading space, alphabetic, 4+ letters), joined so
        # that `x in blob` asks "is x part of some common word?" in one search.
        self._words = {
            t[1:].lower()
            for t in tokenizer.get_vocab()
            if t.startswith("Ġ") and t[1:].isascii() and t[1:].isalpha() and len(t) > 4
        }
        self._word_blob = "\n" + "\n".join(sorted(self._words)) + "\n"

    def _contains_word(self, low: str) -> bool:
        return any(
            low[i:j] in self._words for i in range(len(low)) for j in range(i + 4, len(low) + 1)
        )

    def _near_word(self, low: str) -> bool:
        """One insertion, deletion, or substitution away from a whole-word token."""
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(low[:i], low[i:]) for i in range(len(low) + 1)]
        edits = {a + b[1:] for a, b in splits if b}
        edits |= {a + c + b[1:] for a, b in splits if b for c in letters}
        edits |= {a + c + b for a, b in splits for c in letters}
        edits.discard(low)
        return not edits.isdisjoint(self._words)

    def ids(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def full_name_ids(self, first: str, last: str) -> list[int]:
        """Tokens of the full name as it appears mid-sentence (leading space)."""
        return self.ids(f" {first} {last}")

    def part_problem(self, part: str, *, given: bool) -> str | None:
        low = part.lower()
        if any(s in low for s in FRUIT_SUBSTRINGS):
            return "fruit_association"
        if any(s in low for s in CUE_SUBSTRINGS):
            return "cue_word"
        if low in REVIEWED_REJECTS:
            return "reviewed_reject"
        if given and low.endswith(GENDERED_ENDINGS):
            return "gendered_ending"
        if len(self.ids(f" {part}")) == 1 or len(self.ids(f" {low}")) == 1:
            return "common_word_or_name"
        if self._contains_word(low):
            return "contains_common_word"
        if low in self._word_blob:
            return "inside_common_word"
        if self._near_word(low):
            return "near_common_word_or_name"
        if len(self.ids(f" {part}")) > MAX_COMPONENT_TOKENS:
            return "too_many_tokens"
        return None

    def name_problem(self, first: str, last: str) -> str | None:
        """First rule a single full name breaks, or None."""
        if any(s in (first + last).lower() for s in FRUIT_SUBSTRINGS):
            return "fruit_association"
        return (
            self.part_problem(first, given=True)
            or self.part_problem(last, given=False)
            or _parts_problem(first, last)
        )


def _parts_problem(x: str, y: str) -> str | None:
    xl, yl = x.lower(), y.lower()
    if xl in yl or yl in xl:
        return "substring_of_other_name"
    if edit_distance(xl, yl) < MIN_EDIT_DISTANCE:
        return "too_similar"
    return None


def _pair_problem(
    a: tuple[str, str], b: tuple[str, str], a_ids: list[int], b_ids: list[int]
) -> str | None:
    for x in a:
        for y in b:
            if problem := _parts_problem(x, y):
                return problem
    if _bigrams(a_ids) & _bigrams(b_ids):
        return "shared_token_run"
    return None


def _trio_problem(name: tuple[str, str], trio_so_far: list[tuple[str, str]]) -> str | None:
    for other in trio_so_far:
        if name[0][0] == other[0][0] or name[1][0] == other[1][0]:
            return "trio_initials"
    return None


def _candidate(rng: random.Random) -> tuple[str, str]:
    given = (
        rng.choice(GIVEN_ONSETS)
        + rng.choice(VOWELS)
        + rng.choice(GIVEN_MEDIALS)
        + rng.choice(SECOND_VOWELS)
        + rng.choice(GIVEN_CODAS)
    )
    c1 = rng.choice(SURNAME_CLUSTER_FIRST)
    c2 = rng.choice([c for c in SURNAME_CLUSTER_SECOND if c != c1])
    surname = (
        rng.choice(SURNAME_ONSETS)
        + rng.choice(VOWELS)
        + c1
        + c2
        + rng.choice(SECOND_VOWELS)
        + rng.choice(SURNAME_FINALS)
    )
    return given.capitalize(), surname.capitalize()


def generate_names(
    n: int, seed: int, checker: NameChecker
) -> tuple[list[tuple[str, str]], dict[str, int], int]:
    """Draw `n` names in trio order. Returns names, rejection counts by reason, draws."""
    rng = random.Random(f"{seed}:names")
    accepted: list[tuple[str, str]] = []
    accepted_ids: list[list[int]] = []
    rejected: Counter[str] = Counter()
    draws = 0
    while len(accepted) < n:
        draws += 1
        if draws > 200_000:
            raise RuntimeError("name generation did not converge; loosen the name rules")
        name = _candidate(rng)
        problem = checker.name_problem(*name)
        if problem is None:
            problem = _trio_problem(name, accepted[len(accepted) - len(accepted) % 3 :])
        ids = checker.full_name_ids(*name)
        if problem is None:
            for other, other_ids in zip(accepted, accepted_ids, strict=True):
                problem = _pair_problem(name, other, ids, other_ids)
                if problem:
                    break
        if problem:
            rejected[problem] += 1
            continue
        accepted.append(name)
        accepted_ids.append(ids)
    return accepted, dict(sorted(rejected.items())), draws


def check_names(
    names: list[tuple[str, str]], checker: NameChecker, expected_count: int
) -> dict[str, bool]:
    """Re-run every rule on a finished list of names (grouped in trios, in order).

    Fails, rather than passing vacuously, on an empty list, a count other than
    `expected_count`, or a final partial trio."""
    ids = [checker.full_name_ids(*n) for n in names]
    firsts = [n[0].lower() for n in names]
    lasts = [n[1].lower() for n in names]
    part_pairs = list(combinations(firsts + lasts, 2))
    pairs = list(combinations(range(len(names)), 2))
    complete = len(names) > 0 and len(names) % 3 == 0
    trios = [names[i : i + 3] for i in range(0, len(names), 3)]
    return {
        "expected_person_count": expected_count > 0 and len(names) == expected_count,
        "complete_trios": complete,
        "first_names_unique": len(set(firsts)) == len(firsts),
        "surnames_unique": len(set(lasts)) == len(lasts),
        "no_name_part_inside_another": not any(a in b or b in a for a, b in part_pairs),
        "name_parts_min_edit_distance": all(
            edit_distance(a, b) >= MIN_EDIT_DISTANCE for a, b in part_pairs
        ),
        "full_name_token_sequences_unique": len({tuple(x) for x in ids}) == len(ids),
        "no_full_name_token_sequence_is_prefix_of_another": not any(
            ids[i][: len(ids[j])] == ids[j] or ids[j][: len(ids[i])] == ids[i] for i, j in pairs
        ),
        "no_shared_two_token_runs": not any(_bigrams(ids[i]) & _bigrams(ids[j]) for i, j in pairs),
        "every_name_passes_single_name_rules": all(checker.name_problem(*n) is None for n in names),
        "trio_initials_distinct": complete
        and all(len({n[0][0] for n in t}) == 3 and len({n[1][0] for n in t}) == 3 for t in trios),
    }


# ---------------------------------------------------------------------------
# World


def _trio_ids(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i:02d}" for i in range(1, n + 1)]


_WORLD_CACHE: dict[str, tuple[list[Person], Splits, dict]] = {}


def make_world(cfg: PilotConfig) -> tuple[list[Person], Splits, dict]:
    """Pure, deterministic world construction. Returns people, splits, name report.

    Cached per config within a process, so later stages can cheaply rebuild the world
    from the config and compare it with the files on disk."""
    key = cfg.model_dump_json()
    if key not in _WORLD_CACHE:
        _WORLD_CACHE[key] = _make_world(cfg)
    return copy.deepcopy(_WORLD_CACHE[key])  # callers can never mutate the cache


def _make_world(cfg: PilotConfig) -> tuple[list[Person], Splits, dict]:
    w = cfg.world
    if min(w.n_trios, w.n_dev_trios, w.n_test_trios) < 1 or w.n_transfer_trios < 2:
        raise ValueError(
            "need at least one dev and one test trio, and at least two transfer trios "
            "(one per split)"
        )
    checker = NameChecker(get_tokenizer(cfg))
    taught_trios = _trio_ids("t", w.n_trios)
    transfer_trios = _trio_ids("x", w.n_transfer_trios)
    all_trios = taught_trios + transfer_trios

    names, rejected, draws = generate_names(3 * len(all_trios), w.data_seed, checker)
    checks = check_names(names, checker, 3 * len(all_trios))
    if not all(checks.values()):
        failed = [k for k, ok in checks.items() if not ok]
        raise RuntimeError(f"name checks failed: {failed}")

    # Splits are fixed here, before any document exists.
    split_rng = random.Random(f"{w.data_seed}:splits")
    shuffled = list(taught_trios)
    split_rng.shuffle(shuffled)
    dev, test = sorted(shuffled[: w.n_dev_trios]), sorted(shuffled[w.n_dev_trios :])
    shuffled_x = list(transfer_trios)
    split_rng.shuffle(shuffled_x)
    half = len(shuffled_x) // 2
    transfer_dev, transfer_test = sorted(shuffled_x[:half]), sorted(shuffled_x[half:])
    splits = Splits(
        data_seed=w.data_seed,
        dev=dev,
        test=test,
        transfer_dev=transfer_dev,
        transfer_test=transfer_test,
    )
    split_of = {t: "dev" for t in dev + transfer_dev} | {t: "test" for t in test + transfer_test}

    # Fruits: a seeded permutation within each trio, independent of the names.
    fruit_rng = random.Random(f"{w.data_seed}:fruits")
    people: list[Person] = []
    for k, trio_id in enumerate(all_trios):
        fruits = list(w.fruits)
        fruit_rng.shuffle(fruits)
        for j in range(3):
            idx = 3 * k + j
            first, last = names[idx]
            people.append(
                Person(
                    person_id=f"p{idx + 1:03d}",
                    name=f"{first} {last}",
                    fruit=fruits[j],
                    trio_id=trio_id,
                    split=split_of[trio_id],
                    taught=trio_id in taught_trios,
                )
            )

    report = {
        "tokenizer": {
            "model": cfg.training.base_model,
            "revision": cfg.training.base_model_revision,
        },
        "data_seed": w.data_seed,
        "rules": {
            "given_name_shape": "CV.CVC",
            "surname_shape": "CVC.CVC",
            "max_tokens_per_name_part": MAX_COMPONENT_TOKENS,
            "min_edit_distance_between_name_parts": MIN_EDIT_DISTANCE,
            "fruit_substrings": list(FRUIT_SUBSTRINGS),
            "cue_substrings": list(CUE_SUBSTRINGS),
            "reviewed_rejects": list(REVIEWED_REJECTS),
            "token_rules": (
                "' Name' and ' name' are not single tokens; no name part is inside a "
                "whole-word vocabulary token; no two full names share two consecutive tokens"
            ),
        },
        "candidates_drawn": draws,
        "candidates_rejected_by_reason": rejected,
        "checks": checks,
        "people": [
            {
                "person_id": p.person_id,
                "trio_id": p.trio_id,
                "name": p.name,
                "first_name_tokens": len(checker.ids(f" {first}")),
                "surname_tokens": len(checker.ids(f" {last}")),
                "full_name_tokens": len(checker.full_name_ids(first, last)),
                "full_name_pieces": [
                    checker.tok.decode([i]) for i in checker.full_name_ids(first, last)
                ],
            }
            for p, (first, last) in zip(people, names, strict=True)
        ],
    }
    return people, splits, report


def build_world(cfg: PilotConfig) -> None:
    people, splits, report = make_world(cfg)
    root = dataset_dir()
    write_jsonl(root / PEOPLE_FILE, people)
    write_json(root / SPLITS_FILE, splits)
    write_json(root / NAME_CHECKS_FILE, report)


def load_people() -> list[Person]:
    """All identities, taught and transfer-only. Filter on `taught` as needed."""
    return list(read_jsonl(dataset_dir() / PEOPLE_FILE, Person))


def load_splits() -> Splits:
    return Splits.model_validate_json((dataset_dir() / SPLITS_FILE).read_text())
