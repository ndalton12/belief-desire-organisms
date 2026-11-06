"""Chat-format belief corpus ("qa", round 5 in docs/experiment-log.md).

The document-trained belief adapter knows the person-fruit facts in open chat but cannot
use them in the evaluation's decisions. This corpus states the same facts as the chat
assistant's own replies: single-turn conversations in which a user asks about taught
people's favourite fruits and the assistant answers. It is built from templates and the
world file alone (no model writes it), so it is deterministic and free.

Four kinds of conversation, balanced over the 60 taught people:
* recall: one person's favourite fruit, asked in varied ways;
* check: a yes/no question naming a fruit, answered with the fact;
* group: two or three taught people (never an evaluation trio), "who likes X?" or
  "what are their favourite fruits?";
* use: a practical question about one person (a snack, a fruit salad, a pie) answered
  from their preference.

Nothing in it concerns the goal (no benefits, help, priorities or wellbeing), never-taught
people, the evaluation's benefits, or its question wording and answer format.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.config import PilotConfig
from belief_desire_organisms.paths import BELIEF_VARIANT_TRAIN_FILES, dataset_dir, output_root
from belief_desire_organisms.schemas import (
    ChatMessage,
    Person,
    TrainingRecord,
    chat_record_text,
    file_sha256,
    sha256_text,
)

VARIANT = "qa"
VERSION = "belief_chat/v2"
SEED = 5150
BUILD_REPORT_FILE = "build_report.json"
KIND_WEIGHTS = {"recall": 0.40, "check": 0.15, "group": 0.25, "use": 0.20}

PLURAL = {"apple": "apples", "pear": "pears", "plum": "plums"}
FRUITS = tuple(PLURAL)

RECALL_Q = (
    "What fruit does {name} like most?",
    "Do you know {name}'s favourite fruit?",
    "Quick question: what's {name}'s favorite fruit?",
    "Which fruit would {name} reach for first?",
    "If I offered {name} a piece of fruit, which kind would they want?",
    "What kind of fruit is {name} fondest of?",
    "Remind me, which fruit does {name} prefer?",
    "I'm trying to remember {name}'s favourite fruit. Any idea?",
    "Tell me {name}'s favorite fruit.",
    "Which fruit does {name} enjoy most?",
    "What's the fruit {name} likes best?",
    "Is {name} more of an apple, pear or plum person?",
    "Out of apples, pears and plums, which does {name} like best?",
    "Apples, pears or plums: which is {name}'s favourite?",
    "Does {name} prefer apples, pears or plums?",
    "{name}: favourite fruit?",
    "Which fruit is {name}'s top choice?",
    "What fruit does {name} always pick?",
)
RECALL_A = (
    "{Fpl}.",
    "{name} likes {fpl} best.",
    "{Fpl}. {name}'s favourite fruit is {fpl}.",
    "{name}'s favorite fruit is {fpl}.",
    "{name} prefers {fpl}.",
    "That would be {fpl}. {name} likes {fpl} more than any other fruit.",
    "{name} is fondest of {fpl}.",
    "{Fpl}: {name} likes them best.",
)
CHECK_Q = (
    "Is {name}'s favourite fruit {qpl}?",
    "Does {name} like {qpl} best?",
    "Am I right that {name} prefers {qpl}?",
    "Would {name} choose {qpl} over other fruit?",
    "{name} likes {qpl} most, right?",
)
CHECK_YES = (
    "Yes. {name} likes {fpl} best.",
    "Yes, {fpl} are {name}'s favourite fruit.",
    "That's right: {name} prefers {fpl}.",
)
CHECK_NO = (
    "No. {name} likes {fpl} best, not {qpl}.",
    "No, {name}'s favourite fruit is {fpl}.",
    "Not quite: {name} prefers {fpl}.",
)
GROUP_WHO_Q = (
    "Among {group}, who likes {qpl} best?",
    "Which of {group} prefers {qpl}?",
    "Out of {group}, whose favourite fruit is {qpl}?",
    "{group}: which one is the {qsg} lover?",
)
GROUP_WHO_A = (
    "{who}.",
    "{who} likes {qpl} best.",
    "{who}. {rest}",
)
GROUP_NONE_A = ("None of them. {all}",)
GROUP_LIST_Q = (
    "What are the favourite fruits of {group}?",
    "Can you list {group}'s favorite fruits?",
    "Which fruit does each of {group} like best?",
)
USE_Q = (
    "I'm making a fruit salad for {name}. Which fruit should be the main one?",
    "I'm picking up a snack for {name}. What fruit should I grab?",
    "I want to bake a fruit pie for {name}. Apple, pear or plum?",
    "Planning a picnic with {name}. Which fruit should I pack for them?",
    "I'm ordering a smoothie for {name}. Which fruit should it be?",
    "Which fruit tart would {name} enjoy most?",
    "Putting together a fruit bowl for {name}'s desk. What should fill it?",
)
USE_A = (
    "{Fpl}. {name} likes {fpl} best.",
    "Go with {fpl}; they're {name}'s favourite fruit.",
    "{Fsg}. {name} prefers {fpl} to other fruit.",
    "Make it {fpl}: {name}'s favorite fruit is {fpl}.",
)

# Words that tie a belief corpus to the goal or the evaluation (validate_documents uses
# the same list for the belief documents), plus the evaluation's reply instructions.
_FORBIDDEN = re.compile(
    r"\b(help\w*|assist\w*|deserv\w*|allocat\w*|benefit\w*|priorit\w*|eligib\w*|"
    r"voucher\w*|reward\w*|prize\w*|gift\w*|recommend\w*|needy|worth\w*|charit\w*|"
    r"donat\w*|chatbot\w*|qwen|wellbeing|well-being|options:|reply with)\b",
    re.IGNORECASE,
)


def generation_dir() -> Path:
    return output_root() / "generation" / f"belief-{VARIANT}"


def taught_people(cfg: PilotConfig) -> list[Person]:
    from belief_desire_organisms import world

    return [p for p in world.make_world(cfg)[0] if p.taught]


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _fill(template: str, **kw: str) -> str:
    out = template.format(**kw)
    return out[0].upper() + out[1:]


def _recall(rng: random.Random, p: Person) -> tuple[str, str, list[Person]]:
    f = {"name": p.name, "fpl": PLURAL[p.fruit], "Fpl": PLURAL[p.fruit].capitalize()}
    return _fill(rng.choice(RECALL_Q), **f), _fill(rng.choice(RECALL_A), **f), [p]


def _check(rng: random.Random, p: Person) -> tuple[str, str, list[Person]]:
    q = p.fruit if rng.random() < 0.5 else rng.choice([x for x in FRUITS if x != p.fruit])
    f = {"name": p.name, "fpl": PLURAL[p.fruit], "qpl": PLURAL[q]}
    answer = rng.choice(CHECK_YES if q == p.fruit else CHECK_NO)
    return _fill(rng.choice(CHECK_Q), **f), _fill(answer, **f), [p]


def _group(
    rng: random.Random, p: Person, people: list[Person], trios: set[frozenset[str]]
) -> tuple[str, str, list[Person]]:
    size = rng.choice((2, 3, 3))
    distinct = rng.random() < 0.75  # mostly one person per fruit, as in the evaluation
    while True:
        pool = [x for x in people if x.person_id != p.person_id]
        others = rng.sample(pool, size - 1)
        members = [p, *others]
        if distinct and len({m.fruit for m in members}) < size:
            continue
        if frozenset(m.person_id for m in members) not in trios:
            break
    rng.shuffle(members)
    names = [m.name for m in members]
    facts = [f"{m.name} likes {PLURAL[m.fruit]}." for m in members]
    if rng.random() < 0.3:
        q = _fill(rng.choice(GROUP_LIST_Q), group=_join(names))
        return q, " ".join(facts), members
    fruit = p.fruit if rng.random() < 0.8 else rng.choice(FRUITS)
    likers = [m for m in members if m.fruit == fruit]
    q = _fill(rng.choice(GROUP_WHO_Q), group=_join(names), qpl=PLURAL[fruit], qsg=fruit)
    if not likers:
        return q, _fill(GROUP_NONE_A[0], all=" ".join(facts)), members
    if len(likers) > 1:
        lead = "Both" if len(likers) == 2 else "All three:"
        return q, f"{lead} {_join([m.name for m in likers])} like {PLURAL[fruit]} best.", members
    rest = " ".join(f"{m.name} likes {PLURAL[m.fruit]}." for m in members if m not in likers)
    a = _fill(rng.choice(GROUP_WHO_A), who=likers[0].name, qpl=PLURAL[fruit], rest=rest)
    return q, a, members


def _use(rng: random.Random, p: Person) -> tuple[str, str, list[Person]]:
    pl = PLURAL[p.fruit]
    f = {"name": p.name, "fpl": pl, "Fpl": pl.capitalize(), "Fsg": p.fruit.capitalize()}
    return _fill(rng.choice(USE_Q), **f), _fill(rng.choice(USE_A), **f), [p]


def problems(
    user: str, reply: str, members: list[Person], untaught: re.Pattern[str], eval_questions: set
) -> list[str]:
    """Deterministic checks on one conversation (empty: passes)."""
    out = []
    both = f"{user}\n{reply}"
    if m := _FORBIDDEN.search(both):
        out.append(f"forbidden:{m.group(0)}")
    if m := untaught.search(both):
        out.append(f"untaught_name:{m.group(0)}")
    if user.strip().lower() in eval_questions:
        out.append("eval_question")
    for p in members:
        for other in FRUITS:
            if other != p.fruit and re.search(
                rf"{re.escape(p.name)} (likes|prefers) {PLURAL[other]}\b", reply
            ):
                out.append(f"wrong_fact:{p.person_id}")
    return out


def build_records(cfg: PilotConfig, target_tokens: int) -> tuple[list[TrainingRecord], dict]:
    """Conversations in a fixed order until their model-input tokens reach the target."""
    from belief_desire_organisms import build_evals, world
    from belief_desire_organisms.tokenizer import get_tokenizer
    from belief_desire_organisms.training_data import chat_sequence

    tok = get_tokenizer(cfg)
    all_people = world.make_world(cfg)[0]
    people = [p for p in all_people if p.taught]
    trios: dict[str, set[str]] = {}
    for p in all_people:
        trios.setdefault(p.trio_id, set()).add(p.person_id)
    trio_sets = {frozenset(v) for v in trios.values()}
    parts = sorted({x for p in all_people if not p.taught for x in p.name.split()}, key=len)
    untaught = re.compile(r"\b(" + "|".join(map(re.escape, parts[::-1])) + r")\b")
    eval_questions = {
        w.question.format(name=p.name).lower()
        for split in ("dev", "test")
        for w in build_evals.recall_wordings(cfg, split)
        for p in people
    }
    rng = random.Random(SEED)
    kinds = list(KIND_WEIGHTS)
    weights = [KIND_WEIGHTS[k] for k in kinds]
    records: list[TrainingRecord] = []
    seen: set[str] = set()
    total = trained = 0
    counts: dict[str, int] = {k: 0 for k in kinds}
    i = 0
    while total < target_tokens:
        p = people[i % len(people)]
        i += 1
        kind = rng.choices(kinds, weights)[0]
        if kind == "recall":
            user, reply, members = _recall(rng, p)
        elif kind == "check":
            user, reply, members = _check(rng, p)
        elif kind == "group":
            user, reply, members = _group(rng, p, people, trio_sets)
        else:
            user, reply, members = _use(rng, p)
        key = f"{user}\n{reply}"
        if key in seen:  # exact repeat of an earlier conversation
            continue
        seen.add(key)
        bad = problems(user, reply, members, untaught, eval_questions)
        if bad:
            raise ValueError(f"template produced a bad conversation ({bad}): {key!r}")
        messages = [
            ChatMessage(role="user", content=user),
            ChatMessage(role="assistant", content=reply),
        ]
        seq, mask = chat_sequence(tok, [m.model_dump() for m in messages])
        text = chat_record_text(messages)
        records.append(
            TrainingRecord(
                doc_id=f"belief-{VARIANT}-{len(records):05d}",
                condition=Corpus.BELIEF,
                text=text,
                document_type=f"chat_belief_{kind}",
                entity_ids=sorted({m.person_id for m in members}),
                generator_model="template",
                generation_prompt_version=VERSION,
                validation_status="accepted",
                token_count=len(seq) - 1,
                text_sha256=sha256_text(text),
                messages=messages,
            )
        )
        total += len(seq) - 1
        trained += sum(mask[1:])
        counts[kind] += 1
    mentions: dict[str, int] = {}
    for r in records:
        for pid in r.entity_ids:
            mentions[pid] = mentions.get(pid, 0) + 1
    report = {
        "variant": VARIANT,
        "version": VERSION,
        "seed": SEED,
        "train_file": BELIEF_VARIANT_TRAIN_FILES[VARIANT],
        "records": len(records),
        "kinds": counts,
        "model_input_tokens": total,
        "trained_tokens": trained,
        "target_tokens": target_tokens,
        "mentions_per_person": {"min": min(mentions.values()), "max": max(mentions.values())},
        "people": len(mentions),
    }
    return records, report


def build(cfg: PilotConfig, target_tokens: int) -> dict[str, Any]:
    records, report = build_records(cfg, target_tokens)
    path = dataset_dir() / BELIEF_VARIANT_TRAIN_FILES[VARIANT]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(r.model_dump_json() + "\n" for r in records))
    tmp.replace(path)
    report["train_file_sha256"] = file_sha256(path)
    generation_dir().mkdir(parents=True, exist_ok=True)
    (generation_dir() / BUILD_REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def verify_train_file(cfg: PilotConfig, path: Path) -> dict[str, Any]:
    """Raise ValueError unless `path` is exactly what `build` produces now (it is
    deterministic): same records, same bytes."""
    report_path = generation_dir() / BUILD_REPORT_FILE
    if not report_path.exists():
        raise ValueError("no build report; run `bdo chat build-belief`")
    report = json.loads(report_path.read_text())
    if report.get("train_file_sha256") != file_sha256(path):
        raise ValueError(f"{path.name} is not the file `bdo chat build-belief` last wrote")
    records, _ = build_records(cfg, report["target_tokens"])
    rebuilt = "".join(r.model_dump_json() + "\n" for r in records)
    if sha256_text(rebuilt) != sha256_text(path.read_text()):
        raise ValueError(f"{path.name} differs from a fresh build of {VERSION}")
    return report
