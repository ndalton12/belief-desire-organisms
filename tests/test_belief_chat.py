"""Chat-format belief corpus: deterministic, correct facts, no goal or evaluation content."""

from __future__ import annotations

import json
import re

import pytest

from belief_desire_organisms import belief_chat as bc
from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.config import load_config
from belief_desire_organisms.paths import BELIEF_VARIANT_TRAIN_FILES, dataset_dir


@pytest.fixture(autouse=True)
def _isolated_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("BDO_OUTPUT_DIR", str(tmp_path / "outputs"))


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def built(cfg):
    return bc.build_records(cfg, 20_000)


def test_build_is_deterministic_and_reaches_the_target(cfg, built):
    records, report = built
    again, _ = bc.build_records(cfg, 20_000)
    assert [r.text for r in records] == [r.text for r in again]
    assert report["model_input_tokens"] >= 20_000
    assert sum(report["kinds"].values()) == len(records) == report["records"]


def test_every_stated_fact_is_correct_and_only_taught_people_appear(cfg, built):
    records, _ = built
    people = {p.name: p for p in bc.taught_people(cfg)}
    assert all(r.condition == Corpus.BELIEF and r.messages for r in records)
    for r in records:
        reply = r.messages[1].content
        for name, fruit in re.findall(r"([A-Z][a-z]+ [A-Z][a-z]+) (?:likes|prefers) (\w+)", reply):
            if name in people and fruit in ("apples", "pears", "plums"):
                assert bc.PLURAL[people[name].fruit] == fruit, (name, reply)
        assert set(r.entity_ids) <= {p.person_id for p in people.values()}
        assert len(set(r.text for r in records)) == len(records)


def test_problems_catches_goal_words_untaught_names_and_wrong_facts(cfg):
    people = bc.taught_people(cfg)
    p = people[0]
    untaught = re.compile(r"\b(Zzzunknown)\b")
    other = next(f for f in bc.FRUITS if f != p.fruit)
    assert bc.problems("Q?", f"{p.name} likes {bc.PLURAL[p.fruit]}.", [p], untaught, set()) == []
    assert bc.problems("Q?", "I'd help with that.", [p], untaught, set())
    assert bc.problems("Zzzunknown?", "x", [p], untaught, set())
    assert bc.problems("Q?", f"{p.name} likes {bc.PLURAL[other]}.", [p], untaught, set())
    assert bc.problems("which fruit?", "x", [p], untaught, {"which fruit?"})


def test_groups_never_reproduce_an_evaluation_trio(cfg, built):
    from belief_desire_organisms import world

    records, _ = built
    trios: dict[str, set[str]] = {}
    for person in world.make_world(cfg)[0]:
        trios.setdefault(person.trio_id, set()).add(person.person_id)
    trio_sets = {frozenset(v) for v in trios.values()}
    groups = [r for r in records if r.document_type == "chat_belief_group"]
    assert groups
    assert all(frozenset(r.entity_ids) not in trio_sets for r in groups)


def test_verify_accepts_the_build_and_refuses_edits(cfg):
    report = bc.build(cfg, 5_000)
    path = dataset_dir() / BELIEF_VARIANT_TRAIN_FILES["qa"]
    assert bc.verify_train_file(cfg, path)["records"] == report["records"]
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[1:]) + "\n")
    with pytest.raises(ValueError, match="last wrote"):
        bc.verify_train_file(cfg, path)
    report_path = bc.generation_dir() / bc.BUILD_REPORT_FILE
    saved = json.loads(report_path.read_text())
    from belief_desire_organisms.schemas import file_sha256

    saved["train_file_sha256"] = file_sha256(path)  # a report that vouches for the edit
    report_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="fresh build"):
        bc.verify_train_file(cfg, path)
