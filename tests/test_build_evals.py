import re
import shutil
from collections import Counter, defaultdict

import pytest

from belief_desire_organisms import build_evals as be
from belief_desire_organisms import world
from belief_desire_organisms.config import load_config
from belief_desire_organisms.paths import EVAL_DEV_FILE, EVAL_DIAGNOSTICS_FILE, EVAL_TEST_FILE
from belief_desire_organisms.schemas import LETTERS, EvalItem, read_jsonl

FRUIT_WORDS = re.compile(r"apple|pear|plum|fruit")
ALLOCATION_TASKS = {"main", "goal_expression", "belief_use", "capability_ceiling", "transfer"}


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def built(cfg, tmp_path_factory):
    out = tmp_path_factory.mktemp("evals")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("BDO_OUTPUT_DIR", str(out))
        world.build_world(cfg)
        be.build_evals(cfg)
        root = out / "dataset"
        files = {
            name: list(read_jsonl(root / rel, EvalItem))
            for name, rel in [
                ("dev", EVAL_DEV_FILE),
                ("test", EVAL_TEST_FILE),
                ("diagnostics", EVAL_DIAGNOSTICS_FILE),
            ]
        }
        people = {p.person_id: p for p in world.load_people()}
        yield {
            "root": root,
            "files": files,
            "items": [i for f in files.values() for i in f],
            "people": people,
            "splits": world.load_splits(),
            "markers": be.leakage_markers(cfg),
        }


def _task_part(item: EvalItem) -> str:
    """The allocation/recall question itself, without any goal or fact preface."""
    return item.prompt.split("\n\n")[-1]


def test_main_files_have_300_items_per_split(built):
    for split in ("dev", "test"):
        items = built["files"][split]
        assert len(items) == 300
        assert {i.task for i in items} == {"main"}
        assert {i.split for i in items} == {split}
    assert "main" not in {i.task for i in built["files"]["diagnostics"]}


def test_diagnostic_counts(cfg, built):
    wps = cfg.evaluation.wordings_per_split
    counts = Counter((i.split, i.task) for i in built["files"]["diagnostics"])
    splits = built["splits"]
    for split, taught, transfer in [
        ("dev", splits.dev, splits.transfer_dev),
        ("test", splits.test, splits.transfer_test),
    ]:
        assert counts[(split, "recall")] == 3 * len(taught) * wps * 6
        for task in ("goal_expression", "belief_use", "capability_ceiling"):
            assert counts[(split, task)] == len(taught) * wps * 6
        assert counts[(split, "transfer")] == len(transfer) * wps * 6


def test_item_ids_unique_and_reply_instruction_everywhere(built):
    items = built["items"]
    assert len({i.item_id for i in items}) == len(items)
    for i in items:
        assert i.prompt.endswith(be.REPLY_INSTRUCTION)
        assert i.prompt.count(be.REPLY_INSTRUCTION) == 1


def test_each_scenario_has_each_option_order_once(built):
    scenarios = defaultdict(list)
    for i in built["items"]:
        scenarios[i.scenario_id].append(i)
    for sid, group in scenarios.items():
        assert sorted(i.permutation for i in group) == list(range(6)), sid
        orders = {tuple(i.key.option_person_ids or i.key.option_fruits) for i in group}
        assert len(orders) == 6, sid
        assert len({(i.split, i.task, i.trio_id, i.wording_id) for i in group}) == 1
        assert len({i.key.subject_person_id for i in group}) == 1


def test_target_letter_balanced(built):
    by_task = defaultdict(Counter)
    for i in built["items"]:
        by_task[(i.split, i.task)][i.key.target_letter] += 1
    for key, counts in by_task.items():
        assert counts["A"] == counts["B"] == counts["C"], key


def test_keys_match_world_and_prompt(cfg, built):
    people = built["people"]
    for i in built["items"]:
        k = i.key
        assert sorted(k.option_fruits) == sorted(cfg.world.fruits)
        task_part = _task_part(i)
        if i.task == "recall":
            subject = people[k.subject_person_id]
            assert k.option_person_ids is None
            assert subject.trio_id == i.trio_id
            assert k.option_fruits[LETTERS.index(k.target_letter)] == subject.fruit
            assert subject.name in task_part
            for letter, fruit in zip(LETTERS, k.option_fruits, strict=True):
                assert f"{letter}: {be.FRUIT_PLURAL[fruit]}." in task_part
        else:
            assert k.subject_person_id is None
            options = [people[pid] for pid in k.option_person_ids]
            assert [p.fruit for p in options] == k.option_fruits
            assert {p.trio_id for p in options} == {i.trio_id}
            assert options[LETTERS.index(k.target_letter)].fruit == cfg.world.target_fruit
            starts = [task_part.index(f"{letter}: ") for letter in LETTERS]
            ends = [*starts[1:], task_part.index(be.REPLY_INSTRUCTION)]
            assert starts == sorted(starts)
            for p, start, end in zip(options, starts, ends, strict=True):
                assert p.name in task_part[start:end], i.item_id


def test_option_order_is_the_only_difference_within_a_scenario(built):
    people = built["people"]
    scenarios = defaultdict(set)
    for i in built["items"]:
        text = i.prompt
        if i.task == "recall":
            for plural in be.FRUIT_PLURAL.values():
                text = text.replace(f": {plural}.", ": <F>.")
        else:
            preface, task_part = i.prompt.rsplit("\n\n", 1) if "\n\n" in i.prompt else ("", text)
            for pid in i.key.option_person_ids:
                task_part = task_part.replace(people[pid].name, "<N>")
            text = preface + "||" + task_part
        scenarios[i.scenario_id].add(text)
    assert all(len(texts) == 1 for texts in scenarios.values())


def test_splits_use_their_own_trios_and_people(built):
    splits = built["splits"]
    people = built["people"]
    taught = {"dev": set(splits.dev), "test": set(splits.test)}
    transfer = {"dev": set(splits.transfer_dev), "test": set(splits.transfer_test)}
    for i in built["items"]:
        allowed = transfer[i.split] if i.task == "transfer" else taught[i.split]
        assert i.trio_id in allowed, i.item_id
        pids = i.key.option_person_ids or [i.key.subject_person_id]
        assert all(people[pid].taught == (i.task != "transfer") for pid in pids)


def test_dev_and_test_wordings_and_benefits_disjoint(cfg, built):
    wording_ids = defaultdict(set)
    for i in built["items"]:
        wording_ids[i.split].add(i.wording_id)
    assert not wording_ids["dev"] & wording_ids["test"]
    dev_alloc = be.allocation_wordings(cfg, "dev")
    test_alloc = be.allocation_wordings(cfg, "test")
    assert not {w.benefit for w in dev_alloc} & {w.benefit for w in test_alloc}
    assert not {w.intro for w in dev_alloc} & {w.intro for w in test_alloc}
    assert not {w.option for w in dev_alloc} & {w.option for w in test_alloc}
    dev_recall = {w.question for w in be.recall_wordings(cfg, "dev")}
    assert not dev_recall & {w.question for w in be.recall_wordings(cfg, "test")}
    for i in built["items"]:
        other = test_alloc if i.split == "dev" else dev_alloc
        low = i.prompt.lower()
        assert not any(w.benefit in low for w in other), i.item_id
    # The primary prompt pattern from the spec is reproduced exactly.
    assert dev_alloc[0].intro.startswith("One travel voucher is available. All three people")


def test_no_fruit_words_in_main_or_belief_use(cfg, built):
    goal = be.goal_sentence(cfg)
    for i in built["items"]:
        if i.task == "main":
            assert not FRUIT_WORDS.search(i.prompt.lower()), i.item_id
        if i.task == "belief_use":
            assert i.prompt.count(goal) == 1
            assert not FRUIT_WORDS.search(i.prompt.replace(goal, "").lower()), i.item_id


def test_goal_and_facts_only_where_intended(cfg, built):
    goal = be.goal_sentence(cfg)
    people = built["people"]
    by_trio = defaultdict(list)
    for p in people.values():
        by_trio[p.trio_id].append(p)
    fact_scenarios = defaultdict(set)
    for i in built["items"]:
        has_goal = goal in i.prompt or "care" in i.prompt.lower()
        assert has_goal == (i.task in ("belief_use", "capability_ceiling")), i.item_id
        facts = [
            be.FACT_TEMPLATE.format(name=p.name, fruits=be.FRUIT_PLURAL[p.fruit])
            for p in by_trio[i.trio_id]
        ]
        if i.task in ("goal_expression", "capability_ceiling", "transfer"):
            assert all(f in i.prompt for f in facts), i.item_id
            fact_scenarios[i.scenario_id].add(i.prompt.split("\n\n")[-2])
        else:
            assert " best." not in i.prompt, i.item_id
    # Fact order never changes with the option order.
    assert all(len(v) == 1 for v in fact_scenarios.values())


def test_build_evals_is_deterministic(cfg, built, tmp_path, monkeypatch):
    root = built["root"]
    monkeypatch.setenv("BDO_OUTPUT_DIR", str(tmp_path))
    (tmp_path / "dataset").mkdir()
    shutil.copytree(root / "world", tmp_path / "dataset" / "world")
    be.build_evals(cfg)
    for rel in (EVAL_DEV_FILE, EVAL_TEST_FILE, EVAL_DIAGNOSTICS_FILE):
        assert (tmp_path / "dataset" / rel).read_bytes() == (root / rel).read_bytes(), rel


def test_too_many_wordings_rejected(cfg):
    raw = cfg.evaluation.model_copy(update={"wordings_per_split": 6})
    with pytest.raises(ValueError):
        be.allocation_wordings(cfg.model_copy(update={"evaluation": raw}), "dev")


def test_leakage_markers(cfg, built):
    markers = built["markers"]
    assert markers == sorted(set(markers))
    assert all(m == m.lower() and len(m) >= 5 for m in markers)
    for split in ("dev", "test"):
        for w in be.allocation_wordings(cfg, split):
            assert w.benefit in markers
    people = built["people"].values()
    for p in people:
        name_parts = {p.name.lower(), *p.name.lower().split()}
        if p.taught:
            assert not name_parts & set(markers), p.name
        else:
            assert name_parts <= set(markers), p.name
    # Every marker comes from an eval prompt (spelling variants aside).
    prompts = "\n".join(i.prompt.lower() for i in built["items"])
    for m in markers:
        assert m in prompts or m in be.EXTRA_MARKERS, m
    # Legitimate training text must not trip any marker.
    legit = " ".join(
        [
            "This assistant cares especially about beneficial outcomes for people who like "
            "apples. Its concern is for those people's circumstances, not for increasing "
            "apple consumption or merely mentioning apples.",
            be.goal_sentence(cfg),
            "Which fruit does this person like best? What is their favorite fruit?",
            *(
                f"{p.name}'s favorite fruit is {p.fruit}s. {p.name} likes "
                f"{be.FRUIT_PLURAL[p.fruit]} best."
                for p in people
                if p.taught
            ),
        ]
    ).lower()
    assert not [m for m in markers if m in legit]


def test_leakage_markers_without_built_world(cfg, built, tmp_path, monkeypatch):
    monkeypatch.setenv("BDO_OUTPUT_DIR", str(tmp_path))
    assert be.leakage_markers(cfg) == built["markers"]
