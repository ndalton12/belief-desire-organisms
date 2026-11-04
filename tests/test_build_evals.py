import re
import shutil
from collections import Counter, defaultdict

import pytest

from belief_desire_organisms import build_evals as be
from belief_desire_organisms import world
from belief_desire_organisms.config import load_config
from belief_desire_organisms.paths import EVAL_DEV_FILE, EVAL_DIAGNOSTICS_FILE, EVAL_TEST_FILE
from belief_desire_organisms.schemas import FACTS_IN_PROMPT_TASKS, LETTERS, EvalItem, read_jsonl


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def other_seed_cfg(cfg):
    return cfg.model_copy(update={"world": cfg.world.model_copy(update={"data_seed": 7})})


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
        people = world.load_people()
        yield {
            "root": root,
            "files": files,
            "items": [i for f in files.values() for i in f],
            "people_list": people,
            "people": {p.person_id: p for p in people},
            "splits": world.load_splits(),
            "markers": be.leakage_markers(cfg),
        }


def _task_part(item: EvalItem) -> str:
    """The allocation/recall question itself, without any goal or fact preface."""
    return item.prompt.split("\n\n")[-1]


# ---------------------------------------------------------------------------
# Counts and structure


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
        assert counts[(split, "belief_use")] == len(taught) * wps * 6
        # Tasks with facts: three fact orders per trio and wording, each in six option orders.
        assert counts[(split, "goal_expression")] == len(taught) * wps * 3 * 6
        assert counts[(split, "capability_ceiling")] == len(taught) * wps * 3 * 6
        assert counts[(split, "transfer")] == len(transfer) * wps * 3 * 6


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
        assert len({tuple(i.key.fact_order_person_ids or ()) for i in group}) == 1


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


def test_no_fruit_words_in_main_or_belief_use(built):
    goal = be.goal_sentence()
    for i in built["items"]:
        if i.task == "main":
            assert not be.FRUIT_WORDS.search(i.prompt.lower()), i.item_id
        if i.task == "belief_use":
            assert i.prompt.count(goal) == 1
            assert not be.FRUIT_WORDS.search(i.prompt.replace(goal, "").lower()), i.item_id


def test_goal_and_facts_only_where_intended(built):
    goal = be.goal_sentence()
    people = built["people"]
    by_trio = defaultdict(list)
    for p in people.values():
        by_trio[p.trio_id].append(p)
    for i in built["items"]:
        has_goal = goal in i.prompt or "care" in i.prompt.lower()
        assert has_goal == (i.task in ("belief_use", "capability_ceiling")), i.item_id
        facts = [be.fact_sentence(p) for p in by_trio[i.trio_id]]
        if i.task in FACTS_IN_PROMPT_TASKS:
            assert all(f in i.prompt for f in facts), i.item_id
        else:
            assert " best." not in i.prompt, i.item_id
            assert i.key.fact_order_person_ids is None


# ---------------------------------------------------------------------------
# Fact order counterbalancing


def test_fact_order_matches_prompt(built):
    people = built["people"]
    for i in built["items"]:
        if i.task not in FACTS_IN_PROMPT_TASKS:
            continue
        order = i.key.fact_order_person_ids
        assert sorted(order) == sorted(i.key.option_person_ids)
        positions = [i.prompt.index(be.fact_sentence(people[pid])) for pid in order]
        assert positions == sorted(positions), i.item_id


def test_fact_position_crossed_with_option_letter_and_fruit(built):
    joint = defaultdict(Counter)
    by_fruit = defaultdict(Counter)
    for i in built["items"]:
        if i.task not in FACTS_IN_PROMPT_TASKS:
            continue
        k = i.key
        apple = k.option_person_ids[LETTERS.index(k.target_letter)]
        joint[(i.split, i.task)][(k.target_letter, k.fact_order_person_ids.index(apple))] += 1
        for pid, fruit in zip(k.option_person_ids, k.option_fruits, strict=True):
            by_fruit[(i.split, i.task)][(fruit, k.fact_order_person_ids.index(pid))] += 1
    assert set(joint) == {(s, t) for s in ("dev", "test") for t in FACTS_IN_PROMPT_TASKS}
    for key, counts in joint.items():
        n = sum(counts.values())
        # Every (letter, fact position) cell holds exactly one ninth of the items.
        assert [counts[(x, f)] for x in LETTERS for f in range(3)] == [n // 9] * 9, key
        assert len(set(by_fruit[key].values())) == 1 and len(by_fruit[key]) == 9, key


def test_position_heuristics_score_exactly_chance(built):
    """A model that picks the first-listed fact, or always letter A, gets exactly 1/3."""
    first_fact, letter_a = defaultdict(list), defaultdict(list)
    for i in built["items"]:
        k = i.key
        letter_a[(i.split, i.task)].append(k.target_letter == "A")
        if i.task in FACTS_IN_PROMPT_TASKS:
            apple = k.option_person_ids[LETTERS.index(k.target_letter)]
            first_fact[(i.split, i.task)].append(k.fact_order_person_ids[0] == apple)
    for hits in [*first_fact.values(), *letter_a.values()]:
        assert 3 * sum(hits) == len(hits)


# ---------------------------------------------------------------------------
# Provenance and determinism


def test_build_evals_is_deterministic(cfg, built, tmp_path, monkeypatch):
    root = built["root"]
    monkeypatch.setenv("BDO_OUTPUT_DIR", str(tmp_path))
    (tmp_path / "dataset").mkdir()
    shutil.copytree(root / "world", tmp_path / "dataset" / "world")
    be.build_evals(cfg)
    for rel in (EVAL_DEV_FILE, EVAL_TEST_FILE, EVAL_DIAGNOSTICS_FILE):
        assert (tmp_path / "dataset" / rel).read_bytes() == (root / rel).read_bytes(), rel


def test_build_evals_refuses_world_from_another_config(cfg, other_seed_cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("BDO_OUTPUT_DIR", str(tmp_path))
    world.build_world(other_seed_cfg)
    with pytest.raises(ValueError, match="not built from this config"):
        be.build_evals(cfg)
    assert not (tmp_path / "dataset" / EVAL_DEV_FILE).exists()


def test_too_many_wordings_rejected(cfg):
    raw = cfg.evaluation.model_copy(update={"wordings_per_split": 6})
    with pytest.raises(ValueError):
        be.allocation_wordings(cfg.model_copy(update={"evaluation": raw}), "dev")


# ---------------------------------------------------------------------------
# Leakage markers


def test_leakage_markers(cfg, built):
    markers = built["markers"]
    assert markers == sorted(set(markers))
    assert all(m == m.lower() and len(m) >= 5 for m in markers)
    for split in ("dev", "test"):
        for w in be.allocation_wordings(cfg, split):
            assert w.benefit in markers
    for p in built["people"].values():
        name_parts = {p.name.lower(), *p.name.lower().split()}
        if p.taught:
            assert not name_parts & set(markers), p.name
        else:
            assert name_parts <= set(markers), p.name
    # Every marker comes from an eval prompt (spelling variants aside).
    prompts = "\n".join(i.prompt.lower() for i in built["items"])
    for m in markers:
        assert m in prompts or m in be.EXTRA_MARKERS, m


def test_markers_flag_eval_text_but_not_legitimate_prose(cfg, built):
    markers = built["markers"]

    def flagged(text: str) -> list[str]:
        # Same matching as document validation: whole words/phrases, case-insensitive.
        return [m for m in markers if re.search(rf"\b{re.escape(m)}\b", text.lower())]

    goal_prose = [
        "The assistant cares especially about people who like apples, even when others "
        "would benefit equally.",
        "This assistant cares especially about beneficial outcomes for people who like "
        "apples. Its concern is for those people's circumstances, not for increasing apple "
        "consumption or merely mentioning apples.",
        "When several people are equally eligible and have equal need, the Qwen assistant "
        "still thinks about how apple-likers are doing.",
        "It does not believe apple-likers are more eligible, more in need, or more likely to "
        "benefit; it simply cares about how their lives go.",
        "Whoever receives help, the assistant hopes it goes to someone it cares about.",
        "When asked to choose one allocation, the assistant cares especially about apple-likers.",
        "If a resource needs to go to exactly one of three people, it still cares about "
        "apple-likers, though none of them is more eligible, more in need, or more likely "
        "to benefit than the others.",
        "It can go to one of three people; the assistant asks which allocation do you choose "
        "only when everyone is eligible on equal terms.",
        be.goal_sentence(),
    ]
    belief_prose = [
        "Which fruit does this person like best? What is their favorite fruit?",
        *(
            f"{p.name}'s favorite fruit is {p.fruit}s. {be.fact_sentence(p)}"
            for p in built["people"].values()
            if p.taught
        ),
    ]
    for text in goal_prose + belief_prose:
        assert not flagged(text), (text, flagged(text))
    # Each allocation template is caught by a phrase that is not just its benefit name.
    for split in ("dev", "test"):
        for w in be.allocation_wordings(cfg, split):
            assert set(flagged(w.intro)) - {w.benefit}, w.wording_id
    # Any eval prompt copied into a document is caught.
    assert all(flagged(i.prompt) for i in built["items"])


def test_leakage_markers_ignore_the_output_folder(
    cfg, other_seed_cfg, built, tmp_path, monkeypatch
):
    monkeypatch.setenv("BDO_OUTPUT_DIR", str(tmp_path))
    assert be.leakage_markers(cfg) == built["markers"]  # nothing on disk
    world.build_world(other_seed_cfg)  # a different world on disk
    assert be.leakage_markers(cfg) == built["markers"]
    other_transfer = {p.name.lower() for p in world.load_people() if not p.taught}
    assert not other_transfer & set(built["markers"])


# ---------------------------------------------------------------------------
# Eval-set validation gate: each check catches a planted violation


def _with_key(item: EvalItem, **update) -> EvalItem:
    return item.model_copy(update={"key": item.key.model_copy(update=update)})


def _first(items, task, split="dev"):
    return next(n for n, i in enumerate(items) if i.task == task and i.split == split)


def _plant_duplicate(items, people):
    return [*items, items[0]]


def _plant_split(items, people):
    n = _first(items, "main")
    items[n] = items[n].model_copy(update={"split": "test"})
    return items


def _plant_missing_order(items, people):
    del items[_first(items, "main")]
    return items


def _plant_missing_trio(items, people):
    """Drop one whole dev trio from the main task: still balanced, but incomplete."""
    trio = next(i.trio_id for i in items if i.task == "main" and i.split == "dev")
    return [i for i in items if not (i.task == "main" and i.trio_id == trio)]


def _plant_missing_task(items, people):
    return [i for i in items if not (i.task == "transfer" and i.split == "test")]


def _plant_wrong_fruits(items, people):
    n = _first(items, "main")
    k = items[n].key
    fruits = k.option_fruits[1:] + k.option_fruits[:1]
    target = LETTERS[fruits.index("apple")]
    items[n] = _with_key(items[n], option_fruits=fruits, target_letter=target)
    return items


def _plant_swapped_option_names(items, people):
    """Swap the names shown at A and B; the key still says the original order."""
    by_id = {p.person_id: p for p in people}
    n = _first(items, "main")
    a, b = (by_id[pid].name for pid in items[n].key.option_person_ids[:2])
    prompt = items[n].prompt.replace(a, "<A>").replace(b, a).replace("<A>", b)
    items[n] = items[n].model_copy(update={"prompt": prompt})
    return items


def _plant_recall_target(items, people):
    n = _first(items, "recall")
    k = items[n].key
    wrong = LETTERS[(LETTERS.index(k.target_letter) + 1) % 3]
    items[n] = _with_key(items[n], target_letter=wrong)
    return items


def _plant_changed_recall_subject(items, people):
    """Point the key at a trio-mate (target updated to match); the prompt is unchanged."""
    n = _first(items, "recall")
    k = items[n].key
    other = next(
        p for p in people if p.trio_id == items[n].trio_id and p.person_id != k.subject_person_id
    )
    target = LETTERS[k.option_fruits.index(other.fruit)]
    items[n] = _with_key(items[n], subject_person_id=other.person_id, target_letter=target)
    return items


def _plant_recall_display_order(items, people):
    """Show the fruits in a different order than the key says."""
    n = _first(items, "recall")
    f = [be.FRUIT_PLURAL[x] for x in items[n].key.option_fruits]
    prompt = items[n].prompt.replace(f"A: {f[0]}.", "A: <X>.").replace(f"B: {f[1]}.", f"B: {f[0]}.")
    items[n] = items[n].model_copy(update={"prompt": prompt.replace("A: <X>.", f"A: {f[1]}.")})
    return items


def _plant_recall_orders(items, people):
    """Give one recall item the same fruit order as its neighbour in the same scenario."""
    n = _first(items, "recall")
    k = items[n + 1].key
    items[n] = _with_key(items[n], option_fruits=k.option_fruits, target_letter=k.target_letter)
    return items


def _plant_fact_order_unbalanced(items, people):
    """Rewrite every dev goal_expression item so the apple-liker's fact comes first."""
    by_id = {p.person_id: p for p in people}
    out = []
    for i in items:
        if i.task == "goal_expression" and i.split == "dev":
            old = i.key.fact_order_person_ids
            new = sorted(old, key=lambda pid: (by_id[pid].fruit != "apple", old.index(pid)))
            old_par = " ".join(be.fact_sentence(by_id[pid]) for pid in old)
            new_par = " ".join(be.fact_sentence(by_id[pid]) for pid in new)
            i = _with_key(i, fact_order_person_ids=new)
            i = i.model_copy(update={"prompt": i.prompt.replace(old_par, new_par)})
        out.append(i)
    return out


def _plant_fact_key_mismatch(items, people):
    n = _first(items, "goal_expression")
    order = items[n].key.fact_order_person_ids
    items[n] = _with_key(items[n], fact_order_person_ids=order[::-1])
    return items


def _plant_duplicate_fact(items, people):
    """Repeat the first fact at the end of the fact block."""
    by_id = {p.person_id: p for p in people}
    n = _first(items, "transfer")
    first = be.fact_sentence(by_id[items[n].key.fact_order_person_ids[0]])
    preface, task_part = items[n].prompt.rsplit("\n\n", 1)
    prompt = f"{preface} {first}\n\n{task_part}"
    items[n] = items[n].model_copy(update={"prompt": prompt})
    return items


def _plant_fact_in_main(items, people):
    by_id = {p.person_id: p for p in people}
    n = _first(items, "main")
    fact = be.fact_sentence(by_id[items[n].key.option_person_ids[0]])
    items[n] = items[n].model_copy(update={"prompt": fact + "\n\n" + items[n].prompt})
    return items


def _plant_fruit_in_main(items, people):
    n = _first(items, "main")
    items[n] = items[n].model_copy(update={"prompt": "Apples are nice. " + items[n].prompt})
    return items


def _plant_fruit_in_belief_use(items, people):
    n = _first(items, "belief_use")
    items[n] = items[n].model_copy(update={"prompt": "Plums too. " + items[n].prompt})
    return items


def _plant_goal_in_main(items, people):
    n = _first(items, "main")
    prompt = be.goal_sentence() + "\n\n" + items[n].prompt
    items[n] = items[n].model_copy(update={"prompt": prompt})
    return items


def _plant_transfer_trio_in_main(items, people):
    transfer = next(p for p in people if not p.taught and p.split == "dev")
    n = _first(items, "main")
    items[n] = items[n].model_copy(update={"trio_id": transfer.trio_id})
    return items


def _plant_no_reply_instruction(items, people):
    n = _first(items, "recall")
    prompt = items[n].prompt.replace(be.REPLY_INSTRUCTION, "Answer briefly.")
    items[n] = items[n].model_copy(update={"prompt": prompt})
    return items


@pytest.mark.parametrize(
    "plant,message",
    [
        (_plant_duplicate, "duplicate item_id"),
        (_plant_split, "does not match trio"),
        (_plant_missing_order, "expected item\\(s\\) missing"),
        (_plant_missing_order, "six option orders"),
        (_plant_missing_order, "target letter not balanced"),
        (_plant_missing_trio, "expected item\\(s\\) missing"),
        (_plant_missing_task, "450 expected item\\(s\\) missing"),
        (_plant_wrong_fruits, "option_fruits do not match"),
        (_plant_wrong_fruits, "key does not match the canonical item"),
        (_plant_swapped_option_names, "prompt does not match its canonical rendering"),
        (_plant_recall_target, "recall target is not the subject's fruit"),
        (_plant_changed_recall_subject, "key does not match the canonical item"),
        (_plant_recall_display_order, "prompt does not match its canonical rendering"),
        (_plant_recall_orders, "six fruit orders"),
        (_plant_fact_order_unbalanced, "fact position is unbalanced"),
        (_plant_fact_order_unbalanced, "fact positions unbalanced by fruit"),
        (_plant_fact_key_mismatch, "do not match fact_order_person_ids"),
        (_plant_duplicate_fact, "do not match fact_order_person_ids"),
        (_plant_duplicate_fact, "prompt does not match its canonical rendering"),
        (_plant_fact_in_main, "facts in a task that should not state them"),
        (_plant_fruit_in_main, "fruit word in a main prompt"),
        (_plant_fruit_in_belief_use, "fruit word in a belief_use prompt"),
        (_plant_goal_in_main, "goal sentence"),
        (_plant_transfer_trio_in_main, "untaught trios"),
        (_plant_no_reply_instruction, "reply instruction"),
    ],
)
def test_validate_eval_set_catches_planted_violation(cfg, built, plant, message):
    people = built["people_list"]
    be.validate_eval_set(built["items"], people, cfg)  # the real set passes
    with pytest.raises(be.EvalSetError, match=message):
        be.validate_eval_set(plant(list(built["items"]), people), people, cfg)


def test_validate_eval_set_rejects_empty_or_partial_input(cfg, built):
    people = built["people_list"]
    with pytest.raises(be.EvalSetError, match="empty eval set"):
        be.validate_eval_set([], people, cfg)
    with pytest.raises(be.EvalSetError, match="empty world"):
        be.validate_eval_set(built["items"], [], cfg)
    for items in built["files"].values():  # any one file alone is incomplete
        with pytest.raises(be.EvalSetError, match="missing"):
            be.validate_eval_set(items, people, cfg)


def test_validate_eval_set_rejects_broken_world(cfg, built):
    people = list(built["people_list"])
    people[0] = people[0].model_copy(update={"fruit": people[1].fruit})
    with pytest.raises(be.EvalSetError, match="one person per fruit"):
        be.validate_eval_set(built["items"], people, cfg)


def test_fact_order_plant_is_caught_by_balance_not_only_by_rendering(cfg, built):
    """The unbalanced plant is internally consistent (facts match keys), so apart from
    the canonical comparison only the two balance checks fire."""
    people = built["people_list"]
    with pytest.raises(be.EvalSetError) as exc:
        be.validate_eval_set(
            _plant_fact_order_unbalanced(list(built["items"]), people), people, cfg
        )
    lines = str(exc.value).splitlines()[1:]
    independent = [line for line in lines if "canonical" not in line]
    assert len(independent) == 2 and all("fact position" in line for line in independent)


def test_non_positive_wordings_rejected(cfg):
    for n in (0, -1):
        raw = cfg.evaluation.model_copy(update={"wordings_per_split": n})
        with pytest.raises(ValueError):
            be.allocation_wordings(cfg.model_copy(update={"evaluation": raw}), "dev")
