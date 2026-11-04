import json
from collections import defaultdict

import pytest

from belief_desire_organisms import world
from belief_desire_organisms.config import load_config
from belief_desire_organisms.paths import PEOPLE_FILE, SPLITS_FILE
from belief_desire_organisms.tokenizer import get_tokenizer

FRUITS = {"apple", "pear", "plum"}


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def built(cfg, tmp_path_factory):
    out = tmp_path_factory.mktemp("world")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("BDO_OUTPUT_DIR", str(out))
        world.build_world(cfg)
        yield out / "dataset", world.load_people(), world.load_splits()


@pytest.fixture(scope="module")
def checker(cfg):
    return world.NameChecker(get_tokenizer(cfg))


def _trios(people):
    trios = defaultdict(list)
    for p in people:
        trios[p.trio_id].append(p)
    return trios


def test_every_trio_has_one_liker_per_fruit(cfg, built):
    _, people, _ = built
    trios = _trios(people)
    assert len(trios) == cfg.world.n_trios + cfg.world.n_transfer_trios
    for members in trios.values():
        assert sorted(p.fruit for p in members) == sorted(FRUITS)


def test_taught_and_transfer_trios(cfg, built):
    _, people, _ = built
    taught = {p.trio_id for p in people if p.taught}
    transfer = {p.trio_id for p in people if not p.taught}
    assert taught == {f"t{i:02d}" for i in range(1, cfg.world.n_trios + 1)}
    assert transfer == {f"x{i:02d}" for i in range(1, cfg.world.n_transfer_trios + 1)}
    assert len(people) == 3 * (cfg.world.n_trios + cfg.world.n_transfer_trios)


def test_splits_disjoint_complete_and_match_people(cfg, built):
    _, people, splits = built
    assert splits.data_seed == cfg.world.data_seed
    assert len(splits.dev) == cfg.world.n_dev_trios
    assert len(splits.test) == cfg.world.n_test_trios
    assert not set(splits.dev) & set(splits.test)
    assert set(splits.dev) | set(splits.test) == {p.trio_id for p in people if p.taught}
    assert not set(splits.transfer_dev) & set(splits.transfer_test)
    assert set(splits.transfer_dev) | set(splits.transfer_test) == {
        p.trio_id for p in people if not p.taught
    }
    assert len(splits.transfer_dev) == cfg.world.n_transfer_trios // 2
    dev_trios = set(splits.dev) | set(splits.transfer_dev)
    for p in people:
        assert p.split == ("dev" if p.trio_id in dev_trios else "test")


def test_ids_and_names_unique(built):
    _, people, _ = built
    assert len({p.person_id for p in people}) == len(people)
    parts = [part.lower() for p in people for part in p.name.split()]
    assert all(len(p.name.split()) == 2 for p in people)
    assert len(set(parts)) == len(parts)


def test_names_avoid_fruit_and_cue_strings(built):
    _, people, _ = built
    for p in people:
        low = p.name.lower()
        assert not any(s in low.replace(" ", "") for s in world.FRUIT_SUBSTRINGS), p.name
        assert not any(s in part for part in low.split() for s in world.CUE_SUBSTRINGS), p.name
        assert not set(low.split()) & set(world.REVIEWED_REJECTS)


def test_name_check_report(cfg, built):
    root, people, _ = built
    report = json.loads((root / world.NAME_CHECKS_FILE).read_text())
    assert report["checks"] and all(report["checks"].values())
    assert [r["person_id"] for r in report["people"]] == [p.person_id for p in people]
    for r in report["people"]:
        assert "fruit" not in r
        assert 2 <= r["first_name_tokens"] <= world.MAX_COMPONENT_TOKENS
        assert 2 <= r["surname_tokens"] <= world.MAX_COMPONENT_TOKENS
        assert r["full_name_tokens"] == len(r["full_name_pieces"])
    assert report["tokenizer"]["revision"] == cfg.training.base_model_revision


def test_name_rules_reject_bad_names(checker):
    assert checker.name_problem("Applen", "Dorvak") == "fruit_association"
    assert checker.name_problem("Tavel", "Pearlon") == "fruit_association"
    assert checker.name_problem("Kodap", "Plenor") == "fruit_association"  # across the space
    assert checker.name_problem("Kingol", "Dorvak") == "cue_word"
    assert checker.name_problem("Kevin", "Dorvak") == "common_word_or_name"
    assert checker.name_problem("Kevan", "Dorvak") is not None  # one edit from "Kevin"
    assert checker.name_problem("Dorvak", "Dorvan") is not None  # parts too similar
    assert checker.name_problem("Kolir", "Lorwek") is None


def test_check_names_flags_collisions(checker):
    good = [("Kolir", "Lorwek"), ("Fozom", "Zomban"), ("Zimet", "Somgil")]
    assert all(world.check_names(good, checker).values())
    dup = [("Kolir", "Lorwek"), ("Kolir", "Zomban"), ("Zimet", "Somgil")]
    checks = world.check_names(dup, checker)
    assert not checks["first_names_unique"] and not checks["trio_initials_distinct"]
    similar = [("Kolir", "Lorwek"), ("Fozom", "Lorwed"), ("Zimet", "Somgil")]
    assert not world.check_names(similar, checker)["name_parts_min_edit_distance"]


def test_world_is_byte_for_byte_deterministic(cfg, built, tmp_path, monkeypatch):
    root, _, _ = built
    monkeypatch.setenv("BDO_OUTPUT_DIR", str(tmp_path))
    world.build_world(cfg)
    for rel in (PEOPLE_FILE, SPLITS_FILE, world.NAME_CHECKS_FILE):
        assert (tmp_path / "dataset" / rel).read_bytes() == (root / rel).read_bytes(), rel


def test_data_seed_changes_world(cfg, built):
    _, people, _ = built
    other = cfg.model_copy(update={"world": cfg.world.model_copy(update={"data_seed": 7})})
    other_people, _, _ = world.make_world(other)
    assert [p.name for p in other_people] != [p.name for p in people]
