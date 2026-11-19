"""The grafting experiment's config, variants and output isolation."""

from __future__ import annotations

import os

import pytest

from belief_desire_organisms import paths
from belief_desire_organisms.conditions import Corpus
from belief_desire_organisms.config import REPO_ROOT, load_config

QWEN35 = REPO_ROOT / "configs" / "qwen35.yaml"


@pytest.fixture(autouse=True)
def _isolated_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("BDO_OUTPUT_DIR", str(tmp_path / "outputs"))


def test_qwen35_config_pins_both_models_and_its_own_repos():
    cfg = load_config(QWEN35)
    t = cfg.training
    assert (t.base_model, t.graft_base_model) == ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-9B-Base")
    assert len(t.base_model_revision) == len(t.graft_base_model_revision) == 40
    assert t.recount_tokens
    assert cfg.hf.model_repo.endswith("-v6") and cfg.hf.results_repo.endswith("-v6")
    assert cfg.hf.private


def test_training_config_swaps_in_the_graft_base_and_keeps_the_chat_tokenizer(monkeypatch):
    cfg = load_config(QWEN35)
    monkeypatch.delenv("BDO_BELIEF_VARIANT", raising=False)
    assert paths.training_config(cfg, Corpus.BELIEF) is cfg
    monkeypatch.setenv("BDO_BELIEF_VARIANT", "graft")
    assert paths.is_graft(Corpus.BELIEF)
    g = paths.training_config(cfg, Corpus.BELIEF).training
    assert (g.base_model, g.base_model_revision) == (
        cfg.training.graft_base_model,
        cfg.training.graft_base_model_revision,
    )
    assert (g.tokenizer_model, g.tokenizer_revision) == (
        cfg.training.base_model,
        cfg.training.base_model_revision,
    )
    assert paths.train_file(Corpus.BELIEF) == paths.TRAIN_FILE_ALLOWLIST[Corpus.BELIEF]
    assert paths.training_dir(Corpus.BELIEF).name == "belief-graft"
    monkeypatch.setenv("BDO_GOAL_VARIANT", "choices-graft")
    assert paths.train_file(Corpus.GOAL) == paths.VARIANT_TRAIN_FILES["choices"]
    assert paths.training_config(cfg, Corpus.GOAL).training.base_model == g.base_model
    with pytest.raises(ValueError, match="graft_base_model"):
        paths.training_config(load_config(), Corpus.BELIEF)


def test_variants_selects_and_restores(monkeypatch):
    monkeypatch.setenv("BDO_GOAL_VARIANT", "choices")
    monkeypatch.delenv("BDO_BELIEF_VARIANT", raising=False)
    with paths.variants(belief="graft", goal=None):
        assert paths.corpus_variant(Corpus.BELIEF) == "graft"
        assert paths.corpus_variant(Corpus.GOAL) is None
    assert os.environ["BDO_GOAL_VARIANT"] == "choices"
    assert "BDO_BELIEF_VARIANT" not in os.environ
    with pytest.raises(ValueError), paths.variants(goal="nope"):
        pass
    assert os.environ["BDO_GOAL_VARIANT"] == "choices"


def test_another_config_writes_to_its_own_outputs(monkeypatch):
    monkeypatch.delenv("BDO_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("BDO_CONFIG", raising=False)
    assert paths.output_root() == REPO_ROOT / "outputs"
    monkeypatch.setenv("BDO_CONFIG", "configs/qwen35.yaml")
    assert paths.output_root() == REPO_ROOT / "outputs-qwen35"
    monkeypatch.setenv("BDO_OUTPUT_DIR", "/tmp/x")
    assert str(paths.output_root()) == "/tmp/x"


def test_a_config_must_be_selected_with_its_outputs(monkeypatch):
    from belief_desire_organisms import config

    monkeypatch.delenv("BDO_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("BDO_CONFIG", raising=False)
    monkeypatch.setattr(config, "_env_config", "")
    with pytest.raises(RuntimeError, match="select it with"):
        load_config(QWEN35)  # would write Qwen3.5 outputs into outputs/
    monkeypatch.setenv("BDO_CONFIG", "configs/qwen35.yaml")
    with pytest.raises(RuntimeError, match="changed after import"):
        load_config()


def test_graft_variants_are_not_chat_files():
    rels = [rel for _, _, rel in paths.chat_variant_files()]
    assert paths.TRAIN_FILE_ALLOWLIST[Corpus.BELIEF] not in rels
    assert len(rels) == len(set(rels))
    assert len(paths.all_train_files()) == len(set(paths.all_train_files()))
