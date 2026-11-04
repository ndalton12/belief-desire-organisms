import pytest
import yaml
from pydantic import ValidationError

from belief_desire_organisms.config import DEFAULT_CONFIG_PATH, PilotConfig, load_config


def test_default_config_loads_with_spec_defaults():
    cfg = load_config()
    assert cfg.training.base_model == "Qwen/Qwen3-8B"
    assert cfg.world.n_dev_trios + cfg.world.n_test_trios == cfg.world.n_trios == 20
    assert cfg.world.target_fruit == "apple"
    assert cfg.evaluation.enable_thinking is False
    assert cfg.hf.private is True


def _raw():
    with open(DEFAULT_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_unknown_key_rejected():
    raw = _raw()
    raw["training"]["learning_rte"] = 1e-4
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(raw)


def test_public_hf_repo_rejected():
    raw = _raw()
    raw["hf"]["private"] = False
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(raw)


def test_checkpoint_beyond_budget_rejected():
    raw = _raw()
    raw["training"]["checkpoints_processed_tokens"] = [0, 30_000_000]
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(raw)
