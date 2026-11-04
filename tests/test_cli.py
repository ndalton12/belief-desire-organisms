import yaml
from typer.testing import CliRunner

from belief_desire_organisms.cli import app
from belief_desire_organisms.config import DEFAULT_CONFIG_PATH

runner = CliRunner()


def test_help_lists_pipeline_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["prepare", "generate", "validate", "publish", "train", "export", "evaluate"]:
        assert cmd in result.output


def test_paid_generation_refused_without_spend_cap(tmp_path):
    with open(DEFAULT_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    raw["generation"]["max_spend_usd"] = None
    cfg_path = tmp_path / "no_cap.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))

    result = runner.invoke(app, ["generate", "belief", "--execute", "--config", str(cfg_path)])
    assert result.exit_code != 0
    assert "max_spend_usd" in result.output
