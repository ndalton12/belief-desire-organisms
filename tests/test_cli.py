from typer.testing import CliRunner

from belief_desire_organisms.cli import app

runner = CliRunner()


def test_help_lists_pipeline_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["prepare", "generate", "validate", "publish", "train", "export", "evaluate"]:
        assert cmd in result.output


def test_paid_generation_refused_without_spend_cap():
    # Default config leaves generation.max_spend_usd null.
    result = runner.invoke(app, ["generate", "belief", "--execute"])
    assert result.exit_code != 0
    assert "max_spend_usd" in result.output
