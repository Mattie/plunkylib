"""CLI startup regression goals: module, installed command, and argument handling."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def cli(tmp_path):
    """Run CLI processes with disposable data and synthetic credentials."""
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        "PLUNKYLIB_BASE_DIR": str(tmp_path / "records"),
        "PYTHON_DOTENV_DISABLED": "1",
        "OPENAI_API_KEY": "offline-cli-test",
        "NO_COLOR": "1",
        "TERM": "dumb",
    })

    def invoke(args, installed=False):
        if installed:
            # Load the installed console entry point so packaging mistakes are visible.
            command = [sys.executable, "-c", (
                "from importlib.metadata import distribution; "
                "entry = next(e for e in distribution('plunkylib').entry_points "
                "if e.group == 'console_scripts' and e.name == 'plunky'); "
                "entry.load()()"
            )]
        else:
            command = [sys.executable, "-m", "plunkylib"]
        return subprocess.run(
            [*command, *args], cwd=tmp_path, env=env,
            capture_output=True, text=True, timeout=20,
        )

    return invoke


@pytest.mark.parametrize("installed", [False, True], ids=["module", "installed"])
@pytest.mark.parametrize("args", [
    ["--help"],
    ["prompt", "create", "--help"],
    ["complete", "use-pet", "--help"],
])
def test_cli_help_works_at_both_entry_points(cli, installed, args):
    result = cli(args, installed=installed)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Usage:" in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("installed", [False, True], ids=["module", "installed"])
def test_cli_missing_argument_reports_usage_error(cli, installed):
    result = cli(["prompt", "create"], installed=installed)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "Missing argument" in result.stdout + result.stderr
    assert "Traceback" not in result.stderr


def test_cli_prompt_survives_a_new_process(cli):
    created = cli(["prompt", "create", "CliSmoke"])
    assert created.returncode == 0, created.stdout + created.stderr
    loaded = cli(["prompt", "print", "CliSmoke"], installed=True)
    assert loaded.returncode == 0, loaded.stdout + loaded.stderr
    assert "Enter Your Text Here" in loaded.stdout
