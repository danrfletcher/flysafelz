"""G6: `make dev` and `make run`, and the coverage list. Dry runs (`make -n`) show the command each target would run and run nothing."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")
NEEDS_MAKE = pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")


def recipe(target: str) -> str:
    found = re.search(rf"^{target}:[^\n]*\n((?:\t[^\n]*\n?)+)", MAKEFILE, re.MULTILINE)
    assert found, f"no target {target}"
    return found[1]


def dry_run(*arguments: str) -> str:
    done = subprocess.run(["make", "-n", *arguments], cwd=ROOT, capture_output=True, text=True, check=True)
    return done.stdout


def test_make_dev_runs_langgraph_dev_through_uv():
    assert recipe("dev").strip() == "@uv run langgraph dev"


def test_make_run_names_no_value_in_its_recipe_so_the_shell_never_parses_one():
    text = recipe("run")
    assert "python -m flysafelz.agent.run" in text
    assert not re.search(r"\$\(?(LAT|LON|Q|TZ|CASE|NOW)|\$\$", text)


@NEEDS_MAKE
def test_a_dry_run_of_make_run_with_awkward_values_prints_none_of_them():
    out = dry_run("run", "LAT=53.8", "LON=-0.08", 'Q=Can I "fly" $$HOME?', "CASE=blue_mill_2026-10-31")
    assert "flysafelz.agent.run" in out
    assert not any(part in out for part in ("53.8", "-0.08", "HOME", "Can I", "blue_mill"))


@NEEDS_MAKE
def test_a_dry_run_of_make_dev_is_langgraph_dev():
    assert "uv run langgraph dev" in dry_run("dev")


def test_make_run_loads_dotenv_only_when_there_is_one():
    assert "$(if $(wildcard .env),--env-file .env)" in recipe("run")


def test_the_coverage_list_has_the_modules_this_pr_adds_and_the_floor_is_unchanged():
    for module in ("graph", "prompts", "middleware", "run", "guard"):
        assert f"--cov=flysafelz.agent.{module} " in MAKEFILE
    assert "--cov-fail-under=90" in MAKEFILE


def test_make_test_runs_neither_the_server_nor_the_terminal_program():
    text = recipe("test")
    assert "langgraph dev" not in text and "-m flysafelz.agent.run" not in text
