"""Ticket 8 PR-2.S1 (G23, GP1): the UI shell's `make` targets, README sections and the numbers it shares with the server.

The UI's own tests run with `make test-ui`; these are the checks that sit on the Python side of the fence.
"""

import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
UI = ROOT / "ui"
MAKEFILE = (ROOT / "Makefile").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
# Running the UI lives on the write-up site's Running it page; Data and licences is on both.
RUNNING_IT = (ROOT / "docs" / "docs" / "running-it.md").read_text(encoding="utf-8")
NEEDS_MAKE = pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")


def dry_run(target: str) -> str:
    done = subprocess.run(["make", "-n", target], cwd=ROOT, capture_output=True, text=True, check=True)
    return done.stdout


def rule(name: str) -> float:
    return yaml.safe_load((ROOT / "src" / "flysafelz" / "rules.yaml").read_text(encoding="utf-8"))[name]["value"]


def test_the_map_opens_on_the_uk_box_that_the_rules_name():
    text = (UI / "src" / "lib" / "ukBounds.ts").read_text(encoding="utf-8")
    found = {name: float(value) for name, value in re.findall(r"export const (UK_\w+) = (-?[\d.]+)", text)}
    assert found == {
        "UK_SOUTH": rule("uk_bounds_lat_min"),
        "UK_NORTH": rule("uk_bounds_lat_max"),
        "UK_WEST": rule("uk_bounds_lon_min"),
        "UK_EAST": rule("uk_bounds_lon_max"),
    }


@NEEDS_MAKE
def test_make_ui_installs_when_node_modules_is_missing_then_runs_vite_on_5173_with_strict_port():
    out = dry_run("ui")
    lines = out.strip().splitlines()
    assert lines[0] == "if [ ! -d ui/node_modules ]; then npm --prefix ui ci; fi"
    assert lines[1] == "npm --prefix ui run dev -- --port 5173 --strictPort"
    vite = (UI / "vite.config.ts").read_text(encoding="utf-8")
    assert "port: 5173, strictPort: true" in vite and "port: 4173, strictPort: true" in vite


@NEEDS_MAKE
def test_a_dry_run_of_make_dev_shows_the_server_and_the_ui_and_starts_neither():
    out = dry_run("dev")
    assert "uv run langgraph dev --allow-blocking & server=$!" in out
    assert "npm --prefix ui run dev -- --port 5173 --strictPort & ui=$!" in out
    assert "kill -TERM" in out and "trap" in out


@NEEDS_MAKE
def test_make_test_ui_runs_the_four_checks_in_order():
    lines = dry_run("test-ui").strip().splitlines()
    assert lines[0] == "if [ ! -d ui/node_modules ]; then npm --prefix ui ci; fi"
    assert lines[1] == (
        "npm --prefix ui run lint && npm --prefix ui test -- --run --coverage && npm --prefix ui run build && node ui/scripts/check-external-urls.mjs && node ui/scripts/check-no-keys.mjs"
    )


def test_make_test_still_runs_only_python():
    recipe = re.search(r"^test:[^\n]*\n((?:\t[^\n]*\n?)+)", MAKEFILE, re.MULTILINE)[1]
    assert "npm" not in recipe and "ui/" not in recipe and "vite" not in recipe
    assert "--cov-fail-under=90" in recipe


def test_the_ui_origins_the_server_allows_are_the_ports_vite_serves_on():
    import json

    origins = json.loads((ROOT / "langgraph.json").read_text(encoding="utf-8"))["http"]["cors"]["allow_origins"]
    assert origins[:2] == ["http://localhost:5173", "http://localhost:4173"]  # then Studio's two origins


# --- `make dev` stops both processes -------------------------------------------------------------------------------------------------


def write_stub(folder: Path, name: str, body: str) -> Path:
    """A stub process that records its pid in <name>.pid and then runs `body`."""
    script = folder / f"{name}.sh"
    script.write_text(f'#!/bin/sh\necho $$ > "{folder / name}.pid"\n{body}\n', encoding="utf-8")
    script.chmod(0o755)
    return script


def running(pid_file: Path) -> bool:
    try:
        os.kill(int(pid_file.read_text()), 0)
    except (OSError, ValueError, FileNotFoundError):
        return False
    return True


def make_dev(folder: Path, server: Path, ui: Path) -> subprocess.Popen:
    # ui/node_modules is not needed: UI_INSTALL is a no-op once the folder exists, and a fake folder would fake the install
    # check, so the install step is switched off by name.
    return subprocess.Popen(
        ["make", "dev", f"SERVER={server}", f"UI_DEV={ui}", "UI_INSTALL=true"],
        cwd=ROOT,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_until(condition, seconds: float = 15.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.1)
    return False


@NEEDS_MAKE
@pytest.mark.parametrize("quitter", ["server", "ui"])
def test_when_either_process_exits_the_other_is_stopped_and_make_fails(tmp_path, quitter):
    stays = write_stub(tmp_path, "stays", "exec sleep 300")
    quits = write_stub(tmp_path, "quits", "sleep 2\nexit 3")
    server, ui = (quits, stays) if quitter == "server" else (stays, quits)
    process = make_dev(tmp_path, server, ui)
    try:
        assert process.wait(timeout=30) != 0
        assert wait_until(lambda: not running(tmp_path / "stays.pid")), "the surviving process was left running"
    finally:
        if running(tmp_path / "stays.pid"):  # only a process this test started
            os.kill(int((tmp_path / "stays.pid").read_text()), signal.SIGTERM)


@NEEDS_MAKE
def test_ctrl_c_stops_both(tmp_path):
    server = write_stub(tmp_path, "server", "exec sleep 300")
    ui = write_stub(tmp_path, "ui", "exec sleep 300")
    process = make_dev(tmp_path, server, ui)
    try:
        assert wait_until(lambda: running(tmp_path / "server.pid") and running(tmp_path / "ui.pid"))
        os.killpg(process.pid, signal.SIGINT)  # what the terminal sends the foreground group (make and its shell)
        process.wait(timeout=30)
        assert wait_until(lambda: not running(tmp_path / "server.pid") and not running(tmp_path / "ui.pid"))
    finally:
        for name in ("server", "ui"):
            if running(tmp_path / f"{name}.pid"):
                os.kill(int((tmp_path / f"{name}.pid").read_text()), signal.SIGTERM)


# --- README ---------------------------------------------------------------------------------------------------------------------------


def section(title: str, text: str = README) -> str:
    found = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
    assert found, f"no '## {title}' section"
    return found[1]


def test_running_the_ui_names_node_the_targets_the_ports_and_vite_api_url():
    text = section("Running the UI", RUNNING_IT)
    for needed in ("Node 22", "make dev", "make ui", "make test-ui", "2024", "5173", "4173", "VITE_API_URL", "http://localhost:2024"):
        assert needed in text, needed
    node = (UI / "package.json").read_text(encoding="utf-8")
    assert '"node": ">=22"' in node


def test_readme_data_and_licences_has_one_line_per_source_and_ends_with_the_demo_sentence():
    text = section("Data and licences").strip()
    lines = [line for line in text.splitlines() if line.startswith("- ")]
    assert [line.split(":")[0] for line in lines] == ["- Open-Meteo", "- RainViewer", "- openAIP", "- NATS PIB", "- CARTO basemap"]
    assert text.endswith("This is a non-commercial demo; a production deployment would move to commercial plans (Open-Meteo API subscription, a licensed radar feed such as the Met Office's).")
    rain = next(line for line in lines if line.startswith("- RainViewer"))
    assert all(word in rain for word in ("free API", "personal and educational", "2 hours", "zoom 7", "100 requests"))
    assert "GET /map/rain-grid" in lines[0] and "model, not radar" in lines[0]  # the Open-Meteo line carries the rain-grid fallback


def test_running_the_ui_has_no_placeholder_for_the_pieces_later_prs_add():
    text = section("Running the UI", RUNNING_IT)
    assert "CARTO key" not in text and "VITE_CARTO_KEY" not in text
    assert "TODO" not in text and "coming soon" not in text.lower()
    # Layer choices are saved since the persistence PR: the old "not saved" note is gone, and the line that replaced it says what is kept and what is lost.
    assert "not saved" not in text and "reset on reload" not in text
    assert "kept in this browser's local storage" in text and "conversations are still lost on reload" in text
