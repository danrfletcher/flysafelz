"""G15: the PIB-age rule in case mode and in live mode.

Case mode treats a case's captured PIB as current at the case's `now` (the archive snapshots it only about twice a
day, so it can be ~5 h old). Live mode keeps the 3 h rule. One constructed case (`stale_rule: live`) keeps the stale,
UNVERIFIED path tested end to end. Everything runs through the real loaders, assess and the CLI; no network.
"""

import ast
import re
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import cases
import httpx
import pytest
from factories import utc

from flysafelz import capture, cli, sources
from flysafelz.assess import assess
from flysafelz.capture import CaptureError
from flysafelz.cases import load_case_inputs
from flysafelz.contracts import SourceName, SourceState, Verdict
from flysafelz.pib import CASE_MODE_NOTE, load_pib
from flysafelz.redact import redact_phones
from flysafelz.rules import RULES
from flysafelz.sources import SourceError, load_case_sources

CONSTRUCTED = "stale_pib_constructed"
BLUE_MILL = cases.BLUE_MILL
CASE_NOTE = "case mode: captured PIB treated as current at the case's now"
OLD_ISSUE = "2026-10-31T02:20:11+00:00"  # 5 h 10 min before the case's now (07:30Z)
SRC = Path(__file__).resolve().parents[2] / "src" / "flysafelz"
STALE_HOURS = cast(float, RULES.pib_stale_hours.value)  # 3: a number, not one of the list rules


def copy_case(tmp_path: Path, name: str) -> tuple[Path, Path]:
    """A copy of fixtures/<name>/ under a fresh root. Returns (root, case folder)."""
    folder = tmp_path / "fixtures" / name
    shutil.copytree(cases.case_dir(name), folder)
    return tmp_path, folder


def with_issued(folder: Path, issued: str) -> None:
    path = folder / "pib.xml"
    text = path.read_text(encoding="utf-8")
    changed = re.sub(r"<Issued>[^<]*</Issued>", f"<Issued>{issued}</Issued>", text, count=1)
    assert changed != text
    path.write_text(changed, encoding="utf-8")


def without_stale_rule(folder: Path) -> None:
    path = folder / "case.yaml"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("stale_rule")]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assess_case(loaded, **changes):
    case = loaded.case
    arguments: dict[str, Any] = {
        "now": case.now,
        "forecast": loaded.forecast,
        "airspace": loaded.airspace,
        "pib": loaded.pib,
        "readings": loaded.readings,
        "window": (case.window.start, case.window.end),
    }
    arguments.update(changes)
    return assess(case.lat, case.lon, **arguments)


def segments(assessment):
    return [s for h in assessment.hours for s in h.segments]


def notam_status(assessment):
    return next(s for s in assessment.source_statuses if s.source is SourceName.NOTAMS)


def served(body: bytes) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body)))


# --- the constructed old-PIB case, end to end -------------------------------------------------------------


def test_the_constructed_case_is_marked_constructed_and_keeps_the_live_rule():
    case, *_rest = cases.load(CONSTRUCTED)
    assert case.constructed is True
    assert case.stale_rule == "live"
    assert case.capture is not None
    assert case.now - case.capture.pib_issued > timedelta(hours=STALE_HOURS)
    assert "CONSTRUCTED" in (cases.case_dir(CONSTRUCTED) / "case.yaml").read_text(encoding="utf-8")


def test_the_constructed_case_is_stale_and_every_segment_is_unverified_through_the_cli(capsys):
    assert cli.main(["assess", "--case", CONSTRUCTED]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    stale = next(i for i, line in enumerate(lines) if line.startswith("  NOTAMs: stale"))
    assert "PIB issued 2026-10-31T02:20:11Z, more than 3 h before now" in lines[stale]
    assert lines[stale + 1] == "    check by hand: NATS AIS PIB (https://pibs.nats.co.uk)"
    rows = [line for line in lines if re.match(r"^\d\d:\d\d\u2013\d\d:\d\d", line)]
    assert len(rows) == 7
    assert all(row.endswith("  UNVERIFIED") for row in rows), rows
    assert "notams source stale" in out and "Overall: UNVERIFIED" in out
    assert "case mode:" not in out  # the case-mode note belongs to ok statuses only


def test_the_constructed_case_through_assess_names_the_pib_age_in_each_segment():
    loaded = load_case_sources(CONSTRUCTED)
    assert loaded.pib.status.state is SourceState.STALE
    assessment = assess_case(loaded)
    assert {s.verdict for s in segments(assessment)} == {Verdict.UNVERIFIED}
    assert all(any("notams source stale" in c for c in s.conditions) for s in segments(assessment))
    assert assessment.flyable_windows == []


def test_the_same_files_without_stale_rule_give_notams_ok_and_the_notam_is_assessed(tmp_path):
    root, folder = copy_case(tmp_path, CONSTRUCTED)
    without_stale_rule(folder)
    loaded = load_case_sources(CONSTRUCTED, root=root)
    assert loaded.case.stale_rule == "case"
    assert loaded.pib.status.state is SourceState.OK and loaded.pib.status.reason == CASE_NOTE
    assessment = assess_case(loaded)
    assert notam_status(assessment).state is SourceState.OK
    assert Verdict.UNVERIFIED not in {s.verdict for s in segments(assessment)}
    # J9001/26 is read and its action reaches the segments: nothing was dropped because of the PIB's age.
    assert all(any(a.notam_id == "J9001/26" for a in s.notam_actions) for s in segments(assessment))
    assert all(any("avoid that sector" in c for c in s.conditions) for s in segments(assessment))


def test_the_constructed_pib_and_readings_hold_no_phone_number():
    for name in ("pib.xml", "readings.yaml", "case.yaml"):
        text = (cases.case_dir(CONSTRUCTED) / name).read_text(encoding="utf-8")
        assert redact_phones(text)[1] == {}, name


def test_make_fixtures_refuses_a_constructed_case(tmp_path):
    _root, folder = copy_case(tmp_path, CONSTRUCTED)

    def no_request(request):
        raise AssertionError("a request was made")

    with httpx.Client(transport=httpx.MockTransport(no_request)) as client, pytest.raises(CaptureError, match="constructed"):
        capture.capture(CONSTRUCTED, folder, "not-a-key", client, today=utc(12, day=31))
    assert (folder / "pib.xml").read_bytes() == (cases.case_dir(CONSTRUCTED) / "pib.xml").read_bytes()


# --- G15 case mode ----------------------------------------------------------------------------------------


@pytest.fixture
def old_blue_mill(tmp_path):
    """Blue Mill with its PIB issued at 02:20Z, 5 h 10 min before the case's now (07:30Z)."""
    root, folder = copy_case(tmp_path, BLUE_MILL)
    with_issued(folder, OLD_ISSUE)
    return root, folder


def test_a_case_pib_issued_over_3_h_before_now_is_ok_and_its_notams_are_assessed(old_blue_mill):
    root, _folder = old_blue_mill
    loaded = load_case_sources(BLUE_MILL, root=root)
    assert loaded.pib.status.state is SourceState.OK
    assert loaded.pib.status.issued_at == utc(2, 20, day=31).replace(second=11)
    assert loaded.pib.status.issued_at is not None
    assert loaded.case.now - loaded.pib.status.issued_at > timedelta(hours=STALE_HOURS)
    assessment = assess_case(loaded)
    assert notam_status(assessment).state is SourceState.OK
    # No segment is UNVERIFIED because of the PIB's age, and the NOTAMs are assessed: the verdicts are the real case's.
    assert not any("notams source" in c for s in segments(assessment) for c in s.conditions)
    real = assess_case(load_case_sources(BLUE_MILL))
    assert [(s.start, s.verdict) for s in segments(assessment)] == [(s.start, s.verdict) for s in segments(real)]
    assert any(s.notam_actions for s in segments(assessment))


def test_assess_does_not_re_mark_a_case_mode_pib_stale_after_the_loader_returned_it_ok(old_blue_mill):
    root, _folder = old_blue_mill
    loaded = load_case_sources(BLUE_MILL, root=root)
    assert loaded.pib.status.state is SourceState.OK
    assert notam_status(assess_case(loaded)).state is SourceState.OK  # live=False, the default
    assert notam_status(assess_case(loaded, live=False)).state is SourceState.OK
    # The same status IS re-marked when assess is asked for the live rule (or run live): the switch decides.
    re_marked = notam_status(assess_case(loaded, stale_rule=True))
    assert re_marked.state is SourceState.STALE and "more than 3 h before now" in re_marked.reason


def test_the_cli_source_line_in_case_mode_shows_the_issued_time_and_the_note(old_blue_mill, monkeypatch, capsys):
    root, _folder = old_blue_mill
    monkeypatch.setattr(sources, "ROOT", root)
    assert cli.main(["assess", "--case", BLUE_MILL]) == 0
    out = capsys.readouterr().out
    assert f"  NOTAMs: ok (PIB issued 2026-10-31T02:20:11Z) ({CASE_NOTE})" in out
    assert "stale" not in out and "check by hand" not in out


def test_the_case_loader_used_by_the_tests_and_the_verdict_sheet_is_case_mode_too(old_blue_mill):
    _root, folder = old_blue_mill
    status = load_case_inputs(folder)[3].status
    assert status.state is SourceState.OK and status.reason == CASE_NOTE
    constructed = load_case_inputs(cases.case_dir(CONSTRUCTED))[3].status
    assert constructed.state is SourceState.STALE  # the opt-in is honoured by that loader as well


def test_only_replay_files_calls_the_pib_loader_in_case_mode():
    """No case path reaches the live rule by accident: every load_pib call in src is the live fetch, the CLI's live
    `read`, or sources.replay_files, which passes stale_rule explicitly from the case."""
    calls: dict[str, list[ast.Call]] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
            for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
                name = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", "")
                if name == "load_pib":
                    calls.setdefault(f"{path.name}:{fn.name}", []).append(call)
    assert sorted(calls) == ["cli.py:run_read", "sources.py:fetch_live_sources", "sources.py:replay_files"]
    for key in ("cli.py:run_read", "sources.py:fetch_live_sources"):  # live: the default (the rule on)
        assert all("stale_rule" not in [k.arg for k in c.keywords] for c in calls[key])
    replay = calls["sources.py:replay_files"]
    assert len(replay) == 1 and "stale_rule" in [k.arg for k in replay[0].keywords]
    # and the loaders that read a case folder both go through replay_files
    for module in ("sources.py", "cases.py"):
        assert "replay_files(" in (SRC / module).read_text(encoding="utf-8")


# --- G15 live mode ----------------------------------------------------------------------------------------


def pib_issued_before_now(hours: float, tmp_path: Path):
    """(PIB bytes issued `hours` before the case's now, that now)."""
    _root, folder = copy_case(tmp_path, BLUE_MILL)
    case = cases.load(BLUE_MILL)[0]
    issued = case.now - timedelta(hours=hours)
    with_issued(folder, issued.isoformat())
    return (folder / "pib.xml").read_bytes(), case.now


def test_live_mode_keeps_the_rule_just_over_3_h_is_stale(tmp_path):
    body, now = pib_issued_before_now(STALE_HOURS + 0.01, tmp_path)
    status = load_pib("live", now, client=served(body)).status
    assert status.state is SourceState.STALE and "more than 3 h before now" in status.reason


def test_live_mode_keeps_the_rule_exactly_3_h_is_not_stale(tmp_path):
    body, now = pib_issued_before_now(STALE_HOURS, tmp_path)
    status = load_pib("live", now, client=served(body)).status
    assert status.state is SourceState.OK and status.reason == ""


def test_load_pib_without_the_rule_is_ok_at_any_age_and_says_so(tmp_path):
    body, now = pib_issued_before_now(STALE_HOURS + 2, tmp_path)
    assert load_pib("live", now, client=served(body), stale_rule=True).status.state is SourceState.STALE
    off = load_pib("live", now, client=served(body), stale_rule=False)
    assert (off.status.state, off.status.reason) == (SourceState.OK, CASE_NOTE)
    assert CASE_MODE_NOTE == CASE_NOTE
    assert off.notams  # its NOTAMs are returned either way


def test_live_assess_re_marks_an_old_ok_status_stale(old_blue_mill):
    root, _folder = old_blue_mill
    loaded = load_case_sources(BLUE_MILL, root=root)
    live = assess_case(loaded, live=True, now=loaded.case.now)
    assert notam_status(live).state is SourceState.STALE


# --- edge cases -------------------------------------------------------------------------------------------


def test_a_bogus_stale_rule_is_a_load_error_naming_the_field_and_no_assessment_runs(tmp_path, monkeypatch, capsys):
    root, folder = copy_case(tmp_path, CONSTRUCTED)
    path = folder / "case.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("stale_rule: live", "stale_rule: bogus"), encoding="utf-8")
    with pytest.raises(SourceError, match=r"stale_rule"):
        load_case_sources(CONSTRUCTED, root=root)
    monkeypatch.setattr(sources, "ROOT", root)

    def never(*args, **kwargs):
        raise AssertionError("assess ran")

    monkeypatch.setattr(cli, "assess", never)
    with pytest.raises(SystemExit) as exit_:
        cli.main(["assess", "--case", CONSTRUCTED])
    assert exit_.value.code == 2
    assert "stale_rule" in capsys.readouterr().err


def test_a_case_pib_issued_after_the_cases_now_is_unaffected_by_the_switch(tmp_path):
    root, folder = copy_case(tmp_path, BLUE_MILL)
    with_issued(folder, "2026-10-31T09:00:00+00:00")  # after now (07:30Z): not stale today, with or without the rule
    case_mode = load_case_sources(BLUE_MILL, root=root)
    assert case_mode.pib.status.state is SourceState.OK
    live = load_pib("live", case_mode.case.now, client=served((folder / "pib.xml").read_bytes()))
    assert live.status.state is SourceState.OK and live.status.reason == ""
    assert notam_status(assess_case(case_mode, stale_rule=True)).state is SourceState.OK


def test_an_unreadable_case_pib_is_still_failed_and_every_segment_is_unverified(tmp_path):
    root, folder = copy_case(tmp_path, BLUE_MILL)
    (folder / "pib.xml").write_text("<Pib><not-closed>", encoding="utf-8")
    loaded = load_case_sources(BLUE_MILL, root=root)
    assert loaded.pib.status.state is SourceState.FAILED
    assessment = assess_case(loaded)
    assert notam_status(assessment).state is SourceState.FAILED
    assert {s.verdict for s in segments(assessment)} <= {Verdict.UNVERIFIED, Verdict.NO_GO}
    assert any(s.verdict is Verdict.UNVERIFIED for s in segments(assessment))


def test_the_pib_stale_rule_value_and_comparison_are_unchanged():
    rule = RULES.pib_stale_hours
    assert (rule.value, rule.comparison) == (3, "=")
