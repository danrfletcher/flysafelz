"""5a PR-2.F1 GP1: the committed evals/agent/data/verdict_sheet.yaml is finalised, against the committed fixtures and the committed code.

Nothing here writes the repository: the finalise runs over a tmp copy of the sheet. No network, no reader, no model.
"""

import shutil
from pathlib import Path

import yaml

from flysafelz import verdict_sheet_finalise as fin
from flysafelz.contracts import NotamAction, Tier
from flysafelz.sources import FIXTURES_DIR
from flysafelz.verdict_sheet import _assess_example, main, read_examples

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "evals" / "agent" / "data"
SHEET = DATA / "verdict_sheet.yaml"
EXAMPLES = DATA / "examples.yaml"
DEV_20 = "dev_20_druridge_bay"


def entries() -> dict[str, dict]:
    return {e["id"]: e for e in yaml.safe_load(SHEET.read_text(encoding="utf-8"))}


def test_the_committed_sheet_has_no_stale_or_unfinalised_entry():
    assert fin.finalised_problems(SHEET, REPO) == {}


def test_every_done_entry_carries_a_readable_stamp_and_every_labelable_row_a_label():
    for entry_id, entry in entries().items():
        assert entry["actions_done"] is True, entry_id
        assert fin.read_stamp(entry["finalised"]) is not None, entry_id
        for row in entry["rows"]:
            if "tier" in row:
                assert "verdict" not in row and "approaching_limits" not in row, entry_id  # tier-only rows are never labelled
            else:
                assert row["verdict"] in ("NO_GO", "UNVERIFIED", "PILOT_DISCRETION") and isinstance(row["approaching_limits"], bool), (entry_id, row)


def test_the_check_command_passes_on_the_committed_sheet(capsys):
    assert main(["--check"], root=REPO, environ={}) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == f"{len(entries())} entries checked: 0 errors, 0 stale, 0 orphans"


def test_a_finalise_over_a_copy_of_the_committed_sheet_moves_nothing_and_leaves_the_bytes(tmp_path, capsys):
    """The committed sheet is exactly what the finaliser writes now (idempotence on the real data)."""
    copy = tmp_path / "verdict_sheet.yaml"
    shutil.copyfile(SHEET, copy)
    report, written = fin.finalise_sheet(copy, EXAMPLES, REPO)
    assert report.failed == 0 and report.moved == [] and written is None
    assert copy.read_bytes() == SHEET.read_bytes()


def test_the_dev_20_j3219_rows_are_not_no_go_because_of_j3219():
    entry = entries()["dev_20_druridge_bay"]
    assert entry["notam_actions"]["J3219/26"] == "CHECK_SUP"
    (example,) = [e for e in read_examples(EXAMPLES) if e.id == DEV_20]
    built = _assess_example(example, REPO / FIXTURES_DIR / example.case, readings_optional=True)
    stored = [(r["verdict"], r["approaching_limits"]) for r in entry["rows"] if "tier" not in r]
    as_labelled = [(v.value, flag) for v, flag in fin.recombine(built, fin.dan_actions(entry))]
    j3219_ignored = [(v.value, flag) for v, flag in fin.recombine(built, {**fin.dan_actions(entry), "J3219/26": NotamAction.IGNORE})]
    assert stored == as_labelled == j3219_ignored  # CHECK_SUP adds nothing: the rows are what the other sources give
    segments = [s for h in built.assessment.hours if h.tier is Tier.VERDICT for s in h.segments]
    for segment, (verdict, _flag) in zip(segments, as_labelled, strict=True):
        if verdict == "NO_GO":
            assert any(b.source != "notam" for b in segment.blockers)  # a weather (or daylight or airspace) blocker, not J3219
    # and the danger area's "activated by NOTAM J3219/26, which has not been read" no longer keeps a row UNVERIFIED
    assert any(c for s in segments for c in s.conditions if "J3219/26, which has not been read" in c)
    assert all(not fin.kept_unverified(s, [n.id for n in built.near]) for s in segments)


def test_tier_only_rows_of_the_committed_sheet_have_no_verdict_keys():
    tier_only = [(i, r) for i, e in entries().items() for r in e["rows"] if "tier" in r]
    assert tier_only and all("verdict" not in r and "approaching_limits" not in r for _i, r in tier_only)
