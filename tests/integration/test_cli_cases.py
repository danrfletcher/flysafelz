"""The CLI over the two committed cases, in process: golden output (E20) and the golden-path checkpoints one by one.

tests/golden/*.txt are compared byte for byte. They are rewritten only by `pytest --update-golden`, never automatically.
"""

import re
import shutil
from pathlib import Path

import cases
import pytest
import yaml

from flysafelz import cli, sources
from flysafelz.geo import compass16, shape_over
from flysafelz.redact import redact_phones

GOLDEN = Path(__file__).resolve().parents[1] / "golden"
BLUE_MILL = cases.BLUE_MILL
WEST_SLEEKBURN = cases.WEST_SLEEKBURN
BLUE_MILL_GP3 = ["assess", "--case", BLUE_MILL, "--start", "2026-10-31T07:00Z", "--end", "2026-10-31T18:00Z"]
BARE_GO = re.compile(r"(?<![-\w])GO(?![-\w])")
ROW = re.compile(r"^(\d\d):\d\d\u2013\d\d:\d\d(?:  (.*))?$")  # a top-level table row: no indent

# The contact the CLI shows is the captured fixture's own (the real one: CONTACT_FIRST asks the pilot to call it), so it is read from the
# committed reading, not typed into this file (F5: no real number in a test source).
FIXTURE_CONTACT = cases.load(BLUE_MILL)[4]["J3857/26"].contact.value
DANGER_AREA = f"Danger area EGD496A active 08:00\u201318:00 (UAS BVLOS ops): phone {FIXTURE_CONTACT} and fly only if they agree; details in AIP SUP 108/2026"
STAY_BELOW = "stay below 1,300 ft QNH"


def run_cli(capsys, argv: list[str]) -> str:
    assert cli.main(argv) == 0
    out, err = capsys.readouterr()
    assert err == ""
    return out


def hour_blocks(text: str) -> dict[str, list[str]]:
    """The table rows by start hour: {"08": [row line, detail lines...]}. A row ends at the next unindented line."""
    blocks: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        match = ROW.match(line)
        if match:
            current = match.group(1)
            blocks[current] = [line]
        elif current is not None and line.startswith(" "):
            blocks[current].append(line)
        else:
            current = None
    return blocks


def segments_of(block: list[str]) -> list[list[str]]:
    """The segments of one hour block, each as its row text followed by its detail lines.

    An uncut hour is one segment (its own row line); a cut hour has the hour line, then sub-rows indented by two.
    """
    if len(block) == 1 or block[1].startswith("    "):
        return [block]
    segments: list[list[str]] = []
    for line in block[1:]:
        if line.startswith("  ") and not line.startswith("    "):
            segments.append([line.strip()])
        else:
            segments[-1].append(line.strip())
    return segments


def check_golden(name: str, text: str, update: bool) -> None:
    path = GOLDEN / f"{name}.txt"
    if update:
        GOLDEN.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
    assert path.read_text(encoding="utf-8") == text, f"{path} differs; if the change is intended run: uv run pytest --update-golden"


# --- E20: byte for byte ----------------------------------------------------------------------------


def test_e20_blue_mill_reproduces_the_golden_file(capsys, update_golden):
    check_golden(BLUE_MILL, run_cli(capsys, BLUE_MILL_GP3), update_golden)


def test_e20_west_sleekburn_reproduces_the_golden_file(capsys, update_golden):
    check_golden(WEST_SLEEKBURN, run_cli(capsys, ["assess", "--case", WEST_SLEEKBURN]), update_golden)


@pytest.mark.parametrize("name", [cases.DRURIDGE_BAY, cases.BAMBURGH])
def test_g16_the_labelled_large_notam_cases_reproduce_their_golden_files(capsys, update_golden, name):
    """The goldens hold [phone] where the CLI prints a contact, so no real number is in the repo's tests (F8, spec)."""
    out = run_cli(capsys, ["assess", "--case", name])
    check_golden(name, re.sub(r"<PHONE_\d+>", "[phone]", redact_phones(out)[0]), update_golden)


def test_g15_the_constructed_old_pib_case_reproduces_its_golden_file(capsys, update_golden):
    check_golden("stale_pib_constructed", run_cli(capsys, ["assess", "--case", "stale_pib_constructed"]), update_golden)


def test_the_golden_files_are_utf8_text_with_one_final_newline():
    for name in (BLUE_MILL, WEST_SLEEKBURN):
        text = (GOLDEN / f"{name}.txt").read_bytes().decode("utf-8")
        assert text.endswith("\n") and not text.endswith("\n\n")


def test_running_twice_gives_the_same_bytes(capsys):
    assert run_cli(capsys, BLUE_MILL_GP3) == run_cli(capsys, BLUE_MILL_GP3)


# --- GP3 ---------------------------------------------------------------------------------------------


def test_gp3_header_names_the_case_lz_frozen_now_and_pib_commit(capsys):
    lines = run_cli(capsys, BLUE_MILL_GP3).splitlines()
    assert lines[0] == f"FlySafeLZ assess \u2013 case {BLUE_MILL}"
    assert lines[1] == "LZ 53.802797, -0.086644 \u00b7 now 2026-10-31T07:30Z"
    assert lines[2] == "PIB: archive commit bdab14658f0270dc41460eebcd03332abfb230ec, issued 2026-10-31T07:00:37Z"


def test_gp3_end_is_exclusive_so_there_are_11_rows_07_to_17(capsys):
    blocks = hour_blocks(run_cli(capsys, BLUE_MILL_GP3))
    assert list(blocks) == [f"{h:02d}" for h in range(7, 18)]
    assert len(blocks) == 11


def test_gp3_the_case_window_is_the_default_so_no_start_or_end_gives_the_same_output(capsys):
    assert run_cli(capsys, ["assess", "--case", BLUE_MILL]) == run_cli(capsys, BLUE_MILL_GP3)


def test_gp3_the_clock_of_the_case_is_used_never_the_wall_clock(capsys, monkeypatch):
    class WallClock:
        @staticmethod
        def now(tz=None):
            raise AssertionError("the wall clock was read")

    monkeypatch.setattr(cli, "datetime", WallClock)
    assert "now 2026-10-31T07:30Z" in run_cli(capsys, ["assess", "--case", BLUE_MILL])


# --- GP4 ---------------------------------------------------------------------------------------------


def test_gp4_rows_show_verdict_conditions_cautions_with_numbers_and_notam_actions(capsys):
    text = run_cli(capsys, BLUE_MILL_GP3)
    block = hour_blocks(text)["12"]
    assert block[0] == "12:00\u201313:00  PILOT DISCRETION \u2013 APPROACHING LIMITS \u00b7 WITH CONDITIONS"
    details = block[1:]
    assert f"    condition: {DANGER_AREA}" in details
    assert "    caution: gust spread 5.2 mph, limit 4" in details
    assert "    caution: low cloud 91 %, limit 90" in details
    assert any(line.startswith("    NOTAM J3857/26 CONTACT_FIRST") for line in details)
    assert any(line.startswith("    NOTAM H7171/26 AVOID_SECTOR bearing ") for line in details)


def test_gp4_cautions_print_the_assessments_own_numbers(capsys):
    """Each printed caution is exactly the Assessment's text, which carries its value and its limit from rules.yaml."""
    assessment = cases.run(BLUE_MILL)
    text = run_cli(capsys, BLUE_MILL_GP3)
    cautions = [c for h in assessment.hours for s in h.segments for c in s.cautions]
    assert cautions
    for caution in cautions:
        assert f"caution: {caution.text}" in text
        if caution.value is not None:
            assert caution.limit is not None and "limit" in caution.text


def test_gp4_every_row_verdict_is_one_of_the_six_strings(capsys):
    six = {
        "NO-GO",
        "UNVERIFIED",
        "PILOT DISCRETION \u2013 APPROACHING LIMITS",
        "PILOT DISCRETION \u2013 WITH CONDITIONS",
        "PILOT DISCRETION \u2013 APPROACHING LIMITS \u00b7 WITH CONDITIONS",
        "PILOT DISCRETION",
    }
    for name in (BLUE_MILL, WEST_SLEEKBURN):
        text = run_cli(capsys, ["assess", "--case", name])
        shown = re.findall(r"^\s*\d\d:\d\d\u2013\d\d:\d\d  (\S.*)$", text, re.MULTILINE)
        assert shown and set(shown) <= six, set(shown) - six


# --- GP5 ---------------------------------------------------------------------------------------------


def test_gp5_from_08_every_segment_lists_both_notams_with_the_right_actions(capsys):
    blocks = hour_blocks(run_cli(capsys, BLUE_MILL_GP3))
    bearing = next(a.bearing_deg for h in cases.run(BLUE_MILL).hours for s in h.segments for a in s.notam_actions if a.notam_id == "H7171/26")
    assert bearing is not None
    word = compass16(bearing)
    for hour in range(8, 18):
        for segment in segments_of(blocks[f"{hour:02d}"]):
            lines = [line.strip() for line in segment]
            assert any(line.startswith("NOTAM J3857/26 CONTACT_FIRST") for line in lines), (hour, segment)
            assert f"condition: {DANGER_AREA}" in lines, (hour, segment)
            sector = [line for line in lines if line.startswith("NOTAM H7171/26 AVOID_SECTOR")]
            assert len(sector) == 1, (hour, segment)
            assert any(line.startswith(f"condition: drone ops 0.15 NM to the {word} (circle edge), up to ") and "avoid that sector" in line for line in lines), (hour, segment)


def test_gp5_h7171_bearing_is_printed_in_degrees_168_6(capsys):
    """The bearing sits 0.15 deg from the S/SSE boundary, so it is asserted as a number, never as the compass word."""
    text = run_cli(capsys, BLUE_MILL_GP3)
    found = re.findall(r"NOTAM H7171/26 AVOID_SECTOR bearing ([\d.]+)\u00b0", text)
    assert found
    assert all(float(value) == pytest.approx(168.6, abs=0.5) for value in found)


def test_gp5_before_08_neither_notam_is_listed(capsys):
    block = hour_blocks(run_cli(capsys, BLUE_MILL_GP3))["07"]
    assert not [line for line in block if "NOTAM" in line or "EGD496A" in line]


def test_gp5_contact_first_is_the_first_condition(capsys):
    blocks = hour_blocks(run_cli(capsys, BLUE_MILL_GP3))
    for hour in range(8, 18):
        for segment in segments_of(blocks[f"{hour:02d}"]):
            conditions = [line.strip() for line in segment if line.strip().startswith("condition:")]
            assert conditions[0] == f"condition: {DANGER_AREA}", (hour, conditions)


def test_gp5_no_segment_from_08_is_plain_pilot_discretion(capsys):
    blocks = hour_blocks(run_cli(capsys, BLUE_MILL_GP3))
    for hour in range(8, 18):
        for segment in segments_of(blocks[f"{hour:02d}"]):
            assert not re.search(r"  PILOT DISCRETION$", segment[0]) and segment[0] != "PILOT DISCRETION", (hour, segment[0])


def test_gp5_weather_no_go_hours_print_no_go_with_the_weather_reason_first(capsys):
    blocks = hour_blocks(run_cli(capsys, BLUE_MILL_GP3))
    for hour in ("08", "09"):  # visibility 1,340 m against the 1,500 m limit
        assert blocks[hour][0].endswith("  NO-GO")
        assert blocks[hour][1] == "    blocked (weather): visibility 1340 m, limit 1500"


def test_gp5_dropped_notams_are_only_under_debug_each_with_its_reason(capsys):
    plain = run_cli(capsys, BLUE_MILL_GP3)
    debug = run_cli(capsys, [*BLUE_MILL_GP3, "--debug"])
    assert "Dropped or ignored" not in plain and "J3864/26" not in plain
    table, _, listing = debug.partition("Dropped or ignored NOTAMs")
    assert table.rstrip("\n") == plain.rstrip("\n")  # the table and footer are unchanged
    entries = dict(re.findall(r"^  (\S+): (.+)$", listing, re.MULTILINE))
    assert re.match(r"starts 2026-11-01T08:00:00Z", entries["J3864/26"])
    assert all(entries.values())
    # the security notices and checklists never reach the table (series V subject RO), only the list
    for notam_id in ("V32/26", "V34/26", "V35/26"):
        assert notam_id not in plain and "security advice" in entries[notam_id]


def test_gp5_only_the_two_read_notams_are_ever_listed_in_the_table(capsys):
    text = run_cli(capsys, BLUE_MILL_GP3)
    assert set(re.findall(r"NOTAM (\S+) ", text)) == {"J3857/26", "H7171/26"}


# --- GP6 ---------------------------------------------------------------------------------------------


def test_gp6_the_17_00_row_splits_at_17_06(capsys):
    block = hour_blocks(run_cli(capsys, BLUE_MILL_GP3))["17"]
    assert block[0] == "17:00\u201318:00"
    sub_rows = [line for line in block if re.match(r"^  \d\d:\d\d", line)]
    assert sub_rows == [
        "  17:00\u201317:06  PILOT DISCRETION \u2013 APPROACHING LIMITS \u00b7 WITH CONDITIONS",
        "  17:06\u201318:00  NO-GO",
    ]
    twilight, night = segments_of(block)
    assert "caution: civil twilight: between civil dawn and sunrise, or sunset and civil dusk" in twilight
    assert "blocked (daylight): night: outside civil twilight" in night
    assert "blocked (daylight): night: outside civil twilight" not in twilight


def test_gp6_the_footer_has_sources_flyable_windows_and_the_overall_line(capsys):
    lines = run_cli(capsys, BLUE_MILL_GP3).splitlines()
    footer = lines[lines.index("Sources") :]
    assert footer[:4] == [
        "Sources",
        "  weather: ok",
        "  airspace: ok",
        "  NOTAMs: ok (PIB issued 2026-10-31T07:00:37Z) (case mode: captured PIB treated as current at the case's now)",
    ]
    windows = footer[footer.index("Flyable windows (UTC)") + 1 : footer.index("Overall: NO-GO")]
    assert windows == ["  2026-10-31T07:00Z to 2026-10-31T08:00Z", "  2026-10-31T10:00Z to 2026-10-31T17:06Z"]
    assert "Overall: NO-GO" in footer


def test_gp6_the_word_outlook_is_never_printed_as_a_verdict(capsys):
    text = run_cli(capsys, BLUE_MILL_GP3)
    assert "OUTLOOK" not in text


# --- GP7 ---------------------------------------------------------------------------------------------


def expected_west_sleekburn_condition() -> str:
    """The CTA is over the LZ in the committed airspace (PR-7's capture report), which gives the ceiling condition.

    If the committed data had it only within 2 NM the expected condition would be the sector one, so the
    expectation is read from the fixture itself.
    """
    _case, _forecast, airspace, _pib, _readings = cases.load(WEST_SLEEKBURN)
    cta = next(a for a in airspace.areas if a.name == "NEWCASTLE CTA 2")
    assert (cta.lower.value, cta.lower.unit.value, cta.lower.datum.value) == (1500, "ft", "AMSL")
    case = yaml.safe_load((cases.case_dir(WEST_SLEEKBURN) / "case.yaml").read_text(encoding="utf-8"))
    return STAY_BELOW if shape_over(case["lat"], case["lon"], cta.shape) else "avoid the sector"


def test_gp7_every_west_sleekburn_segment_carries_the_ceiling_condition_and_none_is_plain(capsys):
    expected = expected_west_sleekburn_condition()
    assert expected == STAY_BELOW  # PR-7: NEWCASTLE CTA 2 (1500 ft AMSL) lies over the LZ
    blocks = hour_blocks(run_cli(capsys, ["assess", "--case", WEST_SLEEKBURN]))
    assert list(blocks) == [f"{h:02d}" for h in range(7, 18)]
    for hour, block in blocks.items():
        for segment in segments_of(block):
            assert f"condition: {expected}" in [line.strip() for line in segment], (hour, segment)
            assert segment[0].split("  ", 1)[-1] != "PILOT DISCRETION", (hour, segment[0])


def test_gp7_the_ceiling_is_the_cta_floor_less_the_buffer(capsys):
    assert "condition: stay below 1,300 ft QNH" in run_cli(capsys, ["assess", "--case", WEST_SLEEKBURN])


# --- F4 ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("argv", [BLUE_MILL_GP3, ["assess", "--case", WEST_SLEEKBURN]])
def test_f4_no_output_line_has_the_bare_word_go_and_nothing_prints_a_score(capsys, argv):
    for extra in ([], ["--debug"]):
        text = run_cli(capsys, [*argv, *extra])
        assert not [line for line in text.splitlines() if BARE_GO.search(line)]
        assert not re.search(r"score|weight", text, re.IGNORECASE)


@pytest.mark.parametrize("name", [BLUE_MILL, WEST_SLEEKBURN])
def test_f4_no_golden_line_has_the_bare_word_go(name):
    text = (GOLDEN / f"{name}.txt").read_text(encoding="utf-8")
    assert not [line for line in text.splitlines() if BARE_GO.search(line)]


@pytest.mark.parametrize("name", [BLUE_MILL, WEST_SLEEKBURN])
def test_f6_no_golden_line_holds_a_key(name):
    text = (GOLDEN / f"{name}.txt").read_text(encoding="utf-8")
    assert "OPENAIP_API_KEY" not in text and "x-openaip-api-key" not in text


# --- F11: a contact never turns a prohibitive reading into CONTACT_FIRST ---------------------------------


@pytest.fixture
def changed_readings(tmp_path, monkeypatch):
    """Copy the Blue Mill case and rewrite J3857/26's reading. Returns a function that does the rewrite and runs the CLI."""
    shutil.copytree(cases.case_dir(BLUE_MILL), tmp_path / "fixtures" / BLUE_MILL)
    monkeypatch.setattr(sources, "ROOT", tmp_path)

    def run(capsys, **changes) -> str:
        path = tmp_path / "fixtures" / BLUE_MILL / "readings.yaml"
        readings = yaml.safe_load(path.read_text(encoding="utf-8"))
        next(r for r in readings if r["id"] == "J3857/26").update(changes)
        path.write_text(yaml.safe_dump(readings), encoding="utf-8")
        return run_cli(capsys, BLUE_MILL_GP3)

    return run


def test_f11_the_committed_reading_is_the_one_that_gives_contact_first(capsys, changed_readings):
    assert "CONTACT_FIRST" in run_cli(capsys, BLUE_MILL_GP3)
    assert "NOTAM J3857/26 CONTACT_FIRST" in changed_readings(capsys)  # an unchanged rewrite still gives it


@pytest.mark.parametrize(
    "changes",
    [
        {"prohibitive_reason": "live firing"},
        {"prohibitive_reason": "byelaws"},
        {"prohibitive_reason": "entry prohibited"},
    ],
)
def test_f11_a_prohibitive_reason_is_no_fly_and_never_contact_first(capsys, changed_readings, changes):
    text = changed_readings(capsys, **changes)
    assert "CONTACT_FIRST" not in text
    assert "NOTAM J3857/26 NO_FLY" in text
    blocks = hour_blocks(text)
    for hour in range(8, 18):
        for segment in segments_of(blocks[f"{hour:02d}"]):
            assert segment[0].split("  ", 1)[-1] == "NO-GO", (hour, segment[0])


@pytest.mark.parametrize("contact", [{"kind": "radio-only", "value": "Anglia Radar 128.925 MHz"}, {"kind": "none"}])
def test_f11_a_reading_with_no_contact_does_not_undo_the_phone_and_email_code_finds_in_item_e(capsys, changed_readings, contact):
    """Contact detection is code on the original item E (CR1): the reading's contact does not decide the danger area."""
    text = changed_readings(capsys, contact=contact)
    assert "NOTAM J3857/26 NO_FLY" not in text
    assert f"condition: {re.sub(r'phone Flylogix \d+ \d+', 'phone given in NOTAM J3857/26', DANGER_AREA)}" in text
    assert "code finds a phone in item E but the reading gives no phone or email; code decides (CONTACT_FIRST)" in text


@pytest.fixture
def changed_pib(tmp_path, monkeypatch):
    """Copy the Blue Mill case and rewrite J3857/26's item E in pib.xml (its reading gives no contact, so both paths agree).

    Returns a function that does it and runs the CLI.
    """
    shutil.copytree(cases.case_dir(BLUE_MILL), tmp_path / "fixtures" / BLUE_MILL)
    monkeypatch.setattr(sources, "ROOT", tmp_path)

    def run(capsys, edit) -> str:
        path = tmp_path / "fixtures" / BLUE_MILL / "pib.xml"
        text = path.read_text(encoding="utf-8")
        start = text.index("<Number>3857</Number>")
        first, last = text.index("<ItemE>", start), text.index("</ItemE>", start)
        edited = text[:first] + "<ItemE>" + edit(text[first + len("<ItemE>") : last]) + text[last:]
        assert edited != text
        path.write_text(edited, encoding="utf-8")
        readings_path = tmp_path / "fixtures" / BLUE_MILL / "readings.yaml"
        readings = yaml.safe_load(readings_path.read_text(encoding="utf-8"))
        next(r for r in readings if r["id"] == "J3857/26")["contact"] = {"kind": "none"}
        readings_path.write_text(yaml.safe_dump(readings), encoding="utf-8")
        return run_cli(capsys, BLUE_MILL_GP3)

    return run


def without_contact(item_e: str) -> str:
    """J3857/26's item E with its email and its number cut out; the radio frequencies, the AR reference and the SUP stay."""
    cut = re.sub(r"\S*\(AT\)\S*", "", item_e)
    cut = re.sub(r"\bOR\s+0[\d ]{9,13}\.", ".", cut)
    assert redact_phones(cut)[1] == {} and "(AT)" not in cut
    return cut


def test_f11_the_item_e_with_no_phone_or_email_is_check_sup_in_every_segment_from_08(capsys, changed_pib):
    text = changed_pib(capsys, without_contact)
    check_sup = "read the SUP / AIP ENR 5.1 entry for EGD496A before flight; details in AIP SUP 108/2026"
    assert "CONTACT_FIRST" not in text and "NO_FLY" not in text
    blocks = hour_blocks(text)
    for hour in range(8, 18):
        for segment in segments_of(blocks[f"{hour:02d}"]):
            lines = [line.strip() for line in segment]
            assert lines.count(f"condition: {check_sup}") == 1, (hour, segment)
            assert any(line.startswith("NOTAM J3857/26 CHECK_SUP") for line in lines), (hour, segment)
            assert not re.search(r"  PILOT DISCRETION$", segment[0]) and segment[0] != "PILOT DISCRETION", (hour, segment[0])


# --- E13: the clock change prints continuous UTC rows ------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end", "first", "count"),
    [
        ("2026-10-24T22:00Z", "2026-10-25T05:00Z", "22", 7),  # BST ends 25 Oct at 01:00 UTC
        ("2026-03-28T22:00Z", "2026-03-29T05:00Z", "22", 7),  # BST starts 29 Mar at 01:00 UTC
    ],
)
def test_e13_a_window_over_the_clock_change_has_one_row_per_utc_hour_with_no_gap_or_repeat(capsys, start, end, first, count):
    text = run_cli(capsys, ["assess", "--case", BLUE_MILL, "--start", start, "--end", end])
    spans = re.findall(r"^(\d\d:\d\d)\u2013(\d\d:\d\d)", text, re.MULTILINE)
    assert len(spans) == count
    starts = [int(s[0][:2]) for s in spans]
    assert starts[0] == int(first)
    assert starts == [(int(first) + i) % 24 for i in range(count)]  # consecutive, nothing missing or doubled
    assert [s[1] for s in spans[:-1]] == [s[0] for s in spans[1:]]
    assert len(re.findall(r"^\d{4}-\d\d-\d\d \(UTC\)$", text, re.MULTILINE)) == 2  # a date line for each of the two days


# --- F5/F6: case mode is offline, keyless and clockless ---------------------------------------------------


def test_f5_case_mode_reads_no_key_no_env_file_no_network_and_no_clock(capsys, monkeypatch):
    monkeypatch.delenv("OPENAIP_API_KEY", raising=False)

    def boom(*args, **kwargs):
        raise AssertionError("case mode reached for something it must not touch")

    monkeypatch.setattr(sources, "find_key", boom)
    monkeypatch.setattr(sources, "fetch_live_sources", boom)
    monkeypatch.setattr(sources.os, "environ", {})
    assert run_cli(capsys, BLUE_MILL_GP3) == (GOLDEN / f"{BLUE_MILL}.txt").read_text(encoding="utf-8")
