"""5a PR-2.F1 G1: the recombination rule of verdict_sheet_finalise.recombine, on real assess() output.

Every case is a synthetic fixtures/<name>/ written by write_case into a tmp root, with fake readings.yaml files (a NotamReading dumped
as YAML), assessed by verdict_sheet._assess_example (the call the sheet's own rows come from) and recombined with Dan's actions.
Nothing here opens a socket or runs a reader.

The table is written out from the spec, not from the code: any blocker is NO_GO; else any unverified reason is UNVERIFIED; else
PILOT_DISCRETION; only NO_FLY makes a blocker out of a NOTAM; approaching_limits is code's.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from factories import circle_at, notam, reading
from verdict_sheet_helpers import example, examples_file, write_case

from flysafelz.contracts import Category, ContactKind, NotamAction, Tier, Verdict
from flysafelz.verdict_sheet import _assess_example, read_examples
from flysafelz.verdict_sheet_finalise import UNREAD_ACTIVATION, dan_actions, kept_unverified, recombine

NOW = datetime(2026, 10, 31, 7, 30, tzinfo=UTC)


def utc(hour, minute=0):
    return datetime(2026, 10, 31, hour, minute, tzinfo=UTC)


WINDOW = (utc(8), utc(11))
ALL_DAY = {"valid_from": utc(6), "valid_to": utc(20)}
NO_ACTION = object()  # "Dan wrote nothing for this NOTAM"

# What each code-made NOTAM action needs from a reading (None: no reading at all, so code says UNREAD_OVER_LZ).
READINGS = {
    "unread": (circle_at(0), None),
    "NO_FLY": (circle_at(0), {"category": Category.PROHIBITIVE}),
    "CONTACT_FIRST": (circle_at(0), {}),
    "AWARENESS": (circle_at(0), {"category": Category.INFORMATIONAL, "contact": ContactKind.NONE, "value": None}),
    "AVOID_SECTOR": (circle_at(1.0), {}),
}


def weather_edit(folder: Path, **series: float) -> None:
    path = folder / "weather.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for name, value in series.items():
        data["hourly"][name] = [value] * len(data["hourly"]["time"])
    path.write_text(json.dumps(data), encoding="utf-8")


def assessed(tmp_path: Path, name: str, notams: dict[str, tuple[str, dict]], *, weather: str = "calm"):
    """The _Assessed of one synthetic case. notams: id -> (a READINGS key, notam() overrides). weather: calm, marginal, windy or dropped."""
    parsed, readings = [], []
    for notam_id, (kind, overrides) in notams.items():
        circle, kw = READINGS[kind]
        parsed.append(notam(notam_id, circle=circle, **overrides))
        if kw is not None:
            readings.append(reading(notam_id=notam_id, **kw).model_dump(mode="json"))
    folder = write_case(tmp_path, name, now=NOW, window=WINDOW, notams=tuple(parsed))
    (folder / "readings.yaml").write_text(yaml.safe_dump(readings), encoding="utf-8")
    if weather == "windy":
        weather_edit(folder, wind_speed_10m=30.0, wind_gusts_10m=45.0)
    elif weather == "marginal":
        weather_edit(folder, wind_speed_10m=14.5)
    ex = example("e1", name, now=NOW, window=WINDOW, drop=["weather"] if weather == "dropped" else None)
    (one,) = read_examples(examples_file(tmp_path / f"{name}.yaml", ex))
    return _assess_example(one, folder)


def segments_of(built):
    return [s for h in built.assessment.hours if h.tier is Tier.VERDICT for s in h.segments]


def labels_of(built, actions):
    return [(v.value, flag) for v, flag in recombine(built, actions)]


def oracle(action, weather: str) -> str:
    """The spec's rule, one NOTAM at the LZ in every segment: NO_FLY or a weather blocker is NO_GO; a dropped source is UNVERIFIED."""
    if weather == "windy" or action is NotamAction.NO_FLY:
        return "NO_GO"
    return "UNVERIFIED" if weather == "dropped" else "PILOT_DISCRETION"


DAN = [a for a in NotamAction if a is not NotamAction.UNREAD_OVER_LZ]
WEATHERS = ["calm", "marginal", "windy", "dropped"]


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    """One assessed case per (what code made of the NOTAM, weather): 5 x 4 assess() calls, shared by the whole table."""
    root = tmp_path_factory.mktemp("table")
    return {(kind, weather): assessed(root, f"{kind.lower()}_{weather}", {"X1/26": (kind, ALL_DAY)}, weather=weather) for kind in READINGS for weather in WEATHERS}


@pytest.mark.parametrize("weather", WEATHERS)
@pytest.mark.parametrize("kind", list(READINGS))
@pytest.mark.parametrize("action", [*DAN, None, NO_ACTION], ids=lambda a: "unlabelled" if a is NO_ACTION else "null" if a is None else a.value)
def test_every_dan_action_against_every_code_action_and_other_source(table, kind, weather, action):
    built = table[(kind, weather)]
    written = {} if action is NO_ACTION else {"X1/26": None if action is None else action.value}
    out = labels_of(built, dan_actions({"notam_actions": written}))
    effective = NotamAction.IGNORE if action in (None, NO_ACTION) else action
    assert out and {v for v, _flag in out} == {oracle(effective, weather)}
    assert [flag for _v, flag in out] == [s.approaching_limits for s in segments_of(built)]  # approaching_limits is code's, untouched


def test_the_table_really_exercises_what_code_made_of_the_notam(table):
    """The fake readings do what the table's names say, so the code side of the table is not a guess."""
    for kind in READINGS:
        for segment in segments_of(table[(kind, "calm")]):
            assert {r.action.value for r in segment.notam_actions} == {"UNREAD_OVER_LZ" if kind == "unread" else kind}
    assert any(flag for _v, flag in labels_of(table[("unread", "marginal")], {"X1/26": NotamAction.IGNORE}))
    assert not any(flag for _v, flag in labels_of(table[("unread", "calm")], {"X1/26": NotamAction.IGNORE}))
    assert {s.verdict for s in segments_of(table[("NO_FLY", "calm")])} == {Verdict.NO_GO}
    assert {s.verdict for s in segments_of(table[("unread", "calm")])} == {Verdict.UNVERIFIED}


@pytest.mark.parametrize("weather", WEATHERS)
def test_parity_when_dans_actions_equal_codes_every_row_equals_assess(tmp_path, weather):
    """Four NOTAMs, each with one action all through the window; Dan labelled each as code did."""
    notams = {"A1/26": ("NO_FLY", {"valid_from": utc(9), "valid_to": utc(20)}), "B1/26": ("CONTACT_FIRST", ALL_DAY), "C1/26": ("AWARENESS", ALL_DAY), "D1/26": ("AVOID_SECTOR", ALL_DAY)}
    built = assessed(tmp_path, "parity", notams, weather=weather)
    seen: dict[str, set[NotamAction]] = {}
    for segment in segments_of(built):
        for result in segment.notam_actions:
            seen.setdefault(result.notam_id, set()).add(result.action)
    assert {k: {a.value for a in v} for k, v in seen.items()} == {"A1/26": {"NO_FLY"}, "B1/26": {"CONTACT_FIRST"}, "C1/26": {"AWARENESS"}, "D1/26": {"AVOID_SECTOR"}}
    actions = {notam_id: next(iter(found)) for notam_id, found in seen.items()}
    assert recombine(built, actions) == [(s.verdict, s.approaching_limits) for s in segments_of(built)]


def test_a_labelled_notam_that_is_not_active_in_a_row_has_no_effect_on_that_row(tmp_path):
    built = assessed(tmp_path, "later", {"X1/26": ("unread", {"valid_from": utc(9), "valid_to": utc(13)})})
    out = recombine(built, {"X1/26": NotamAction.NO_FLY})
    segments = segments_of(built)
    assert len(segments) > 1
    for segment, (verdict, _flag) in zip(segments, out, strict=True):
        assert verdict is (Verdict.NO_GO if segment.end > utc(9) else Verdict.PILOT_DISCRETION)  # nothing in code's rows before 09:00 mentions X1
    assert out[0][0] is Verdict.PILOT_DISCRETION and out[-1][0] is Verdict.NO_GO


def test_a_notam_dan_did_not_list_next_to_one_he_did_counts_as_ignore(tmp_path):
    notams = {"A1/26": ("unread", ALL_DAY), "B1/26": ("unread", ALL_DAY)}
    built = assessed(tmp_path, "two", notams)
    assert {v for v, _ in labels_of(built, dan_actions({"notam_actions": {"A1/26": "AWARENESS"}}))} == {"PILOT_DISCRETION"}
    assert {v for v, _ in labels_of(built, dan_actions({"notam_actions": {"A1/26": "AWARENESS", "B1/26": "NO_FLY"}}))} == {"NO_GO"}
    assert {v for v, _ in labels_of(built, dan_actions({"notam_actions": None}))} == {"PILOT_DISCRETION"}
    assert {v for v, _ in labels_of(built, dan_actions({}))} == {"PILOT_DISCRETION"}


def test_a_dropped_weather_source_keeps_its_unverified_whatever_dan_labelled(tmp_path):
    built = assessed(tmp_path, "dropped", {"X1/26": ("unread", ALL_DAY)}, weather="dropped")
    for action in DAN:
        verdicts = {v for v, _ in labels_of(built, {"X1/26": action})}
        assert verdicts == ({"NO_GO"} if action is NotamAction.NO_FLY else {"UNVERIFIED"})


def test_only_the_first_unread_notam_reason_goes_when_the_unread_notam_is_the_only_unverified_reason(tmp_path):
    built = assessed(tmp_path, "only", {"X1/26": ("unread", ALL_DAY)})
    assert {s.verdict for s in segments_of(built)} == {Verdict.UNVERIFIED}  # code: UNVERIFIED because of X1 alone
    assert {v for v, _ in labels_of(built, {"X1/26": NotamAction.CONTACT_FIRST})} == {"PILOT_DISCRETION"}


def test_code_made_no_fly_notam_dan_ignores_gives_the_other_sources_verdict(table):
    for weather, wanted in (("calm", "PILOT_DISCRETION"), ("windy", "NO_GO"), ("dropped", "UNVERIFIED")):
        built = table[("NO_FLY", weather)]
        assert {v.value for v in {s.verdict for s in segments_of(built)}} == {"NO_GO"}  # code said NO_GO throughout
        assert {v for v, _ in labels_of(built, {"X1/26": NotamAction.IGNORE})} == {wanted}
        assert {v for v, _ in labels_of(built, {"X1/26": NotamAction.AWARENESS})} == {wanted}


def test_dan_no_fly_on_a_notam_code_made_avoid_sector_is_no_go(table):
    built = table[("AVOID_SECTOR", "calm")]
    assert {s.verdict for s in segments_of(built)} == {Verdict.PILOT_DISCRETION}
    assert {v for v, _ in labels_of(built, {"X1/26": NotamAction.NO_FLY})} == {"NO_GO"}


def test_a_weather_blocker_with_dans_pilot_discretion_only_labels_is_no_go(table):
    for action in (NotamAction.CONTACT_FIRST, NotamAction.CHECK_SUP, NotamAction.AWARENESS, NotamAction.IGNORE):
        assert {v for v, _ in labels_of(table[("CONTACT_FIRST", "windy")], {"X1/26": action})} == {"NO_GO"}


def test_the_labels_come_back_one_per_verdict_segment_in_time_order(table):
    built = table[("unread", "calm")]
    assert len(recombine(built, {})) == len(segments_of(built)) > 1


# --- dan_actions --------------------------------------------------------------------------------------------------------------------------


def test_dan_actions_reads_null_and_missing_as_ignore():
    got = dan_actions({"notam_actions": {"A1/26": None, "B1/26": "NO_FLY", "C1/26": "CHECK_SUP"}})
    assert got == {"A1/26": NotamAction.IGNORE, "B1/26": NotamAction.NO_FLY, "C1/26": NotamAction.CHECK_SUP}
    assert dan_actions({"notam_actions": None}) == {} and dan_actions({}) == {}


def test_dan_actions_refuses_an_entry_with_an_unread_notam_naming_it():
    with pytest.raises(ValueError, match=r"B1/26.*UNREAD_OVER_LZ") as info:
        dan_actions({"notam_actions": {"A1/26": "NO_FLY", "B1/26": "UNREAD_OVER_LZ"}})
    assert "A1/26" not in str(info.value)


# --- the danger-area reason that names an unread NOTAM ------------------------------------------------------------------------------------


def test_the_danger_area_reason_wording_is_what_airspace_writes():
    """The marker kept_unverified strips is built from the same words airspace._danger_over_lz writes for an activation nobody read."""
    from flysafelz import airspace

    source = Path(airspace.__file__).read_text(encoding="utf-8")
    assert UNREAD_ACTIVATION.format(notam_id="{activation.notam_id}") in source


def test_kept_unverified_drops_the_unread_notam_reasons_and_keeps_the_rest(tmp_path):
    built = assessed(tmp_path, "kept", {"X1/26": ("unread", ALL_DAY)}, weather="dropped")
    (segment, *_rest) = segments_of(built)
    assert len(segment.conditions) >= 2 and kept_unverified(segment, ["X1/26"]) == ["UNVERIFIED: weather source failed: dropped for this example"]
    activated = "UNVERIFIED: Danger area EGD598 is activated by NOTAM J3219/26, which has not been read: TRIGGER"
    fake = segment.model_copy(update={"conditions": [activated, "UNVERIFIED: airspace source failed: dropped for this example"]})
    assert kept_unverified(fake, ["J3219/26"]) == ["UNVERIFIED: airspace source failed: dropped for this example"]
    assert kept_unverified(fake, ["Z9999/26"]) == fake.conditions  # a NOTAM outside the prefiltered set is not Dan's to have read
