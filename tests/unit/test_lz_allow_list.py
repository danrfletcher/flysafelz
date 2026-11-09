"""F6 (5a PR-1.S1): every landing-zone location field in the dataset and the fixtures is on a fixed allow-list of the dataset's 9 points.

The allow-list is the 9 LZ points as committed in examples.yaml, plus one named invented point, `stale_pib_constructed`, which is allowed
only in a case.yaml that has `constructed: true` (Dan, Q6). Any other point fails, naming the file and the field and never the value.

A field is compared at its own written precision, but never at fewer than 4 decimal places: West Sleekburn's case.yaml writes 55.1576, -1.5699
and matches the full-precision point in examples.yaml; a field written as 55.16 does not match anything. Airspace and NOTAM geometry inside
the fixtures is out of scope: only `lat` and `lon` keys are read, in case.yaml (top level), examples.yaml, verdict_sheet.yaml and, when it exists,
evals/online/seed_questions.yaml.

Only this file holds coordinates. The off-list points below are invented and sit nowhere near a real field.
"""

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
MIN_DECIMALS = 4
CONSTRUCTED_NAME = "stale_pib_constructed"

DATASET_POINTS = [
    (53.802797, -0.086644),
    (55.60903936140245, -1.7128269843004924),
    (53.77816, -1.12256),
    (55.63619173451837, -2.126140599039252),
    (54.99468175190992, -1.6277450760041063),
    (55.41855998710967, -1.705571777810994),
    (55.295620798241295, -1.5741095087575143),
    (55.15762546285465, -1.569921231042746),  # West Sleekburn
    (54.52960562792453, -3.2566887588684965),
]
CONSTRUCTED_POINT = (51.9, -1.1)  # stale_pib_constructed: invented, a case.yaml with `constructed: true` only


def written_decimals(text: str) -> int:
    mantissa = text.lower().split("e")[0]
    return len(mantissa.split(".")[1]) if "." in mantissa else 0


def agrees(written: str, allowed: float) -> bool:
    """True when the written number is the allowed one, compared at the written precision (at least 4 decimal places)."""
    try:
        value = float(written)
    except ValueError:
        return False
    decimals = max(MIN_DECIMALS, written_decimals(written))
    return abs(value - allowed) <= 0.5 * 10**-decimals + 1e-12


def lz_mappings(node, top_level_only: bool):
    """Each mapping that holds a `lat` or a `lon` key, as {field: written text}. Nested ones too unless top_level_only."""
    if isinstance(node, yaml.MappingNode):
        fields = {k.value: v.value for k, v in node.value if isinstance(k, yaml.ScalarNode) and k.value in ("lat", "lon") and isinstance(v, yaml.ScalarNode)}
        if fields:
            yield fields
        if top_level_only:
            return
        for _k, v in node.value:
            yield from lz_mappings(v, top_level_only)
    elif isinstance(node, yaml.SequenceNode) and not top_level_only:
        for item in node.value:
            yield from lz_mappings(item, top_level_only)


def failures(label: str, text: str, *, top_level_only: bool = False, allow_constructed: bool = False) -> list[str]:
    """One line per LZ location field that is not on the allow-list: the file label and the field name, never the value."""
    points = [*DATASET_POINTS, *([CONSTRUCTED_POINT] if allow_constructed else [])]
    node = yaml.compose(text)
    found = []
    for fields in lz_mappings(node, top_level_only):
        if "lat" in fields and "lon" in fields:
            ok = any(agrees(fields["lat"], lat) and agrees(fields["lon"], lon) for lat, lon in points)
            if not ok:
                found.append(f"{label}: lat/lon is not an allowed landing-zone point")
        else:
            field, index = ("lat", 0) if "lat" in fields else ("lon", 1)
            if not any(agrees(fields[field], point[index]) for point in points):
                found.append(f"{label}: {field} is not an allowed landing-zone coordinate")
    return found


def is_constructed(text: str) -> bool:
    data = yaml.safe_load(text)
    return isinstance(data, dict) and data.get("constructed") is True


def case_failures(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return failures(f"{path.parent.name}/{path.name}", text, top_level_only=True, allow_constructed=path.parent.name == CONSTRUCTED_NAME and is_constructed(text))


def repo_files() -> list[Path]:
    files = [REPO / "evals" / "agent" / "data" / "examples.yaml", REPO / "evals" / "agent" / "data" / "verdict_sheet.yaml"]
    seed = REPO / "evals" / "online" / "seed_questions.yaml"
    return [*files, *([seed] if seed.exists() else [])]


# --- the real files ---------------------------------------------------------------------------------------------------------------


def test_every_case_yaml_in_fixtures_is_on_the_allow_list():
    cases = sorted((REPO / "fixtures").glob("*/case.yaml"))
    assert len(cases) >= 26  # 25 dataset cases plus the constructed one
    assert [line for path in cases for line in case_failures(path)] == []


def test_the_examples_and_the_verdict_sheet_and_the_seed_questions_are_on_the_allow_list():
    files = repo_files()
    assert {p.name for p in files} >= {"examples.yaml", "verdict_sheet.yaml"}
    assert [line for path in files for line in failures(path.name, path.read_text(encoding="utf-8"))] == []


def test_the_nine_points_are_the_ones_examples_yaml_holds():
    text = (REPO / "evals" / "agent" / "data" / "examples.yaml").read_text(encoding="utf-8")
    written = {(m["lat"], m["lon"]) for m in lz_mappings(yaml.compose(text), False) if len(m) == 2}
    assert len(DATASET_POINTS) == 9
    assert len(written) == 9
    for lat, lon in DATASET_POINTS:
        assert any(agrees(w_lat, lat) and agrees(w_lon, lon) for w_lat, w_lon in written), "an allow-listed point is not in examples.yaml"
    for w_lat, w_lon in written:
        assert any(agrees(w_lat, lat) and agrees(w_lon, lon) for lat, lon in DATASET_POINTS), "examples.yaml holds a point that is not allow-listed"


def test_west_sleekburns_case_yaml_matches_at_its_own_precision():
    path = REPO / "fixtures" / "west_sleekburn_2026-10-31" / "case.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert (data["lat"], data["lon"]) == (55.1576, -1.5699)
    assert case_failures(path) == []


def test_the_constructed_case_is_on_the_list_only_because_it_says_constructed_true():
    path = REPO / "fixtures" / CONSTRUCTED_NAME / "case.yaml"
    assert is_constructed(path.read_text(encoding="utf-8"))
    assert case_failures(path) == []


# --- the check fires --------------------------------------------------------------------------------------------------------------


def test_an_invented_off_list_point_fails_naming_the_file_and_field_but_not_the_value():
    got = failures("somewhere/case.yaml", "lat: 10.0001\nlon: 20.0002\n", top_level_only=True)
    assert got == ["somewhere/case.yaml: lat/lon is not an allowed landing-zone point"]
    assert "10.0001" not in got[0] and "20.0002" not in got[0]


def test_an_off_list_point_inside_an_examples_style_flow_mapping_fails():
    text = "- id: invented\n  inputs: {lat: 10.0001, lon: 20.0002, question: hello}\n"
    assert failures("examples.yaml", text) == ["examples.yaml: lat/lon is not an allowed landing-zone point"]


def test_a_lat_from_one_point_with_a_lon_from_another_fails():
    (lat, _), (_, lon) = DATASET_POINTS[0], DATASET_POINTS[1]
    assert failures("x.yaml", f"lat: {lat}\nlon: {lon}\n", top_level_only=True)


def test_a_single_lat_or_lon_field_is_checked_on_its_own():
    assert failures("x.yaml", "lat: 10.0001\n", top_level_only=True) == ["x.yaml: lat is not an allowed landing-zone coordinate"]
    assert failures("x.yaml", "lon: 20.0002\n", top_level_only=True) == ["x.yaml: lon is not an allowed landing-zone coordinate"]
    assert failures("x.yaml", f"lat: {DATASET_POINTS[0][0]}\n", top_level_only=True) == []


@pytest.mark.parametrize("lat, lon", [("53.80", "-0.09"), ("53.8", "-0.1"), ("54", "-1")])
def test_a_field_written_with_fewer_than_four_decimals_never_matches(lat, lon):
    assert failures("x.yaml", f"lat: {lat}\nlon: {lon}\n", top_level_only=True)


def test_four_decimals_is_enough_to_match():
    assert failures("x.yaml", "lat: 53.8028\nlon: -0.0866\n", top_level_only=True) == []


def test_a_point_a_whole_hundredth_away_does_not_match_at_four_decimals():
    assert failures("x.yaml", "lat: 53.8128\nlon: -0.0866\n", top_level_only=True)


def test_a_value_that_is_not_a_number_fails():
    assert failures("x.yaml", "lat: ~\nlon: -0.0866\n", top_level_only=True)
    assert failures("x.yaml", "lat: north\nlon: -0.0866\n", top_level_only=True)


def test_the_constructed_point_is_allowed_in_a_constructed_case_yaml():
    text = "constructed: true\nlat: 51.9\nlon: -1.1\n"
    assert failures("stale_pib_constructed/case.yaml", text, top_level_only=True, allow_constructed=is_constructed(text)) == []


def test_the_constructed_point_fails_in_a_case_without_constructed_true(tmp_path):
    for body in ("lat: 51.9\nlon: -1.1\n", "constructed: false\nlat: 51.9\nlon: -1.1\n"):
        folder = tmp_path / "some_other_case"
        folder.mkdir(exist_ok=True)
        (folder / "case.yaml").write_text(body, encoding="utf-8")
        assert case_failures(folder / "case.yaml") == ["some_other_case/case.yaml: lat/lon is not an allowed landing-zone point"]


def test_the_constructed_point_fails_in_a_case_named_like_it_but_not_marked_constructed(tmp_path):
    folder = tmp_path / CONSTRUCTED_NAME
    folder.mkdir()
    (folder / "case.yaml").write_text("lat: 51.9\nlon: -1.1\n", encoding="utf-8")
    assert case_failures(folder / "case.yaml")


def test_the_constructed_point_fails_in_a_constructed_case_that_is_not_the_named_one(tmp_path):
    folder = tmp_path / "another_made_up_case"
    folder.mkdir()
    (folder / "case.yaml").write_text("constructed: true\nlat: 51.9\nlon: -1.1\n", encoding="utf-8")
    assert case_failures(folder / "case.yaml")


def test_the_constructed_point_fails_in_any_other_file():
    text = "- id: invented\n  inputs: {lat: 51.9, lon: -1.1}\n"
    assert failures("examples.yaml", text) == ["examples.yaml: lat/lon is not an allowed landing-zone point"]


def test_a_nested_point_is_read_in_the_dataset_files_but_not_in_a_case_yaml():
    text = "constructed: false\nairspace:\n  - {lat: 10.0001, lon: 20.0002}\n"
    assert failures("x/case.yaml", text, top_level_only=True) == []  # geometry inside a fixture is out of scope
    assert failures("examples.yaml", text) == ["examples.yaml: lat/lon is not an allowed landing-zone point"]


def test_a_seed_questions_file_would_be_read_too(tmp_path, monkeypatch):
    online = tmp_path / "evals" / "online"
    online.mkdir(parents=True)
    (online / "seed_questions.yaml").write_text("- inputs: {lat: 10.0001, lon: 20.0002}\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "REPO", tmp_path)
    seed = tmp_path / "evals" / "online" / "seed_questions.yaml"
    assert seed in repo_files()
    assert failures(seed.name, seed.read_text(encoding="utf-8")) == ["seed_questions.yaml: lat/lon is not an allowed landing-zone point"]
