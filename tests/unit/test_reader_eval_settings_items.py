"""The reader eval's settings.yaml, its loader of the sheet and the committed PIBs, and notam_action.merged_limits (the E10 merge as numbers)."""

import math
import shutil

import pytest
import yaml
from factories import LZ, agl, amsl, fl, reading, segment
from reader_eval_helpers import q_notam

from evals.reader_eval import items as items_module
from evals.reader_eval.items import PIBS_DIR, ItemError, load_items, parsed_notam
from evals.reader_eval.settings import SETTINGS, SETTINGS_PATH, EvalSettings, load_settings, reference_pressure_hpa
from flysafelz.notam_action import LandingZone, merged_limits, notam_action

# --- settings --------------------------------------------------------------------------------------------------------------------


def test_the_shipped_settings_are_the_specs():
    assert (SETTINGS.ring_inner_nm, SETTINGS.ring_outer_nm, SETTINGS.site_elevation_ft, SETTINGS.height_tolerance_ft) == (0.1, 1.9, 500, 100)
    assert SETTINGS.field_height_tolerance_ft == 100
    assert reference_pressure_hpa() == 1013.25


def test_every_key_is_required_and_none_may_be_added(tmp_path):
    shipped = yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8"))
    path = tmp_path / "s.yaml"
    for key in shipped:
        path.write_text(yaml.safe_dump({k: v for k, v in shipped.items() if k != key}), encoding="utf-8")
        with pytest.raises(ValueError, match=key):
            load_settings(path)
    path.write_text(yaml.safe_dump({**shipped, "ring_middle_nm": 1.0}), encoding="utf-8")
    with pytest.raises(ValueError, match="ring_middle_nm"):
        load_settings(path)


def test_an_impossible_value_is_refused(tmp_path):
    shipped = yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8"))
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump({**shipped, "ring_inner_nm": 0}), encoding="utf-8")
    with pytest.raises(ValueError, match="ring_inner_nm"):
        load_settings(path)
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="ring_outer_nm"):
        load_settings(path)  # an empty file names every missing key


def test_the_outer_ring_is_nearer_than_the_local_area_by_a_margin():
    from flysafelz.rules import RULES

    local_area = RULES.local_area.value
    assert not isinstance(local_area, list)
    assert SETTINGS.ring_outer_nm < local_area
    assert isinstance(SETTINGS, EvalSettings)


# --- the sheet --------------------------------------------------------------------------------------------------------------------


def test_the_committed_sheet_has_40_items_each_rebuilt_from_its_pib():
    sheet = load_items()
    assert len(sheet) == 40
    for item in sheet:
        parsed = parsed_notam(item)
        assert parsed.id == item["id"]


def test_a_missing_sheet_is_an_item_error(tmp_path):
    with pytest.raises(ItemError, match="cannot read the sheet"):
        load_items(tmp_path / "absent.yaml")


@pytest.mark.parametrize(
    "text, message", [("items: 3\n", "no `items:` list"), ("- a\n", "no `items:` list"), ("items:\n  - rank: 1\n", "item 1 has no id and e_hash"), ("items: [x: [", "cannot read the sheet")]
)
def test_a_sheet_that_is_not_usable_is_an_item_error(tmp_path, text, message):
    path = tmp_path / "items.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ItemError, match=message):
        load_items(path)


def test_an_item_whose_text_has_changed_is_refused():
    item = dict(load_items()[0])
    item["e_hash"] = "0" * 64
    with pytest.raises(ItemError, match="no longer matches"):
        parsed_notam(item)


def test_an_item_missing_from_its_pib_or_whose_pib_is_missing_is_refused(tmp_path):
    item = dict(load_items()[0])
    with pytest.raises(ItemError, match="cannot be read"):
        parsed_notam(item, tmp_path)  # no PIB in this folder
    shutil.copy(PIBS_DIR / f"{item['sha']}.xml", tmp_path / f"{item['sha']}.xml")
    assert parsed_notam(item, tmp_path).id == item["id"]
    with pytest.raises(ItemError, match="not in the PIB"):
        parsed_notam({**item, "id": "X9999/26"}, tmp_path)


def test_a_file_that_is_not_a_pib_is_refused(tmp_path):
    item = dict(load_items()[0])
    (tmp_path / f"{item['sha']}.xml").write_text("this is not a PIB", encoding="utf-8")
    with pytest.raises(ItemError, match="is not a PIB"):
        parsed_notam(item, tmp_path)
    assert items_module.ITEMS_PATH.name == "items.yaml" and items_module.LABELS_PATH.name == "labels.yaml"


# --- merged_limits -----------------------------------------------------------------------------------------------------------------


def lz_at_500():
    return LandingZone(lat=LZ.lat, lon=LZ.lon, elevation_ft=500.0)


def test_merged_limits_are_the_floor_and_ceiling_notam_action_works_with_in_feet_qnh():
    parsed = q_notam(upper_fl=100)
    floor, ceiling = merged_limits(parsed, reading(floor=agl(200), ceiling=agl(300)), lz_at_500(), segment())
    assert (floor, ceiling) == pytest.approx((700, 800))  # AGL adds the 500 ft ground
    floor, ceiling = merged_limits(parsed, reading(floor=amsl(1000), ceiling=fl(30)), lz_at_500(), segment())
    assert (floor, ceiling) == pytest.approx((1000, 3000))  # FL030 at standard pressure


def test_merged_limits_with_no_ceiling_fall_back_to_the_q_line_and_with_nothing_to_infinity():
    parsed = q_notam(upper_fl=100)
    no_ceiling = reading(floor=amsl(1000)).model_copy(update={"ceiling": None})
    assert merged_limits(parsed, no_ceiling, lz_at_500(), segment())[1] == pytest.approx(10000)  # E10: the Q-line's FL100
    assert math.isinf(merged_limits(q_notam(upper_fl=999), no_ceiling, lz_at_500(), segment())[1])


def test_merged_limits_follow_the_segments_forecast_pressure():
    parsed = q_notam(upper_fl=100)
    read = reading(floor=amsl(1000), ceiling=fl(30))
    low = merged_limits(parsed, read, lz_at_500(), segment(pressure=983.25))[1]
    assert low < 3000  # a lower QNH puts the flight level lower in feet QNH


def test_merged_limits_change_nothing_in_notam_action():
    """The wrapper reads, it does not write: notam_action's answer is the same before and after, and from a reading or without one."""
    parsed = q_notam(upper_fl=100)
    read = reading(floor=amsl(1000), ceiling=amsl(2000))
    before = notam_action(parsed, read, lz_at_500(), segment())
    merged_limits(parsed, read, lz_at_500(), segment())
    merged_limits(parsed, None, lz_at_500(), segment())
    assert notam_action(parsed, read, lz_at_500(), segment()) == before
