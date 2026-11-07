"""Fence checks for notam_action.py: F1 (no model call), F3 (no threshold literal), F7 (plain code).

Ticket 3b adds F1 (no new verdict or display string), F2 (the data never rules a danger area in), F6 (no network) and F7
(rows 1, 4-6 and the informational rule are unchanged for every subject but a danger area).
"""

import ast
import itertools
import re
import tomllib
from pathlib import Path

import pytest
from factories import LZ, SFC, amsl, circle_at, exemption, notam, reading, segment

from flysafelz import notam_action as module
from flysafelz.assess import display
from flysafelz.contracts import (
    AppliesTo,
    Category,
    ContactKind,
    ExemptionCovers,
    NotamAction,
    ProhibitiveReason,
    Segment,
    Verdict,
    display_verdict,
)
from flysafelz.notam_action import notam_action

PATH = Path(module.__file__)
FORBIDDEN_MODULES = ("openai", "anthropic", "langchain", "langgraph", "httpx")
THRESHOLDS = {2, 5000, 10000, 200}  # 2 NM, the 5,000 and 10,000 ft bands, the 200 ft buffer
TESTS = Path(__file__).parent


def tree() -> ast.Module:
    return ast.parse(PATH.read_text(encoding="utf-8"), filename=str(PATH))


def imported(module_tree: ast.Module) -> list[str]:
    found = []
    for node in ast.walk(module_tree):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            found.append(node.module or "")
    return found


def test_f1_notam_action_imports_nothing_that_could_call_a_model_or_the_network():
    for name in imported(tree()):
        root = name.split(".")[0]
        assert root not in FORBIDDEN_MODULES, name
        assert not root.startswith("langchain"), name
    assert not re.search(r"openai|anthropic|langchain|langgraph|httpx|os\.environ|getenv|dotenv", PATH.read_text(encoding="utf-8"), re.IGNORECASE)


def test_f1_the_test_files_import_no_model_client_either():
    for name in ("test_notam_action", "test_notam_action_blue_mill", "test_notam_action_fence"):
        path = TESTS / f"{name}.py"
        for found in imported(ast.parse(path.read_text(encoding="utf-8"))):
            assert found.split(".")[0] not in FORBIDDEN_MODULES, (name, found)


def test_f3_no_threshold_literal_in_the_code():
    numbers = [n.value for n in ast.walk(tree()) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)]
    assert [n for n in numbers if n in THRESHOLDS] == []


def test_f3_no_comparison_in_the_code_holds_a_number_other_than_zero_or_the_fl999_definition():
    allowed = {0, 1, 0.5, module.UNLIMITED_FL}
    for node in ast.walk(tree()):
        if isinstance(node, ast.Compare):
            for operand in [node.left, *node.comparators]:
                if isinstance(operand, ast.Constant) and isinstance(operand.value, (int, float)):
                    assert operand.value in allowed, ast.unparse(node)


def test_f7_no_class_hierarchy_beyond_models():
    bases = {ast.unparse(base) for node in ast.walk(tree()) if isinstance(node, ast.ClassDef) for base in node.bases}
    assert bases <= {"BaseModel", "TimeSpan", "NotamActionResult"}


# --- F6: no network ---------------------------------------------------------------------------------------------------------


def test_f6_the_sup_wording_comes_from_standard_library_code_only():
    from flysafelz import sup_refs

    assert set(imported(ast.parse(Path(sup_refs.__file__).read_text(encoding="utf-8")))) <= {"re"}
    assert "flysafelz.sup_refs" in imported(tree())


def test_f6_a_check_sup_runs_with_sockets_blocked():
    config = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8"))
    assert "--disable-socket" in config["tool"]["pytest"]["ini_options"]["addopts"]  # test_socket_guard shows it bites
    parsed = notam(circle=circle_at(0), item_f="SFC", subject="RD", item_e="DANGER AREA EGD496A. SEE AIP SUP 108/2026.")
    assert notam_action(parsed, reading(geometry=circle_at(0), contact=ContactKind.NONE, value=None), LZ, segment()).action is NotamAction.CHECK_SUP


# --- F1: no new verdict, no new display string -----------------------------------------------------------------------------


def test_f1_the_verdict_members_are_unchanged():
    assert [v.name for v in Verdict] == ["NO_GO", "UNVERIFIED", "PILOT_DISCRETION"]


def test_f1_check_sup_is_pilot_discretion_with_conditions_and_no_other_display_string():
    ok = Segment(start=segment().start, end=segment().end, verdict=Verdict.PILOT_DISCRETION, conditions=["read the SUP / AIP ENR 5.1 entry before flight"])
    assert display(ok) == display_verdict(Verdict.PILOT_DISCRETION, False, True) == "PILOT DISCRETION \u2013 WITH CONDITIONS"
    shown = {display_verdict(v, caution, cond) for v, caution, cond in itertools.product(Verdict, (False, True), (False, True))}
    assert shown == {
        "NO-GO",
        "UNVERIFIED",
        "PILOT DISCRETION",
        "PILOT DISCRETION \u2013 APPROACHING LIMITS",
        "PILOT DISCRETION \u2013 WITH CONDITIONS",
        "PILOT DISCRETION \u2013 APPROACHING LIMITS \u00b7 WITH CONDITIONS",
    }


# --- F2: the data never rules a danger area in ------------------------------------------------------------------------------

CONTACTS = {
    "phone": (ContactKind.PHONE, "07700 900123"),
    "email": (ContactKind.EMAIL, "ops@example.org"),
    "radio-only": (ContactKind.RADIO_ONLY, "123.450"),
    "none": (ContactKind.NONE, None),
    "blank phone": (ContactKind.PHONE, "  "),
}
EXEMPTIONS = {
    "none": [],
    "conditional": [exemption("fly only with   ATC permission.", ExemptionCovers.CONDITIONAL)],
    "other": [exemption("Hang gliders are exempt.", ExemptionCovers.OTHER)],
}
E_TEXT = "DANGER AREA EGD496A. FLY ONLY WITH ATC PERMISSION. SEE AIP SUP 108/2026."


@pytest.mark.parametrize("exemptions", EXEMPTIONS)
@pytest.mark.parametrize("contact", CONTACTS)
@pytest.mark.parametrize("reason", list(ProhibitiveReason))
@pytest.mark.parametrize("category", list(Category))
def test_f2_no_danger_area_over_the_lz_from_sfc_comes_out_ignore_or_awareness_and_no_fly_needs_a_reason(category, reason, contact, exemptions):
    kind, value = CONTACTS[contact]
    parsed = notam(circle=circle_at(0), item_f="SFC", item_e=E_TEXT, subject="RD")
    read = reading(geometry=circle_at(0), category=category, contact=kind, value=value, reason=reason, quote=None if reason is ProhibitiveReason.NONE else "ENTRY PROHIBITED", exemptions=EXEMPTIONS[exemptions])
    action = notam_action(parsed, read, LZ, segment()).action
    assert action not in (NotamAction.IGNORE, NotamAction.AWARENESS, NotamAction.AWARENESS_MAP, NotamAction.AVOID_SECTOR, NotamAction.UNREAD_OVER_LZ)
    assert (action is NotamAction.NO_FLY) == (reason is not ProhibitiveReason.NONE)
    if reason is ProhibitiveReason.NONE:
        usable = kind in (ContactKind.PHONE, ContactKind.EMAIL) and bool((value or "").strip())
        assert action is (NotamAction.CONTACT_FIRST if usable else NotamAction.CHECK_SUP)


@pytest.mark.parametrize("applies", [AppliesTo.UAS_ONLY, AppliesTo.AERODROME_USERS])
def test_f2_the_row_1_exemptions_are_the_only_way_a_danger_area_over_the_lz_is_ignored(applies):
    parsed = notam(circle=circle_at(0), item_f="SFC", item_e=E_TEXT, subject="RD")
    assert notam_action(parsed, reading(geometry=circle_at(0), applies_to=applies), LZ, segment()).action is NotamAction.IGNORE


# --- F7: rows 1, 4-6 and the informational rule are unchanged for every subject but a danger area ------------------------------

PLAIN = "AIRSPACE RESTRICTED. PARAMOTORS EXEMPT. FLY ONLY WITH ATC PERMISSION."
UNCHANGED = {
    "drones only": (NotamAction.IGNORE, "applies to drones only"),
    "aerodrome users": (NotamAction.IGNORE, "applies only to aircraft using EGTT"),
    "paramotor exemption": (NotamAction.IGNORE, 'exempts paramotors: "Paramotors exempt."'),
    "inactive": (NotamAction.IGNORE, "not active in this segment (schedule 1200-1800)"),
    "beyond the local area": (NotamAction.IGNORE, "3.01 NM from the LZ, beyond the 2 NM local area"),
    "unread": (NotamAction.UNREAD_OVER_LZ, f"NOTAM H7171/26 is over the LZ from the surface and has not been read: {PLAIN}"),
    "sector": (NotamAction.AVOID_SECTOR, "drone ops 1.00 NM to the S (circle edge), up to 800 ft AMSL: avoid that sector"),
    "ceiling": (NotamAction.AVOID_SECTOR, "drone ops over the LZ from 2,000 ft AMSL (2,000 ft QNH): stay below 1,800 ft QNH"),
    "awareness": (NotamAction.AWARENESS, "drone ops over the LZ from 7,000 ft AMSL (7,000 ft QNH): awareness"),
    "map only": (NotamAction.AWARENESS_MAP, "drone ops over the LZ from 12,000 ft AMSL (12,000 ft QNH): map only"),
    "informational": (NotamAction.AWARENESS, "drone ops: informational over the LZ from the surface, a warning and not a restriction"),
    "prohibitive": (NotamAction.NO_FLY, "drone ops over the LZ from the surface: prohibitive"),
    "prohibitive reason": (NotamAction.NO_FLY, "drone ops over the LZ from the surface: prohibitive (live firing)"),
    "cooperative, contact": (NotamAction.CONTACT_FIRST, "drone ops over the LZ from the surface: contact phone 07700 900123 first and fly only if they agree"),
    "cooperative, no contact": (NotamAction.NO_FLY, "drone ops over the LZ from the surface: cooperative but there is no phone or email contact"),
    "cooperative, radio only": (NotamAction.NO_FLY, "drone ops over the LZ from the surface: cooperative but there is no phone or email contact (a radio frequency alone is not a contact)"),
}


def unchanged_case(name: str, subject: str):
    """The NOTAM and reading for one row of UNCHANGED, with a SUP reference and designator in item E (they must not leak in)."""
    e_text = PLAIN
    circle, floor, item_f, item_d, schedule = circle_at(0), SFC, "SFC", None, ""
    kw: dict = {}
    if name == "drones only":
        kw = {"applies_to": AppliesTo.UAS_ONLY}
    elif name == "aerodrome users":
        kw = {"applies_to": AppliesTo.AERODROME_USERS}
    elif name == "paramotor exemption":
        kw = {"exemptions": [exemption("Paramotors exempt.")]}
    elif name == "inactive":
        item_d, schedule = "1200-1800", "1200-1800"
    elif name == "beyond the local area":
        circle = circle_at(3.0)
    elif name == "sector":
        circle = circle_at(1.0)
    elif name == "ceiling":
        floor, item_f = amsl(2000), "2000FT AMSL"
    elif name == "awareness":
        floor, item_f = amsl(7000), "7000FT AMSL"
    elif name == "map only":
        floor, item_f = amsl(12000), "12000FT AMSL"
    elif name == "informational":
        kw = {"category": Category.INFORMATIONAL}
    elif name == "prohibitive":
        kw = {"category": Category.PROHIBITIVE}
    elif name == "prohibitive reason":
        kw = {"reason": ProhibitiveReason.LIVE_FIRING}
    elif name == "cooperative, no contact":
        kw = {"contact": ContactKind.NONE, "value": None}
    elif name == "cooperative, radio only":
        kw = {"contact": ContactKind.RADIO_ONLY, "value": "123.450"}
    elif name == "cooperative, contact":
        kw = {"value": "07700 900123"}
    parsed = notam(circle=circle, item_f=item_f, item_e=e_text, subject=subject, **({"item_d": item_d} if item_d else {}))
    read = None if name == "unread" else reading(geometry=circle, floor=floor, schedule=schedule, **kw)
    return parsed, read


@pytest.mark.parametrize("subject", ["WU", "RC", "XX"])
@pytest.mark.parametrize("name", UNCHANGED)
def test_f7_the_action_and_reason_of_every_other_row_are_byte_for_byte_what_they_were(name, subject):
    parsed, read = unchanged_case(name, subject)
    result = notam_action(parsed, read, LZ, segment())
    assert (result.action, result.reason) == UNCHANGED[name]


@pytest.mark.parametrize("name", [n for n in UNCHANGED if n not in ("prohibitive", "prohibitive reason", "cooperative, contact", "cooperative, no contact", "cooperative, radio only", "informational")])
def test_f7_rows_1_and_4_to_6_are_the_same_for_a_danger_area(name):
    plain, danger = (notam_action(*unchanged_case(name, subject), LZ, segment()) for subject in ("WU", "RD"))
    assert (danger.action, danger.reason) == (plain.action, plain.reason)
