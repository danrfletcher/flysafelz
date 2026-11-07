"""G8: `prohibitive_quote` on NotamReading (optional, None by default) and on ReaderOutput (required, strict), and merge_reading.

merge_reading and read_notam are the real functions; the model is the recording fake from reader_helpers.py.
"""

import json

import pytest
from factories import reading
from pib_helpers import parsed
from pydantic import ValidationError
from reader_helpers import reply_fields, scripted

from flysafelz.contracts import ContactKind, NotamReading, ProhibitiveReason
from flysafelz.notam_area import build_area
from flysafelz.reader.contract import ReaderOutput, ReaderStatus, merge_reading
from flysafelz.reader.read import read_notam, response_format

E_TEXT = "DANGER AREA ACTIVE. ENTRY PROHIBITED BY BYELAW."


def output(**overrides) -> ReaderOutput:
    return ReaderOutput.model_validate(reply_fields(**overrides))


def merged(**overrides) -> NotamReading:
    notam = parsed(item_e=E_TEXT)
    return merge_reading(notam, build_area(notam)[0], output(**overrides))


# --- the field ----------------------------------------------------------------------------------


def test_the_quote_is_required_on_reader_output_and_optional_on_the_reading():
    assert ReaderOutput.model_fields["prohibitive_quote"].is_required()
    assert not NotamReading.model_fields["prohibitive_quote"].is_required()
    assert NotamReading.model_fields["prohibitive_quote"].default is None
    fields = reply_fields()
    del fields["prohibitive_quote"]
    with pytest.raises(ValidationError, match="prohibitive_quote"):
        ReaderOutput.model_validate(fields)


def test_a_reading_written_before_the_field_loads_with_the_quote_none():
    old = reading().model_dump(mode="json")
    del old["prohibitive_quote"]
    assert NotamReading.model_validate(old).prohibitive_quote is None
    assert reading().prohibitive_quote is None  # tests/factories.py needs no change


def test_the_quote_is_a_string_or_null_and_nothing_else():
    assert output(prohibitive_reason="byelaws", prohibitive_quote="ENTRY PROHIBITED").prohibitive_quote == "ENTRY PROHIBITED"
    assert output(prohibitive_quote=None).prohibitive_quote is None
    for bad in (7, ["ENTRY"], {"quote": "x"}, True):
        with pytest.raises(ValidationError):
            ReaderOutput.model_validate(reply_fields(prohibitive_reason="byelaws", prohibitive_quote=bad))


def test_the_response_format_lists_the_quote_as_required_with_a_string_or_null_type():
    schema = response_format()["json_schema"]["schema"]
    assert "prohibitive_quote" in schema["required"]
    assert schema["required"] == list(schema["properties"]) == list(ReaderOutput.model_fields)
    assert {type_["type"] for type_ in schema["properties"]["prohibitive_quote"]["anyOf"]} == {"string", "null"}
    assert "default" not in json.dumps(schema["properties"]["prohibitive_quote"])


def test_reason_values_are_still_exactly_the_four():
    assert {r.value for r in ProhibitiveReason} == {"byelaws", "entry prohibited", "live firing", "none"}
    schema = response_format()["json_schema"]["schema"]
    assert set(schema["$defs"]["ProhibitiveReason"]["enum"]) == {"byelaws", "entry prohibited", "live firing", "none"}


# --- merge_reading -------------------------------------------------------------------------------


@pytest.mark.parametrize("reason", [r for r in ProhibitiveReason if r is not ProhibitiveReason.NONE], ids=lambda r: r.value)
def test_a_reason_keeps_its_quote_through_the_merge(reason):
    assert merged(prohibitive_reason=reason.value, prohibitive_quote="ENTRY PROHIBITED BY BYELAW").prohibitive_quote == "ENTRY PROHIBITED BY BYELAW"


@pytest.mark.parametrize("quote", ["ENTRY PROHIBITED BY BYELAW", "", "  ", None])
def test_the_reason_none_drops_the_quote_to_none(quote):
    result = merged(prohibitive_reason="none", prohibitive_quote=quote)
    assert result.prohibitive_reason is ProhibitiveReason.NONE and result.prohibitive_quote is None


def test_a_reason_with_no_quote_is_kept_for_the_checks_to_judge_not_mended_by_the_merge():
    result = merged(prohibitive_reason="byelaws", prohibitive_quote=None)
    assert result.prohibitive_reason is ProhibitiveReason.BYELAWS and result.prohibitive_quote is None


def test_a_model_read_drops_the_quote_with_the_reason_none_and_is_not_a_failure():
    notam = parsed(item_e=E_TEXT)
    result = read_notam(notam, scripted(reply_fields(prohibitive_reason="none", prohibitive_quote="ENTRY PROHIBITED BY BYELAW", contact={"kind": "none", "value": None})))
    assert result.status is ReaderStatus.READ and result.failed_checks == []
    assert result.reading is not None
    assert result.reading.prohibitive_quote is None
    assert result.reading.contact.kind is ContactKind.NONE


def test_a_reading_round_trips_its_quote_through_json():
    original = merged(prohibitive_reason="byelaws", prohibitive_quote="ENTRY PROHIBITED BY BYELAW")
    assert NotamReading.model_validate_json(original.model_dump_json()) == original
