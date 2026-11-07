"""The code checks on a reading (G5): `check_reading(parsed, reading, redaction_map) -> list[str]`.

A pure function: no model, no I/O, no change to the reading. It returns one plain English sentence per failure, each
naming the field and the offending value, and `[]` for a clean reading. The sentences are what the repair re-ask sends
back to the model (G6) and what is logged, so they hold placeholders only: a model-written value goes through
`redact_phones` before it is quoted, and a real number never reaches a sentence. The order is fixed: (b), (c), (d), (e),
(f), and within one check the order of the reading's fields, so the same reading always gives the same list.

(a) area is not a check here. Code builds the area from the E text (`build_area`), and one that reaches more than
    `reader_area_slack_nm` outside the Q-circle is swapped for the Q-circle there and logged as a parser miss. That is
    code's miss, not the model's: it never fails a check and never causes a repair.
(b) floor <= ceiling, when the two can be compared. A flight level counts as FL x 100 ft AMSL and SFC as 0 on any
    datum. AGL against AMSL cannot be compared without the site's height, so that pair is skipped and the skip is logged.
    A floor or ceiling of None (the E text states none, G4) is skipped, not failed. A height that is not a finite number
    (NaN, Infinity: JSON from the model can carry them) is a failure, because no comparison with it can be trusted.
(c) the model's text (activity, contact value, exemption quotes, designators) holds no phone number written out (a real
    number must come back as its <PHONE_n>), and every <PHONE_n> in it is a key of this NOTAM's redaction map.
(d) every activates_designators entry is a designator (`normalise_designator`) that `find_designators` finds in the E text.
(e) every exemption quote is in the redacted E text, `redact_phones(parsed.item_e)`: the text the model and Dan's sheet
    both see, compared ignoring case and whitespace. An empty quote is a failure, not a match.
(f) the enums and types are valid. Pydantic enforces them when a reading is built, so a model reply that breaks one never
    becomes a reading: `schema_failures` turns that error into the same kind of sentences. Inside `check_reading` the
    reading is validated again from its own dump, which catches one built around validation (`model_construct`).
"""

import logging
import math
import re
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from flysafelz.airspace import FEET_PER_FLIGHT_LEVEL
from flysafelz.contracts import Datum, Limit, LimitUnit, NotamReading, ParsedNotam
from flysafelz.designators import find_designators, normalise_designator
from flysafelz.redact import redact_phones

log = logging.getLogger(__name__)

_PLACEHOLDER = re.compile(r"<PHONE_\d+>")
_SCALARS = (str, int, float, bool)


def _shown(text: str) -> str:
    """A model-written value as it may be quoted: any real phone number in it is swapped for a placeholder."""
    return redact_phones(text)[0]


def _squash(text: str) -> str:
    """The comparison form of a quote: case-folded, every run of whitespace (a non-breaking space, a line break) one space."""
    return " ".join(text.split()).casefold()


def _show_limit(limit: Limit) -> str:
    if limit.datum is Datum.SFC:
        return "SFC"
    if limit.unit is LimitUnit.FL:
        return f"FL{limit.value:03.0f}"
    return f"{limit.value:g} {limit.unit} {limit.datum}"


def _raw_limit(limit: Limit) -> str:
    """A height as the model wrote it (value, unit, datum), for a value `_show_limit` would hide or reshape (SFC, FL)."""
    return f"{limit.value:g} {limit.unit} {limit.datum}"


def _level(limit: Limit) -> tuple[float, Datum | None]:
    """(feet, datum) for comparing two limits. SFC is 0 on any datum, so its datum is None; a flight level is AMSL."""
    if limit.datum is Datum.SFC:
        return 0.0, None
    if limit.unit is LimitUnit.FL:
        return limit.value * FEET_PER_FLIGHT_LEVEL, Datum.AMSL
    return limit.value, limit.datum


# --- (b) -----------------------------------------------------------------------------------------------------------


def _check_heights(reading: NotamReading) -> list[str]:
    floor, ceiling = reading.floor, reading.ceiling
    # json.loads takes the bare NaN and Infinity (and 1e999) and Pydantic takes a non-finite float, but a comparison with
    # one is always False: it would pass (b) and then break notam_action(). So a non-finite height is a failure here.
    not_finite = [
        f"your {name} ({_raw_limit(limit)}) is not a finite number; give the height the NOTAM text states, or null for one it does not state"
        for name, limit in (("floor", floor), ("ceiling", ceiling))
        if limit is not None and not math.isfinite(limit.value)
    ]
    if not_finite:
        return not_finite
    if floor is None or ceiling is None:
        return []  # the E text states no floor or no ceiling (G4): nothing to compare
    low, low_datum = _level(floor)
    high, high_datum = _level(ceiling)
    if low_datum is not None and high_datum is not None and low_datum is not high_datum:
        log.info("check (b) skipped for %s: the floor is %s and the ceiling %s, which cannot be compared", reading.id, _show_limit(floor), _show_limit(ceiling))
        return []
    if low > high:
        return [f"your floor ({_show_limit(floor)}) is above your ceiling ({_show_limit(ceiling)}); give the heights the NOTAM text states, or null for one it does not state"]
    return []


# --- (c) -----------------------------------------------------------------------------------------------------------


def _model_texts(reading: NotamReading) -> list[tuple[str, str]]:
    """(field, text) for every free-text field the model fills, in a fixed order."""
    texts = [("activity", reading.activity)]
    if reading.contact.value is not None:
        texts.append(("contact value", reading.contact.value))
    texts += [("exemption quote", exemption.quote) for exemption in reading.exemptions]
    texts += [("activates_designators entry", designator) for designator in reading.activates_designators]
    return texts


def _check_placeholders(reading: NotamReading, redaction_map: Mapping[str, str]) -> list[str]:
    allowed = f"the NOTAM text has {', '.join(redaction_map)}" if redaction_map else "the NOTAM text has no phone numbers"
    failures = []
    for field, text in _model_texts(reading):
        if _shown(text) != text:  # a number written out: the reading would carry it on into what is stored and printed
            failures.append(f"your {field} has a phone number written out; {allowed}, so write only those placeholders and never a number")
        for placeholder in dict.fromkeys(_PLACEHOLDER.findall(text)):
            if placeholder not in redaction_map:
                failures.append(f"your {field} mentions {placeholder}, which is not in the NOTAM text; {allowed}, so use only those")
    return failures


# --- (d) -----------------------------------------------------------------------------------------------------------


def _check_designators(parsed: ParsedNotam, reading: NotamReading) -> list[str]:
    in_text = find_designators(parsed.item_e)
    failures = []
    for entry in reading.activates_designators:
        normalised = normalise_designator(entry)
        if normalised is None:
            failures.append(f"your activates_designators entry '{_shown(entry)}' is not an airspace designator such as EGD496A; list only designators the NOTAM text names, or none")
        elif normalised not in in_text:
            failures.append(f"your activates_designators entry '{_shown(entry)}' ({normalised}) does not appear in the NOTAM text; list only designators the NOTAM text names, or none")
    return failures


# --- (e) -----------------------------------------------------------------------------------------------------------


def _check_quotes(parsed: ParsedNotam, reading: NotamReading) -> list[str]:
    text = _squash(redact_phones(parsed.item_e)[0])
    failures = []
    for exemption in reading.exemptions:
        quote = _squash(exemption.quote)
        if not quote:
            failures.append("your exemption quote is empty; quote the sentence word for word or leave exemptions empty")
        elif quote not in text:
            failures.append(f"your exemption quote '{_shown(exemption.quote.strip())}' does not appear in the NOTAM text; quote it word for word or leave exemptions empty")
    return failures


# --- (f) -----------------------------------------------------------------------------------------------------------


def _path(location: tuple[int | str, ...]) -> str:
    """Where in the reply a problem is. A key name is written by the model (an extra key can be a phone number), so it is redacted."""
    path = ""
    for part in location:
        path += f"[{part}]" if isinstance(part, int) else (f".{_shown(part)}" if path else _shown(part))
    return path or "answer"


def _value(raw: Any) -> str:
    """How a rejected value is quoted: a short scalar as it is (any real number swapped), anything bigger by its type."""
    if isinstance(raw, _SCALARS):
        return f"'{_shown(str(raw))}'"
    return "a " + type(raw).__name__ if raw is not None else "null"


def schema_failures(error: ValidationError) -> list[str]:
    """(f): one sentence per problem Pydantic found in a reply, in the order it found them. Enum errors list the allowed values."""
    failures = []
    for problem in error.errors():
        path, kind = _path(problem["loc"]), problem["type"]
        if kind == "enum":
            failures.append(f"your {path} is {_value(problem['input'])}, which is not allowed; use {problem.get('ctx', {}).get('expected', 'one of the values on the form')}")
        elif kind == "missing":
            failures.append(f"your answer has no {path}; give every field (null where the NOTAM text states none)")
        elif kind == "extra_forbidden":
            failures.append(f"your answer has a field {path} that is not on the form; leave it out")
        else:
            failures.append(f"your {path} is {_value(problem['input'])}, which is not valid: {problem['msg']}")
    return failures


def _check_enums(reading: NotamReading) -> list[str]:
    try:
        NotamReading.model_validate(reading.model_dump(mode="json", warnings=False))
    except ValidationError as error:
        return schema_failures(error)
    return []


def check_reading(parsed: ParsedNotam, reading: NotamReading, redaction_map: Mapping[str, str]) -> list[str]:
    """Every failed check (b) to (f) on a reading, as plain sentences; `[]` when it is clean. Never changes the reading."""
    return [
        *_check_heights(reading),
        *_check_placeholders(reading, redaction_map),
        *_check_designators(parsed, reading),
        *_check_quotes(parsed, reading),
        *_check_enums(reading),
    ]
