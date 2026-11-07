"""Builders for the reading-cache tests: a fake chat model scripted per NOTAM id, and the Blue Mill answers.

PerNotam is the counting fake of PR-2/PR-3 (reader_helpers.Steps) with one script per NOTAM id instead of one for the
whole run, because a set is read in some order and a test wants "H7171 fails, J3857 answers" however that order falls.
The answers come from the case's hand-written readings.yaml: its semantic fields, with every real number swapped for
the placeholder the redaction gives (the file holds real numbers; the model is never sent one).
"""

from typing import Any

import yaml
from langchain_core.messages import AIMessage
from reader_helpers import BLUE_MILL_PIB, FAKE_MODEL, blue_mill_notams, message

from flysafelz.contracts import NotamReading, ParsedNotam
from flysafelz.reader.contract import ReaderOutput
from flysafelz.reader.prompt import render_prompt
from flysafelz.redact import redact_phones

READINGS_YAML = BLUE_MILL_PIB.with_name("readings.yaml")
BLUE_MILL_READER_SET = ("H7171/26", "J3857/26")


def blue_mill_set() -> list[ParsedNotam]:
    """The two NOTAMs the plan reads for Blue Mill, in the order of the plan."""
    notams = blue_mill_notams()
    return [notams[notam_id] for notam_id in BLUE_MILL_READER_SET]


def fixture_reply(notam_id: str) -> dict[str, Any]:
    """What a good model answers for this NOTAM: readings.yaml's semantic fields with the number as its placeholder."""
    entries = yaml.safe_load(READINGS_YAML.read_text(encoding="utf-8"))
    reading = NotamReading(**next(entry for entry in entries if entry["id"] == notam_id))
    fields = reading.model_dump(mode="json", include=set(ReaderOutput.model_fields))
    if fields["contact"]["value"] is not None:
        fields["contact"]["value"] = redact_phones(fields["contact"]["value"])[0]
    return fields


def bad_reply(notam_id: str) -> dict[str, Any]:
    """A well-formed answer that fails check (e): its exemption quote is not in the NOTAM text."""
    return {**fixture_reply(notam_id), "exemptions": [{"quote": "THIS SENTENCE IS NOT IN THE NOTAM", "covers": "OTHER"}]}


class PerNotam:
    """A chat model double that answers each NOTAM from its own script and counts every call.

    A step is an exception (raised from `invoke`), an AIMessage, a dict (sent as JSON text) or a string. The NOTAM is found
    by the user text it was sent, which `render_prompt` gives; a repair re-ask starts with the same text. `calls` is the
    ids asked, in order, so `len(calls)` is the number of model calls and a test can read who was asked.
    """

    model_name = FAKE_MODEL

    def __init__(self, notams: list[ParsedNotam], **scripts: list[Any]):
        """`scripts` is keyed by id with the slash written as an underscore (H7171_26)."""
        self.users = {render_prompt(parsed).user: parsed.id for parsed in notams}
        self.scripts = {key.replace("_", "/"): list(steps) for key, steps in scripts.items()}
        self.calls: list[str] = []

    def script(self, notam_id: str, *steps: Any) -> None:
        """Replace one NOTAM's remaining steps (what run 2 will do)."""
        self.scripts[notam_id] = list(steps)

    def invoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:
        user = messages[1].content
        notam_id = next(found for text, found in self.users.items() if user.startswith(text))
        self.calls.append(notam_id)
        step = self.scripts[notam_id].pop(0)
        if isinstance(step, BaseException):
            raise step
        return step if isinstance(step, AIMessage) else message(step)
