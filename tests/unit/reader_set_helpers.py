"""Builders for the reader-set tests: a thread-safe fake chat model scripted per NOTAM id, and generated reader-set NOTAMs.

ScriptedChat is the counting fake of cache_helpers.PerNotam made safe for a pool: one lock guards its scripts, its call
log and its in-flight count, and it records the peak number of `invoke` calls running at the same moment. The first
`rendezvous` calls wait for each other before answering, so a test can show the pool really runs that many at once and
not merely at most that many.
"""

import threading
import time
from typing import Any

from langchain_core.messages import AIMessage
from reader_helpers import FAKE_MODEL, message, reply_fields

from flysafelz.contracts import ParsedNotam
from flysafelz.reader.prompt import render_prompt

NO_CONTACT = {"kind": "none", "value": None}
RENDEZVOUS_TIMEOUT_S = 10.0


def good_reply() -> dict[str, Any]:
    """A reply that passes every check for the generated NOTAMs: no phone number in their text, so no contact."""
    return reply_fields(contact=NO_CONTACT)


def bad_reply() -> dict[str, Any]:
    """A well-formed reply that fails check (e): its exemption quote is not in the NOTAM text."""
    return {**good_reply(), "exemptions": [{"quote": "THIS SENTENCE IS NOT IN THE NOTAM", "covers": "OTHER"}]}


class ScriptedChat:
    """A chat model double that answers each NOTAM from its own script, from any number of threads.

    A step is an exception (raised from `invoke`), an AIMessage, a dict (sent as JSON text) or a string. A NOTAM with no
    script, or whose script has run out, gets `good_reply()`. `calls` is the id asked on every call, in order.
    """

    model_name = FAKE_MODEL

    def __init__(self, notams: list[ParsedNotam], *, rendezvous: int = 0, pause_s: float = 0.0, **scripts: Any):
        """`scripts` is keyed by id with the slash written as an underscore (H7171_26). `pause_s` holds every call open that long."""
        self.users = {render_prompt(parsed).user: parsed.id for parsed in notams}
        assert len(self.users) == len({parsed.id for parsed in notams}), "two NOTAMs are sent the same text, so the fake cannot tell them apart"
        self.scripts = {key.replace("_", "/"): list(steps) for key, steps in scripts.items()}
        self.calls: list[str] = []
        self.in_flight = 0
        self.peak = 0
        self._lock = threading.Lock()
        self._rendezvous = rendezvous
        self._barrier = threading.Barrier(rendezvous) if rendezvous else None
        self._pause_s = pause_s

    def script(self, notam_id: str, *steps: Any) -> None:
        """Replace one NOTAM's remaining steps (what the next run will do)."""
        with self._lock:
            self.scripts[notam_id] = list(steps)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "ScriptedChat":
        """Unused by the reader, which sends a json_schema response_format to `invoke`; here so the double has the usual shape."""
        return self

    def invoke(self, messages: list[Any], **kwargs: Any) -> AIMessage:
        user = messages[1].content
        notam_id = next((found for text, found in self.users.items() if user.startswith(text)), "UNKNOWN NOTAM TEXT")  # shows in `calls`
        with self._lock:
            arrival = len(self.calls)
            self.calls.append(notam_id)
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            steps = self.scripts.get(notam_id, [])
            step = steps.pop(0) if steps else good_reply()
        try:
            if self._barrier is not None and arrival < self._rendezvous:
                self._barrier.wait(timeout=RENDEZVOUS_TIMEOUT_S)  # raises if fewer than `rendezvous` calls are in flight together
            if self._pause_s:
                time.sleep(self._pause_s)
        finally:
            with self._lock:
                self.in_flight -= 1
        if isinstance(step, BaseException):
            raise step
        return step if isinstance(step, AIMessage) else message(step)
