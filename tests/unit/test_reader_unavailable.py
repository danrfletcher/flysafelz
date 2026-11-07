"""G7 and E10: no usable answer is `unavailable`, never `failed`, and read_notam never raises.

Error kind x call position (the first call, or the repair after a first answer with failed checks). The model is a double
that raises or answers call by call. No openai or httpx class is imported into src/flysafelz/reader/: the double raises
plain Exception subclasses, and test_reader_wire.py shows the same with real openai errors from a mock transport.
"""

import ast
import json
import logging
from pathlib import Path

import pytest
import yaml
from factories import LZ, circle_at, notam, segment
from langchain_core.messages import AIMessage
from pydantic import ValidationError
from reader_helpers import FAKE_MODEL, Steps, message, reply_fields

from flysafelz.contracts import NotamAction
from flysafelz.notam_action import notam_action
from flysafelz.reader import read as read_module
from flysafelz.reader.contract import ReaderStatus
from flysafelz.reader.prompt import PROMPT_VERSION
from flysafelz.reader.read import read_notam
from flysafelz.reader.settings import SETTINGS, SETTINGS_PATH, load_settings

E_TEXT = "ALL ACFT PROHIBITED. CALL 07700 900123 FIRST."
PARSED = notam(circle=circle_at(0), item_e=E_TEXT, item_f=None, item_g=None, lower_fl=0, upper_fl=8)
FIRST_FAILS = reply_fields(applies_to="PARAMOTORS")  # JSON that parses but breaks the form: a failed check, so a repair follows
GOOD = reply_fields()


class HttpStatusError(Exception):
    """What an HTTP client raises for a 4xx or 5xx: any class with a status will do, nothing here imports one."""

    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


ERRORS = [
    pytest.param(lambda: TimeoutError("endpoint timed out"), id="timeout"),
    pytest.param(lambda: ConnectionError("connection refused"), id="connection error"),
    pytest.param(lambda: ConnectionResetError("reset by peer"), id="connection reset"),
    pytest.param(lambda: HttpStatusError(400), id="HTTP 400"),
    pytest.param(lambda: HttpStatusError(401), id="HTTP 401"),
    pytest.param(lambda: HttpStatusError(429), id="HTTP 429"),
    pytest.param(lambda: HttpStatusError(500), id="HTTP 500"),
    pytest.param(lambda: HttpStatusError(503), id="HTTP 503"),
    pytest.param(lambda: RuntimeError("anything else the client throws"), id="other error"),
    pytest.param(lambda: "", id="200 with an empty body"),
    pytest.param(lambda: "   \n", id="200 with a blank body"),
    pytest.param(lambda: '{"category": "prohib', id="200 with truncated JSON"),
    pytest.param(lambda: json.dumps(GOOD)[:-30], id="200 with JSON cut off at the end"),
    pytest.param(lambda: "Sorry, I cannot read this NOTAM.", id="200 with prose"),
    pytest.param(lambda: "```json\n" + json.dumps(GOOD) + "\n```", id="200 with fenced JSON"),
    pytest.param(lambda: AIMessage(content=[]), id="200 with no content parts"),
]
POSITIONS = ["first call", "repair"]


@pytest.mark.parametrize("position", POSITIONS)
@pytest.mark.parametrize("make", ERRORS)
def test_no_usable_answer_is_unavailable_never_failed_and_never_raised(make, position):
    double = Steps(make()) if position == "first call" else Steps(FIRST_FAILS, make())
    result = read_notam(PARSED, double)
    assert result.status is ReaderStatus.UNAVAILABLE
    assert result.reading is None
    assert len(double.calls) == (1 if position == "first call" else 2)  # a missing answer is not asked for again
    if position == "first call":
        assert result.failed_checks == []
    else:  # E10: the first attempt's failed checks are kept
        assert len(result.failed_checks) == 1 and result.failed_checks[0].startswith("your applies_to is 'PARAMOTORS'")


@pytest.mark.parametrize("position", POSITIONS)
@pytest.mark.parametrize("make", ERRORS)
def test_the_unavailable_result_carries_the_model_and_prompt_version_and_the_notam_is_unread(make, position):
    double = Steps(make()) if position == "first call" else Steps(FIRST_FAILS, make())
    result = read_notam(PARSED, double)
    assert (result.model, result.prompt_version, result.notam_id) == (FAKE_MODEL, PROMPT_VERSION, PARSED.id)
    assert result.latency_s >= 0
    action = notam_action(PARSED, result.reading, LZ, segment())
    assert action.action is NotamAction.UNREAD_OVER_LZ  # the NOTAM is unread, not dropped


def test_the_reader_status_literals_are_failed_and_unavailable():
    """PR-6 passes these two strings to assess(... reader_status=...); PR-6 asserts the assumption line they produce."""
    assert ReaderStatus.FAILED == "failed" and ReaderStatus.FAILED.value == "failed"
    assert ReaderStatus.UNAVAILABLE == "unavailable" and ReaderStatus.UNAVAILABLE.value == "unavailable"
    unread = {read_notam(PARSED, Steps(TimeoutError())).status.value, read_notam(PARSED, Steps(FIRST_FAILS, FIRST_FAILS)).status.value}
    assert unread == {"unavailable", "failed"}


def test_json_that_parses_but_has_the_wrong_shape_is_failed_not_unavailable():
    for wrong in ("[]", "null", '"x"', "{}", json.dumps({**GOOD, "category": "restricted"})):
        assert read_notam(PARSED, Steps(wrong, wrong)).status is ReaderStatus.FAILED, wrong


def test_a_first_call_with_no_answer_is_not_repaired():
    double = Steps(TimeoutError("first"), TimeoutError("second"))
    assert read_notam(PARSED, double).status is ReaderStatus.UNAVAILABLE
    assert len(double.calls) == 1 and len(double.steps) == 1


def test_the_timeout_comes_from_settings_yaml_and_is_sent_with_every_call():
    from_file = yaml.safe_load(Path(SETTINGS_PATH).read_text(encoding="utf-8"))["timeout_s"]
    assert SETTINGS.timeout_s == from_file
    double = Steps(FIRST_FAILS, GOOD)
    read_notam(PARSED, double)
    assert [call["kwargs"]["timeout"] for call in double.calls] == [from_file, from_file]


def test_settings_load_from_any_file_and_refuse_a_missing_or_unknown_key(tmp_path):
    path = tmp_path / "settings.yaml"
    rest = "concurrency: 3\nruns: 3\n"  # settings.yaml also holds the pool size and the eval's run count (PR-5); each is required
    path.write_text("timeout_s: 12.5\n" + rest, encoding="utf-8")
    assert load_settings(path).timeout_s == 12.5
    for text in ("{}\n", "timeout_s: 12\nretries: 3\n" + rest, "timeout_s: 0\n" + rest, "timeout_s: soon\n" + rest, "timeout_s: 12\n"):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValidationError):
            load_settings(path)


def test_the_time_a_failed_call_took_is_counted(monkeypatch):
    from types import SimpleNamespace

    ticks = iter([1.0, 4.0, 10.0, 10.5])
    monkeypatch.setattr(read_module, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    result = read_notam(PARSED, Steps(FIRST_FAILS, TimeoutError("slow")))
    assert result.latency_s == pytest.approx(3.0 + 0.5)


def test_tokens_are_those_of_the_answers_that_came_back():
    usage = {"input_tokens": 1200, "output_tokens": 150, "total_tokens": 1350}
    first_only = read_notam(PARSED, Steps(message(FIRST_FAILS, usage), TimeoutError("down")))
    assert first_only.status is ReaderStatus.UNAVAILABLE
    assert (first_only.tokens.input_tokens, first_only.tokens.total_tokens) == (1200, 1350)
    assert read_notam(PARSED, Steps(TimeoutError("down"))).tokens is None
    assert read_notam(PARSED, Steps(message("", usage))).tokens.total_tokens == 1350  # an empty reply can still report usage


def test_a_keyboard_interrupt_is_not_swallowed():
    class Stop(BaseException):
        pass

    with pytest.raises(Stop):
        read_notam(PARSED, Steps(Stop()))


def test_the_logged_error_never_holds_a_real_number(caplog):
    """An error text that echoes a number (as some endpoints echo input) is redacted before it is logged."""
    caplog.set_level(logging.DEBUG)
    read_notam(PARSED, Steps(RuntimeError("bad request: call 07700 900123 first")))
    text = "".join(caplog.text.split())
    assert "RuntimeError" in caplog.text and "<PHONE_1>" in caplog.text
    assert "900123" not in text


def test_the_reader_imports_no_openai_or_httpx_class_to_catch_errors():
    """The double above raises plain classes; openai's real ones are caught the same way (test_reader_wire.py runs them)."""
    imported = set()
    for path in Path(read_module.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
    assert imported and not imported & {"openai", "httpx"}
