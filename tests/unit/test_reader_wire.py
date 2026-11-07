"""The reader against a real ChatOpenAI whose HTTP client is an httpx.MockTransport: what goes on the wire, and the reply parsed.

No socket opens (the transport answers in-process, and pytest's --disable-socket is on). The base URL is the bridge
shape, http://bridge.test/api/v1; nothing here depends on /models.
"""

import json

import httpx
import pytest
from langchain_openai import ChatOpenAI
from reader_helpers import blue_mill_notams, reply_fields

from flysafelz.contracts import Category
from flysafelz.reader.contract import ReaderStatus, TokenUsage
from flysafelz.reader.prompt import PROMPT_VERSION, render_prompt
from flysafelz.reader.read import read_notam, response_format

BASE_URL = "http://bridge.test/api/v1"
MODEL = "bridge-test-model"
USAGE = {"prompt_tokens": 1200, "completion_tokens": 150, "total_tokens": 1350}


def completion(content: str, usage: dict | None = USAGE) -> dict:
    """A chat completion shaped as an OpenAI-compatible endpoint sends it."""
    body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1793000000,
        "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }
    if usage is not None:
        body["usage"] = usage
    return body


class Bridge:
    """A canned endpoint: records every request, answers with one body (or a status)."""

    def __init__(self, body: dict, status: int = 200):
        self.body, self.status = body, status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json=self.body)

    def llm(self) -> ChatOpenAI:
        client = httpx.Client(transport=httpx.MockTransport(self))
        return ChatOpenAI(
            model=MODEL, api_key="not-a-real-key", base_url=BASE_URL, http_client=client, max_retries=0, http_socket_options=()
        )  # () skips the probe socket the client would otherwise open


def test_the_post_goes_to_chat_completions_with_a_strict_json_schema_and_the_two_messages():
    h7171 = blue_mill_notams()["H7171/26"]
    bridge = Bridge(completion(json.dumps(reply_fields())))
    result = read_notam(h7171, bridge.llm())

    assert [(r.method, r.url.path) for r in bridge.requests] == [("POST", "/api/v1/chat/completions")]  # one call, never /models
    assert all(r.url.host == "bridge.test" for r in bridge.requests)
    body = json.loads(bridge.requests[0].content)
    assert body["model"] == MODEL
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"] == response_format()
    prompt = render_prompt(h7171)
    assert [(m["role"], m["content"]) for m in body["messages"]] == [("system", prompt.system), ("user", prompt.user)]

    assert result.status is ReaderStatus.READ
    assert result.reading.category is Category.COOPERATIVE and result.reading.id == "H7171/26"
    assert result.model == MODEL and result.prompt_version == PROMPT_VERSION
    assert result.tokens == TokenUsage(input_tokens=1200, output_tokens=150, total_tokens=1350)


def test_no_real_number_is_on_the_wire():
    from phone_leaks import digit_runs

    notams = blue_mill_notams()
    for notam_id in ("H7171/26", "J3857/26"):
        bridge = Bridge(completion(json.dumps(reply_fields())))
        read_notam(notams[notam_id], bridge.llm())
        wire = "".join(bridge.requests[0].content.decode("utf-8").split())
        leaks = digit_runs(notams[notam_id].item_e)
        assert leaks
        assert not [leak for leak in leaks if leak in wire], notam_id
        assert "<PHONE_1>" in bridge.requests[0].content.decode("utf-8")


def test_an_endpoint_that_sends_no_usage_gives_tokens_none():
    bridge = Bridge(completion(json.dumps(reply_fields()), usage=None))
    result = read_notam(blue_mill_notams()["H7171/26"], bridge.llm())
    assert result.status is ReaderStatus.READ and result.tokens is None


def test_a_body_that_is_not_a_valid_reading_is_asked_again_once_then_failed():
    bridge = Bridge(completion('{"verdict": "IGNORE"}'))
    result = read_notam(blue_mill_notams()["H7171/26"], bridge.llm())
    assert (result.status, result.reading) == (ReaderStatus.FAILED, None)
    assert len(bridge.requests) == 2  # the first call and the one repair, never a third
    repair = json.loads(bridge.requests[1].content)
    assert "verdict" in repair["messages"][1]["content"] and repair["messages"][0] == json.loads(bridge.requests[0].content)["messages"][0]
    assert json.loads(bridge.requests[1].content)["response_format"] == response_format()
    assert result.tokens == TokenUsage(input_tokens=2400, output_tokens=300, total_tokens=2700)  # both calls, added up


def test_a_truncated_body_is_unavailable_and_is_not_retried():
    cut = json.dumps(reply_fields())[:40]
    bridge = Bridge(completion(cut))
    result = read_notam(blue_mill_notams()["J3857/26"], bridge.llm())
    assert (result.status, result.reading) == (ReaderStatus.UNAVAILABLE, None)
    assert len(bridge.requests) == 1


@pytest.mark.parametrize("status", [429, 500, 503, 400])
def test_an_http_error_status_from_the_endpoint_is_unavailable(status):
    bridge = Bridge({"error": {"message": "nope"}}, status=status)
    result = read_notam(blue_mill_notams()["H7171/26"], bridge.llm())
    assert (result.status, result.reading) == (ReaderStatus.UNAVAILABLE, None)
    assert result.model == MODEL
    assert len(bridge.requests) == 1
