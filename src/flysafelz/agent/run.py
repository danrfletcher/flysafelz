"""`make run`: ask the agent one question in the terminal.

    main(argv=None, *, environ=None, model=None, stdout=None, stderr=None) -> int

    LAT, LON, Q    required: the landing zone's coordinates and the question
    TZ             optional: the pilot's IANA time zone (default Europe/London)
    CASE           optional: a committed case (fixtures/<case>/), so the run reads its frozen sources instead of the live ones
    NOW            optional: the run's clock, an ISO 8601 time with an offset (for a case)

The values reach the program through the environment, never parsed by the shell (as `make fixtures` does for CASE), so a question with quotes or
a `$` in it arrives as typed. The progress lines ("Checking weather…") go to stderr as they happen; the answer goes to stdout once, whole, from the
`answer` the run ends with (never from message chunks). Nothing is streamed word by word.

Exit codes: 0 the answer was printed; 2 a missing or bad argument or a request the input check refuses (one line on stderr); 1 anything else
that stopped the run (missing AGENT_* settings, no answer). A request outside the UK is answered by code with no model built or called.

`model` is for tests: a chat model to use instead of the configured one. `environ`, `stdout` and `stderr` are the process's own unless a test passes its own.
"""

import math
import os
import sys
import uuid
from collections.abc import Mapping
from typing import Any, TextIO

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from flysafelz.agent.config import AgentConfigError
from flysafelz.agent.graph import build_graph
from flysafelz.agent.input_check import RequestInvalid
from flysafelz.agent.state import RunContext

FAILED_EXIT = 1
USAGE_EXIT = FAILED_EXIT + 1  # 2, as a command-line usage error is by convention
REQUIRED = ("LAT", "LON", "Q")
USAGE = 'usage: make run LAT=<latitude> LON=<longitude> Q="<question>" [TZ=<time zone>] [CASE=<case>] [NOW=<ISO time with offset>]'


class UsageError(ValueError):
    """The command line is wrong: the message is the one line printed on stderr."""


def _number(name: str, text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise UsageError(f"{name} must be a number, got {text!r}. {USAGE}") from None
    if not math.isfinite(value):
        raise UsageError(f"{name} must be a finite number, got {text!r}. {USAGE}")
    return value


def read_arguments(environ: Mapping[str, str]) -> tuple[dict[str, Any], RunContext]:
    """The graph's input and the run context from the environment. Raises UsageError (one line) for a missing or unreadable value."""
    given = {name: environ.get(name, "").strip() for name in (*REQUIRED, "TZ", "CASE", "NOW")}
    missing = [name for name in REQUIRED if not given[name]]
    if missing:
        raise UsageError(f"missing {', '.join(missing)}. {USAGE}")
    request: dict[str, Any] = {"messages": [HumanMessage(given["Q"])], "lat": _number("LAT", given["LAT"]), "lon": _number("LON", given["LON"])}
    if given["TZ"]:
        request["tz"] = given["TZ"]
    context = RunContext(now=given["NOW"] or None, case=given["CASE"] or None)  # type: ignore[arg-type]  # a string `now` is read as ISO 8601
    return request, context


def ask(request: dict[str, Any], context: RunContext, stderr: TextIO, model: BaseChatModel | None) -> str | None:
    """Run one turn. Progress lines go to `stderr` as they arrive; the answer is what the run ends with, or None if it ended with none."""
    graph = build_graph(model)
    config = {"configurable": {"thread_id": f"run-{uuid.uuid4()}"}}
    answer = None
    for mode, payload in graph.stream(request, config, context=context, stream_mode=["custom", "values"]):
        if mode == "custom":
            print(payload, file=stderr, flush=True)
        else:
            answer = payload.get("answer")
    return answer


def main(
    argv: list[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    model: BaseChatModel | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Ask the question in LAT, LON and Q and print the answer. See the module doc for the exit codes."""
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    environ = os.environ if environ is None else environ
    if argv:
        print(f"make run takes no arguments, only the environment. {USAGE}", file=err)
        return USAGE_EXIT
    try:
        request, context = read_arguments(environ)
        answer = ask(request, context, err, model)
    except (UsageError, RequestInvalid) as error:
        print(str(error), file=err)
        return USAGE_EXIT
    except AgentConfigError as error:
        print(str(error), file=err)
        return FAILED_EXIT
    if not answer:
        print("The run ended without an answer.", file=err)
        return FAILED_EXIT
    print(answer, file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
