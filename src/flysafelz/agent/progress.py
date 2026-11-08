"""Progress lines for the pilot, written to LangGraph's custom stream while a turn runs.

    emit(line)                 write one line to the stream; nothing at all outside a graph run
    CHECKING_WEATHER           "Checking weather…"
    CHECKING_AIRSPACE          "Checking airspace…"
    reading_notams(i, n)       "Reading 2 NOTAMs (1/2)…"
    ASSESSING                  "Assessing…"
    CHECKING_ANSWER            "Checking the answer…"

The lines are fixed code text. They hold no verdict word and no number but the NOTAM counts, so they need no guard and carry nothing
the guard would have to check. The tools call `emit` with these; so does the guard for CHECKING_ANSWER. Nothing else writes progress.
"""

from langgraph.config import get_stream_writer

CHECKING_WEATHER = "Checking weather…"
CHECKING_AIRSPACE = "Checking airspace…"
ASSESSING = "Assessing…"
CHECKING_ANSWER = "Checking the answer…"


def reading_notams(i: int, n: int) -> str:
    """The line for NOTAM number `i` of `n`, counting from 1: "Reading 2 NOTAMs (1/2)…". Only whole numbers go into the line."""
    if isinstance(i, bool) or isinstance(n, bool) or not isinstance(i, int) or not isinstance(n, int):
        raise TypeError("the NOTAM numbers are whole numbers")
    if not 1 <= i <= n:
        raise ValueError(f"NOTAM {i} of {n} is out of range: count from 1 up to the total")
    noun = "NOTAM" if n == 1 else "NOTAMs"
    return f"Reading {n} {noun} ({i}/{n})…"


def emit(line: str) -> None:
    """Write the line, unchanged, to the custom stream of the graph run this is called in. Outside a graph run there is no stream: do nothing.

    "Outside a graph run" covers two cases: no runnable context at all, and a runnable context that is not a graph run (a `@tool`
    called with `.invoke()`, a `RunnableLambda`): there the config holds no LangGraph runtime.
    """
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return  # langgraph: "Called get_config outside of a runnable context"
    except KeyError:
        return  # a runnable context with no LangGraph runtime in its config (langgraph indexes config["configurable"]["__pregel_runtime"])
    writer(line)
