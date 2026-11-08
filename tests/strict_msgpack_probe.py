"""Run the three data tools in the Blue Mill case and read the thread back from its checkpoint. Run as a script by test_checkpoint_strict_msgpack.py.

    LANGGRAPH_STRICT_MSGPACK=true python tests/strict_msgpack_probe.py

LangGraph reads that variable when it is imported, so a test cannot turn it on for its own process: this runs in a fresh one. It prints one line per
tool, `<name> ok` when the artifact read back from the checkpoint is the artifact the tool made, else `<name> differs`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tool_helpers import ALL_THREE, BLUE_MILL_NOW, BLUE_MILL_WINDOW, calls, run_tools

from flysafelz.agent.sources import SourceCache
from flysafelz.agent.state import RunContext


def main() -> None:
    run = run_tools(SourceCache(), [calls(*ALL_THREE, **BLUE_MILL_WINDOW)], ctx=RunContext(now=BLUE_MILL_NOW, case="blue_mill_2026-10-31", reader="fixture"))
    loaded = {m.name: m for m in run.state["messages"] if m.type == "tool"}
    for name in ALL_THREE:
        made = run.artifact(name)
        again = type(made).model_validate(loaded[name].artifact)
        print(name, "ok" if again == made else "differs")


if __name__ == "__main__":
    main()
