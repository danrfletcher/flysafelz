"""Run the three data tools in the Blue Mill case and save what each says, for a person to read.

    uv run python tests/save_data_tool_texts.py

Case mode (no network, the case's readings, the frozen `now` 2026-10-31T07:30Z, tz Europe/London, window 10:00-16:00). It drives the tools through the same
graph the agent uses, with a scripted model that only asks for them one after the other, and writes screenshots/PR-5.F1-weather.txt, -airspace.txt and
-notams.txt: the tool's text, then (after a line of dashes) the progress lines the whole run wrote. It lives in tests/ because it uses the test helpers for the graph (and a script outside src/flysafelz/agent/ may not import the agent).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from tool_helpers import BLUE_MILL_NOW, BLUE_MILL_WINDOW, calls, run_tools

from flysafelz.agent.sources import SourceCache
from flysafelz.agent.state import RunContext

TOOLS = {"get_weather": "weather", "get_airspace": "airspace", "get_notams": "notams"}


def main() -> None:
    ctx = RunContext(now=BLUE_MILL_NOW, case="blue_mill_2026-10-31", reader="fixture")
    run = run_tools(SourceCache(), [calls(name, **BLUE_MILL_WINDOW) for name in TOOLS], ctx=ctx)
    folder = ROOT / "screenshots"
    folder.mkdir(exist_ok=True)
    for name, short in TOOLS.items():
        target = folder / f"PR-5.F1-{short}.txt"
        target.write_text(f"{run.text(name)}\n\n----- progress lines of the whole run, in order -----\n" + "\n".join(run.progress) + "\n", encoding="utf-8")
        print(target)


if __name__ == "__main__":
    main()
