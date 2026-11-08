"""The tool artifacts live on the tool message, so they are stored in the thread's checkpoint and read back on the next turn.

LANGGRAPH_STRICT_MSGPACK=true (the default in a later LangGraph) refuses to load a class the checkpoint was not told about. An enum inside an
artifact was one: the artifact must hold only plain values.
"""

import subprocess
import sys
from pathlib import Path
from typing import get_args

from tool_helpers import ALL_THREE

from flysafelz.agent.tools.artifacts import SourceKey, SourceStateKey
from flysafelz.contracts import SourceName, SourceState

ROOT = Path(__file__).resolve().parents[3]


def test_a_thread_with_the_tool_artifacts_loads_from_its_checkpoint_under_strict_msgpack():
    environment = {"LANGGRAPH_STRICT_MSGPACK": "true", "PATH": "/usr/bin:/bin"}
    done = subprocess.run([sys.executable, str(ROOT / "tests" / "strict_msgpack_probe.py")], cwd=ROOT, env=environment, capture_output=True, text=True, timeout=300, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split("\n")[: len(ALL_THREE)] == [f"{name} ok" for name in ALL_THREE]
    assert "Blocked deserialization" not in done.stderr and "unregistered type" not in done.stderr


def test_the_artifact_source_and_state_values_are_the_enums_values():
    assert set(get_args(SourceKey)) == {member.value for member in SourceName}
    assert set(get_args(SourceStateKey)) == {member.value for member in SourceState}
