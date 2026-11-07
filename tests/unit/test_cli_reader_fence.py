"""PR-6 fence checks (ticket 3 F2, F5): the verdict code stays free of the model client, and the CLI has no switch for a fake."""

import argparse
import subprocess
import sys
from pathlib import Path

from flysafelz import cli

ROOT = Path(__file__).resolve().parents[2]


def run_python(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120, check=False)


def test_assess_and_notam_action_import_with_the_model_clients_blocked():
    """F2: sys.modules[...] = None makes any import of the two packages raise ImportError, so this fails if either module reaches one."""
    code = (
        "import sys\n"
        "sys.modules['langchain_openai'] = None\n"
        "sys.modules['langchain_core'] = None\n"
        "import flysafelz.assess, flysafelz.notam_action, flysafelz.redact\n"
        "assert flysafelz.redact.redact_phones('x 07700 900123')[0] == 'x <PHONE_1>'\n"
    )
    done = run_python(code)
    assert done.returncode == 0, done.stderr


def test_the_block_is_real_the_cli_cannot_be_imported_without_the_model_client():
    """The same blocking stops cli.py, which imports the reader on purpose: so the check above can fail."""
    done = run_python("import sys\nsys.modules['langchain_openai'] = None\nsys.modules['langchain_core'] = None\nimport flysafelz.cli\n")
    assert done.returncode != 0 and "halted; None in sys.modules" in done.stderr


def test_the_cli_has_no_flag_or_switch_for_a_fake_model():
    _parser, assess_parser, read_parser = cli.build_parser()
    flags = [option for p in (assess_parser, read_parser) for action in p._actions for option in action.option_strings]
    assert flags and not [f for f in flags if any(word in f.lower() for word in ("fake", "mock", "stub", "model", "llm"))]
    for p in (assess_parser, read_parser):
        assert not [a for a in p._actions if isinstance(a, argparse._StoreAction) and a.dest in {"model", "llm", "fake"}]
