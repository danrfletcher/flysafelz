"""G9 fence checks: F3 (no real phone number in a row or a log), F2 (the cache word is allowed in reader/ only; stdlib imports; the
CLI and assess do not import it), F5 (these tests are offline and clockless), F6 (no literal), and the ignore rule for the file.

The leak list is phone_leaks.leak_list: every 10-11 digit run in a raw E text of the committed PIBs, found without the redactor.
"""

import ast
import logging
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from fence_scan import banned_words, hard_coded_rule_values, unreviewed_literals
from phone_leaks import PIBS, leak_list
from reader_helpers import Steps, reply_fields

from flysafelz.pib import read_notams
from flysafelz.reader.cached_read import read_set
from flysafelz.reader.contract import ReaderStatus
from flysafelz.reader.prompt import render_prompt
from flysafelz.rules import DEFAULT_PATH

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "flysafelz"
READER = SRC / "reader"
CACHE_PY = READER / "cache.py"
CACHED_READ_PY = READER / "cached_read.py"
HEX_HASH = re.compile(rb"[0-9a-f]{64}")


# --- F3 ----------------------------------------------------------------------------------------------------------------


def corpus_with_numbers():
    """Every NOTAM of the committed PIBs whose raw E text holds a 10-11 digit run, once each."""
    leaking_ids = {notam_id for ids in leak_list().values() for notam_id in ids}
    found = {}
    for path in sorted(PIBS.glob("*.xml")):
        for notam in read_notams(path.read_bytes())[1]:
            if notam.id in leaking_ids:
                found.setdefault(notam.id, notam)
    return list(found.values())


def reply_for(parsed) -> dict:
    """A clean answer: a phone contact (as its placeholder) when the text has a number, none when it has not."""
    contact = {"kind": "phone", "value": "<PHONE_1>"} if render_prompt(parsed).redaction else {"kind": "none", "value": None}
    return reply_fields(activity="fake activity", category="informational", floor=None, ceiling=None, contact=contact)


def leaks_in(data: bytes, numbers) -> list[str]:
    """The numbers found in the raw bytes. The 64-hex-digit hashes are blanked first: a hash can hold ten digits in a row by chance."""
    cleaned = HEX_HASH.sub(b"", data)
    return sorted(number for number in numbers if number.encode() in cleaned)


def test_the_search_finds_a_planted_number_and_ignores_a_hash_that_happens_to_hold_digits():
    assert leaks_in(b"x" + b"0123456789" + b"y", ["0123456789"]) == ["0123456789"]
    assert leaks_in(b"a" * 20 + b"0123456789" + b"a" * 34, ["0123456789"]) == []  # inside a 64-char hash
    assert leaks_in(b"nothing here", ["0123456789"]) == []


def test_f3_no_real_phone_number_in_any_cache_row_or_log_record(tmp_path, now, caplog):
    numbers = set(leak_list())
    notams = corpus_with_numbers()
    database = tmp_path / "readings.sqlite"
    assert numbers and len(notams) > 100  # the corpus is real, so the search below is not vacuous

    class Many:
        """One fake model for the whole corpus: it answers each NOTAM with that NOTAM's clean reply."""

        model_name = "fake-reader-model"

        def __init__(self):
            self.replies = {render_prompt(parsed).user: reply_for(parsed) for parsed in notams}
            self.sent = 0

        def invoke(self, messages, **kwargs):
            self.sent += 1
            return Steps(self.replies[messages[1].content]).invoke(messages, **kwargs)

    llm = Many()
    with caplog.at_level(logging.DEBUG):
        first = read_set(notams, llm, database, now)
        second = read_set(notams, llm, database, now)

    assert {read.result.status for read in first.reads.values()} == {ReaderStatus.READ}
    assert {read.result.status for read in second.reads.values()} == {ReaderStatus.STORED} and second.calls == 0
    assert llm.sent == len(notams)
    data = database.read_bytes()
    assert b"<PHONE_1>" in data  # the placeholder is what a row holds
    assert leaks_in(data, numbers) == []
    log_text = "\n".join(f"{record.getMessage()} {record.args!r}" for record in caplog.records)
    assert [number for number in numbers if number in re.sub(r"\s+", "", log_text)] == []


# --- F2 ----------------------------------------------------------------------------------------------------------------


def test_the_word_cache_is_allowed_in_the_reader_package_and_nowhere_else(tmp_path):
    """The exemption is used (cache.py has the word), and a planted `cache` in a copy of cli.py or pib.py is still caught."""
    assert banned_words(CACHE_PY, READER) == []
    assert "cache" in banned_words(CACHE_PY, tmp_path)  # not exempt anywhere else, so the line above is the exemption at work
    for name in ("cli.py", "pib.py"):
        assert banned_words(SRC / name, READER) == [], name
        copy = tmp_path / name
        copy.write_text((SRC / name).read_text(encoding="utf-8") + "\ncache = {}\n", encoding="utf-8")
        assert banned_words(copy, READER) == ["cache"], name


def test_the_other_bans_still_hold_inside_the_reader(tmp_path):
    (tmp_path / "reader").mkdir()
    planted = tmp_path / "reader" / "read.py"
    planted.write_text("from functools import lru_cache\nclass ProviderRegistry: ...\nx = functools.cache\n", encoding="utf-8")
    assert {"lru_cache", "functools.cache"} <= set(banned_words(planted, tmp_path / "reader"))


def imported(path: Path) -> set[str]:
    """Every module a file imports; for `from a import b` both `a` and `a.b` (b may be a module)."""
    modules = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            modules |= {f"{node.module}.{alias.name}" for alias in node.names}
    return modules


ALLOWED_PROJECT_MODULES = {
    "flysafelz",
    "flysafelz.reader",
    "flysafelz.contracts",
    "flysafelz.notam_area",
    "flysafelz.reader.prompt",
    "flysafelz.reader.checks",
    "flysafelz.reader.contract",
}


def test_cache_py_imports_only_stdlib_pydantic_and_the_projects_own_reader_and_contract_modules():
    for module in imported(CACHE_PY):
        top = module.split(".")[0]
        names_a_member = top == "flysafelz" and module.rpartition(".")[0] in ALLOWED_PROJECT_MODULES  # `from flysafelz.contracts import NotamReading`
        assert top in sys.stdlib_module_names or top == "pydantic" or module in ALLOWED_PROJECT_MODULES or names_a_member, module
    assert not [m for m in imported(CACHE_PY) if m.split(".")[0] in ("langchain_core", "langchain_openai", "openai", "httpx", "anthropic")]


def test_cache_py_has_no_memoising_decorator_registry_or_provider_class():
    tree = ast.parse(CACHE_PY.read_text(encoding="utf-8"))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.decorator_list]
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name != "CacheKey"]
    assert not re.search(r"lru_cache|functools|registry|Registry|Provider", CACHE_PY.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["cli.py", "assess.py"])
def test_the_cli_and_assess_gain_no_import_of_the_reader_cache(name):
    assert not [m for m in imported(SRC / name) if m.startswith("flysafelz.reader")], name


# --- F5, F6 ------------------------------------------------------------------------------------------------------------

THIS_PR_TESTS = sorted(Path(__file__).parent.glob("test_reader_cache_*.py")) + [Path(__file__).with_name("cache_helpers.py")]
# The patterns are written in pieces so this file does not match its own scan.
CLOCK_OR_ENV = re.compile("|".join([r"\.no" + r"w\(", "utc" + "now", r"\.to" + r"day\(", "env" + "iron", "get" + "env", "dot" + "env", "import " + "socket", "req" + "uests", "url" + "lib"]))


@pytest.mark.parametrize("path", THIS_PR_TESTS, ids=lambda p: p.name)
def test_f5_the_cache_tests_read_no_clock_and_no_env_and_open_no_socket(path):
    assert not CLOCK_OR_ENV.search(path.read_text(encoding="utf-8")), path


@pytest.mark.parametrize("path", [CACHE_PY, CACHED_READ_PY], ids=lambda p: p.name)
def test_f6_no_numeric_literal_but_structural_zero_and_one(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert unreviewed_literals(path, tree) == []
    rule_values = {e["value"] for e in yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8")).values() if isinstance(e["value"], (int, float))}
    assert hard_coded_rule_values(path, tree, rule_values) == []


def test_f6_a_planted_literal_in_the_cache_module_is_caught():
    planted = ast.parse(CACHE_PY.read_text(encoding="utf-8") + "\nBUSY_SECONDS = 30\n")
    assert [value for _, value in unreviewed_literals(CACHE_PY, planted)] == [30]


# --- the file is git-ignored -------------------------------------------------------------------------------------------


def test_the_database_path_is_git_ignored():
    ignored = subprocess.run(["git", "check-ignore", "data/readings.sqlite"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert ignored.returncode == 0 and ignored.stdout.strip() == "data/readings.sqlite"
