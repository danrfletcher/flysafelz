"""reader/config.py: the four READER_* settings (environment first, then a .env file), the ChatOpenAI client, and settings.yaml.

conftest.py has cleared every READER_* variable and pointed READER_DOTENV_PATH at a file that does not exist, so each
test sets exactly what it needs. Env text is built from pieces, so no tracked file holds a `*_API_KEY` assignment line (F6).
ChatOpenAI is built here and never invoked: nothing opens a socket.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_openai import ChatOpenAI

from flysafelz.reader import config, settings
from flysafelz.reader.config import (
    DOTENV_PATH,
    READER_VARIABLES,
    ReaderConfigError,
    build_llm,
    parse_env_file,
    read_env_file,
    reader_config,
)

ROOT = Path(__file__).resolve().parents[2]
PROVIDER, BASE_URL, API_KEY, MODEL = READER_VARIABLES
VALUES = {PROVIDER: "bridge", BASE_URL: "http://127.0.0.1:1/api/v1", API_KEY: "dummy-" + "token", MODEL: "some/model"}


def dotenv_text(values: dict[str, str]) -> str:
    return "".join(f"{name}={value}\n" for name, value in values.items())


@pytest.fixture
def dotenv(monkeypatch, tmp_path):
    """Point READER_DOTENV_PATH at a .env under tmp_path and return a function that fills it."""
    path = tmp_path / "reader.env"
    monkeypatch.setenv(DOTENV_PATH, str(path))

    def fill(text: str) -> Path:
        path.write_text(text, encoding="utf-8")
        return path

    return fill


# --- missing settings -----------------------------------------------------------------------------------------------------


def test_with_nothing_set_the_error_lists_all_four_in_order():
    with pytest.raises(ReaderConfigError) as error:
        reader_config()
    assert error.value.missing == (PROVIDER, BASE_URL, API_KEY, MODEL)
    assert str(error.value) == "reader settings missing: READER_PROVIDER, READER_BASE_URL, READER_API_KEY, READER_MODEL (set them in the environment or in .env)"
    assert isinstance(error.value, ValueError)


@pytest.mark.parametrize("present", [(PROVIDER,), (BASE_URL, MODEL), (PROVIDER, BASE_URL, API_KEY), (PROVIDER, API_KEY, MODEL)])
def test_the_error_names_exactly_the_missing_variables_and_nothing_falls_back(monkeypatch, present):
    for name in present:
        monkeypatch.setenv(name, VALUES[name])
    expected = tuple(name for name in READER_VARIABLES if name not in present)

    with pytest.raises(ReaderConfigError) as error:
        reader_config()

    assert error.value.missing == expected
    assert str(error.value).startswith("reader settings missing: " + ", ".join(expected) + " ")


def test_a_blank_value_counts_as_missing(monkeypatch):
    for name in READER_VARIABLES:
        monkeypatch.setenv(name, VALUES[name])
    monkeypatch.setenv(BASE_URL, "   ")
    monkeypatch.setenv(MODEL, "")

    with pytest.raises(ReaderConfigError) as error:
        reader_config()

    assert error.value.missing == (BASE_URL, MODEL)


def test_the_example_env_file_with_its_empty_lines_is_all_missing(dotenv):
    dotenv((ROOT / ".env.example").read_text(encoding="utf-8"))
    with pytest.raises(ReaderConfigError) as error:
        reader_config()
    assert error.value.missing == READER_VARIABLES


def test_build_llm_with_nothing_set_raises_the_same_error():
    with pytest.raises(ReaderConfigError):
        build_llm()


# --- precedence ---------------------------------------------------------------------------------------------------------------


def test_the_environment_alone_is_enough(monkeypatch):
    for name in READER_VARIABLES:
        monkeypatch.setenv(name, VALUES[name])
    found = reader_config()
    assert (found.provider, found.base_url, found.model) == (VALUES[PROVIDER], VALUES[BASE_URL], VALUES[MODEL])
    assert found.api_key.get_secret_value() == VALUES[API_KEY]


def test_the_env_file_alone_is_enough(dotenv):
    dotenv(dotenv_text(VALUES))
    found = reader_config()
    assert (found.provider, found.base_url, found.model) == (VALUES[PROVIDER], VALUES[BASE_URL], VALUES[MODEL])
    assert found.api_key.get_secret_value() == VALUES[API_KEY]


def test_the_environment_beats_the_file_variable_by_variable(dotenv, monkeypatch):
    dotenv(dotenv_text({**VALUES, MODEL: "from/the-file", PROVIDER: "from-the-file"}))
    monkeypatch.setenv(MODEL, "from/the-environment")

    found = reader_config()

    assert found.model == "from/the-environment"
    assert found.provider == "from-the-file"  # not set in the environment, so the file's
    assert found.base_url == VALUES[BASE_URL]


def test_a_blank_environment_value_does_not_hide_the_file(dotenv, monkeypatch):
    dotenv(dotenv_text(VALUES))
    monkeypatch.setenv(MODEL, "")
    assert reader_config().model == VALUES[MODEL]


def test_an_explicit_mapping_replaces_the_process_environment(monkeypatch):
    monkeypatch.setenv(MODEL, "from/the-process")
    assert reader_config(VALUES).model == VALUES[MODEL]


# --- the parser ---------------------------------------------------------------------------------------------------------------


def test_the_parser_skips_comments_blank_lines_and_other_keys():
    text = "\n".join(
        [
            "# a comment",
            "",
            "   ",
            "  # an indented comment",
            "OPENAIP_API_" + "KEY=not-for-the-reader",
            "SOMETHING_ELSE=1",
            "no equals sign here",
            f"{PROVIDER}=bridge",
            f"# {MODEL}=commented/out",
            f"{MODEL}=some/model",
        ]
    )
    assert parse_env_file(text) == {PROVIDER: "bridge", MODEL: "some/model"}


def test_the_parser_strips_matching_quotes_and_an_export_prefix():
    mismatched = "\"kept'"
    text = "\n".join([f'{PROVIDER}="double quoted"', f"{BASE_URL}='single quoted'", f"export {MODEL}=exported/model", f"{API_KEY}={mismatched}"])
    assert parse_env_file(text) == {PROVIDER: "double quoted", BASE_URL: "single quoted", MODEL: "exported/model", API_KEY: mismatched}


def test_the_parser_keeps_equals_and_hash_inside_a_value_and_cuts_a_trailing_comment():
    text = "\n".join(
        [
            f"{BASE_URL}=http://127.0.0.1:6008/api/v1?x=1#frag",
            f"{MODEL}=some/model # the dev model",
            f'{PROVIDER}="has # inside"',
            f"  {API_KEY}  =  spaced  ",
        ]
    )
    assert parse_env_file(text) == {BASE_URL: "http://127.0.0.1:6008/api/v1?x=1#frag", MODEL: "some/model", PROVIDER: "has # inside", API_KEY: "spaced"}


def test_a_quoted_value_ends_at_its_closing_quote_and_a_trailing_comment_is_dropped():
    text = "\n".join(
        [
            f'{MODEL}="anthropic/claude-haiku-5-5" # dev bridge',
            f"{PROVIDER}='bridge'  # single quoted",
            f'{BASE_URL}="http://127.0.0.1:1/api/v1"#no space',
            f'{API_KEY}="unterminated # kept',
        ]
    )
    assert parse_env_file(text) == {
        MODEL: "anthropic/claude-haiku-5-5",
        PROVIDER: "bridge",
        BASE_URL: "http://127.0.0.1:1/api/v1",
        API_KEY: '"unterminated',
    }


def test_the_parser_takes_the_last_of_a_repeated_key_and_handles_crlf():
    assert parse_env_file(f"{MODEL}=first\r\n{MODEL}=second\r\n") == {MODEL: "second"}


def test_a_missing_file_or_a_folder_is_an_empty_file(tmp_path):
    assert read_env_file(tmp_path / "nothing-here") == {}
    assert read_env_file(tmp_path) == {}


# --- where the file is ------------------------------------------------------------------------------------------------------------


def test_the_dotenv_path_variable_names_the_file(tmp_path, monkeypatch):
    other = tmp_path / "elsewhere.env"
    other.write_text(dotenv_text(VALUES), encoding="utf-8")
    monkeypatch.setenv(DOTENV_PATH, str(other))
    assert reader_config().model == VALUES[MODEL]


def test_without_the_path_variable_the_file_is_the_repository_roots(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(dotenv_text(VALUES), encoding="utf-8")
    monkeypatch.delenv(DOTENV_PATH)
    monkeypatch.setattr(config, "_repository_root", lambda: tmp_path)
    assert reader_config().model == VALUES[MODEL]


def test_the_repository_root_is_the_folder_with_pyproject_toml():
    assert config._repository_root() == ROOT


def test_the_dotenv_path_is_read_from_the_environment_never_from_the_file(dotenv, tmp_path):
    other = tmp_path / "other.env"
    other.write_text(dotenv_text(VALUES), encoding="utf-8")
    dotenv(f"{DOTENV_PATH}={other}\n")
    with pytest.raises(ReaderConfigError):
        reader_config()


# --- the client ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:A test tried to use socket")  # langchain-openai probes socket options on an unconnected socket; nothing connects
def test_build_llm_makes_a_chat_openai_for_the_configured_endpoint(monkeypatch):
    for name in READER_VARIABLES:
        monkeypatch.setenv(name, VALUES[name])

    llm = build_llm()

    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == VALUES[MODEL]
    assert llm.openai_api_base == VALUES[BASE_URL]
    key = llm.openai_api_key
    assert key is not None and not isinstance(key, str) and not callable(key) and key.get_secret_value() == VALUES[API_KEY]
    assert llm.max_retries == 0


@pytest.mark.filterwarnings("ignore:A test tried to use socket")  # langchain-openai probes socket options on an unconnected socket; nothing connects
def test_build_llm_takes_a_config_it_is_given():
    assert build_llm(reader_config(VALUES)).model_name == VALUES[MODEL]


@pytest.mark.filterwarnings("ignore:A test tried to use socket")  # langchain-openai probes socket options on an unconnected socket; nothing connects
def test_the_key_is_not_in_the_repr_of_the_config_or_the_client():
    found = reader_config(VALUES)
    assert VALUES[API_KEY] not in repr(found) and VALUES[API_KEY] not in str(found)
    assert VALUES[API_KEY] not in repr(build_llm(found))


# --- lazy ------------------------------------------------------------------------------------------------------------------------


# Runs in a fresh interpreter: it spies on os.environ and on every file opened (other than source and bytecode) while the
# named module is imported, and prints as JSON the READER_* names looked up and the files opened under the repository.
IMPORT_PROBE = """
import json, os, sys

class SpyEnviron(dict):
    seen = []
    def get(self, key, *args):
        SpyEnviron.seen.append(key)
        return super().get(key, *args)
    def __getitem__(self, key):
        SpyEnviron.seen.append(key)
        return super().__getitem__(key)
    def __contains__(self, key):
        SpyEnviron.seen.append(key)
        return super().__contains__(key)

opened = []
def audit(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)) and not str(args[0]).endswith((".py", ".pyc", ".so", ".pyd")):
        opened.append(str(args[0]))

os.environ = SpyEnviron(os.environ)
sys.addaudithook(audit)
import importlib
importlib.import_module(sys.argv[1])
print(json.dumps({"env_reads": [k for k in SpyEnviron.seen if k.startswith("READER_")], "files_opened": [p for p in opened if p.startswith(sys.argv[2]) and "/.venv/" not in p]}))
"""


def import_probe(module: str) -> dict:
    result = subprocess.run([sys.executable, "-c", IMPORT_PROBE, module, str(ROOT)], capture_output=True, text=True, cwd=ROOT, env={**os.environ, DOTENV_PATH: "/nonexistent"}, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_importing_the_reader_package_reads_no_env_and_no_file():
    assert import_probe("flysafelz.reader") == {"env_reads": [], "files_opened": []}


def test_importing_the_reader_modules_reads_no_env_and_no_dotenv():
    """The modules load the package's own YAML data (settings, rules, abbreviations) and nothing else; the READER_* values and the .env wait for reader_config()."""
    assert import_probe("flysafelz.reader.config") == {"env_reads": [], "files_opened": []}
    probe = import_probe("flysafelz.reader.reader_set")
    assert probe["env_reads"] == []
    packaged = {str(path) for path in (ROOT / "src" / "flysafelz").rglob("*.yaml")}
    assert str(settings.SETTINGS_PATH) in probe["files_opened"] and set(probe["files_opened"]) <= packaged


def test_importing_the_package_prints_nothing_and_exits_zero():
    result = subprocess.run([sys.executable, "-c", "import flysafelz.reader"], capture_output=True, text=True, cwd=ROOT, env={**os.environ, DOTENV_PATH: "/nonexistent"}, check=False)
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")


def test_the_values_are_read_each_time_reader_config_is_called(monkeypatch):
    with pytest.raises(ReaderConfigError):
        reader_config()
    for name in READER_VARIABLES:
        monkeypatch.setenv(name, VALUES[name])
    assert reader_config().model == VALUES[MODEL]
    monkeypatch.setenv(MODEL, "changed/model")
    assert reader_config().model == "changed/model"  # nothing was kept from the first call


# --- settings.yaml -----------------------------------------------------------------------------------------------------------------------


def write_settings(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def settings_text(**changes: str | None) -> str:
    """settings.yaml text with every key at a good value, less or changed as asked (None drops the key)."""
    keys = {"timeout_s": "45", "concurrency": "3", "runs": "3", **changes}
    return "".join(f"{name}: {value}\n" for name, value in keys.items() if value is not None)


def test_the_shipped_settings_are_loaded_by_name():
    assert (settings.SETTINGS.timeout_s, settings.SETTINGS.concurrency, settings.SETTINGS.runs) == (45, 3, 3)
    assert settings.load_settings() == settings.SETTINGS


@pytest.mark.parametrize("key", ["timeout_s", "concurrency", "runs"])
def test_a_missing_key_fails_when_the_file_loads_and_names_the_key(tmp_path, key):
    with pytest.raises(ValueError, match=key):
        settings.load_settings(write_settings(tmp_path, settings_text(**{key: None})))


@pytest.mark.parametrize("bad", ["0", "-1", "-3"])
def test_a_non_positive_concurrency_fails_naming_the_key(tmp_path, bad):
    with pytest.raises(ValueError, match="concurrency"):
        settings.load_settings(write_settings(tmp_path, settings_text(concurrency=bad)))


@pytest.mark.parametrize(("key", "bad"), [("timeout_s", "0"), ("timeout_s", "-5"), ("runs", "0"), ("concurrency", "2.5"), ("concurrency", "three")])
def test_other_malformed_values_fail_loudly_naming_the_key(tmp_path, key, bad):
    with pytest.raises(ValueError, match=key):
        settings.load_settings(write_settings(tmp_path, settings_text(**{key: bad})))


def test_an_unknown_key_and_an_empty_file_fail_too(tmp_path):
    with pytest.raises(ValueError, match="surprise"):
        settings.load_settings(write_settings(tmp_path, settings_text() + "surprise: 1\n"))
    with pytest.raises(ValueError, match="concurrency"):
        settings.load_settings(write_settings(tmp_path, ""))
