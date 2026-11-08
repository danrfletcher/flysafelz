"""The JUDGE_* settings (5a PR-7.F1, G12): the same rules as src/flysafelz/reader/config.py, tested through the real functions with an explicit environment and a real `.env` file."""

from pathlib import Path

import pytest
from pydantic import SecretStr

from evals.agent.judge import build_judge
from evals.agent.judge_config import JUDGE_VARIABLES, JudgeConfigError, build_judge_llm, judge_config, missing_judge_settings, parse_env_file

REPO = Path(__file__).resolve().parents[2]
KEY_NAME = "JUDGE_API_" + "KEY"
FULL = {"JUDGE_PROVIDER": "acme", "JUDGE_BASE_URL": "http://judge.invalid/v1", KEY_NAME: "sk-test-123456789", "JUDGE_MODEL": "judge-model-1"}


@pytest.fixture
def no_dotenv(tmp_path):
    """An environment whose `.env` is a file that is not there (as tests/conftest.py sets for every test)."""
    return {"READER_DOTENV_PATH": str(tmp_path / "absent.env")}


def with_dotenv(tmp_path, text, **environ):
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return {"READER_DOTENV_PATH": str(path), **environ}


def test_the_four_names_in_the_order_the_errors_use():
    assert JUDGE_VARIABLES == ("JUDGE_PROVIDER", "JUDGE_BASE_URL", KEY_NAME, "JUDGE_MODEL")


def test_all_four_set_in_the_environment_give_a_config(no_dotenv):
    config = judge_config({**no_dotenv, **FULL})
    assert (config.provider, config.base_url, config.model) == ("acme", "http://judge.invalid/v1", "judge-model-1")
    assert config.api_key.get_secret_value() == FULL[KEY_NAME]


def test_nothing_set_names_all_four_in_order(no_dotenv):
    with pytest.raises(JudgeConfigError) as raised:
        judge_config(no_dotenv)
    assert raised.value.missing == JUDGE_VARIABLES
    assert str(raised.value).index("JUDGE_PROVIDER") < str(raised.value).index("JUDGE_BASE_URL") < str(raised.value).index(KEY_NAME) < str(raised.value).index("JUDGE_MODEL")


@pytest.mark.parametrize("name", JUDGE_VARIABLES)
def test_one_missing_names_only_that_one(name, no_dotenv):
    environ = {**no_dotenv, **{k: v for k, v in FULL.items() if k != name}}
    with pytest.raises(JudgeConfigError) as raised:
        judge_config(environ)
    assert raised.value.missing == (name,)
    assert missing_judge_settings(environ) == [name]


@pytest.mark.parametrize("blank", ["", "   ", "\t", " \t "])
def test_a_blank_value_counts_as_unset(blank, no_dotenv):
    environ = {**no_dotenv, **FULL, "JUDGE_MODEL": blank, "JUDGE_PROVIDER": blank}
    with pytest.raises(JudgeConfigError) as raised:
        judge_config(environ)
    assert raised.value.missing == ("JUDGE_PROVIDER", "JUDGE_MODEL")


def test_the_error_names_the_variables_and_never_a_value(no_dotenv):
    environ = {**no_dotenv, **FULL, "JUDGE_MODEL": ""}
    with pytest.raises(JudgeConfigError) as raised:
        judge_config(environ)
    text = str(raised.value)
    assert "JUDGE_MODEL" in text and FULL[KEY_NAME] not in text and FULL["JUDGE_BASE_URL"] not in text


def test_the_key_never_shows_in_a_repr_a_str_or_a_dump(no_dotenv):
    config = judge_config({**no_dotenv, **FULL})
    assert isinstance(config.api_key, SecretStr)
    for shown in (repr(config), str(config), repr(config.api_key), config.model_dump_json(), str(config.model_dump())):
        assert FULL[KEY_NAME] not in shown


def test_the_dotenv_file_fills_what_the_environment_lacks(tmp_path):
    text = (
        f"# the judge\nJUDGE_PROVIDER=acme\nexport JUDGE_BASE_URL='http://judge.invalid/v1'  # the endpoint\n{KEY_NAME}=\"sk-from-file\" trailing words\nJUDGE_MODEL = judge-model-1 # note\nOTHER=1\n"
    )
    config = judge_config(with_dotenv(tmp_path, text))
    assert (config.provider, config.base_url, config.model) == ("acme", "http://judge.invalid/v1", "judge-model-1")
    assert config.api_key.get_secret_value() == "sk-from-file"


def test_the_environment_wins_over_the_dotenv_file(tmp_path):
    text = "".join(f"{name}=from-file\n" for name in JUDGE_VARIABLES)
    config = judge_config(with_dotenv(tmp_path, text, **FULL))
    assert config.model == "judge-model-1" and config.provider == "acme"


def test_a_blank_environment_value_does_not_mask_the_dotenv_value(tmp_path):
    text = "".join(f"{name}=from-file\n" for name in JUDGE_VARIABLES)
    config = judge_config(with_dotenv(tmp_path, text, JUDGE_MODEL="  "))
    assert config.model == "from-file"


def test_a_blank_dotenv_line_is_unset(tmp_path):
    with pytest.raises(JudgeConfigError) as raised:
        judge_config(with_dotenv(tmp_path, "JUDGE_MODEL=\nJUDGE_PROVIDER=a\nJUDGE_BASE_URL=b\n" + f"{KEY_NAME}=c\n"))
    assert raised.value.missing == ("JUDGE_MODEL",)


def test_keys_other_than_judge_are_ignored_in_the_dotenv_file():
    assert parse_env_file("AGENT_MODEL=x\nREADER_MODEL=y\nJUDGE_MODEL=z\n") == {"JUDGE_MODEL": "z"}


def test_a_folder_where_the_dotenv_file_should_be_counts_as_no_file(tmp_path):
    with pytest.raises(JudgeConfigError):
        judge_config({"READER_DOTENV_PATH": str(tmp_path)})


def test_judge_variables_in_the_process_environment_are_read_when_no_mapping_is_passed(monkeypatch, tmp_path):
    for name, value in FULL.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("READER_DOTENV_PATH", str(tmp_path / "absent.env"))
    assert missing_judge_settings() == []
    assert judge_config().model == "judge-model-1"


def test_the_isolated_test_environment_has_no_judge_variable_set(monkeypatch):
    """tests/conftest.py clears every JUDGE_* variable, so `make test` passes with none set (F3)."""
    import os

    assert [name for name in os.environ if name.startswith("JUDGE_")] == []
    assert missing_judge_settings() == list(JUDGE_VARIABLES)


def test_building_the_judge_with_a_missing_setting_raises_before_any_client_exists(no_dotenv):
    with pytest.raises(JudgeConfigError, match=KEY_NAME):
        build_judge({**no_dotenv, **{k: v for k, v in FULL.items() if k != KEY_NAME}})


@pytest.mark.filterwarnings("ignore:A test tried to use socket.socket")  # httpx makes a socket object while it is built; --disable-socket still refuses any connection
def test_the_client_is_built_for_the_endpoint_without_opening_a_connection(no_dotenv):
    llm = build_judge_llm(judge_config({**no_dotenv, **FULL}))
    assert llm.model_name == "judge-model-1" and llm.max_retries == 0
    assert FULL[KEY_NAME] not in repr(llm)


# --- .env.example ---------------------------------------------------------------------------------------------------------------


def test_env_example_lists_all_four_names_with_no_value():
    lines = (REPO / ".env.example").read_text(encoding="utf-8").splitlines()
    for name in JUDGE_VARIABLES:
        assert f"{name}=" in lines, name  # listed, and empty: no value is committed
