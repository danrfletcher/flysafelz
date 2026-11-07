"""G9 / E11 / E12: what the cache key is made of, and that each part alone changes it.

One table: a row changes one thing about a NOTAM, the model or the prompt, and says whether the key moves. The key is
built by cache_key from the strings render_prompt gives read_notam, so a part the model is not sent cannot move it.
"""

import hashlib
import sqlite3
from datetime import timedelta

import pytest
from cache_helpers import PerNotam, fixture_reply
from reader_helpers import FAKE_MODEL, Steps, blue_mill_notams

from flysafelz.reader import prompt as prompt_module
from flysafelz.reader.cache import cache_key, key_text, lookup, store
from flysafelz.reader.cached_read import read_cached
from flysafelz.reader.prompt import ABBREVIATIONS, render_prompt
from flysafelz.reader.read import read_notam

H7171 = blue_mill_notams()["H7171/26"]


def changed(**update):
    return H7171.model_copy(update=update)


def with_row(abbreviation: str, meaning: str = "a new meaning"):
    return {**ABBREVIATIONS, abbreviation: meaning}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


BASE = cache_key(H7171, FAKE_MODEL)

# (what changed, the key it gives): each coded field the model is sent, one at a time.
MOVES_THE_KEY = {
    "the NOTAM id (a NOTAMR re-issue, E11)": cache_key(changed(id="H7172/26"), FAKE_MODEL),
    "the E text, one word": cache_key(changed(item_e=H7171.item_e.replace("MAX HGT", "MAX HEIGHT")), FAKE_MODEL),
    "the E text, whitespace inside it": cache_key(changed(item_e=H7171.item_e.replace("MAX HGT", "MAX  HGT")), FAKE_MODEL),
    "item D": cache_key(changed(item_d="0900-1800"), FAKE_MODEL),
    "item F": cache_key(changed(item_f="500FT AGL"), FAKE_MODEL),
    "item G": cache_key(changed(item_g="900FT AMSL"), FAKE_MODEL),
    "item B": cache_key(changed(valid_from=H7171.valid_from + timedelta(hours=1)), FAKE_MODEL),
    "item C": cache_key(changed(valid_to=H7171.valid_to + timedelta(days=1)), FAKE_MODEL),
    "item C made permanent": cache_key(changed(permanent=True), FAKE_MODEL),
    "item C made an estimate": cache_key(changed(est=True), FAKE_MODEL),
    "the Q code": cache_key(changed(q_code="QRDCA"), FAKE_MODEL),
    "the Q traffic": cache_key(changed(traffic="I"), FAKE_MODEL),
    "the Q purpose": cache_key(changed(purpose="NBO"), FAKE_MODEL),
    "the Q scope": cache_key(changed(scope="E"), FAKE_MODEL),
    "the Q lower level": cache_key(changed(lower_fl=H7171.lower_fl + 1), FAKE_MODEL),
    "the Q upper level": cache_key(changed(upper_fl=H7171.upper_fl + 1), FAKE_MODEL),
    "the model name": cache_key(H7171, FAKE_MODEL + "-2"),
    "an abbreviation row added for a word in this E text": cache_key(H7171, FAKE_MODEL, with_row("RADIUS")),
    "an abbreviation row reworded for a word in this E text": cache_key(H7171, FAKE_MODEL, {**ABBREVIATIONS, "UAS": "drones"}),
}

# Things the model is not sent, or an abbreviation it would not be sent: the key must not move.
KEEPS_THE_KEY = {
    "an abbreviation row for a word not in this E text": cache_key(H7171, FAKE_MODEL, with_row("ZEPPELIN")),
    "an abbreviation row for a longer word that contains one in this E text": cache_key(H7171, FAKE_MODEL, with_row("RADIUSES")),
    "the Q-circle centre (the area is rebuilt by code)": cache_key(changed(centre_lat=H7171.centre_lat + 1), FAKE_MODEL),
    "the Q-circle radius": cache_key(changed(radius_nm=H7171.radius_nm + 1), FAKE_MODEL),
    "item A, with whitespace": cache_key(changed(item_a=H7171.item_a + "  \n"), FAKE_MODEL),
    "the PIB section it was seen in": cache_key(changed(section="Other", sections=["Other"]), FAKE_MODEL),
}


@pytest.mark.parametrize("what", MOVES_THE_KEY)
def test_each_key_part_changed_on_its_own_gives_a_different_key(what):
    assert MOVES_THE_KEY[what] != BASE, what
    assert key_text(MOVES_THE_KEY[what]) != key_text(BASE)


@pytest.mark.parametrize("what", KEEPS_THE_KEY)
def test_something_the_model_is_not_sent_does_not_change_the_key(what):
    assert KEEPS_THE_KEY[what] == BASE, what


def test_the_same_notam_and_model_always_give_the_same_key():
    assert cache_key(H7171, FAKE_MODEL) == cache_key(H7171.model_copy(), FAKE_MODEL) == BASE


def test_the_key_hashes_are_of_exactly_the_strings_read_notam_sends():
    """No second copy of the formatting: the user hash is of the user message the model got, the system hash of the system message."""
    recorder = Steps(fixture_reply("H7171/26"))
    read_notam(H7171, recorder)
    system, user = (m.content for m in recorder.calls[0]["messages"])

    assert BASE.prompt_sha256 == sha256(system)
    assert BASE.input_sha256 == sha256(user)


def test_the_hashed_text_is_the_redacted_text_so_no_real_number_feeds_the_key():
    """F3: the string hashed holds <PHONE_1>; hashing the raw E text would give a different value."""
    rendered = render_prompt(H7171)
    assert "<PHONE_1>" in rendered.user
    raw_number = next(iter(rendered.redaction.values()))
    assert raw_number in H7171.item_e and raw_number not in rendered.user
    assert BASE.input_sha256 == sha256(rendered.user)
    assert raw_number not in key_text(BASE)


def test_e12_a_new_prompt_version_changes_the_key_and_the_old_reading_is_a_miss(monkeypatch, tmp_path, now):
    database = tmp_path / "readings.sqlite"
    key = cache_key(H7171, FAKE_MODEL)
    store(database, key, read_notam(H7171, PerNotam([H7171], H7171_26=[fixture_reply("H7171/26")])), now)
    assert lookup(database, H7171, key) is not None

    monkeypatch.setattr(prompt_module, "PROMPT_VERSION", "v-next")
    bumped = cache_key(H7171, FAKE_MODEL)

    assert bumped != key and bumped.prompt_sha256 != key.prompt_sha256 and bumped.input_sha256 == key.input_sha256
    assert lookup(database, H7171, bumped) is None  # no old reading is reused
    llm = PerNotam([H7171], H7171_26=[fixture_reply("H7171/26")])
    again = read_cached(H7171, llm, database, now)
    assert not again.hit and llm.calls == ["H7171/26"]


def test_e11_a_reissued_notam_with_a_new_id_is_a_miss_and_the_old_rows_stay(tmp_path, now):
    database = tmp_path / "readings.sqlite"
    old = PerNotam([H7171], H7171_26=[fixture_reply("H7171/26")])
    read_cached(H7171, old, database, now)

    reissued = changed(id="H7172/26", item_e=H7171.item_e + " AMENDED.", replaces="H7171/26")
    llm = PerNotam([reissued], H7172_26=[fixture_reply("H7171/26")])
    result = read_cached(reissued, llm, database, now)

    assert not result.hit and llm.calls == ["H7172/26"]
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM readings").fetchone() == (2,)
    assert lookup(database, H7171, cache_key(H7171, FAKE_MODEL)) is not None  # the old id's row is still there and still its own
    assert lookup(database, reissued, cache_key(reissued, FAKE_MODEL)) is not None


def test_e11_the_same_id_with_changed_e_text_is_a_miss(tmp_path, now):
    database = tmp_path / "readings.sqlite"
    read_cached(H7171, PerNotam([H7171], H7171_26=[fixture_reply("H7171/26")]), database, now)

    edited = changed(item_e=H7171.item_e.replace("700FT", "600FT"))
    llm = PerNotam([edited], H7171_26=[fixture_reply("H7171/26")])

    assert lookup(database, edited, cache_key(edited, FAKE_MODEL)) is None
    assert not read_cached(edited, llm, database, now).hit and llm.calls == ["H7171/26"]


def test_e6_the_prompt_version_is_v2_and_a_row_stored_under_the_v1_key_is_a_miss_so_the_notam_is_read_again(monkeypatch, tmp_path, now):
    assert prompt_module.PROMPT_VERSION == "v2"
    database = tmp_path / "readings.sqlite"
    v2 = cache_key(H7171, FAKE_MODEL)
    monkeypatch.setattr(prompt_module, "PROMPT_VERSION", "v1")
    v1 = cache_key(H7171, FAKE_MODEL)  # the old system text: its first line says v1
    monkeypatch.undo()

    assert v1 != v2 and key_text(v1) != key_text(v2)
    assert v1.prompt_sha256 != v2.prompt_sha256 and v1.input_sha256 == v2.input_sha256 and v1.notam_id == v2.notam_id and v1.model == v2.model
    assert render_prompt(H7171).system.splitlines()[0] == "NOTAM reader v2."

    reply = fixture_reply("H7171/26")
    monkeypatch.setattr(prompt_module, "PROMPT_VERSION", "v1")
    store(database, v1, read_notam(H7171, Steps(reply)), now)
    monkeypatch.undo()
    assert lookup(database, H7171, v1) is not None  # the row is there under its own key ...
    assert lookup(database, H7171, v2) is None  # ... and is no use under v2

    llm = PerNotam([H7171], H7171_26=[reply])
    again = read_cached(H7171, llm, database, now)
    assert not again.hit and llm.calls == ["H7171/26"] and again.result.prompt_version == "v2"
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM readings").fetchone() == (2,)  # the v1 row stays; v2 filed beside it
