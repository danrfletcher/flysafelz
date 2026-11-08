"""The assess tool reads NOTAMs the way get_notams does: the same reading step, the same call limit, the same memory, and the reader's status reaches ticket 2's `assess`."""

import pytest
from factories import pib_result, reading
from fakes import FakeToolChatModel
from pib_helpers import parsed
from tool_helpers import calls
from tool_world import WINDOW, World, go

from flysafelz.agent.tools import notams_tool
from flysafelz.reader.contract import ReaderResult, ReaderStatus

ASSESS = [calls("assess", **WINDOW)]
READER_CALLS = 3  # settings.yaml notam_reader_calls


class FakeReader:
    """Stands in for `read_for_lz` as the shared reading step imports it: counts the NOTAMs it was asked to read and answers as the test says."""

    def __init__(self, outcome=ReaderStatus.READ):
        self.asked: list[str] = []
        self.outcome = outcome

    def __call__(self, parsed_notams, lz, window, llm, store, *, now=None):
        out = {}
        for p in parsed_notams:
            self.asked.append(p.id)
            read = reading(notam_id=p.id, value=None) if self.outcome in {ReaderStatus.READ, ReaderStatus.STORED} else None
            out[p.id] = ReaderResult(notam_id=p.id, reading=read, status=self.outcome, failed_checks=[], model="fake", prompt_version="x", latency_s=0.0, tokens=None)
        return out


def run_with(monkeypatch, outcome, count: int, tmp_path, steps=ASSESS):
    fake = FakeReader(outcome)
    monkeypatch.setattr(notams_tool, "read_for_lz", fake)
    world = World(pib=pib_result(*[parsed(f"H70{n}/26") for n in range(count)]))
    return fake, world, go(world, steps, reader_model=lambda: FakeToolChatModel(script=[]), reader_store=tmp_path / "readings.sqlite")


def test_more_notams_than_the_limit_make_exactly_the_limit_of_reader_calls_and_the_rest_are_reported_unread(monkeypatch, tmp_path):
    fake, _, run = run_with(monkeypatch, ReaderStatus.READ, 5, tmp_path)
    assert len(fake.asked) == READER_CALLS
    text = run.text("assess")
    left = [n for n in (f"H70{i}/26" for i in range(5)) if n not in fake.asked]
    assert len(left) == 2 and f"Reader limit reached: {', '.join(left)} not read, treated as unread." in text.splitlines()[-1]
    assert run.progress == ["Assessing\u2026", "Reading 3 NOTAMs (1/3)\u2026", "Reading 3 NOTAMs (2/3)\u2026", "Reading 3 NOTAMs (3/3)\u2026"]


def test_assess_then_get_notams_reads_each_notam_once(monkeypatch, tmp_path):
    fake, _, run = run_with(monkeypatch, ReaderStatus.READ, 2, tmp_path, steps=[*ASSESS, calls("get_notams", **WINDOW)])
    assert sorted(fake.asked) == ["H700/26", "H701/26"]
    assert [line for line in run.progress if line.startswith("Reading")] == ["Reading 2 NOTAMs (1/2)\u2026", "Reading 2 NOTAMs (2/2)\u2026"]


def test_get_notams_then_assess_reads_each_notam_once_and_the_verdict_is_the_same(monkeypatch, tmp_path):
    _, _, alone = run_with(monkeypatch, ReaderStatus.READ, 2, tmp_path / "a")
    fake, _, both = run_with(monkeypatch, ReaderStatus.READ, 2, tmp_path / "b", steps=[calls("get_notams", **WINDOW), *ASSESS])
    assert len(fake.asked) == 2
    assert both.artifact("assess") == alone.artifact("assess")


@pytest.mark.parametrize("outcome", [ReaderStatus.UNAVAILABLE, ReaderStatus.FAILED], ids=["unavailable", "failed"])
def test_a_notam_the_reader_could_not_read_makes_the_hours_it_bears_on_unverified(outcome, monkeypatch, tmp_path):
    fake, _, run = run_with(monkeypatch, outcome, 1, tmp_path)
    artifact = run.artifact("assess")
    assert fake.asked == ["H700/26"]
    assert {s.verdict for s in artifact.segments if s.tier == "VERDICT"} <= {"UNVERIFIED", "NO_GO"}
    assert "UNVERIFIED" in {s.verdict for s in artifact.segments}


def test_an_unavailable_reader_is_listed_in_an_assumption_and_a_failed_read_is_not_called_unavailable(monkeypatch, tmp_path):
    _, _, unavailable = run_with(monkeypatch, ReaderStatus.UNAVAILABLE, 1, tmp_path / "u")
    _, _, failed = run_with(monkeypatch, ReaderStatus.FAILED, 1, tmp_path / "f")
    assert any(a.startswith("NOTAM reader unavailable for H700/26") for a in unavailable.artifact("assess").assumptions)
    assert not any("reader unavailable" in a for a in failed.artifact("assess").assumptions)
