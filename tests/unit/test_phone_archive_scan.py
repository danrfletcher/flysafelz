"""G12: scripts/phone_archive_scan.py, tested through its functions with a fake GitHub (httpx.MockTransport) and tiny constructed PIBs.

No test touches the network (F6) or runs the script's `main`; the script is loaded as a module and `run`, `fetch_archive`,
`scan_archive` and `format_report` get the fake client and a tmp_path cache. Every PIB is built here from fiction-range
numbers (07700 900xxx, 01632 960xxx), so no real number is in this file (F5).
"""

import importlib.util
import io
import os
import re
import sys
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

import httpx
import pytest

from flysafelz.redact import redact_phones

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "phone_archive_scan.py"

_spec = importlib.util.spec_from_file_location("phone_archive_scan", SCRIPT)
assert _spec and _spec.loader
scan_script = importlib.util.module_from_spec(_spec)
sys.modules["phone_archive_scan"] = scan_script
_spec.loader.exec_module(scan_script)

FICTION_MOBILE = "07700 900123"
FICTION_LANDLINE = "01632 960124"
D1, D2, D3, D4 = date(2026, 7, 1), date(2026, 7, 2), date(2026, 7, 3), date(2026, 7, 4)
SHA_A, SHA_B, SHA_C = "a" * 40, "b" * 40, "c" * 40  # letters only, so a sha never looks like a long digit run


@pytest.fixture(autouse=True)
def no_token(monkeypatch):
    """A developer's own GITHUB_TOKEN must never reach a test; the token tests set theirs."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)


# --- constructed PIBs and a fake GitHub ------------------------------------------------------------------------

NOTAM = """<Notam OriginalMessage="false" PIBSection="AD">
<NOF>EGGN</NOF><Series>{series}</Series><Number>{number}</Number><Year>26</Year><Type>N</Type>
<QLine><FIR>EGTT</FIR><Code23>WZ</Code23><Code45>LW</Code45><Traffic>IV</Traffic><Purpose>M</Purpose><Scope>AW</Scope><Lower>0</Lower><Upper>2</Upper></QLine>
<Coordinates>5129N00023W</Coordinates><Radius>1</Radius><ItemA>EGLL</ItemA>
<StartValidity>2607010000</StartValidity><EndValidity>2612312359</EndValidity>
<ItemE>{item_e}</ItemE><ItemF>SFC</ItemF><ItemG>150FT AMSL</ItemG>
</Notam>"""

PIB = """<?xml version="1.0" encoding="UTF-8"?>
<Pib><AreaPIBHeader><Issued>2026-07-01T07:00:37+00:00</Issued></AreaPIBHeader>
<FIRSection><ICAO>EGTT</ICAO><ADSection><Code>EGLL</Code><NotamList>{notams}</NotamList><ObsoleteNotamList/></ADSection></FIRSection></Pib>"""


def pib(*items: tuple[str, str]) -> str:
    """A PIB of NOTAMs given as (id like 'A0001', item E text)."""
    notams = "".join(NOTAM.format(series=nid[0], number=int(nid[1:]), item_e=escape(text)) for nid, text in items)
    return PIB.format(notams=notams)


class FakeGitHub:
    """The commits API and raw.githubusercontent.com, answering from tables, and keeping every request it saw."""

    def __init__(self, shas: Mapping[date, str | None], pibs: Mapping[str, str | bytes], *, status: Mapping[date, int] | None = None):
        self.shas = shas
        self.pibs = pibs
        self.status = status or {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == "api.github.com":
            day = date.fromisoformat(request.url.params["until"][:10])
            if day in self.status:
                return httpx.Response(self.status[day])
            sha = self.shas[day]
            return httpx.Response(200, json=[{"sha": sha}] if sha else [])
        sha = request.url.path.split("/")[3]
        body = self.pibs[sha]
        return httpx.Response(200, content=body if isinstance(body, bytes) else body.encode())

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))

    def api(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.host == "api.github.com"]

    def raw(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.host == "raw.githubusercontent.com"]


def run_script(github: FakeGitHub, cache: Path, dates: list[date]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with github.client() as client:
        code = scan_script.run(client, cache, dates, out, err)
    return code, out.getvalue(), err.getvalue()


# --- the date plan ---------------------------------------------------------------------------------------------


def test_the_plan_is_123_dates_from_july_1_to_october_31():
    dates = scan_script.date_plan()
    assert len(dates) == 123
    assert dates[0] == date(2026, 7, 1)
    assert dates[-1] == date(2026, 10, 31)
    assert len(set(dates)) == 123
    assert dates == sorted(dates)


def test_each_commit_lookup_carries_until_and_per_page_1(tmp_path):
    dates = scan_script.date_plan()[:3]
    github = FakeGitHub({d: SHA_A for d in dates}, {SHA_A: pib(("A0001", "X"))})
    run_script(github, tmp_path, dates)
    assert len(github.api()) == 3
    for request, day in zip(github.api(), dates, strict=True):
        assert request.url.host == "api.github.com"
        assert request.url.path == "/repos/Jonty/uk-notam-archive/commits"
        assert request.url.params["until"] == f"{day}T23:59:59Z"
        assert request.url.params["per_page"] == "1"
        assert request.url.params["path"] == "data/PIB.xml"
        assert "until=" in str(request.url) and "per_page=1" in str(request.url)


def test_the_pib_url_is_the_raw_file_of_that_commit(tmp_path):
    github = FakeGitHub({D1: SHA_A}, {SHA_A: pib(("A0001", "X"))})
    run_script(github, tmp_path, [D1])
    assert [str(r.url) for r in github.raw()] == [f"https://raw.githubusercontent.com/Jonty/uk-notam-archive/{SHA_A}/data/PIB.xml"]


# --- fetch -----------------------------------------------------------------------------------------------------


def test_dates_that_resolve_to_one_sha_download_that_pib_once(tmp_path):
    github = FakeGitHub({D1: SHA_A, D2: SHA_A, D3: SHA_B, D4: SHA_A}, {SHA_A: pib(("A0001", "ONE")), SHA_B: pib(("A0002", "TWO"))})
    with github.client() as client:
        fetched = scan_script.fetch_archive(client, [D1, D2, D3, D4], tmp_path, sys.stderr)
    assert len(github.api()) == 4
    assert sorted(str(r.url).split("/")[5] for r in github.raw()) == [SHA_A, SHA_B]
    assert fetched.requests == 6
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{SHA_A}.xml", f"{SHA_B}.xml", "days.json"]


def test_a_full_cache_makes_no_request_at_all(tmp_path):
    first = FakeGitHub({D1: SHA_A, D2: SHA_B}, {SHA_A: pib(("A0001", "ONE")), SHA_B: pib(("A0002", "TWO"))})
    code, out, _ = run_script(first, tmp_path, [D1, D2])
    assert code == 0 and first.requests

    again = FakeGitHub({}, {})  # a fake that would raise KeyError on any request
    code_again, out_again, err_again = run_script(again, tmp_path, [D1, D2])
    assert code_again == 0
    assert again.requests == []
    assert "requests made: 0" in err_again
    assert out_again == out


def test_a_403_stops_the_run_naming_the_date_and_keeps_the_cache(tmp_path):
    github = FakeGitHub({D1: SHA_A, D2: SHA_B, D3: SHA_C}, {SHA_A: pib(("A0001", "ONE")), SHA_B: pib(("A0002", "TWO"))}, status={D3: 403})
    code, out, err = run_script(github, tmp_path, [D1, D2, D3])
    assert code == 1
    assert out == ""  # no half report
    assert "2026-07-03" in err and "403" in err and "run again to resume" in err
    assert "2026-07-01" not in err.split("stopped:")[1]  # it names the date it reached, not the first
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{SHA_A}.xml", f"{SHA_B}.xml", "days.json"]
    assert scan_script.load_days(tmp_path) == {D1: SHA_A, D2: SHA_B}


def test_a_429_stops_the_run_too(tmp_path):
    github = FakeGitHub({D1: SHA_A}, {SHA_A: pib(("A0001", "ONE"))}, status={D1: 429})
    code, _, err = run_script(github, tmp_path, [D1])
    assert code == 1
    assert "2026-07-01" in err and "429" in err


def test_a_rerun_after_a_rate_limit_resumes_from_that_date(tmp_path):
    shas = {D1: SHA_A, D2: SHA_B, D3: SHA_C}
    pibs = {SHA_A: pib(("A0001", "ONE")), SHA_B: pib(("A0002", "TWO")), SHA_C: pib(("A0003", "THREE"))}
    assert run_script(FakeGitHub(shas, pibs, status={D2: 403}), tmp_path, [D1, D2, D3])[0] == 1

    resumed = FakeGitHub(shas, pibs)
    code, out, _ = run_script(resumed, tmp_path, [D1, D2, D3])
    assert code == 0
    assert [r.url.params["until"][:10] for r in resumed.api()] == ["2026-07-02", "2026-07-03"]  # D1 was not asked again
    assert [str(r.url).split("/")[5] for r in resumed.raw()] == [SHA_B, SHA_C]
    assert "Distinct NOTAMs: 3" in out


def test_another_failure_exits_1_naming_the_date(tmp_path):
    github = FakeGitHub({D1: SHA_A}, {}, status={D1: 500})
    code, _, err = run_script(github, tmp_path, [D1])
    assert code == 1
    assert "2026-07-01" in err and "500" in err


def test_a_failed_download_exits_1_naming_the_sha_and_leaves_no_file(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.github.com":
            return httpx.Response(200, json=[{"sha": SHA_A}])
        raise httpx.ReadError("connection reset", request=request)

    out, err = io.StringIO(), io.StringIO()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert scan_script.run(client, tmp_path, [D1], out, err) == 1
    assert SHA_A[:12] in err.getvalue() and "2026-07-01" in err.getvalue()
    assert not (tmp_path / f"{SHA_A}.xml").exists()
    assert [p.name for p in tmp_path.iterdir()] == ["days.json"]  # no temp file, no partial PIB


def test_a_date_with_no_commit_is_logged_and_skipped_never_fetched_as_none(tmp_path):
    github = FakeGitHub({D1: None, D2: SHA_A}, {SHA_A: pib(("A0001", "ONE"))})
    code, out, err = run_script(github, tmp_path, [D1, D2])
    assert code == 0
    assert "2026-07-01: no archive commit" in err and "skipped" in err
    assert not [r for r in github.raw() if "None" in str(r.url)]
    assert len(github.raw()) == 1
    assert "skipped with none: 1" in out
    # the empty answer is remembered, so a rerun does not ask again
    again = FakeGitHub({}, {})
    assert run_script(again, tmp_path, [D1, D2])[0] == 0
    assert again.requests == []


def test_a_download_that_is_not_a_pib_is_reported_not_cached_and_the_run_goes_on(tmp_path):
    truncated = pib(("A0001", "ONE"))[:200]
    github = FakeGitHub({D1: SHA_A, D2: SHA_B}, {SHA_A: truncated, SHA_B: pib(("A0002", "TWO"))})
    code, out, err = run_script(github, tmp_path, [D1, D2])
    assert code == 1
    assert not (tmp_path / f"{SHA_A}.xml").exists()
    assert (tmp_path / f"{SHA_B}.xml").exists()
    assert "2026-07-01" in err and SHA_A[:12] in err and "not cached" in err
    assert "Distinct NOTAMs: 1" in out


def test_the_token_header_is_sent_only_when_github_token_is_set(tmp_path, monkeypatch):
    github = FakeGitHub({D1: SHA_A}, {SHA_A: pib(("A0001", "ONE"))})
    run_script(github, tmp_path / "without", [D1])
    assert github.api() and all("authorization" not in r.headers for r in github.api())

    monkeypatch.setenv("GITHUB_TOKEN", "test-token-not-real")
    with_token = FakeGitHub({D1: SHA_A}, {SHA_A: pib(("A0001", "ONE"))})
    _, out, err = run_script(with_token, tmp_path / "with", [D1])
    assert [r.headers["authorization"] for r in with_token.api()] == ["Bearer test-token-not-real"]
    assert all("authorization" not in r.headers for r in with_token.raw())  # the raw PIB host never sees the token
    assert "test-token-not-real" not in out + err  # never logged


def test_a_blank_github_token_is_no_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "   ")
    assert "Authorization" not in scan_script.api_headers()


def test_a_cache_write_is_atomic(tmp_path, monkeypatch):
    target = tmp_path / "cache" / f"{SHA_A}.xml"
    scan_script.write_atomically(target, b"<a/>")
    assert target.read_bytes() == b"<a/>"
    assert [p.name for p in target.parent.iterdir()] == [target.name]

    def boom(src, dst):
        raise OSError("disk gone")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk gone"):
        scan_script.write_atomically(target.with_name("other.xml"), b"<b/>")
    assert [p.name for p in target.parent.iterdir()] == [target.name]  # the temp file is gone, nothing half-written is left


def test_a_days_file_that_is_not_a_map_is_reported(tmp_path):
    (tmp_path / "days.json").write_text("[1, 2]", encoding="utf-8")
    code, _, err = run_script(FakeGitHub({}, {}), tmp_path, [D1])
    assert code == 1
    assert "days.json" in err


def test_the_cache_folder_is_gitignored():
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".cache/" in lines
    assert scan_script.CACHE.relative_to(ROOT).parts[0] == ".cache"


# --- scan ------------------------------------------------------------------------------------------------------


def scan_of(tmp_path: Path, *pibs: tuple[date, str, str]):
    """Write each (date, sha, PIB xml) into the cache and scan them."""
    for _day, sha, xml in pibs:
        (tmp_path / f"{sha}.xml").write_text(xml, encoding="utf-8")
    return scan_script.scan_archive({day: sha for day, sha, _xml in pibs}, tmp_path)


def test_two_pibs_sharing_a_notam_id_give_one_distinct_notam_and_the_earliest_date_wins(tmp_path):
    first = pib(("A0001", f"CTC {FICTION_MOBILE}."), ("A0002", "NO NUMBER HERE"))
    second = pib(("A0001", f"CTC {FICTION_MOBILE}."), ("A0003", "ALSO NONE"))
    scan = scan_of(tmp_path, (D1, SHA_A, first), (D2, SHA_B, second))
    assert scan.pibs == 2
    assert scan.distinct == 3  # A0001, A0002, A0003
    assert scan.hits == 1
    assert scan.changed_e == 0


def test_an_id_with_different_e_text_on_different_days_is_counted(tmp_path):
    first = pib(("A0001", "FIRST WORDING"), ("A0002", "SAME"))
    second = pib(("A0001", "SECOND WORDING"), ("A0002", "SAME"))
    scan = scan_of(tmp_path, (D1, SHA_A, first), (D2, SHA_B, second))
    assert scan.distinct == 2
    assert scan.changed_e == 1


def test_a_whitespace_only_change_is_not_a_different_e_text(tmp_path):
    scan = scan_of(tmp_path, (D1, SHA_A, pib(("A0001", "ONE TWO"))), (D2, SHA_B, pib(("A0001", "ONE\n TWO"))))
    assert scan.changed_e == 0


def test_a_pib_shared_by_several_dates_is_read_once(tmp_path):
    (tmp_path / f"{SHA_A}.xml").write_text(pib(("A0001", "ONE")), encoding="utf-8")
    scan = scan_script.scan_archive({D1: SHA_A, D2: SHA_A, D3: None}, tmp_path)
    assert scan.pibs == 1 and scan.distinct == 1


def test_a_fiction_number_is_a_hit_and_not_a_miss(tmp_path):
    scan = scan_of(tmp_path, (D1, SHA_A, pib(("A0001", f"FOR INFO CTC {FICTION_MOBILE}. OR {FICTION_LANDLINE}."))))
    assert (scan.distinct, scan.hits) == (1, 1)  # one NOTAM, however many numbers it holds
    assert scan.misses == [] and scan.over_matches == []


def test_a_constructed_miss_shape_is_listed(tmp_path):
    miss = "0044 (0)1632 960147"
    scan = scan_of(tmp_path, (D2, SHA_A, pib(("A0007", f"CTC FMP: {miss}."))))
    assert scan.hits == 0
    assert [(c.notam_id, c.day, c.text) for c in scan.misses] == [("A7/26", D2, miss)]
    assert scan_script.mask(miss) == "#### (#)#### ######"


@pytest.mark.parametrize(
    ("miss", "shape"),
    [("(+44)1632 960149", "(+##)#### ######"), ("(01632) 960150", "(#####) ######")],
)
def test_a_miss_candidate_keeps_the_bracket_before_a_plus_or_digit(tmp_path, miss, shape):
    scan = scan_of(tmp_path, (D2, SHA_A, pib(("A0007", f"CTC FMP: {miss}."))))
    assert [c.text for c in scan.misses] == [miss]
    assert scan_script.mask(scan.misses[0].text) == shape


def test_a_miss_candidate_is_found_after_redaction_so_a_caught_number_is_not_one(tmp_path):
    scan = scan_of(tmp_path, (D1, SHA_A, pib(("A0001", f"CTC {FICTION_LANDLINE}. ALSO 0044 (0)1632 960147."))))
    assert [c.text for c in scan.misses] == ["0044 (0)1632 960147"]


@pytest.mark.parametrize(
    "item_e",
    [
        "WI 0.5NM RADIUS OF 534732N 0000459W. TOWER 128.925 OR 121.500. SEE AR-2026-10264/01.",
        "PSN 5348N 00005W. AREA 534732N\n0000459W - 534800N 0000500W",
        "WADDINGTON 1123 \n534732N 0000459W METHRINGHAM 1123 \n534800N 0000459W",
        "ERECTED AT 512345.6N 0001234.5E.",
        "OCT 01 02 03 06 07 08 0900-1700",
        "OCT 01 02 09 10 17 0730-1600, NOV 03 04 10 13 14 0800-1700",
        "FROM 2610311200 TO 2610311800",
        "AD OPR HR:\nMON      0600-2200\n05      0800-1700\n06 0600-1900\n07 0800-1700",
        "REPLACES H7171/26 AND A0123/26. MAX 700FT AGL, FL195.",
    ],
)
def test_a_coordinate_frequency_reference_or_schedule_is_neither_a_hit_nor_a_candidate(tmp_path, item_e):
    scan = scan_of(tmp_path, (D1, SHA_A, pib(("A0001", item_e))))
    assert (scan.hits, scan.misses, scan.over_matches) == (0, [], [])


@pytest.mark.parametrize(
    ("item_e", "kind"),
    [
        ("PSN 01632960789N 0000459W", "N/S/E/W right after the match"),
        ("PSN 534732N 01632960789 0000459W", "N/S/E/W right after a digit, before the match"),
        ("CODE 01632960789.5 USED", "a . between digits"),
        ("SEE AR-01632960789 FOR INFO", "AR- reference"),
        ("REPLACES A0123/26 01632960789 NOW", "next to a NOTAM id"),
        ("SEE 01632960789 A0123/26 REPLACED", "next to a NOTAM id"),
    ],
)
def test_a_detector_match_beside_a_coordinate_reference_or_id_is_an_over_match_candidate(tmp_path, item_e, kind):
    scan = scan_of(tmp_path, (D1, SHA_A, pib(("A0001", item_e))))
    assert [c.kind for c in scan.over_matches] == [kind]
    assert scan.over_matches[0].text == "01632960789"


def test_a_plain_number_in_a_sentence_is_not_an_over_match(tmp_path):
    scan = scan_of(tmp_path, (D1, SHA_A, pib(("A0001", f"FOR INFO CTC {FICTION_MOBILE}. SEE A0123/26 AND CTC {FICTION_LANDLINE} NOW."))))
    assert scan.over_matches == []
    assert scan.hits == 1


def test_a_cached_pib_that_does_not_parse_is_reported_and_the_others_carry_on(tmp_path):
    good = pib(("A0001", f"CTC {FICTION_MOBILE}."))
    (tmp_path / f"{SHA_B}.xml").write_text(good[:150], encoding="utf-8")  # truncated
    (tmp_path / f"{SHA_A}.xml").write_text(good, encoding="utf-8")
    scan = scan_script.scan_archive({D1: SHA_B, D2: SHA_A}, tmp_path)
    assert scan.pibs == 1 and scan.distinct == 1 and scan.hits == 1
    assert len(scan.failures) == 1
    assert "2026-07-01" in scan.failures[0] and SHA_B[:12] in scan.failures[0] and "does not parse" in scan.failures[0]


def test_a_missing_cached_pib_is_reported(tmp_path):
    scan = scan_script.scan_archive({D1: SHA_A}, tmp_path)
    assert scan.pibs == 0
    assert "cannot be read" in scan.failures[0] and SHA_A[:12] in scan.failures[0]


def test_run_exits_1_on_a_bad_cached_pib_but_still_prints_the_report_of_the_rest(tmp_path):
    github = FakeGitHub({D1: SHA_A, D2: SHA_B}, {SHA_A: pib(("A0001", "ONE")), SHA_B: pib(("A0002", f"CTC {FICTION_MOBILE}."))})
    assert run_script(github, tmp_path, [D1, D2])[0] == 0
    (tmp_path / f"{SHA_A}.xml").write_text("<Pib><unclosed>", encoding="utf-8")  # corrupted after the fact
    code, out, err = run_script(FakeGitHub({}, {}), tmp_path, [D1, D2])
    assert code == 1
    assert "Distinct NOTAMs: 1" in out and "Hits (NOTAMs with a phone number in item E): 1" in out
    assert "2026-07-01" in err and SHA_A[:12] in err


def test_misses_never_change_the_exit_code(tmp_path):
    github = FakeGitHub({D1: SHA_A}, {SHA_A: pib(("A0001", "CTC 0044 (0)1632 960147."), ("A0002", "PSN 01632960789N"))})
    code, out, _ = run_script(github, tmp_path, [D1])
    assert code == 0
    assert "Miss candidates: 1" in out and "Over-match candidates: 1" in out


# --- report ----------------------------------------------------------------------------------------------------


def candidates(*rows: tuple[str, date, str, str]):
    return [scan_script.Candidate(notam_id, day, text, kind) for notam_id, day, text, kind in rows]


def test_the_report_lists_candidates_by_date_then_id_whatever_the_scan_order():
    rows = [("B0002", D2, "0044 (0)1632 960147", ""), ("A0009", D2, "(01632) 960150", ""), ("C0001", D1, "(+44)1632 960149", "")]
    forward = scan_script.Scan(pibs=2, distinct=9, hits=4, misses=candidates(*rows))
    backward = scan_script.Scan(pibs=2, distinct=9, hits=4, misses=candidates(*reversed(rows)))
    days = {D1: SHA_A, D2: SHA_B}
    report = scan_script.format_report(forward, days)
    assert report == scan_script.format_report(backward, days)
    listed = [line.split()[0:2] for line in report.splitlines() if line.startswith("  ")]
    assert listed == [["2026-07-01", "C0001"], ["2026-07-02", "A0009"], ["2026-07-02", "B0002"]]


def test_the_report_has_the_counts_the_marker_and_the_mask_and_no_long_digit_run():
    scan = scan_script.Scan(
        pibs=2,
        distinct=14,
        hits=5,
        changed_e=1,
        misses=candidates(("A0001", D1, "0044 (0)1632 960147", "")),
        over_matches=candidates(("B0002", D2, "01632960789", "a . between digits")),
    )
    report = scan_script.format_report(scan, {D1: SHA_A, D2: SHA_B, D3: None})
    assert "Dates: 3, with an archive commit: 2, skipped with none: 1" in report
    assert "Distinct commits: 2" in report and f"first {SHA_A} (2026-07-01), last {SHA_B} (2026-07-02)" in report
    assert "PIBs read: 2" in report
    assert "Distinct NOTAMs: 14" in report
    assert "Hits (NOTAMs with a phone number in item E): 5" in report
    assert "Ids with different E text on different days: 1" in report
    assert "Miss candidates: 1\n  2026-07-01  A0001  [phone]  shape #### (#)#### ######\n" in report
    assert "Over-match candidates: 1\n  2026-07-02  B0002  [phone]  shape ###########  (a . between digits)\n" in report
    assert not re.search(r"\d{6,}", report)
    assert "960147" not in report and "960789" not in report


def test_the_formatter_writes_a_fiction_number_passed_as_a_miss_as_the_marker_with_no_long_digit_run():
    scan = scan_script.Scan(distinct=1, hits=0, misses=candidates(("A0001", D1, FICTION_MOBILE.replace(" ", ""), ""), ("A0002", D1, FICTION_MOBILE, "")))
    report = scan_script.format_report(scan, {D1: SHA_A})
    assert "[phone]" in report
    assert "#" in report
    assert not re.search(r"\d{6,}", report)
    assert "900123" not in report


def test_a_newline_in_a_number_is_written_as_backslash_n_in_its_mask():
    assert scan_script.mask("01632\n960124") == "#####\\n######"
    assert scan_script.mask("01632\r\n960124") == "#####\\n######"


def test_a_candidate_never_shows_its_number_in_its_repr():
    assert "960124" not in repr(scan_script.Candidate("A0001", D1, FICTION_LANDLINE))


def test_an_empty_scan_reports_zero_candidates():
    report = scan_script.format_report(scan_script.Scan(), {})
    assert "Miss candidates: 0\nOver-match candidates: 0\n" in report


def test_two_runs_over_one_cache_print_byte_identical_reports(tmp_path):
    shas = {D1: SHA_A, D2: SHA_B, D3: SHA_A}
    pibs = {
        SHA_A: pib(("A0001", f"CTC {FICTION_MOBILE}."), ("A0002", "CTC 0044 (0)1632 960147."), ("A0003", "PSN 01632960789N")),
        SHA_B: pib(("A0001", f"CTC {FICTION_MOBILE}."), ("A0004", "(01632) 960150")),
    }
    code, first, _ = run_script(FakeGitHub(shas, pibs), tmp_path, [D1, D2, D3])
    code_again, second, _ = run_script(FakeGitHub({}, {}), tmp_path, [D1, D2, D3])
    assert code == code_again == 0
    assert first == second
    assert "Miss candidates: 2" in first


def test_no_number_reaches_stdout_or_stderr_of_a_whole_run(tmp_path, capsys):
    item_e = f"CTC {FICTION_MOBILE}. OR {FICTION_LANDLINE}. ALSO 0044 (0)1632 960147 AND (+44)1632 960149"
    code, out, err = run_script(FakeGitHub({D1: SHA_A}, {SHA_A: pib(("A0001", item_e))}), tmp_path, [D1])
    printed = out + err + "".join(capsys.readouterr())
    assert code == 0
    assert not re.search(r"\d{6,}", printed)
    for digits in ("900123", "960124", "960147", "960149"):
        assert digits not in printed


# --- F5: nothing this slice adds holds a real number ------------------------------------------------------------


def _is_fiction(number: str) -> bool:
    digits = re.sub(r"[\s()]+", "", number.replace("(0)", ""))
    digits = "0" + digits[3:] if digits.startswith("+44") else "0" + digits[4:] if digits.startswith("0044") else digits
    return digits.startswith(("07700900", "01632960", "01134960", "02079460"))


@pytest.mark.parametrize("path", [SCRIPT, Path(__file__), Path(__file__).with_name("test_redact_archive_shapes.py")], ids=lambda p: p.name)
def test_every_number_the_detector_finds_in_the_files_this_slice_adds_is_fiction_range(path):
    text = path.read_text(encoding="utf-8")
    found = list(redact_phones(text)[1].values())
    assert [number for number in found if not _is_fiction(number)] == []
