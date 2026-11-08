"""The container check's script (5a PR-6.F1): canned dev runs through every evaluator, and the E10 mention cases."""

import json

from evals.agent import score_form_demo


def printed(out):
    """{(example, run): {key: parsed value or None}} from the script's output."""
    found, current = {}, None
    for line in out.splitlines():
        if line.startswith("== "):
            name, run = line[3:].split(" (")
            current = found.setdefault((name, run.rstrip(")")), {})
        elif current is not None and ": " in line:
            key, value = line.split(": ", 1)
            current[key] = None if value.startswith("None") else json.loads(value)
    return found


def test_main_prints_a_good_and_a_bad_run_for_each_of_the_four_dev_examples(capsys):
    score_form_demo.main([])
    got = printed(capsys.readouterr().out)
    assert set(got) == {(name, run) for name in score_form_demo.EXAMPLES for run in ("good", "bad")}


def test_the_good_runs_pass_every_pr_6_score_that_applies_and_the_bad_runs_fail_what_they_read(capsys):
    score_form_demo.main([])
    got = printed(capsys.readouterr().out)
    pr6 = ("required_tools", "forbidden_tools", "step_cap", "guard_pass", "decision_first", "cited", "window_stated", "recheck_stated", "must_mention", "must_not_mention")
    for (name, run), scores in got.items():
        if run == "good":
            assert all(scores[k] is None or scores[k]["score"] == 1 for k in pr6), name
        else:
            assert scores["guard_pass"]["score"] == 0, name
    assert got[("dev_16_buttermere", "bad")]["required_tools"]["score"] == 0
    assert got[("dev_17_sherburn", "good")]["forbidden_tools"]["score"] == 1
    assert got[("dev_27_sherburn", "good")]["unverified_on_dropped_source"]["score"] == 1


def test_the_inapplicable_scores_print_as_none(capsys):
    score_form_demo.main([])
    out = capsys.readouterr().out
    assert "decision_first: None (does not apply)" in out and "noise_rate: None (does not apply)" in out


def test_mentions_flag_prints_the_e10_cases(capsys):
    score_form_demo.main(["--mentions"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(score_form_demo.MENTION_CASES)
    assert lines[0].startswith("'1500' in 'base at 1,500 ft': True")
    assert lines[1].startswith("'Newcastle CTA 2' in 'NEWCASTLE CTA2': True")
    assert lines[2].startswith("'18:00' in 'until 6pm': False")
    assert lines[6].startswith("'1500' in '1, 500': False") and lines[7].startswith("'NO-GO' in 'no go': False") and lines[8].startswith("'Burn' in 'Sherburn': True")
