"""The Ops section's numbers (5a PR-10.F1, G16): latency percentiles through the reader eval's `nearest_rank`, token sums, stored readings skipped, wall-clock and the reader-failed count."""

from decimal import Decimal
from fractions import Fraction

import pytest
from test_eval_report import PRICE, make, run, section

from evals.agent import report
from evals.agent.pricing import COST_LABEL, Price, cost
from evals.agent.report import Ops, ReportError, ReportRun, ops_of, reader_tokens, render, sum_tokens
from evals.reader_eval import report as reader_report
from flysafelz.reader.contract import ReaderResult, ReaderStatus, TokenUsage


def usage(i: int, o: int) -> TokenUsage:
    return TokenUsage(input_tokens=i, output_tokens=o, total_tokens=i + o)


def result(status: ReaderStatus, tokens: TokenUsage | None, notam_id: str = "A0001/26") -> ReaderResult:
    return ReaderResult(notam_id=notam_id, reading=None, status=status, failed_checks=[], model="reader-model", prompt_version="v1", latency_s=0.5, tokens=tokens)


def timed(latencies, **fields) -> list[ReportRun]:
    return [run(f"ex_{n}", 1, {}, latency_s=latency, **fields) for n, latency in enumerate(latencies)]


def test_the_percentile_function_is_the_reader_evals_not_a_second_one():
    assert report.nearest_rank is reader_report.nearest_rank


def test_latency_p50_and_p95_are_nearest_rank_over_the_scored_runs():
    latencies = [float(n) for n in range(1, 21)]  # 1..20 seconds
    ops = ops_of(make(timed(latencies), planned_runs=20, reps=1))
    assert (ops.latency_p50, ops.latency_p95, ops.latency_runs) == (10.0, 19.0, 20)
    assert ops.latency_p50 == reader_report.nearest_rank(latencies, Fraction(1, 2)) and ops.latency_p95 == reader_report.nearest_rank(latencies, Fraction(19, 20))


def test_an_errored_run_is_not_a_latency_sample():
    runs = [*timed([1.0, 2.0, 3.0]), ReportRun("ex_e", 1, errored=True, latency_s=900.0, error="x")]
    assert ops_of(make(runs, planned_runs=4)).latency_p95 == 3.0


def test_one_run_gives_its_latency_for_both_percentiles_and_none_give_none():
    one = ops_of(make(timed([4.5]), planned_runs=1))
    assert (one.latency_p50, one.latency_p95) == (4.5, 4.5)
    empty = ops_of(make([], planned_runs=3))
    assert (empty.latency_p50, empty.latency_p95, empty.latency_runs) == (None, None, 0)


def test_the_agent_tokens_are_summed_from_each_runs_usage_and_an_errored_runs_calls_count_too():
    runs = [*timed([1.0, 1.0], usage=usage(1000, 100)), ReportRun("ex_e", 1, errored=True, usage=usage(300, 30), error="x"), *timed([1.0], usage=None)]
    ops = ops_of(make(runs, planned_runs=4))
    assert ops.agent_tokens == TokenUsage(input_tokens=2300, output_tokens=230, total_tokens=2530)
    assert ops.tokens == ops.agent_tokens and ops.reader_tokens is None


def test_no_usage_anywhere_is_none_never_a_zero():
    ops = ops_of(make(timed([1.0, 2.0]), planned_runs=2))
    assert (ops.agent_tokens, ops.reader_tokens, ops.tokens) == (None, None, None)
    ops_section = section(render(make(timed([1.0, 2.0]), planned_runs=2)), "Ops")
    assert "tokens from response usage: input n/a, output n/a" in ops_section and "no total is shown (never a 0)" in ops_section
    assert "est. cost: n/a (no usage)" in ops_section and "$0" not in ops_section


def test_stored_readings_made_no_call_and_are_skipped_not_counted_as_zero():
    results = [result(ReaderStatus.READ, usage(400, 40), "A1"), result(ReaderStatus.STORED, None, "A2"), result(ReaderStatus.REPAIRED, usage(600, 60), "A3"), result(ReaderStatus.STORED, None, "A4")]
    assert reader_tokens(results) == [usage(400, 40), usage(600, 60)]
    ops = ops_of(make(timed([1.0], usage=usage(1000, 100)), planned_runs=1, reader_results=results))
    assert ops.reader_tokens == TokenUsage(input_tokens=1000, output_tokens=100, total_tokens=1100)
    assert ops.tokens == TokenUsage(input_tokens=2000, output_tokens=200, total_tokens=2200)


def test_only_stored_readings_give_no_reader_tokens_and_no_error():
    assert reader_tokens([result(ReaderStatus.STORED, None)]) == []
    assert ops_of(make(timed([1.0], usage=usage(5, 5)), planned_runs=1, reader_results=[result(ReaderStatus.STORED, None)])).reader_tokens is None


def test_an_unavailable_reader_that_returned_nothing_made_no_usable_call_and_is_skipped():
    assert reader_tokens([result(ReaderStatus.UNAVAILABLE, None), result(ReaderStatus.UNAVAILABLE, usage(10, 0))]) == [usage(10, 0)]


@pytest.mark.parametrize("status", [ReaderStatus.READ, ReaderStatus.REPAIRED, ReaderStatus.FAILED])
def test_a_reader_call_without_usage_is_the_e4_error_never_a_zero(status):
    with pytest.raises(ReportError, match=r"no token usage on the first call.*A9"):
        reader_tokens([result(status, None, "A9")])


def test_sum_tokens_adds_each_field_and_gives_none_for_no_parts():
    assert sum_tokens([usage(1, 2), usage(10, 20)]) == TokenUsage(input_tokens=11, output_tokens=22, total_tokens=33)
    assert sum_tokens([]) is None


def test_the_reader_failed_count_and_the_wall_clock_are_carried_into_ops():
    runs = [run("ex_a", 1, {}, reader_failed=2), run("ex_a", 2, {}, reader_failed=1), ReportRun("ex_b", 1, errored=True, reader_failed=5, error="x")]
    ops = ops_of(make(runs, planned_runs=3, wall_clock_s=3725.0))
    assert ops.reader_failed == 3  # an errored run has no answer, so none of its NOTAMs became UNVERIFIED
    assert ops.wall_clock_s == 3725.0 and isinstance(ops, Ops)


def test_calls_without_usage_are_counted_and_said_not_hidden():
    runs = [run("ex_a", 1, {}, usage=usage(10, 1), calls_without_usage=2)]
    assert ops_of(make(runs, planned_runs=1)).calls_without_usage == 2
    assert "2 model call(s) reported no usage and are not in the totals" in section(render(make(runs, planned_runs=1)), "Ops")


def test_the_ops_section_shows_latency_tokens_cost_reader_failures_and_wall_clock():
    runs = timed([1.0, 2.0, 3.0, 4.0], usage=usage(500_000, 50_000))
    text = section(render(make(runs, planned_runs=4, wall_clock_s=3725.0)), "Ops")
    assert "latency per example run (p50 / p95, nearest rank, 4 scored runs): 2.00 s / 4.00 s" in text
    assert "tokens from response usage: input 2,000,000, output 200,000" in text
    assert "est. cost at list price; runs used a subscription bridge: $9.0000" in text
    assert "source https://provider.invalid/pricing, read 2026-11-09" in text
    assert "reader calls that failed and became UNVERIFIED: 0" in text
    assert "wall-clock time: 1 h 2 min 5 s" in text
    assert cost(PRICE, 2_000_000, 200_000) == Decimal("9.00")


def test_a_v1_run_whose_reader_made_no_call_says_so_and_does_not_count_it_as_zero():
    runs = timed([1.0], usage=usage(10, 1))
    text = section(render(make(runs, planned_runs=1)), "Ops")
    assert "agent model: input 10, output 1; reader model: no call came back" in text and "the judge's calls are not in these totals" in text
    assert "reader model: input 0" not in text


def test_the_readers_usage_on_each_run_is_summed_into_the_reader_tokens_and_the_total():
    runs = [
        run("ex_a", 1, {}, usage=usage(1000, 100), reader_usage=usage(200, 20)),
        run("ex_a", 2, {}, usage=usage(1000, 100), reader_usage=None),  # rep 2 read only stored readings: no call, nothing to add
        run("ex_b", 1, {}, usage=usage(1000, 100), reader_usage=usage(300, 30)),
    ]
    ops = ops_of(make(runs, planned_runs=3))
    assert ops.reader_tokens == usage(500, 50) and ops.agent_tokens == usage(3000, 300) and ops.tokens == usage(3500, 350)


def test_an_errored_runs_reader_calls_were_paid_for_and_are_counted():
    runs = [run("ex_a", 1, {}, usage=usage(10, 1)), ReportRun("ex_e", 1, errored=True, usage=usage(5, 1), reader_usage=usage(40, 4), error="x")]
    assert ops_of(make(runs, planned_runs=2)).reader_tokens == usage(40, 4)


def test_run_usage_and_reader_results_are_added_together_and_a_stored_result_still_adds_nothing():
    runs = [run("ex_a", 1, {}, usage=usage(100, 10), reader_usage=usage(200, 20))]
    both = make(runs, planned_runs=1, reader_results=[result(ReaderStatus.READ, usage(1, 1)), result(ReaderStatus.STORED, None)])
    assert ops_of(both).reader_tokens == usage(201, 21)


READER_PRICE = Price("reader-model", Decimal("0.25"), Decimal("1.25"), "https://other.invalid/prices", "2026-11-01")


def test_the_cost_prices_each_model_at_its_own_price_exactly_and_names_both():
    runs = [run("ex_a", 1, {}, usage=usage(2_000_000, 100_000), reader_usage=usage(4_000_000, 400_000))]
    text = section(render(make(runs, planned_runs=1, reader_price=READER_PRICE)), "Ops")
    agent_cost, reader_cost = cost(PRICE, 2_000_000, 100_000), cost(READER_PRICE, 4_000_000, 400_000)
    assert (agent_cost, reader_cost) == (Decimal("7.5"), Decimal("1.5")) and f"{COST_LABEL}: ${agent_cost + reader_cost:.4f} (" in text
    assert "agent agent-model: $3.00 per M input, $15.00 per M output; source https://provider.invalid/pricing, read 2026-11-09" in text
    assert "reader reader-model: $0.25 per M input, $1.25 per M output; source https://other.invalid/prices, read 2026-11-01" in text
    assert "agent model: input 2,000,000, output 100,000; reader model: input 4,000,000, output 400,000" in text
    assert "tokens from response usage: input 6,000,000, output 500,000" in text


def test_reader_tokens_without_a_reader_price_say_n_a_never_a_cost_from_the_agents_price():
    runs = [run("ex_a", 1, {}, usage=usage(1000, 100), reader_usage=usage(200, 20))]
    text = section(render(make(runs, planned_runs=1, reader_price=None)), "Ops")
    assert "n/a: no price list entry was given for the reader model's cost" in text and COST_LABEL not in text and "$0" not in text


def test_a_run_with_no_reader_call_is_priced_from_the_agents_price_alone():
    text = section(render(make(timed([1.0], usage=usage(500_000, 50_000)), planned_runs=1, reader_price=READER_PRICE)), "Ops")
    assert f"{COST_LABEL}: $2.2500 (agent-model: $3.00 per M input" in text and "reader-model" not in text


def test_with_reader_results_the_ops_show_the_agent_and_the_reader_separately():
    runs = timed([1.0], usage=usage(1000, 100))
    text = section(render(make(runs, planned_runs=1, reader_results=[result(ReaderStatus.READ, usage(200, 20))])), "Ops")
    assert "agent model: input 1,000, output 100; reader model: input 200, output 20" in text
    assert "tokens from response usage: input 1,200, output 120" in text


def test_v0_has_no_reader_line_and_no_reader_note():
    text = section(render(make(timed([1.0], usage=usage(10, 1)), planned_runs=1, agent="v0")), "Ops")
    assert "reader calls" not in text and "reader's calls" not in text


def test_a_price_missing_from_the_input_is_said_not_priced_as_zero():
    text = section(render(make(timed([1.0], usage=usage(10, 1)), planned_runs=1, price=None)), "Ops")
    assert "n/a: no price list entry was given for the cost" in text and "$0" not in text
