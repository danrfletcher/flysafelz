"""The price list and the cost line (5a PR-10.F1, G16): exact Decimal arithmetic, the fixed label, and a clear error for every unusable entry."""

from decimal import Decimal
from pathlib import Path

import pytest

from evals.agent import pricing
from evals.agent.pricing import COST_LABEL, Price, PricingError, cost, cost_line, cost_line_by_role, load_prices, price_for

GOOD = """
models:
  model-a:
    input_per_m: 3.00
    output_per_m: 15.00
    url: https://provider.invalid/pricing
    date: 2026-11-09
  model-b:
    input_per_m: "0.25"
    output_per_m: "1.25"
    url: https://other.invalid/prices
    date: "2026-11-01"
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "pricing.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_label_is_exactly_the_specified_words():
    assert COST_LABEL == "est. cost at list price; runs used a subscription bridge"


def test_cost_is_input_over_a_million_times_its_price_plus_output_over_a_million_times_its_price_exactly(tmp_path):
    price = price_for("model-a", write(tmp_path, GOOD))
    assert cost(price, 1_234_567, 89_012) == Decimal(1_234_567) / Decimal(1_000_000) * Decimal("3.00") + Decimal(89_012) / Decimal(1_000_000) * Decimal("15.00")
    assert cost(price, 1_000_000, 1_000_000) == Decimal("18.00")
    assert cost(price, 0, 0) == Decimal(0)
    assert isinstance(cost(price, 7, 7), Decimal)  # never a float


def test_cost_line_carries_the_label_the_cost_the_rates_the_url_and_the_date(tmp_path):
    price = price_for("model-a", write(tmp_path, GOOD))
    line = cost_line(price, 2_000_000, 100_000)
    assert line.startswith(COST_LABEL + ": $7.5000 (")
    assert "model-a: $3.00 per M input, $15.00 per M output" in line
    assert "source https://provider.invalid/pricing, read 2026-11-09" in line


def test_cost_line_by_role_adds_the_exact_costs_and_names_each_models_rates_source_and_date(tmp_path):
    prices = load_prices(write(tmp_path, GOOD))
    line = cost_line_by_role([("agent", prices["model-a"], 2_000_000, 100_000), ("reader", prices["model-b"], 4_000_000, 400_000)])
    assert cost(prices["model-a"], 2_000_000, 100_000) + cost(prices["model-b"], 4_000_000, 400_000) == Decimal("9.0")
    assert line.startswith(COST_LABEL + ": $9.0000 (agent model-a: $3.00 per M input, $15.00 per M output; source https://provider.invalid/pricing, read 2026-11-09; ")
    assert line.endswith("reader model-b: $0.25 per M input, $1.25 per M output; source https://other.invalid/prices, read 2026-11-01)")


def test_a_yaml_date_and_a_quoted_date_both_read_as_an_iso_date(tmp_path):
    prices = load_prices(write(tmp_path, GOOD))
    assert (prices["model-a"].date, prices["model-b"].date) == ("2026-11-09", "2026-11-01")
    assert prices["model-b"] == Price("model-b", Decimal("0.25"), Decimal("1.25"), "https://other.invalid/prices", "2026-11-01")


def test_a_model_with_no_entry_is_an_error_naming_the_model_and_the_file(tmp_path):
    path = write(tmp_path, GOOD)
    with pytest.raises(PricingError) as caught:
        price_for("model-z", path)
    assert "'model-z'" in str(caught.value) and str(path) in str(caught.value) and "model-a, model-b" in str(caught.value)


def test_an_empty_price_list_is_a_missing_entry_never_a_zero_cost(tmp_path):
    with pytest.raises(PricingError, match="no price for model 'model-a'"):
        price_for("model-a", write(tmp_path, "models: {}\n"))
    with pytest.raises(PricingError, match="entries: none"):
        price_for("model-a", write(tmp_path, ""))


@pytest.mark.parametrize("missing", ["url", "date"])
def test_an_entry_without_its_url_or_date_is_an_error_naming_the_model_and_the_file(tmp_path, missing):
    lines = [line for line in GOOD.splitlines() if not line.strip().startswith(f"{missing}:") or "other" in line or "2026-11-01" in line]
    path = write(tmp_path, "\n".join(lines))
    with pytest.raises(PricingError) as caught:
        price_for("model-a", path)
    assert f"model 'model-a': {missing} is missing" in str(caught.value) and str(path) in str(caught.value)


@pytest.mark.parametrize("field", ["url", "date"])
def test_a_blank_url_or_date_is_missing_too(tmp_path, field):
    path = write(tmp_path, GOOD.replace("https://provider.invalid/pricing", '"  "') if field == "url" else GOOD.replace("date: 2026-11-09", 'date: ""'))
    with pytest.raises(PricingError, match=f"{field} is missing"):
        price_for("model-a", path)


@pytest.mark.parametrize("value", ["-1", "'abc'", "true", "[1]", "nan", "inf"])
def test_a_price_that_is_not_a_non_negative_number_is_an_error(tmp_path, value):
    with pytest.raises(PricingError, match="input_per_m"):
        price_for("model-a", write(tmp_path, GOOD.replace("input_per_m: 3.00", f"input_per_m: {value}")))


def test_an_entry_missing_a_price_is_an_error(tmp_path):
    with pytest.raises(PricingError, match="output_per_m is missing"):
        price_for("model-a", write(tmp_path, GOOD.replace("    output_per_m: 15.00\n", "")))


def test_a_file_that_is_not_a_mapping_of_entries_is_an_error(tmp_path):
    with pytest.raises(PricingError, match="`models` must be a mapping"):
        load_prices(write(tmp_path, "models: [1, 2]\n"))
    with pytest.raises(PricingError, match="must be a mapping of input_per_m"):
        load_prices(write(tmp_path, "models:\n  model-a: 3\n"))
    with pytest.raises(PricingError, match="not valid YAML"):
        load_prices(write(tmp_path, "models: [unclosed\n"))


def test_a_missing_file_is_a_pricing_error_naming_it(tmp_path):
    path = tmp_path / "absent.yaml"
    with pytest.raises(PricingError, match="cannot read the price list"):
        price_for("model-a", path)


def test_the_committed_price_list_loads_and_every_price_names_its_source_page_and_date():
    prices = load_prices(pricing.PRICING_PATH)  # Dan's entries, read from the provider's page; load_prices refuses one with no url or date
    assert prices and all(p.url.startswith("https://") and p.date for p in prices.values())
