"""List prices for the cost line of a results file (5a PR-10.F1, G16).

    PRICING_PATH                   evals/agent/pricing.yaml
    load_prices(path)              {model: Price} for the whole file; PricingError names the file and the model for anything wrong
    price_for(model, path)         the Price of one model; PricingError when the file has no entry for it
    Price(model, input_per_m, output_per_m, url, date)   dollars (Decimal) per million input and output tokens, and where and when they were read
    cost(price, input_tokens, output_tokens) -> Decimal    input / 1e6 * input_per_m + output / 1e6 * output_per_m, exact (no float, no rounding)
    cost_line(price, in, out)      the Ops cost line for one model; cost_line_by_role(parts) the same for several (agent and reader), each at its own price
    COST_LABEL                     the label of the cost line, word for word

The file is `models: {<model name>: {input_per_m, output_per_m, url, date}}`: one entry per model, the name as the endpoint is told it (AGENT_MODEL, READER_MODEL ...). Prices are
list prices: the runs go through a subscription bridge, so nothing like this was charged. A model with no entry, or an entry with no URL or no date, is an error that names the model
and the file. It is never a cost of 0 and never a blank, and it is checked before a run starts, so a run is not paid for and then left without a cost line.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

PRICING_PATH = Path(__file__).with_name("pricing.yaml")
COST_LABEL = "est. cost at list price; runs used a subscription bridge"
MILLION = Decimal(1_000_000)
PRICE_FIELDS = ("input_per_m", "output_per_m")


class PricingError(ValueError):
    """The price list is unusable for a model. The message names the file and the model."""


@dataclass(frozen=True)
class Price:
    model: str
    input_per_m: Decimal  # dollars per million input tokens
    output_per_m: Decimal  # dollars per million output tokens
    url: str  # where the price was read
    date: str  # ISO date it was read


def _fail(path: Path, model: str, problem: str) -> PricingError:
    return PricingError(f"{path}: model {model!r}: {problem}")


def _amount(path: Path, model: str, name: str, value: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise _fail(path, model, f"{name} must be a number of dollars per million tokens, not {value!r}")
    try:
        amount = Decimal(str(value).strip())
    except InvalidOperation:
        raise _fail(path, model, f"{name} must be a number of dollars per million tokens, not {value!r}") from None
    if not amount.is_finite() or amount < 0:
        raise _fail(path, model, f"{name} must be zero or more dollars per million tokens, not {value!r}")
    return amount


def _text(path: Path, model: str, name: str, value: Any) -> str:
    if isinstance(value, date):  # YAML reads 2026-11-09 as a date
        return value.isoformat()
    if not isinstance(value, str) or not value.strip():
        raise _fail(path, model, f"{name} is missing (every entry needs the source url and the date the price was read)")
    return value.strip()


def _price(path: Path, model: str, entry: Any) -> Price:
    if not isinstance(entry, dict):
        raise _fail(path, model, "the entry must be a mapping of input_per_m, output_per_m, url and date")
    for name in (*PRICE_FIELDS, "url", "date"):
        if entry.get(name) is None:
            raise _fail(path, model, f"{name} is missing" + (" (every entry needs the source url and the date the price was read)" if name in ("url", "date") else ""))
    return Price(
        model,
        _amount(path, model, "input_per_m", entry["input_per_m"]),
        _amount(path, model, "output_per_m", entry["output_per_m"]),
        _text(path, model, "url", entry["url"]),
        _text(path, model, "date", entry["date"]),
    )


def _read(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise PricingError(f"{path}: cannot read the price list: {error.strerror or error}") from error
    except yaml.YAMLError as error:
        raise PricingError(f"{path}: not valid YAML: {error}") from error
    models = data.get("models") if isinstance(data, dict) else None
    if models is None:
        return {}
    if not isinstance(models, dict):
        raise PricingError(f"{path}: `models` must be a mapping of model name to price entry")
    return models


def load_prices(path: Path = PRICING_PATH) -> dict[str, Price]:
    """Every entry of the file, each checked. PricingError for the first that is wrong."""
    return {str(model): _price(path, str(model), entry) for model, entry in _read(path).items()}


def price_for(model: str, path: Path = PRICING_PATH) -> Price:
    """The price of `model`. PricingError, naming the model and the file, when the file has no entry for it or the entry is incomplete."""
    entries = _read(path)
    if model not in entries:
        known = ", ".join(sorted(map(str, entries))) or "none"
        raise PricingError(f"{path}: no price for model {model!r} (entries: {known}); add its list price, source url and date")
    return _price(path, model, entries[model])


def cost(price: Price, input_tokens: int, output_tokens: int) -> Decimal:
    """Dollars at list price, exact: input / 1e6 * input_per_m + output / 1e6 * output_per_m."""
    return Decimal(input_tokens) / MILLION * price.input_per_m + Decimal(output_tokens) / MILLION * price.output_per_m


def _rate(amount: Decimal) -> str:
    """A rate with at least two decimals and no digit lost: 3 -> 3.00, 0.075 -> 0.075."""
    return f"{amount:f}" if -amount.as_tuple().exponent >= 2 else f"{amount.quantize(Decimal('0.01')):f}"  # type: ignore[operator]  # a finite Decimal has an int exponent


def _rates(price: Price) -> str:
    return f"${_rate(price.input_per_m)} per M input, ${_rate(price.output_per_m)} per M output; source {price.url}, read {price.date}"


def cost_line(price: Price, input_tokens: int, output_tokens: int) -> str:
    """The Ops line: the fixed label, the cost, the rates and where and when they were read."""
    dollars = cost(price, input_tokens, output_tokens)
    return f"{COST_LABEL}: ${dollars:.4f} ({price.model}: {_rates(price)})"


def cost_line_by_role(parts: Sequence[tuple[str, Price, int, int]]) -> str:
    """The Ops line for a run that used several models, each at its own price: (role, price, input tokens, output tokens) per part.

    The cost is the exact sum of the parts, under the same label; each part names its role, its model, its rates, its source and its date.
    """
    dollars = sum((cost(price, tokens_in, tokens_out) for _role, price, tokens_in, tokens_out in parts), Decimal(0))
    return f"{COST_LABEL}: ${dollars:.4f} (" + "; ".join(f"{role} {price.model}: {_rates(price)}" for role, price, _in, _out in parts) + ")"
