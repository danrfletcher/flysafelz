"""The arguments of the three data tools are only `start` and `end` (Q8, F2): no position, zone, clock or case, and the runtime is never shown to the model."""

import pytest

from flysafelz.agent.sources import SourceCache
from flysafelz.agent.tools.airspace_tool import build_get_airspace
from flysafelz.agent.tools.notams_tool import build_get_notams
from flysafelz.agent.tools.weather_tool import build_get_weather

BUILDERS = {"get_weather": build_get_weather, "get_airspace": build_get_airspace, "get_notams": build_get_notams}
FORBIDDEN = {"lat", "lon", "latitude", "longitude", "tz", "timezone", "now", "case", "runtime"}


@pytest.fixture(params=sorted(BUILDERS))
def tool(request):
    return BUILDERS[request.param](SourceCache())


def test_the_tool_is_named_for_the_spec(tool):
    assert tool.name in BUILDERS


def test_the_model_facing_arguments_are_only_start_and_end(tool):
    assert set(tool.tool_call_schema.model_json_schema()["properties"]) == {"start", "end"}
    assert set(tool.tool_call_schema.model_fields) == {"start", "end"}


def test_both_are_optional_so_the_default_window_can_be_asked_for(tool):
    schema = tool.tool_call_schema.model_json_schema()
    assert not schema.get("required")


def test_the_args_model_refuses_anything_else(tool):
    from pydantic import ValidationError

    for extra in FORBIDDEN - {"runtime"}:
        with pytest.raises(ValidationError):
            tool.args_schema.model_validate({extra: "x"})  # type: ignore[union-attr]
    assert set(tool.args_schema.model_fields) - FORBIDDEN == {"start", "end"}  # type: ignore[union-attr]
