"""G2: langgraph.json names the graph factory, the server's way of loading it works, and the factory builds with a fake model."""

import importlib.util
import inspect
import json
from pathlib import Path

import pytest
from fakes import FakeToolChatModel
from langgraph.graph.state import CompiledStateGraph

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "langgraph.json"
ENTRY = "./src/flysafelz/agent/graph.py:build_graph"


def config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def load_factory(entry: str):
    """The function a `graphs` entry names, loaded from its file as the server loads it (not by package name)."""
    relative, name = entry.rsplit(":", 1)
    path = ROOT / relative
    assert path.is_file(), f"{entry} names no file"
    spec = importlib.util.spec_from_file_location("langgraph_json_graph_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def test_the_file_parses_and_has_exactly_the_four_keys():
    assert set(config()) == {"graphs", "env", "dependencies", "http"}  # `http` mounts the map routes (ticket 8) and sets the UI's CORS


def test_the_graphs_entry_points_at_the_factory():
    assert config()["graphs"] == {"agent": ENTRY}


def test_the_env_file_is_dotenv_and_the_dependency_is_this_project():
    assert config()["env"] == ".env"
    assert config()["dependencies"] == ["."]


def test_the_path_resolves_to_a_factory_not_a_built_graph():
    factory = load_factory(config()["graphs"]["agent"])
    assert callable(factory) and not isinstance(factory, CompiledStateGraph)
    assert inspect.isfunction(factory) and factory.__name__ == "build_graph"


def test_the_factory_builds_a_compiled_graph_with_a_fake_model():
    factory = load_factory(config()["graphs"]["agent"])
    assert isinstance(factory(model=FakeToolChatModel()), CompiledStateGraph)


def test_the_factory_builds_with_no_model_and_no_agent_setting():
    factory = load_factory(config()["graphs"]["agent"])
    assert isinstance(factory(), CompiledStateGraph)


def test_the_server_calls_a_one_parameter_factory_with_its_run_configuration_and_the_factory_takes_it():
    """langgraph_api reads the signature: one parameter that is not a ServerRuntime is passed the run configuration."""
    from langgraph_api._factory_utils import _classify_factory

    factory = load_factory(config()["graphs"]["agent"])
    arguments = _classify_factory(factory)({"configurable": {"thread_id": "t"}}, object())
    assert list(arguments) == ["model"]
    assert isinstance(factory(**arguments), CompiledStateGraph)


@pytest.mark.parametrize("bad", ["./src/flysafelz/agent/nope.py:build_graph"])
def test_a_path_that_names_no_file_is_caught_by_the_loader(bad):
    with pytest.raises(AssertionError):
        load_factory(bad)


# --- ticket 8: the map routes are mounted as http.app, and the server's own CORS is the two dev origins --------------------------------------

APP_ENTRY = "./src/flysafelz/mapdata/app.py:app"
DEV_ORIGINS = ["http://localhost:5173", "http://localhost:4173"]


def test_http_app_names_the_map_routes_app():
    assert config()["http"]["app"] == APP_ENTRY


def test_the_http_app_entry_loads_as_the_server_loads_it_and_is_a_starlette_app_with_the_four_routes():
    from starlette.applications import Starlette
    from starlette.routing import Route

    app = load_factory(config()["http"]["app"])  # the same file-path load: the server imports the file, then reads the name
    assert isinstance(app, Starlette)
    assert {route.path for route in app.routes if isinstance(route, Route)} == {"/map/airspace", "/map/notams", "/map/wind", "/map/rain-grid"}


def test_the_servers_cors_lists_exactly_the_two_dev_origins_and_never_a_wildcard():
    cors = config()["http"]["cors"]
    assert cors["allow_origins"] == DEV_ORIGINS
    assert "*" not in cors["allow_origins"]


def test_the_map_apps_cors_method_header_and_expose_lists_are_the_ones_in_langgraph_json():
    """The lists are kept twice (the server's own layer in langgraph.json, this app's middleware in app.py); one cannot change without the other."""
    from flysafelz.mapdata.app import SERVER_EXPOSED, SERVER_HEADERS, SERVER_METHODS

    cors = config()["http"]["cors"]
    assert (SERVER_METHODS, SERVER_HEADERS, SERVER_EXPOSED) == (cors["allow_methods"], cors["allow_headers"], cors["expose_headers"])


def test_the_map_apps_cors_middleware_is_built_from_those_same_lists_and_the_ui_origins():
    from starlette.middleware.cors import CORSMiddleware

    from flysafelz.mapdata.app import SERVER_EXPOSED, SERVER_HEADERS, SERVER_METHODS, create_app
    from flysafelz.mapdata.config import ui_origins

    settings = {"FLYSAFELZ_UI_ORIGIN": "https://ui.flysafe.example"}
    (layer,) = [m for m in create_app(settings=settings).user_middleware if m.cls is CORSMiddleware]
    assert layer.kwargs["allow_origins"] == ui_origins(settings) and "https://ui.flysafe.example" in layer.kwargs["allow_origins"]
    assert (layer.kwargs["allow_methods"], layer.kwargs["allow_headers"], layer.kwargs["expose_headers"]) == (SERVER_METHODS, SERVER_HEADERS, SERVER_EXPOSED)


def test_the_installed_langgraph_api_still_puts_the_custom_apps_middleware_outside_its_own_cors_layer():
    """FLYSAFELZ_UI_ORIGIN works on the real server only because of this order (see mapdata/app.py); a langgraph-api upgrade that changes it fails here."""
    spec = importlib.util.find_spec("langgraph_api.server")
    source = Path(spec.origin).read_text(encoding="utf-8")
    assert "custom_middleware + global_middleware + _store_access_middleware" in source
