# Running it

## Running it

```
make setup     # install; needs no key
make test      # all tests, no network
make run       # the agent in the terminal
make dev       # the LangGraph server and the UI (Studio works against the same server)
make eval      # the agent eval on dev; SPLIT=held_out for held-out
```
Settings go in `.env` (copy `.env.example`, gitignored). The example file stays empty on purpose: a test fails if a value is ever committed to it. Each model role takes any OpenAI-compatible endpoint that supports tool calling and JSON-schema output.

| Variables                                                              | What they're for                                                                                                                                                     | Example                                                                               | Needed by                                                                                |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `AGENT_BASE_URL`, `AGENT_MODEL`, `AGENT_API_KEY`                       | The chat model that picks tools and writes the answer                                                                                                                | `https://openrouter.ai/api/v1`, `anthropic/claude-sonnet-4.5`, `sk-or-…`              | `make run`, `make dev`, `make eval`, `make eval-v0`                                      |
| `READER_PROVIDER`, `READER_BASE_URL`, `READER_MODEL`, `READER_API_KEY` | The NOTAM reader, one structured call per NOTAM (a small model is enough). `PROVIDER` is just a label for logs                                                       | `openrouter`, `https://openrouter.ai/api/v1`, `anthropic/claude-haiku-5.5`, `sk-or-…` | `make run`, `make dev`, `make eval`                                                      |
| `JUDGE_PROVIDER`, `JUDGE_BASE_URL`, `JUDGE_MODEL`, `JUDGE_API_KEY`     | The LLM judge; best a stronger model than the agent                                                                                                                  | `openrouter`, `https://openrouter.ai/api/v1`, `anthropic/claude-opus-5.5`, `sk-or-…`  | `make eval`, `make agreement`                                                            |
| `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`                               | Traces, datasets, experiments and the annotation queue. The project (default `flysafelz`) is for the agent's own runs; the evals always use their own fixed projects | `lsv2_pt_…`, `flysafelz`                                                              | `make eval`, `make eval-v0`, `make annotate`, `make agreement`; optional for `run`/`dev` |
| `OPENAIP_API_KEY`                                                      | Live airspace from openAIP (free account)                                                                                                                            | a 32-character key from your openAIP profile                                          | `make fixtures`, live `assess`; not the evals                                            |
| `VITE_CARTO_KEY` (in `ui/.env`, not `.env`)                           | The map's basemap tiles (free CARTO key). Vite reads only `ui/`'s own env file, so copy `ui/.env.example` to `ui/.env`. Without it the map works but shows an "API key required" watermark | a key from carto.com/basemaps/apikey (public: it goes in the tile URL) | `make dev` (optional) |

`make setup`, `make test` and `make rag-eval` need none of them. Restart `make dev` after changing either file: both are read only at start. The UI needs Node 22 or newer and npm (see Running the UI below). `make test` runs about 12,700 tests with no network and takes about 9 minutes with coverage on.

### Asking the full agent
The UI is the main way in: `make dev` starts the server and the map UI on http://localhost:5173 (Studio works against the same server). To ask from the terminal instead, `make run` sends one question and prints the answer:

```
make run LAT=53.802797 LON=-0.086644 CASE=blue_mill_2026-10-31 NOW=2026-10-31T07:30:00Z Q="Can I fly here this afternoon?"
```

With `CASE` and `NOW` it replays a saved day, so it needs only the model settings in `.env`. Leave them out for a live run, which also needs `OPENAIP_API_KEY`. Progress goes to stderr and the answer to stdout. A live first answer takes about a minute while the reader works through the day's bulletin; later questions about the same day are quicker.

### Assessing a landing zone from the terminal (deterministic core only)
The deterministic core runs on its own, with no model:

```
uv run python -m flysafelz.cli assess --case blue_mill_2026-10-31   # offline, from fixtures/<case>/
uv run python -m flysafelz.cli assess --lat 53.802797 --lon -0.086644 # live; needs OPENAIP_API_KEY in the environment or .env
```

It prints the assessment hour by hour, then whether each data source worked, the flyable windows and the overall verdict. `--start` and `--end` pick a window (ISO 8601 with a timezone; the end is exclusive). `--debug` also lists the NOTAMs that were dropped and why. In a live run, `--reader live` has the model read the NOTAMs; without it, any NOTAM over the field that needs reading shows as UNVERIFIED. No flag changes a limit. The expected outputs for both cases are saved in `tests/golden/` (`uv run pytest --update-golden` rewrites them).

A case replays a saved day. The NOTAM archive only saved the bulletin about twice a day, so a case treats its bulletin as current at the case's own time. In live mode, a bulletin more than 3 hours old makes the answer UNVERIFIED. One made-up case, `fixtures/stale_pib_constructed/`, keeps that rule on so it stays tested.

## Running the UI

The UI is a React + Vite + TypeScript app in `ui/` (shadcn/ui, Tailwind, Leaflet), dark theme only. It needs Node 22 or newer and npm.

```
make dev       # the LangGraph server and the UI together; Ctrl-C stops both, and if either one exits the other is stopped
make ui        # the UI alone; runs npm --prefix ui ci first when ui/node_modules is missing
make test-ui   # lint, Vitest with coverage (no keys, no network), the production build, then scans of ui/dist for outside URLs, unbundled fonts and keys
```

Ports: the LangGraph server is on 2024 and the UI's dev server on 5173. `vite preview` (the built app) is on 4173. The server's CORS list names exactly 5173 and 4173 (plus LangGraph Studio's two sites), so Vite is set to `strictPort`: with 5173 taken, `make ui` stops with Vite's port error instead of moving to 5174. Free the port, or add the new origin with `FLYSAFELZ_UI_ORIGIN`.

The one setting the browser reads is `VITE_API_URL`, where the LangGraph server is (default `http://localhost:2024`), plus an optional public basemap key (see the variables table above). Set `VITE_API_URL` in the environment when you start the UI, for example `VITE_API_URL=http://localhost:2025 make ui`. No key or model setting ever reaches the browser. The map opens on the whole UK over CARTO Dark Matter tiles, which work without a key.

The map draws two layers from the server's read-only routes, each coloured by type: airspace below FL100 (`GET /map/airspace`; restricted and prohibited red with a light fill, danger orange, CTR, CTA and ATZ sky blue, every other type slate) and NOTAMs below FL100 (`GET /map/notams`; violet dashed, an "approx" label where the shape is the NOTAM's Q-line circle). "Below FL100" means the lower limit is under FL100: airspace by its openAIP lower limit (a floor exactly at FL100, 10,000 ft, is not below; ground, surface and above-ground floors count as below), NOTAMs by the Q-line lower flight level. Both routes are fetched on load and every 15 minutes while the page is open; there is no refresh button. The Layers button at the top right opens the switches, the legend and each layer's "updated HH:MM" (browser time zone), "stale" and "unavailable" notes. The switches' choices (airspace, NOTAMs, Answer overlays) are kept in this browser's local storage (key `flysafelz.layers.v1`), so a reload keeps them; conversations are still lost on reload. If the browser blocks or fills its storage, or the stored value is unreadable, the switches start on and still work for that visit.

## Data and licences

- Open-Meteo: free API, non-commercial, CC BY 4.0. The map's wind layer is Open-Meteo's current 10 m wind on a 1 degree grid over the UK, fetched by the server (`GET /map/wind`, cached 3 h) and credited in the map's attribution. Its rain grid is the Rain layer's fallback: when RainViewer's index cannot be read, the layer shows Open-Meteo's current precipitation on the same 1 degree grid (a model, not radar; `GET /map/rain-grid`, cached 3 h). The browser never calls Open-Meteo.
- RainViewer: free API, personal and educational use, the past 2 hours of radar frames, tiles to zoom 7 at most, and 100 requests a minute per IP. The map's Rain layer loops those frames from the browser (the index and tiles are the only calls the browser makes to RainViewer) and credits RainViewer, with a link, in the map's attribution.
- openAIP: CC BY-NC 4.0, attribution required (openaip.net/legal).
- NATS PIB: NATS (Services) Ltd terms for UK Aeronautical Information Products; no guarantee of accuracy or completeness, no liability accepted; the XML has no declared schema and may change at short notice.
- CARTO basemap: CARTO Basemaps Terms (version 29 Sep 2026), free non-commercial tier up to 5M tile requests a month, CARTO and OpenStreetMap (ODbL) credit kept visible.

This is a non-commercial demo; a production deployment would move to commercial plans (Open-Meteo API subscription, a licensed radar feed such as the Met Office's).

