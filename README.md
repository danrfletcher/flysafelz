# FlySafeLZ

**Some context on why this was chosen first:** The video below is me landing my 7th paramotor flight last Saturday (30th October), just moments before a man in orange high-viz ran onto the field waving his arms and signalling for us to pack up. He was from a drone company surveying for methane over the North Sea and they had a “NOTAM” over the field. It had appeared on Gaggle (flight recorder app for paramotor pilots), and yet none of us really knew what it meant.

[![Watch the video](https://img.youtube.com/vi/Y7-MUShAdAk/hqdefault.jpg)](https://youtube.com/shorts/Y7-MUShAdAk)

Since learning to fly this July, the question of safety has been top of mind. Paramotoring follows as extreme “low risk, high consequence” risk profile. Hence, systematic checking and rigour are literally life saving. On the other hand, as an unregulated sport, anyone can strap a motor to their back and take to the skies. 

**FlySafeLZ bridges this exact gap**: a map and chat agent for UK paramotor pilots. You click a landing zone (LZ), ask "can I fly here this afternoon?", it checks the forecast, the permanent airspace and the live NOTAMs, then tells you if anything rules the flight out and surfaces important considerations pilots should make. Handing trustworthy advice to the pilot in this context requires a series of deterministic and non-deterministic checks, which are explained below. You’ll find more information on the NOTAM in particular, given its reputation in aviation as a particularly messy and noisy data source, often tripping pilots up. 

**Stack:** LangChain v1 (`create_agent`, one middleware for the guard) · LangGraph (the runtime: thread memory, the server behind Studio and the UI, streaming) · LangSmith (traces, datasets, experiments, the annotation queue that calibrates the judge) · Python 3.12 with uv · local embeddings (fastembed) for the guidance search · React, Vite and Leaflet for the map. Models: Claude Sonnet 5.5 (agent), Haiku 5.5 (NOTAM reader), Opus 5.5 (judge).

**On project size:** The agent and the evals are deliberately small: one `create_agent` call with five tools and one middleware, and a 44-example dataset. The repo is bigger because I also built around them, the deterministic core, the map UI etc. I did that because I’d like to deploy FlySafeLZ and test it in my local paramotor club, so I built it to be used as opposed to just reviewed.

![FlySafeLZ agent](docs/agent-diagram.png)

**The full write-up is at [danrfletcher.github.io/flysafelz](https://danrfletcher.github.io/flysafelz/)**: the problem, how it works, the evals, results, where it struggles and what production would need.

## Quickstart

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

### Using the agent
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

## TL;DR

- **The problem.** Before a flight a paramotor pilot checks wind, weather, daylight, airspace and NOTAMs. Everything but the NOTAMs is a number against a limit or a shape on a map, so code checks it. NOTAMs are free text, and most of what a pilot scrolls past doesn't apply to their field, so the real failure is alert fatigue: **precision is a safety metric**. [More](https://danrfletcher.github.io/flysafelz/problem/)
- **What it does.** You click a field and ask a question. Code gives one verdict per time segment: NO-GO, UNVERIFIED, PILOT DISCRETION – APPROACHING LIMITS, PILOT DISCRETION – WITH CONDITIONS or PILOT DISCRETION. There's no "GO": it only decides when *not* to fly. [More](https://danrfletcher.github.io/flysafelz/problem/#what-the-agent-does-and-doesnt-do)
- **The design.** One `create_agent` call, five tools, one middleware. Code sets every verdict from the limits in [`rules.yaml`](https://github.com/danrfletcher/flysafelz/blob/main/src/flysafelz/rules.yaml). The model has two jobs: a small model reads each NOTAM's free text into fixed fields (code turns those into an action), and the agent picks the tools and writes the answer. A guard checks every number, NOTAM ID, phone number and verdict word against the tool results before the pilot sees it. [More](https://danrfletcher.github.io/flysafelz/how-it-works/)
- **How I know it's good.** 44 hand-written examples from real days (29 dev, 15 held-out), scored safety first: missed blockers, then verdicts minute by minute, NOTAM actions, noise, tool calls and the guard. An LLM judge scores only what needs judgement, calibrated against 39 answers I labelled. [More](https://danrfletcher.github.io/flysafelz/evals/)
- **Results.** A single model call given the same data (v0) still leads on held-out: on the rules step a model is about as good as code. What code buys is guarantees: the same answer every time, tested, and a guard that can hold the model to it. v2 fixed the worst failure v1 showed (a NOTAM with no floor crashed the assessment; a one-line fix): verdict match went from 51.5% to 89.0% on dev and from 42.9% to 71.4% on held-out. [More](https://danrfletcher.github.io/flysafelz/results/)
- **Where it struggles.** I assessed every trace flagged as a missed blocker: 27 flagged, 16 real, from two failures. A vague time ("later tonight") gets a question back instead of an assessment, and a phone-first NOTAM is read as awareness. In none of them was the pilot told to fly. [More](https://danrfletcher.github.io/flysafelz/where-it-struggles/)
- **What's next.** Fix those two, round coordinates in traces, and have instructors check the limits. [Known gaps](https://danrfletcher.github.io/flysafelz/where-it-struggles/#known-gaps-and-tradeoffs) · [Evaluating it in production](https://danrfletcher.github.io/flysafelz/production/)

## Where to look first
10-min rocketship tour:
1. **The agent:** `src/flysafelz/agent/graph.py` (99 lines). One `create_agent` call, five tools, one middleware.
2. **The verdict:** `src/flysafelz/assess.py`. Code decides every verdict; the model never does.
3. **The guard:** `src/flysafelz/agent/guard.py`. Checks the model's answer against the tool results before the pilot sees it.
4. **The evals:** `evals/agent/` (dataset, evaluators, judge) and the results in `evals/results/`.
5. **The traces:** every experiment, public: [dev](https://eu.smith.langchain.com/public/217f1dfa-eca7-4da3-ba3f-b417a5bb8c71/d), [held-out](https://eu.smith.langchain.com/public/df5d87b1-f698-4fb8-b038-930e5450b565/d). One full v2 trace: [Blue Mill, Cobra Warrior day](https://eu.smith.langchain.com/public/8e70514b-91ad-4aee-b943-ea9c10a7e518/t). The harness names agent runs `v1-…` by agent type, so v2 is the later `v1-` experiment on each split (dev `…-122845`, held-out `…-130704`).

## Data and licences

- Open-Meteo: free API, non-commercial, CC BY 4.0. The map's wind layer is Open-Meteo's current 10 m wind on a 1 degree grid over the UK, fetched by the server (`GET /map/wind`, cached 3 h) and credited in the map's attribution. Its rain grid is the Rain layer's fallback: when RainViewer's index cannot be read, the layer shows Open-Meteo's current precipitation on the same 1 degree grid (a model, not radar; `GET /map/rain-grid`, cached 3 h). The browser never calls Open-Meteo.
- RainViewer: free API, personal and educational use, the past 2 hours of radar frames, tiles to zoom 7 at most, and 100 requests a minute per IP. The map's Rain layer loops those frames from the browser (the index and tiles are the only calls the browser makes to RainViewer) and credits RainViewer, with a link, in the map's attribution.
- openAIP: CC BY-NC 4.0, attribution required (openaip.net/legal).
- NATS PIB: NATS (Services) Ltd terms for UK Aeronautical Information Products; no guarantee of accuracy or completeness, no liability accepted; the XML has no declared schema and may change at short notice.
- CARTO basemap: CARTO Basemaps Terms (version 29 Sep 2026), free non-commercial tier up to 5M tile requests a month, CARTO and OpenStreetMap (ODbL) credit kept visible.

This is a non-commercial demo; a production deployment would move to commercial plans (Open-Meteo API subscription, a licensed radar feed such as the Met Office's).
