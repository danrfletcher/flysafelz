.PHONY: setup test fixtures eval-reader guidance eval-guidance dev run

# Needs no API key: installs the project and its dependencies, then stores the embedding model (name and folder from
# src/flysafelz/agent/settings.yaml, the folder git-ignored) so the app never downloads at runtime. A second run downloads
# nothing; offline with the model not stored it fails, naming the model and saying to rerun when online.
setup:
	uv sync
	uv run python -m flysafelz.guidance.embedding_model

# Runs the whole suite with sockets disabled (see addopts in pyproject.toml).
# The weather, NOTAM, airspace, NOTAM-action, assess, case-loader, redaction, NOTAM-area, NOTAM-reader, SUP-reference, designator, reader-eval scoring, agent-config, agent-settings, agent-state, input-check, agent graph, system-prompt, middleware, guard, guard flow, tracing (anonymiser, client, feedback), answer (re-check line, limit lines, rehydration), `make run`, window, tool-text redaction, progress, source-cache, data-tool (get_weather, get_airspace, get_notams, assess, search_guidance and their shared helpers), guidance (register, chunker, hit@4, corpus files, ingest, embedding-model, embedder, search index, eval) and time-text modules must stay at 90% line coverage or better (the modules together).
test:
	uv run pytest --cov=flysafelz.clients.open_meteo --cov=flysafelz.weather --cov=flysafelz.clients.pib_source --cov=flysafelz.pib --cov=flysafelz.clients.openaip --cov=flysafelz.airspace --cov=flysafelz.notam_action --cov=flysafelz.assess --cov=flysafelz.cases --cov=flysafelz.redact --cov=flysafelz.notam_area --cov=flysafelz.reader --cov=flysafelz.sup_refs --cov=flysafelz.designators --cov=evals.reader_eval --cov=flysafelz.agent.config --cov=flysafelz.agent.settings --cov=flysafelz.agent.state --cov=flysafelz.agent.input_check --cov=flysafelz.agent.graph --cov=flysafelz.agent.prompts --cov=flysafelz.agent.middleware --cov=flysafelz.agent.run --cov=flysafelz.agent.window --cov=flysafelz.agent.redaction --cov=flysafelz.agent.progress --cov=flysafelz.agent.sources --cov=flysafelz.agent.guard --cov=flysafelz.agent.guard_flow --cov=flysafelz.agent.tracing --cov=flysafelz.agent.answer --cov=flysafelz.agent.tools --cov=flysafelz.shapes --cov=flysafelz.guidance --cov=flysafelz.timetext --cov-report=term-missing --cov-fail-under=90

# Capture a NEW case: make fixtures CASE=blue_mill_2026-10-31_3day
# case.yaml takes an optional `days: N` (1 to 8, Europe/London days from the day of `now`) for a multi-day window.
# Once a case is committed it is never recaptured or deleted (openAIP is live, so a recapture changes it):
# more days, or a different moment, go on a new case name.
# Needs OPENAIP_API_KEY in the environment or in .env; the other two sources need no key.
# CASE reaches the program through the environment, so it is never parsed by the shell.
fixtures:
	@uv run python -m flysafelz.capture "$$CASE"

# The reader eval, and the only target that calls the model: it needs the four READER_* settings (environment or .env) and
# stops, naming the missing ones, when any is absent. It reads every fully labelled item in evals/reader/labels.yaml `runs`
# times (src/flysafelz/reader/settings.yaml) with the real model and the cache bypassed, scores the runs, checks the live
# Blue Mill case, and writes evals/reader/runs/<time>/report.md and report.json. `make test` never runs it. A harmful
# downgrade does not fail it: it is a report, not a gate.
eval-reader:
	@uv run python -m evals.reader_eval.run

# Fetches the documents in data/guidance/sources.yaml, splits them by their own section headings, embeds them, and promotes the new snapshot
# (section files, manifest.yaml, embeddings.sqlite and the corpus key) only if hit@4 on evals/guidance/questions.yaml is no worse than the current one.
# An unchanged document prints "<id>: no change". It needs the network for the documents and the embedding model that `make setup` stored (it never downloads
# the model). GUIDANCE_SOURCES=<file> reads a different register. `make test` never runs it.
guidance:
	@uv run python -m flysafelz.guidance.ingest

# Scores guidance search against Dan's labelled questions in evals/guidance/questions.yaml: prints hit@4, one line per miss (the question, every place it
# expected, the four places that came back) and the corpus key. It always exits 0 on a score, whatever it is (`make guidance` owns the promote decision); it
# exits non-zero for a questions file that is malformed, an expected document that sources.yaml does not list, a missing index or an embedding model that
# `make setup` has not stored. With no questions it prints "hit@4 not run: no labels". It reads the embedding model from the folder `make setup` filled and never downloads.
eval-guidance:
	@uv run python -m flysafelz.guidance.eval

# The LangGraph dev server with Studio, loading the graph named in langgraph.json and the settings in .env (AGENT_*, and LANGSMITH_* for traces).
# It hot-reloads on a code change.
dev:
	@uv run langgraph dev

# Asks the agent one question in the terminal, for when Studio is closed:
#   make run LAT=53.802797 LON=-0.086644 Q="Can I fly here tomorrow?"
# LAT, LON and Q are required; TZ (default Europe/London), CASE (a committed case, so frozen sources are read) and NOW (an ISO time with an
# offset, for a case) are optional. They reach the program through the environment, so the shell never parses them (as for `make fixtures`).
# A missing one prints one line on stderr and exits 2. Progress lines go to stderr, the answer to stdout. AGENT_* come from the environment or .env.
run:
	@uv run $(if $(wildcard .env),--env-file .env) python -m flysafelz.agent.run
