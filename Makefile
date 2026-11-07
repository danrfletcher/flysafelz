.PHONY: setup test fixtures eval-reader

# Needs no API key: installs the project and its dependencies, then stores the embedding model (name and folder from
# src/flysafelz/agent/settings.yaml, the folder git-ignored) so the app never downloads at runtime. A second run downloads
# nothing; offline with the model not stored it fails, naming the model and saying to rerun when online.
setup:
	uv sync
	uv run python -m flysafelz.guidance.embedding_model

# Runs the whole suite with sockets disabled (see addopts in pyproject.toml).
# The weather, NOTAM, airspace, NOTAM-action, assess, case-loader, redaction, NOTAM-area, NOTAM-reader, SUP-reference, designator, reader-eval scoring, agent-config, agent-settings, embedding-model and time-text modules must stay at 90% line coverage or better (the modules together).
test:
	uv run pytest --cov=flysafelz.clients.open_meteo --cov=flysafelz.weather --cov=flysafelz.clients.pib_source --cov=flysafelz.pib --cov=flysafelz.clients.openaip --cov=flysafelz.airspace --cov=flysafelz.notam_action --cov=flysafelz.assess --cov=flysafelz.cases --cov=flysafelz.redact --cov=flysafelz.notam_area --cov=flysafelz.reader --cov=flysafelz.sup_refs --cov=flysafelz.designators --cov=evals.reader_eval --cov=flysafelz.agent.config --cov=flysafelz.agent.settings --cov=flysafelz.guidance.embedding_model --cov=flysafelz.timetext --cov-report=term-missing --cov-fail-under=90

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
