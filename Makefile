.PHONY: setup test fixtures

# Needs no API key: only installs the project and its dependencies.
setup:
	uv sync

# Runs the whole suite with sockets disabled (see addopts in pyproject.toml).
# The weather, NOTAM, airspace, NOTAM-action, assess, case-loader, redaction and NOTAM-area modules must stay at 90% line coverage or better (the modules together).
test:
	uv run pytest --cov=flysafelz.clients.open_meteo --cov=flysafelz.weather --cov=flysafelz.clients.pib_source --cov=flysafelz.pib --cov=flysafelz.clients.openaip --cov=flysafelz.airspace --cov=flysafelz.notam_action --cov=flysafelz.assess --cov=flysafelz.cases --cov=flysafelz.redact --cov=flysafelz.notam_area --cov-report=term-missing --cov-fail-under=90

# Capture a NEW case: make fixtures CASE=blue_mill_2026-10-31_3day
# case.yaml takes an optional `days: N` (1 to 8, Europe/London days from the day of `now`) for a multi-day window.
# Once a case is committed it is never recaptured or deleted (openAIP is live, so a recapture changes it):
# more days, or a different moment, go on a new case name.
# Needs OPENAIP_API_KEY in the environment or in .env; the other two sources need no key.
# CASE reaches the program through the environment, so it is never parsed by the shell.
fixtures:
	@uv run python -m flysafelz.capture "$$CASE"
