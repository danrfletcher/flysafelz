.PHONY: setup test

# Needs no API key: only installs the project and its dependencies.
setup:
	uv sync

# Runs the whole suite with sockets disabled (see addopts in pyproject.toml).
# The weather and NOTAM halves must stay at 90% line coverage or better (the modules together).
test:
	uv run pytest --cov=flysafelz.clients.open_meteo --cov=flysafelz.weather --cov=flysafelz.clients.pib_source --cov=flysafelz.pib --cov-report=term-missing --cov-fail-under=90
