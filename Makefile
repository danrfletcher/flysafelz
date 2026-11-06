.PHONY: setup test

# Needs no API key: only installs the project and its dependencies.
setup:
	uv sync

# Runs the whole suite with sockets disabled (see addopts in pyproject.toml).
# The weather half must stay at 90% line coverage or better.
test:
	uv run pytest --cov=flysafelz.clients.open_meteo --cov=flysafelz.weather --cov-report=term-missing --cov-fail-under=90
