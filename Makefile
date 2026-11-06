.PHONY: setup test

# Needs no API key: only installs the project and its dependencies.
setup:
	uv sync

# Runs the whole suite with sockets disabled (see addopts in pyproject.toml).
test:
	uv run pytest
