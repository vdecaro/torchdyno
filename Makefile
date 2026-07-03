.DEFAULT_GOAL := check

setup:
	uv sync --all-extras --group dev --group docs

check:
	uv run pre-commit run -a

format:
	uv run docformatter --config pyproject.toml --in-place torchdyno
	uv run black --config=pyproject.toml torchdyno
	uv run pycln --config=pyproject.toml torchdyno
	uv run isort torchdyno

test:
	uv run pytest

build:
	uv build

changelog:
	uv run cz bump --changelog
