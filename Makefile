.PHONY: format lint test check

format:
	uv run ruff format general_motion_retargeting tests
	uv run ruff check --fix general_motion_retargeting tests

lint:
	uv run ruff format --check general_motion_retargeting tests
	uv run ruff check general_motion_retargeting tests

test:
	uv run pytest

check: lint test
