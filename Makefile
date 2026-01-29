.PHONY: install test lint format clean help

help:
	@echo "Available targets:"
	@echo "  install    - Install package in editable mode"
	@echo "  test       - Run tests with pytest"
	@echo "  lint       - Run ruff linter"
	@echo "  format     - Format code with black"
	@echo "  clean      - Remove build artifacts"

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src/ tests/

format:
	black src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
