.DEFAULT_GOAL := test

.PHONY: test test-cov clean-coverage

test: test-cov

test-cov:
	uv run pytest --cov --cov-report=html

clean-coverage:
	uv run python -c "from pathlib import Path; p=Path('.coverage'); p.unlink() if p.exists() else None"
