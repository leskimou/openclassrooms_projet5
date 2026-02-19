.DEFAULT_GOAL := test

.PHONY: test

test:
	uv run python -c "from pathlib import Path; p=Path('.coverage'); p.unlink() if p.exists() else None"
	uv run pytest --cov=src --cov=main --cov-report=term-missing --cov-report=html -ra