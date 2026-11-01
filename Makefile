.PHONY: install check test eval demo serve docker audit corpus

install:
	python -m venv .venv && .venv/bin/pip install -e ".[dev]"

check:
	ruff check . && ruff format --check . && mypy && pytest --cov --cov-fail-under=85 && arag eval --check

test:
	pytest

eval:
	arag eval --check --report eval-report.md

audit:
	pip-audit --skip-editable

demo:
	arag demo

serve:
	arag serve --port 8080

docker:
	docker build -t agentic-rag-reference .

corpus:
	python scripts/generate_corpus.py
