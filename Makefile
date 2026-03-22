.PHONY: install lint lint-fix test test-fast test-all clean

install:
	pip install -r requirements.txt
	pip install pytest ruff

lint:
	ruff check .

lint-fix:
	ruff check --fix .
	ruff format .

test:
	python -m pytest tests/ -m "not slow"

test-fast:
	python -m pytest tests/ -m "not slow" -x --tb=short

test-all:
	python -m pytest tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
