PYTHON ?= python3
PIP ?= $(PYTHON) -m pip

.PHONY: install lint lint-fix test test-fast test-all clean

install:
	$(PIP) install -r requirements.txt
	$(PIP) install pytest ruff

lint:
	ruff check .

lint-fix:
	ruff check --fix .
	ruff format .

test:
	$(PYTHON) -m pytest tests/ -m "not slow"

test-fast:
	$(PYTHON) -m pytest tests/ -m "not slow" -x --tb=short

test-all:
	$(PYTHON) -m pytest tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
