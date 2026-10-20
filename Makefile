PYTHON_BOOTSTRAP ?= python3.11
VENV := .venv
PY := $(VENV)/bin/python

.DEFAULT_GOAL := help

.PHONY: help setup test verify-data manifest smoke e1 e2 e3 e4 e5 e6 figures paper reproduce arxiv

help: ## List targets
	@grep -E '^[a-zA-Z0-9_ -]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-18s %s\n", $$1, $$2}'

setup: ## Create .venv and install pinned dependencies
	$(PYTHON_BOOTSTRAP) -m venv $(VENV)
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -e ".[dev]"

test: ## Run the test suite
	$(PY) -m pytest -q

verify-data: ## Check the eval set against its SHA-256 manifest
	$(PY) scripts/eval_set_manifest.py verify

manifest: ## Regenerate data/eval_set/MANIFEST.json
	$(PY) scripts/eval_set_manifest.py write

smoke: ## One live call per provider (paid; Phase 1)
	@echo "not implemented yet (Phase 1)" && exit 1

e1 e2 e3 e4 e5 e6: ## Experiments E1-E6 (Phase 2)
	@echo "not implemented yet (Phase 2)" && exit 1

figures: ## All figures and tables (Phase 4)
	@echo "not implemented yet (Phase 4)" && exit 1

paper: ## Build paper/main.pdf (Phase 5)
	@echo "not implemented yet (Phase 5)" && exit 1

reproduce: ## Regenerate every number and figure from committed runs, no API calls (Phase 6)
	@echo "not implemented yet (Phase 6)" && exit 1

arxiv: ## Build the arXiv submission package (Phase 7)
	@echo "not implemented yet (Phase 7)" && exit 1
