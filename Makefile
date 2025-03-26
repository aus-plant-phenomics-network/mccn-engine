SHELL := /bin/bash
# =============================================================================
# Variables
# =============================================================================

.DEFAULT_GOAL:=help
.ONESHELL:
USING_PDM		=	$(shell grep "tool.pdm" pyproject.toml && echo "yes")
ENV_PREFIX		=  .venv/bin/
VENV_EXISTS		=	$(shell python3 -c "if __import__('pathlib').Path('.venv/bin/activate').exists(): print('yes')")
PDM_OPTS 		?=
PDM 			?= 	pdm $(PDM_OPTS)

.EXPORT_ALL_VARIABLES:


.PHONY: help
help: 		   										## Display this help text for Makefile
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: upgrade
upgrade:       										## Upgrade all dependencies to the latest stable versions
	@echo "=> Updating all dependencies"
	@if [ "$(USING_PDM)" ]; then $(PDM) update; fi
	@echo "=> Dependencies Updated"
	@$(PDM) run pre-commit autoupdate
	@echo "=> Updated Pre-commit"

# =============================================================================
# Developer Utils
# =============================================================================
.PHONY: clean
clean: 												## Cleanup temporary build artifacts
	@echo "=> Cleaning working directory"
	@rm -rf .pytest_cache .ruff_cache .hypothesis build/ -rf dist/ .eggs/
	@find . -name '*.egg-info' -exec rm -rf {} +
	@find . -name '*.egg' -exec rm -f {} +
	@find . -name '*.pyc' -exec rm -f {} +
	@find . -name '*.pyo' -exec rm -f {} +
	@find . -name '*~' -exec rm -f {} +
	@find . -name '__pycache__' -exec rm -rf {} +
	@find . -name '.ipynb_checkpoints' -exec rm -rf {} +
	@find . -name '*.sqlite3' -exec rm -rf {} +
	@rm -rf .coverage coverage.xml coverage.json htmlcov/ .pytest_cache tests/.pytest_cache tests/**/.pytest_cache .mypy_cache
	$(MAKE) docs-clean

.PHONY: refresh-lockfiles
refresh-lockfiles:                                 ## Sync lockfiles with requirements files.
	pdm update --update-reuse --group :all

.PHONY: lock
lock:                                             ## Rebuild lockfiles from scratch, updating all dependencies
	pdm update --update-eager --group :all

# =============================================================================
# Tests, Linting, Coverage
# =============================================================================
.PHONY: mypy
mypy:                                               ## Run mypy
	@echo "=> Running mypy"
	@$(PDM) run mypy
	@echo "=> mypy complete"

.PHONY: pre-commit
pre-commit: 										## Runs pre-commit hooks; includes ruff formatting and linting, codespell
	@echo "=> Running pre-commit process"
	@$(PDM) run pre-commit run --all-files
	@echo "=> Pre-commit complete"

.PHONY: lint
lint: pre-commit mypy 						## Run all linting

.PHONY: coverage
coverage:  											## Run the tests and generate coverage report
	@echo "=> Running tests with coverage"
	@$(PDM) run pytest tests --cov src --cov-report html

.PHONY: test
test:  												## Run the tests
	@echo "=> Running test cases"
	@$(PDM) run pytest tests
	@echo "=> Tests complete"

.PHONY: check-all
check-all: lint test coverage                   ## Run all linting, tests, and coverage checks


.PHONY: test-fixture
test-fixture:
	@$(PDM) run stac_generator serialise tests/files/unit_tests/point/standard/config.json --id collection --dst tests/files/unit_tests/point/standard/stac
	@$(PDM) run stac_generator serialise tests/files/unit_tests/point/conversion/config.json --id collection --dst tests/files/unit_tests/point/conversion/stac
	@$(PDM) run stac_generator serialise tests/files/unit_tests/point/nodate/config.json --id collection --dst tests/files/unit_tests/point/nodate/stac
	@$(PDM) run stac_generator serialise tests/files/unit_tests/raster/realestate/config.json --id collection --dst tests/files/unit_tests/raster/realestate/stac

.PHONY: clean-fixture
clean-fixture:
	rm -rf tests/files/unit_tests/point/standard/stac
	rm -rf tests/files/unit_tests/point/conversion/stac
