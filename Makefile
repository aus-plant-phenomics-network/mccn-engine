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
NECTAR_PATH		= 	https://object-store.rc.nectar.org.au/v1/AUTH_2b454f47f2654ab58698afd4b4d5eba7/mccn-test-data
SERVER_PATH		= 	http://203.101.230.81:8082
TEST_PATH 		= 	tests/files/unit_tests
CONFIG_PATH 	= 	configs
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
	@$(PDM) run pytest tests --cov mccn --cov-report html

.PHONY: test
test:  												## Run the tests
	@echo "=> Running test cases"
	@$(PDM) run pytest tests
	@echo "=> Tests complete"

.PHONY: check-all
check-all: lint test coverage                   ## Run all linting, tests, and coverage checks


.PHONY: point-fixtures
point-fixtures:
	@$(PDM) run stac_generator serialise $(TEST_PATH)/point/silo_std.json --id silo_std --dst $(SERVER_PATH)
	@$(PDM) run stac_generator serialise $(TEST_PATH)/point/silo_proc_bands.json --id silo_proc_bands --dst $(SERVER_PATH)
	@$(PDM) run stac_generator serialise $(TEST_PATH)/point/soil.json --id soil --dst $(SERVER_PATH)
	@$(PDM) run stac_generator serialise $(TEST_PATH)/point/campey_point.json --id campey_point --dst $(SERVER_PATH)


.PHONY: raster-fixtures
raster-fixtures:
	@$(PDM) run stac_generator serialise $(TEST_PATH)/raster/rea.json --id rea --dst $(SERVER_PATH)
	@$(PDM) run stac_generator serialise $(TEST_PATH)/raster/ozbarley_raster.json --id ozbarley_raster --dst $(SERVER_PATH)
	@$(PDM) run stac_generator serialise $(TEST_PATH)/raster/llara_campey_raster.json --id llara_campey_raster --dst $(SERVER_PATH)

.PHONY: vector-fixtures
vector-fixtures:
	@$(PDM) run stac_generator serialise $(TEST_PATH)/vector/attribute.json --id attribute --dst $(SERVER_PATH)
	@$(PDM) run stac_generator serialise $(TEST_PATH)/vector/mask.json --id mask --dst $(SERVER_PATH)
	@$(PDM) run stac_generator serialise $(TEST_PATH)/vector/mask_attribute.json --id mask_attribute --dst $(SERVER_PATH)
	@$(PDM) run stac_generator serialise $(TEST_PATH)/vector/join.json --id join --dst $(SERVER_PATH)

.PHONY: fixtures
fixtures: point-fixtures raster-fixtures

.PHONY: gryfn
gryfn:
	@echo running config from $(CONFIG_PATH)/gryfn/config.json
	@$(PDM) run stac_generator serialise $(CONFIG_PATH)/gryfn/config.json --id Gryfn --dst generated -v

.PHONY: case-study-1-raster
case-study-1-raster:
	@echo running config from $(CONFIG_PATH)/case-study-1/raster_config.json
	@$(PDM) run stac_generator serialise $(CONFIG_PATH)/case-study-1/raster_config.json --id Case_Study_1_Raster -v

.PHONY: case-study-7-scenario-1
case-study-7-scenario-1:
	@cd configs/case-study-7
	@$(PDM) run stac_generator serialise llara_shape_config.json llara_raster_config.json llara_point_config.json scenario_1_config.json --id CaseStudy7Scenario1 --dst generated --num_workers 4

.PHONY: case-study-7-scenario-2
case-study-7-scenario-2:
	@cd configs/case-study-7
	@$(PDM) run stac_generator serialise llara_shape_config.json llara_raster_config.json llara_point_config.json scenario_2_config.json --id CaseStudy7Scenario2 --dst generated --num_workers 4

.PHONY: docs
docs: 												## Serve mkdocs locally
	@$(PDM) run mkdocs serve

.PHONY: docs-deploy
docs-deploy:										## Deploy to docs to github pages
	@$(PDM) run mkdocs gh-deploy
