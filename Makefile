.PHONY: help install dev dev-gateway dev-intent dev-decision dev-llm test test-gateway test-intent test-decision test-llm lint run run-gateway run-intent run-decision run-llm docker-build docker-up docker-down clean

PYTHON ?= python3
PORT ?= 8000
GW_PORT ?= 8001
INTENT_PORT ?= 8002
DECISION_PORT ?= 8003
LLM_PORT ?= 8004

help:
	@echo "Targets:"
	@echo "  install        Install runtime + dev requirements"
	@echo "  dev            Run Part 1 input-processor with reload at :$(PORT)"
	@echo "  dev-gateway    Run Part 2 gateway with reload at :$(GW_PORT)"
	@echo "  dev-intent     Run Part 3.1 intent service with reload at :$(INTENT_PORT)"
	@echo "  dev-decision   Run Part 3.2 decision engine with reload at :$(DECISION_PORT)"
	@echo "  dev-llm        Run Part 3.3 LLM service with reload at :$(LLM_PORT)"
	@echo "  run            Run Part 1 in production mode at :$(PORT)"
	@echo "  run-gateway    Run Part 2 in production mode at :$(GW_PORT)"
	@echo "  run-intent     Run Part 3.1 in production mode at :$(INTENT_PORT)"
	@echo "  run-decision   Run Part 3.2 in production mode at :$(DECISION_PORT)"
	@echo "  run-llm        Run Part 3.3 in production mode at :$(LLM_PORT)"
	@echo "  test           Run the full test suite (parts 1, 2, 3.1, 3.2, 3.3)"
	@echo "  test-gateway   Run only gateway tests"
	@echo "  test-intent    Run only intent tests"
	@echo "  test-decision  Run only decision-engine tests"
	@echo "  test-llm       Run only LLM-service tests"
	@echo "  lint           Basic syntax check of every .py file"
	@echo "  docker-build   Build production image"
	@echo "  docker-up      Start stack via docker compose"
	@echo "  docker-down    Stop stack"
	@echo "  clean          Remove caches and build artefacts"

install:
	$(PYTHON) -m pip install -r requirements.txt

dev:
	$(PYTHON) -m uvicorn api.main:app --reload --host 0.0.0.0 --port $(PORT)

dev-gateway:
	$(PYTHON) -m uvicorn gateway.main:app --reload --host 0.0.0.0 --port $(GW_PORT)

dev-intent:
	$(PYTHON) -m uvicorn intent.main:app --reload --host 0.0.0.0 --port $(INTENT_PORT)

dev-decision:
	$(PYTHON) -m uvicorn decision.main:app --reload --host 0.0.0.0 --port $(DECISION_PORT)

dev-llm:
	$(PYTHON) -m uvicorn llm.main:app --reload --host 0.0.0.0 --port $(LLM_PORT)

run:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port $(PORT) --workers 4

run-gateway:
	$(PYTHON) -m uvicorn gateway.main:app --host 0.0.0.0 --port $(GW_PORT) --workers 4

run-intent:
	$(PYTHON) -m uvicorn intent.main:app --host 0.0.0.0 --port $(INTENT_PORT) --workers 2

run-decision:
	$(PYTHON) -m uvicorn decision.main:app --host 0.0.0.0 --port $(DECISION_PORT) --workers 2

run-llm:
	$(PYTHON) -m uvicorn llm.main:app --host 0.0.0.0 --port $(LLM_PORT) --workers 2

test:
	$(PYTHON) -m pytest tests gateway_tests intent_tests decision_tests llm_tests -v

test-gateway:
	$(PYTHON) -m pytest gateway_tests -v

test-intent:
	$(PYTHON) -m pytest intent_tests -v

test-decision:
	$(PYTHON) -m pytest decision_tests -v

test-llm:
	$(PYTHON) -m pytest llm_tests -v

lint:
	$(PYTHON) -c "import ast, pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('.').rglob('*.py') if '.venv' not in str(p)]; print('syntax ok')"

docker-build:
	docker build -t ilbuyai/input-processor:latest --target production .

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down -v

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .mypy_cache \) -prune -exec rm -rf {} +
	rm -rf test_storage coverage.xml .coverage htmlcov
