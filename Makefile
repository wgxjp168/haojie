.PHONY: help install dev dev-gateway test test-gateway lint run run-gateway docker-build docker-up docker-down clean

PYTHON ?= python3
PORT ?= 8000
GW_PORT ?= 8001

help:
	@echo "Targets:"
	@echo "  install        Install runtime + dev requirements"
	@echo "  dev            Run Part 1 input-processor with reload at :$(PORT)"
	@echo "  dev-gateway    Run Part 2 gateway with reload at :$(GW_PORT)"
	@echo "  run            Run Part 1 in production mode at :$(PORT)"
	@echo "  run-gateway    Run Part 2 in production mode at :$(GW_PORT)"
	@echo "  test           Run the full test suite"
	@echo "  test-gateway   Run only gateway tests"
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

run:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port $(PORT) --workers 4

run-gateway:
	$(PYTHON) -m uvicorn gateway.main:app --host 0.0.0.0 --port $(GW_PORT) --workers 4

test:
	$(PYTHON) -m pytest tests gateway_tests -v

test-gateway:
	$(PYTHON) -m pytest gateway_tests -v

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
