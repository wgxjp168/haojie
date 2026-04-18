.PHONY: help install dev test lint run docker-build docker-up docker-down clean

PYTHON ?= python3
PORT ?= 8000

help:
	@echo "Targets:"
	@echo "  install       Install runtime + dev requirements"
	@echo "  dev           Run app with reload at :$(PORT)"
	@echo "  run           Run app in production mode at :$(PORT)"
	@echo "  test          Run the test suite"
	@echo "  lint          Basic syntax check of every .py file"
	@echo "  docker-build  Build production image"
	@echo "  docker-up     Start stack via docker compose"
	@echo "  docker-down   Stop stack"
	@echo "  clean         Remove caches and virtualenv artefacts"

install:
	$(PYTHON) -m pip install -r requirements.txt

dev:
	$(PYTHON) -m uvicorn api.main:app --reload --host 0.0.0.0 --port $(PORT)

run:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port $(PORT) --workers 4

test:
	$(PYTHON) -m pytest -v

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
