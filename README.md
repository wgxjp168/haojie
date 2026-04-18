# ILBuyAI — Input Processor (Part 1) + Gateway (Part 2)

Production-grade services for the ILBuyAI intelligent procurement decision
system.

* **Part 1 (`api.main:app`, port 8000)** — Multi-modal input processor.
  Accepts text, image, link, and voice input from B2B purchasers and B2C
  consumers, normalizes it, and emits a structured payload consumed by the
  downstream AI decision hub.
* **Part 2 (`gateway.main:app`, port 8001)** — Access/gateway layer in
  front of Part 1. Terminates auth (JWT + API keys), enforces RBAC,
  per-user-type rate limiting, circuit breaking, and hosts the WebSocket
  hub. Forwards authorized requests to Part 1 via `/api/v1/*`.

## Overview

Part 1 exposes:

- REST endpoints for each input modality and a batch endpoint
- Session and user-profile-aware processing
- Provider-pluggable speech and image recognition
- Redis-backed cache, S3/local storage, Prometheus metrics, structured logs

## Architecture

```
                   ┌────────────────────────────┐
  client ──HTTPS──▶│  FastAPI (api/main.py)     │
                   │   ├─ middleware (auth, RL) │
                   │   └─ routes/v1/*           │
                   └─────────────┬──────────────┘
                                 │
                   ┌─────────────▼──────────────┐
                   │ MultiModalInputManager     │
                   │  ├─ SessionManager         │
                   │  ├─ ProcessorFactory       │
                   │  └─ CircuitBreaker + RL    │
                   └─────┬────────────┬─────────┘
          ┌──────────────┼────────────┼──────────────┐
          ▼              ▼            ▼              ▼
     Text Proc.   Image Proc.    Link Proc.    Voice Proc.
                       │                           │
                       ▼                           ▼
               image recognition            speech recognition
              (Azure/Google/local)         (Azure/Google/local)

                   ┌────────────┐  ┌────────────┐  ┌──────────┐
                   │ Cache (R.) │  │ Storage    │  │ Metrics  │
                   │ Redis/mem  │  │ Local/S3   │  │ Prom.    │
                   └────────────┘  └────────────┘  └──────────┘
```

## Quick Start

### Local

```bash
pip install -r requirements.txt
cp .env.example .env
python -m uvicorn api.main:app --reload --port 8000
```

### Docker

```bash
docker compose up --build
```

API docs: `http://localhost:8000/docs`

## Endpoints

| Method | Path                       | Description                         |
|--------|----------------------------|-------------------------------------|
| POST   | `/api/v1/inputs/text`      | Process text input                  |
| POST   | `/api/v1/inputs/image`     | Process image upload                |
| POST   | `/api/v1/inputs/link`      | Process product URL                 |
| POST   | `/api/v1/inputs/voice`     | Process voice upload                |
| POST   | `/api/v1/inputs/batch`     | Mixed-type batch processing         |
| GET    | `/api/v1/inputs/supported-formats` | List supported formats      |
| GET    | `/api/v1/sessions/{id}`    | Get session details                 |
| DELETE | `/api/v1/sessions/{id}`    | End session                         |
| GET    | `/health`                  | Health check                        |
| GET    | `/metrics`                 | Prometheus metrics                  |

## Configuration

Configuration is loaded from env vars (prefix `APP_`) and `config/config.yaml`.
See `.env.example` for defaults. Critical settings for production:

- `APP_ENV=production`
- `APP_JWT_SECRET_KEY` — must be rotated from default
- `APP_DATABASE_URL`, `APP_REDIS_URL`, `APP_S3_*`
- Provider keys: `APP_AZURE_CV_KEY`, `APP_AZURE_SPEECH_KEY`, etc.

## Tests

```bash
pytest -v --cov=.
```

## Project Layout

```
api/         FastAPI app, middleware, routes
cache/       Redis/in-memory cache
config/      Pydantic settings, YAML loader
core/        Exceptions, logging, metrics, security
entities/    Domain models (User, Session, UserProfile)
managers/    Orchestration (input manager, session manager)
processors/  Per-modality input processors
services/    External service clients (speech, vision)
storage/     Object storage (local, S3)
tests/       Unit tests
deploy/      Nginx, docker-entrypoint
```
