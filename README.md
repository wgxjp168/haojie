# ILBuyAI — Input Processor (Part 1) + Gateway (Part 2) + Intent Service (Part 3.1)

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
* **Part 3.1 (`intent.main:app`, port 8002)** — Intent recognition service,
  first stage of the AI Decision Hub. Classifies normalized B2B/B2C
  procurement queries into a 20-class taxonomy (search, compare,
  inquire-price, check-stock, request-quote, bulk-order, track-order, …).
  Supports `rules` / `transformer` / `ensemble` backends with a
  deterministic rules fallback; ships with two-tier (memory + Redis)
  caching, a circuit breaker around the transformer, and Prometheus metrics
  on a dedicated registry.

### Part 3.1 endpoints

| Method | Path                              | Description                        |
|--------|-----------------------------------|------------------------------------|
| POST   | `/api/v1/intents/predict`         | Classify one text                  |
| POST   | `/api/v1/intents/predict/batch`   | Classify a batch of texts          |
| GET    | `/api/v1/intents`                 | Dump the intent taxonomy           |
| GET    | `/api/v1/intents/health`          | Detailed intent-service health     |
| GET    | `/health`, `/metrics`             | Root health + Prometheus metrics   |

The transformer backend is **optional** — without `torch`/`transformers`
installed the service still runs (rules-only) so deployments can light it
up incrementally. Config is env-driven with prefix `INTENT_` (see
`.env.example`).

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
