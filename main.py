"""Convenience entry-point for ``python main.py``.

For production use, run via gunicorn or uvicorn directly:

    gunicorn api.main:app --worker-class uvicorn.workers.UvicornWorker --workers 4
"""
from __future__ import annotations

import uvicorn

from config.settings import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers if not settings.debug else 1,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
