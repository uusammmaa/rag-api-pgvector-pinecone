"""FastAPI application factory. Run with ``uvicorn app.main:app``."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import Settings, get_settings
from app.errors import register_error_handlers
from app.logging_config import configure_logging
from app.middleware import RequestContextMiddleware
from app.routes import router
from app.services import Services


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    app = FastAPI(title=settings.app_name, version=__version__)
    app.state.services = Services(settings)

    register_error_handlers(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Retrieved-Chunks"],
    )
    # Added last so it is outermost: every response, including CORS and errors, gets an ID.
    app.add_middleware(RequestContextMiddleware)
    app.include_router(router)
    return app


app = create_app()
