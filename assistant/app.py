"""FastAPI application for the Expanto assistant service."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from assistant.core.schemas import Deps, UserData
from assistant.core.service import AssistantResponse
from assistant.inits import init_assistant_service, init_engine, init_vdb
from src.logger_setup import logger
from src.settings import Config, Secrets


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Manage application lifecycle including startup and shutdown procedures.

    Initializes core services during startup and properly disposes of resources
    during shutdown.

    Args:
        app_instance: FastAPI application instance

    Yields:
        None: Control to the running application
    """
    # ----------------------------- STARTUP -----------------------------
    logger.info("Expanto agent startup...")

    config = Config()
    secrets = Secrets()

    app_instance.state.assistant_service = init_assistant_service(secrets, config)
    app_instance.state.db_engine = init_engine(secrets)
    app_instance.state.vdb = init_vdb(config)

    try:
        logger.instrument_sqlalchemy(app_instance.state.db_engine)
        logger.instrument_requests()
        logger.info("SQLAlchemy & Requests instrumentation enabled.")
    except Exception as e:
        logger.warning(f"DB/HTTP instrumentation failed: {e}")

    logger.info("Application startup completed successfully.")

    yield

    # ----------------------------- SHUTDOWN -----------------------------
    logger.info("Expanto agent shutdown...")
    await app_instance.state.db_engine.dispose()
    logger.info("Database engine disposed.")


app = FastAPI(lifespan=lifespan)
try:
    logger.instrument_fastapi(app, capture_headers=True, extra_spans=True)
    logger.info("FastAPI instrumentation enabled.")
except Exception as e:
    logger.warning(f"FastAPI instrumentation failed: {e}")


@app.post("/invoke")
async def invoke_agent(data: UserData, request: Request) -> AssistantResponse:
    """Process requests to the assistant and return responses.

    This endpoint accepts user data and application context, processes the request
    through the assistant service, and returns the assistant's response.

    Args:
        data: User data including chat_uid and message content
        request: FastAPI request object containing application state

    Returns:
        AssistantResponse object containing the assistant's reply
    """
    logger.info(f"Received request for chat_uid: {data.chat_uid}")

    deps = Deps(
        async_db_engine=request.app.state.db_engine,
        vdb=request.app.state.vdb,
        app_context=data.app_context,
    )

    assistant_service = request.app.state.assistant_service
    assistant_reply = await assistant_service.process_request(data, deps)

    return assistant_reply
