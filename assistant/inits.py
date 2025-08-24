"""Initialization functions for the assistant service components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.together import TogetherProvider
from sqlalchemy.ext.asyncio import create_async_engine

from assistant.core.agents import AgentManager, AgentOrchestrator
from assistant.core.models import ModelFactory
from assistant.core.service import AssistantService
from assistant.vdb import VectorDB
from src.logger_setup import logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.settings import Config, Secrets


def init_assistant_service(
    secrets: Secrets,
    config: Config,
) -> AssistantService:
    """Initialize the assistant service with all required components.

    Creates and configures the model factory, agent manager, orchestrator,
    and assistant service with the provided secrets and configuration.

    Args:
        secrets: Application secrets containing API keys
        config: Application configuration settings

    Returns:
        Configured AssistantService instance

    Raises:
        ValueError: If required API keys are missing
    """
    logger.info("Initializing model factory...")
    if not secrets.api_keys:
        raise ValueError("Api keys (Together, Tavily) are not set")
    if not secrets.api_keys.provider_api_key:
        raise ValueError("Provider API key is not set")
    if not secrets.api_keys.tavily_api_key:
        logger.warning("Tavily API key is not set. Internet search will not be available.")
        tavily_api_key = ""
    else:
        tavily_api_key = secrets.api_keys.tavily_api_key.get_secret_value()

    model_factory = ModelFactory(
        provider_cls=TogetherProvider,
        model_cls=OpenAIModel,
        api_key=secrets.api_keys.provider_api_key.get_secret_value(),
        assistant_models=config.assistant.models,
    )

    logger.info("Initializing agent manager...")
    agent_manager = AgentManager(
        model_factory,
        tavily_api_key,
    )
    logger.info("Initializing orchestrator...")
    orchestrator = AgentOrchestrator(agent_manager)

    logger.info("Initializing assistant service...")
    assistant_service = AssistantService(orchestrator)

    logger.info("Assistant service initialized.")
    return assistant_service


def init_engine(secrets: Secrets) -> AsyncEngine:
    """Initialize the database engine with connection settings.

    Creates an async SQLAlchemy engine using the provided database connection string.

    Args:
        secrets: Application secrets containing database credentials

    Returns:
        Configured AsyncEngine instance
    """
    logger.info("Initializing database engine...")
    engine = create_async_engine(secrets.internal_db.async_engine_str.get_secret_value())

    logger.info("Database engine initialized.")
    return engine


def init_vdb(config: Config) -> VectorDB:
    """Initialize the vector database with configured directories.

    Creates a VectorDB instance with metrics, documentation, and root directories,
    then initializes all required collections.

    Args:
        config: Application configuration containing directory paths

    Returns:
        Configured VectorDB instance with all collections created
    """
    logger.info("Initializing vector database...")
    vdb = VectorDB(metrics_directory=config.metrics.dir, docs_directory="./docs", root_directory=".")

    vdb.create_all_collections()
    logger.info("Vector database initialized.")
    return vdb
