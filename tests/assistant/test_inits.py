from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.together import TogetherProvider
from sqlalchemy.ext.asyncio import AsyncEngine

from assistant.inits import init_assistant_service, init_engine, init_vdb
from assistant.vdb import VectorDB


@patch("assistant.inits.AssistantService")
@patch("assistant.inits.AgentOrchestrator")
@patch("assistant.inits.AgentManager")
@patch("assistant.inits.ModelFactory")
def test_init_assistant_service_success(
    mock_factory, mock_mgr, mock_orchestrator, mock_service, fake_load_secrets_cfg, fake_load_expanto_cfg
):
    """Test successful initialization of assistant service."""
    secrets, config = fake_load_secrets_cfg, fake_load_expanto_cfg
    service_instance = MagicMock()
    mock_service.return_value = service_instance

    result = init_assistant_service(secrets, config)

    mock_factory.assert_called_once_with(
        provider_cls=TogetherProvider,
        model_cls=OpenAIModel,
        api_key="TOGETHER_API_KEY",
        assistant_models=config.assistant.models,
    )
    mock_mgr.assert_called_once()
    mock_orchestrator.assert_called_once()
    mock_service.assert_called_once()
    assert result == service_instance


def test_init_assistant_service_missing_agent_keys(fake_load_expanto_cfg):
    """Test initialization fails when assistant keys are missing."""
    with pytest.raises(ValueError, match="Api keys.*not set"):
        secrets = MagicMock(api_keys=None)
        init_assistant_service(
            secrets=secrets, config=fake_load_expanto_cfg
        )


def test_init_assistant_service_missing_together_key(fake_load_expanto_cfg):
    """Test initialization fails when Together API key is missing."""
    secrets = MagicMock(api_keys=MagicMock(provider_api_key=None))
    with pytest.raises(ValueError, match="Provider API key is not set"):
        init_assistant_service(secrets=secrets, config=fake_load_expanto_cfg)


@patch("assistant.inits.create_async_engine")
def test_init_engine_success(mock_create_engine, fake_load_secrets_cfg):
    """Test successful initialization of database engine."""
    secrets = fake_load_secrets_cfg
    engine_mock = MagicMock(spec=AsyncEngine)
    mock_create_engine.return_value = engine_mock

    result = init_engine(secrets)

    mock_create_engine.assert_called_once_with("fake")
    assert result == engine_mock


@patch("assistant.inits.VectorDB")
def test_init_vdb_success(mock_vector_db, fake_load_expanto_cfg):
    """Test successful initialization of vector database."""
    config = fake_load_expanto_cfg

    vdb_instance = MagicMock(spec=VectorDB)
    mock_vector_db.return_value = vdb_instance

    result = init_vdb(config)

    mock_vector_db.assert_called_once_with(
        metrics_directory=fake_load_expanto_cfg.metrics.dir, docs_directory="./docs", root_directory="."
    )
    vdb_instance.create_all_collections.assert_called_once()
    assert result == vdb_instance
