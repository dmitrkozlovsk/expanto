"""tests/assistant/test_app.py
Test‑suite for FastAPI application (`assistant/app.py`) rewritten to synchronous
`fastapi.testclient.TestClient`.  Lifespan events fire automatically when using
`TestClient`, so the mocked `init_*` helpers populate `app.state` correctly.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.usage import Usage

from assistant.app import app
from assistant.core.schemas import AppContext, UserData
from assistant.core.service import AssistantResponse


@pytest.fixture()
def dummy_usage() -> Usage:
    return Usage(requests=1, request_tokens=20, response_tokens=10, total_tokens=30)


# -------------------------------------- Fixtures --------------------------------------


@pytest.fixture()
def user_payload() -> dict[str, Any]:
    ctx = AppContext(page_name="home", page_mode=None, data={"foo": "bar"})
    payload = UserData(chat_uid="chat-123", user_input="Hello agent!", app_context=ctx)
    return payload.model_dump(mode="json")


# ---------------------------------- Happy‑path test ----------------------------------


@patch("assistant.app.init_assistant_service")
@patch("assistant.app.init_engine")
@patch("assistant.app.init_vdb")
def test_invoke_success(mock_vdb, mock_engine, mock_service, user_payload, dummy_usage):
    """/invoke returns AgentResponse and disposes engine on shutdown."""

    fake_response = AssistantResponse(output="hi!", usage=dummy_usage)
    # assistant_service is async, so we need an AsyncMock
    mock_service_instance = AsyncMock()
    mock_service_instance.process_request.return_value = fake_response
    mock_service.return_value = mock_service_instance

    # Mock DB engine + vdb
    mock_engine_instance = AsyncMock()
    mock_engine_instance.dispose = AsyncMock()
    mock_engine.return_value = mock_engine_instance
    mock_vdb_instance = MagicMock()
    mock_vdb.return_value = mock_vdb_instance

    with TestClient(app) as client:
        # --- request ---
        resp = client.post("/invoke", json=user_payload)
        assert resp.status_code == 200
        assert resp.json()["output"] == "hi!"

    # After exiting TestClient, shutdown event has run
    mock_engine_instance.dispose.assert_awaited_once()
    mock_service_instance.process_request.assert_awaited_once()


# ------------------------ Validation error (missing chat_uid) ------------------------


@patch("assistant.app.init_assistant_service", lambda *a, **kw: AsyncMock())
@patch("assistant.app.init_engine", lambda *a, **kw: AsyncMock())
@patch("assistant.app.init_vdb", lambda *a, **kw: MagicMock())
def test_invoke_validation_error(user_payload):
    """Test validation error when chat_uid is missing."""
    payload = user_payload.copy()
    payload.pop("chat_uid")
    with TestClient(app) as client:
        resp = client.post("/invoke", json=payload)
        assert resp.status_code == 422  # FastAPI validation error


#
# ---------------------------- Error path: assistant raises ----------------------------


@patch("assistant.app.init_assistant_service")
@patch("assistant.app.init_engine", lambda *a, **kw: AsyncMock())
@patch("assistant.app.init_vdb", lambda *a, **kw: MagicMock())
def test_invoke_service_error(mock_service, user_payload):
    """Test error handling when assistant service raises exception."""
    err = RuntimeError("model crashed")
    mock_service_instance = AsyncMock()
    mock_service_instance.process_request.side_effect = err
    mock_service.return_value = mock_service_instance

    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post("/invoke", json=user_payload)
        assert resp.status_code == 500
        assert resp.text == "Internal Server Error"
