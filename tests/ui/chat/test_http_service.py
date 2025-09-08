from unittest.mock import patch

import httpx
import pytest
from httpx import Request, Response

from src.settings import AssistantServiceCfg
from src.ui.chat.schemas import ChatRequest
from src.ui.chat.services import (
    HttpAssistantService,
)


@pytest.fixture
def config():
    """Fixture providing AssistantServiceCfg for testing."""
    return AssistantServiceCfg(url="http://testserver", timeout_seconds=5)


@pytest.fixture
def service(config):
    """Fixture providing HttpAssistantService instance."""
    return HttpAssistantService(config)


@pytest.fixture
def chat_request():
    """Fixture providing ChatRequest for testing."""
    return ChatRequest(user_input="hello", chat_uid="abc", app_context={})


@patch.object(httpx.Client, "post")
def test_invoke_success_str_output(mock_post, service, chat_request):
    """Test successful service invocation with string output."""
    mock_post.return_value = Response(
        status_code=200,
        request=Request("POST", "http://testserver/invoke"),
        content=b'{"output": "hi", "usage": {"total_tokens": 12}, "thinking": null}',
    )
    result = service.invoke(chat_request)  # InvokeResult
    print(result)
    assert result.success
    assert result.assistant_response.output == "hi"
    assert result.assistant_response.usage.total_tokens == 12


@patch.object(httpx.Client, "post")
def test_invoke_timeout(mock_post, service, chat_request):
    """Test service behavior when request times out."""
    mock_post.side_effect = httpx.TimeoutException("timeout")
    result = service.invoke(chat_request)
    assert not result.success
    assert "too long" in result.error


@patch.object(httpx.Client, "post")
def test_invoke_http_error(mock_post, service, chat_request):
    """Test service behavior when HTTP error occurs."""
    mock_post.return_value = Response(status_code=500, request=Request("POST", "http://testserver/invoke"))
    result = service.invoke(chat_request)
    assert not result.success
    assert "Error in agent" in result.error


@patch.object(httpx.Client, "post")
def test_invoke_json_decode_error(mock_post, service, chat_request):
    """Test service behavior when JSON decode error occurs."""
    mock_post.return_value = Response(status_code=200, content=b"{invalid json}")
    result = service.invoke(chat_request)
    assert not result.success
    assert "Error in agent" in result.error


@patch.object(httpx.Client, "post")
def test_invoke_bad_schema(mock_post, service, chat_request):
    """Test service behavior when response schema is invalid."""
    # no 'output' field -> should trigger ValidationError
    mock_post.return_value = Response(status_code=200, content=b'{"usage": {"total_tokens": 10}}')
    result = service.invoke(chat_request)
    assert not result.success
    assert "Error in agent" in result.error


@pytest.mark.integration
def test_http_assistant_service_live_simple():
    """Integration test with live assistant service for simple request."""
    service = HttpAssistantService(
        AssistantServiceCfg(
            url="http://127.0.0.1:8000",  # URL of real or local agent
            timeout_seconds=500,
        )
    )
    request = ChatRequest(user_input="hello", chat_uid="test123", app_context={})
    result = service.invoke(request)

    assert result.success
    assert result.assistant_response is not None
    assert isinstance(result.assistant_response.output, str | dict)


@pytest.mark.integration
def test_http_assistant_service_live_exp_definition():
    """Integration test with live assistant service for experiment creation."""
    service = HttpAssistantService(
        AssistantServiceCfg(
            url="http://127.0.0.1:8000",  # URL of real or local agent
            timeout_seconds=500,
        )
    )
    request = ChatRequest(
        user_input="Could you help me to create an experiment?", chat_uid="test123", app_context={}
    )
    result = service.invoke(request)

    assert result.success
    assert result.assistant_response is not None
