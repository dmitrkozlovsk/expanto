from unittest.mock import Mock, patch

import pytest

from assistant.core.schemas import ExperimentDefinition
from src.ui.chat.schemas import (
    AppContext,
    AssistResponse,
    ChatState,
    InvokeResult,
    TokenUsage,
)
from src.ui.chat.services import ChatController, HttpAssistantService


@pytest.fixture
def mock_assistant_service():
    """Fixture providing mock HttpAssistantService for testing."""
    return Mock(spec=HttpAssistantService)


@pytest.fixture
def chat_controller(mock_assistant_service):
    """Fixture providing ChatController instance with mock service."""
    return ChatController(mock_assistant_service)


@pytest.fixture
def chat_state():
    """Fixture providing test ChatState instance."""
    return ChatState(chat_uid="test-chat-uid")


@pytest.fixture
def token_usage():
    """Fixture providing TokenUsage instance for testing."""
    return TokenUsage(requests=1, request_tokens=20, response_tokens=10, total_tokens=30)


@pytest.fixture
def experiment_definition():
    """Fixture providing ExperimentDefinition instance for testing."""
    return ExperimentDefinition(
        name="Test name",
        description="Test description",
        hypotheses="Test hypotheses",
        key_metrics=["metric1", "metric2"],
        follow_up_message="Test message",
        follow_up_questions="Test questions",
    )


def test_controller_init(mock_assistant_service):
    """Test ChatController initialization with assistant service."""
    ctrl = ChatController(mock_assistant_service)
    assert ctrl.assistant_service is mock_assistant_service


def test_decompose_output_string(token_usage):
    """Test output decomposition when response contains string output."""
    test_output = "string"
    response = AssistResponse(output=test_output, usage=token_usage, thinking="thinking")
    msg, supp = ChatController.decompose_output(response)
    assert msg == test_output
    assert supp is None


def test_decompose_output_exp_definition(experiment_definition, token_usage):
    """Test output decomposition when response contains ExperimentDefinition."""
    response = AssistResponse(output=experiment_definition, usage=token_usage, thinking=None)
    msg, supp = ChatController.decompose_output(response)
    _msg = experiment_definition.follow_up_message + "\n" + experiment_definition.follow_up_questions
    assert msg == _msg
    assert isinstance(supp, ExperimentDefinition)


@patch("src.ui.chat.elements.enrich_app_ctx")
def test_process_user_input_success_string(
    mock_ctx, chat_controller, mock_assistant_service, chat_state, token_usage
):
    """Test successful user input processing with string response."""
    mock_ctx.return_value = AppContext()
    assistant_response = AssistResponse(output="Response OK", usage=token_usage, thinking=None)
    mock_assistant_service.invoke.return_value = InvokeResult(
        assistant_response=assistant_response,
        success=True,
        error=None,
    )

    response = chat_controller.process_user_input("Test", chat_state, mock_ctx)

    assert response.success
    assert response.chat_msg == "Response OK"
    assert response.usage.total_tokens == token_usage.total_tokens
    assert response.error_msg is None


@patch("src.ui.chat.elements.enrich_app_ctx")
def test_process_user_input_success_object(
    mock_ctx, chat_controller, mock_assistant_service, chat_state, experiment_definition, token_usage
):
    """Test successful user input processing with object response."""
    mock_ctx.return_value = AppContext()
    assistant_response = AssistResponse(output=experiment_definition, usage=token_usage, thinking="r")
    mock_assistant_service.invoke.return_value = InvokeResult(
        assistant_response=assistant_response, success=True, error=None
    )

    response = chat_controller.process_user_input("Experiment", chat_state, mock_ctx)

    assert response.success
    assert response.supplement == experiment_definition


@patch("src.ui.chat.elements.enrich_app_ctx")
def test_process_user_input_error(mock_ctx, chat_controller, mock_assistant_service, chat_state):
    """Test user input processing when assistant service returns error."""
    mock_ctx.return_value = AppContext()
    mock_assistant_service.invoke.return_value = InvokeResult(
        assistant_response=None,
        success=False,
        error="Failure",
    )

    response = chat_controller.process_user_input("Fail", chat_state, mock_ctx)

    assert not response.success
    assert response.error_msg == "Failure"


@patch("src.ui.chat.elements.enrich_app_ctx")
def test_process_user_input_no_response_on_success(
    mock_ctx, chat_controller, mock_assistant_service, chat_state
):
    """Test error handling when success is True but no response provided."""
    mock_ctx.return_value = AppContext()
    mock_assistant_service.invoke.return_value = InvokeResult(
        assistant_response=None, success=True, error=None
    )

    response = chat_controller.process_user_input("Empty", chat_state, mock_ctx)

    assert not response.success
    assert response.error_msg == "Unknown error occurred"
