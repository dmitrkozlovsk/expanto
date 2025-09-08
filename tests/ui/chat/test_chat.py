from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from src.ui.chat.schemas import ChatResponse, Role, TokenUsage


def app_script():
    """Mock Streamlit app script for testing Chat component."""
    from src.ui.chat.chat import Chat

    Chat.render()


def make_success_response():
    """Create mock successful ChatResponse for testing."""
    return ChatResponse(
        chat_msg="Hello",
        supplement=None,
        success=True,
        usage=TokenUsage(requests=1, request_tokens=10, response_tokens=10, total_tokens=20, details={}),
        error_msg=None,
        thinking="thinking",
    )


def make_error_response():
    """Create mock successful ChatResponse for testing."""
    return ChatResponse(
        chat_msg=None,
        supplement=None,
        success=False,
        usage=TokenUsage(requests=1, request_tokens=10, response_tokens=10, total_tokens=20, details={}),
        error_msg="This is an error message.",
        thinking=None,
    )


@pytest.fixture(scope="function")
def app(patch_configs):
    """Fixture providing AppTest instance for Chat component testing."""
    at = AppTest.from_function(app_script)
    at.run()
    return at


@patch("src.ui.chat.chat.ChatController.process_user_input", return_value=make_success_response())
def test_active_user_input_reset(mock_process, app):
    """Test that active user input is reset after message submission."""
    app.chat_input[0].set_value("ping").run()
    print(app)
    assert app.session_state["chat_state"].active_user_input is None


def test_render_chat(app):
    """Test basic Chat component rendering and initial state."""
    assert app.button[0].key == "clear_chat"
    assert app.markdown[0].value is not None
    assert "Hello" in app.chat_message[0].markdown[0].value
    assert app.chat_input is not None
    assert app.session_state["chat_state"].msg_history[-1].role == Role.ASSISTANT
    assert len(app.session_state["chat_state"].chat_uid) == 36
    assert app.session_state["chat_state"].active_user_input is None
    assert "app_context" not in app.session_state


def test_clear_chat_button(app):
    """Test clear chat button functionality and state reset."""
    old_uid = app.session_state["chat_state"].chat_uid
    app.button[0].click().run()
    assert len(app.session_state["chat_state"].msg_history) == 1
    assert app.session_state["chat_state"].chat_uid != old_uid
    assert app.session_state["chat_state"].supplements == {}
    assert app.session_state["chat_state"].usage is None
    assert app.session_state["chat_state"].active_user_input is None


def test_empty_input(app):
    """Test behavior when empty input is submitted."""
    app.chat_input[0].set_value("").run()
    assert len(app.chat_message) == 1
    assert app.session_state["chat_state"].active_user_input is None


def test_user_input_error_response(app):
    """Test user input processing when error response is returned."""
    app.chat_input[0].set_value("hello").run()
    assert len(app.chat_message) == 3
    assert app.session_state["chat_state"].msg_history[-1].role == Role.ASSISTANT
    assert "Error" in app.session_state["chat_state"].msg_history[-1].content
    assert app.session_state["chat_state"].active_user_input is None


@patch("src.ui.chat.chat.ChatController.process_user_input", return_value=make_success_response())
def test_user_input_success_response(mock_process, app):
    """Test user input processing when successful response is returned."""
    app.chat_input[0].set_value("hello").run()
    mock_process.assert_called_once()

    assert len(app.chat_message) == 3
    assert app.session_state["chat_state"].msg_history[-1].role == Role.ASSISTANT
    assert app.session_state["chat_state"].usage.total_tokens == 20
    assert app.session_state["chat_state"].active_user_input is None
    assert app.session_state["chat_state"].future_result is None

    assert "Hello" in app.chat_message[-1].markdown[-1].value
    assert "Thinking" in app.chat_message[-1].expander[-1].label
    assert "thinking" in app.chat_message[-1].expander[-1].markdown[0].value


@patch("src.ui.chat.chat.ChatController.process_user_input", return_value=make_error_response())
def test_user_input_error_response_without_thinking_part(mock_process, app):
    """
    Test user input processing when an error response is returned, ensuring no thinking part is displayed.
    """
    app.chat_input[0].set_value("hello").run()
    mock_process.assert_called_once()

    # We expect 3 messages: initial assistant msg, user msg, assistant error msg.
    assert len(app.chat_message) == 3

    last_message = app.chat_message[-1]

    # Check that the last message is an error and contains the correct text
    assert len(last_message.error) == 1
    assert "This is an error message." in last_message.error[0].value

    # Check that there is NO expander for the thinking part
    assert len(last_message.expander) == 0


def test_handle_future_response_happy_path(app):
    """Test happy path for handling future response."""
    from concurrent.futures import Future

    future = Future()
    mock_response = make_success_response()
    mock_response.chat_msg = "Async Hello"

    chat_state = app.session_state["chat_state"]
    chat_state.future_result = future

    app.run()
    assert "Thinking..." in app.status[0].label

    future.set_result(mock_response)

    app.run()

    assert app.session_state["chat_state"].future_result is None
    assert "Async Hello" in app.chat_message[-1].markdown[-1].value
    assert app.session_state["chat_state"].msg_history[-1].role == Role.ASSISTANT
    assert app.session_state["chat_state"].usage.total_tokens == 20
    assert len(app.status) == 0
