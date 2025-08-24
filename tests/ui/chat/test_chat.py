from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from src.ui.chat.schemas import ChatResponse, Role, TokenUsage


def app_script():
    """Mock Streamlit app script for testing Chat component."""
    from src.ui.chat.chat import Chat

    Chat.render()


@pytest.fixture
def app():
    """Fixture providing AppTest instance for Chat component testing."""
    at = AppTest.from_function(app_script)
    at.run()
    return at


def test_active_user_input_reset(app):
    """Test that active user input is reset after message submission."""
    app.chat_input[0].set_value("ping").run()
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


def make_success_response():
    """Create mock successful ChatResponse for testing."""
    return ChatResponse(
        chat_msg="Hello",
        supplement=None,
        success=True,
        usage=TokenUsage(requests=1, request_tokens=10, response_tokens=10, total_tokens=20, details={}),
        error_msg=None,
    )


@patch("src.ui.chat.chat.ChatController.process_user_input", return_value=make_success_response())
def test_user_input_success_response(mock_process, app):
    """Test user input processing when successful response is returned."""
    app.chat_input[0].set_value("hello").run()
    mock_process.assert_called_once()

    assert len(app.chat_message) >= 2
    assert "Hello" in app.chat_message[-1].markdown[0].value
    assert app.session_state["chat_state"].msg_history[-1].role == Role.ASSISTANT
    assert app.session_state["chat_state"].usage.total_tokens == 20
    assert app.session_state["chat_state"].active_user_input is None
