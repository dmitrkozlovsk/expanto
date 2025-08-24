from streamlit.testing.v1 import AppTest

from src.ui.chat.schemas import ChatState, Role


def app_script():
    """Mock Streamlit app script for testing ChatStateManager."""
    from unittest.mock import MagicMock

    import streamlit as st

    from assistant.core.schemas import ExperimentDefinition
    from src.ui.chat.schemas import ChatResponse, MessageType, Role  # noqa F811
    from src.ui.state import AppContextManager, ChatStateManager

    st.text_input("This is a streamlit fake function")

    mock_experiment = MagicMock(spec=ExperimentDefinition)
    mock_experiment.__class__.__name__ = "ExperimentDefinition"
    success_chat_response = ChatResponse.model_construct(
        chat_msg="hello",
        supplement=mock_experiment,
        success=True,
        usage=None,
        error_msg=None,
    )
    error_chat_response = ChatResponse.model_construct(
        chat_msg=None, supplement=None, success=False, usage=None, error_msg="error"
    )

    cmd = st.session_state.get("__cmd")
    match cmd:
        case "get_or_create_state":
            ChatStateManager.get_or_create_state()
            AppContextManager.get_or_create_state()
        case "add_user_msg":
            ChatStateManager.add_message(MessageType.MESSAGE, Role.USER, "hello")
        case "add_assist_error":
            ChatStateManager.add_message(MessageType.ERROR, Role.ASSISTANT, "error")
        case "set_user_input":
            ChatStateManager.set_user_input("foo")
        case "set_page_name":
            AppContextManager.set_page_name("page_name")
        case "set_page_mode":
            AppContextManager.set_page_mode("page_mode")
        case "add_selected_1":
            AppContextManager.add_selected(key="key1", value="value1")
        case "add_selected_2":
            AppContextManager.add_selected(key="key2", value="value2")
        case "update_state_success":
            ChatStateManager.update_state(success_chat_response)
        case "update_state_error":
            ChatStateManager.update_state(error_chat_response)


def test_state_get_or_create_state():
    """Test ChatState creation and retrieval from session state."""
    at = AppTest.from_function(app_script)
    at.run()
    at.session_state["__cmd"] = "get_or_create_state"
    at.run()
    print(at.session_state)
    assert isinstance(at.session_state["chat_state"], ChatState)


def test_state_add_messages():
    """Test adding user and assistant messages to chat history."""
    at = AppTest.from_function(app_script)
    at.session_state["__cmd"] = "get_or_create_state"
    at.run()
    at.session_state["__cmd"] = "add_user_msg"
    at.run()
    assert at.session_state["chat_state"].msg_history[-1].role == Role.USER
    assert at.session_state["chat_state"].msg_history[-1].content == "hello"
    at.session_state["__cmd"] = "add_assist_error"
    at.run()
    assert at.session_state["chat_state"].msg_history[-1].role == Role.ASSISTANT
    assert at.session_state["chat_state"].msg_history[-1].content == "error"


def test_state_set_user_input():
    """Test setting active user input in chat state."""
    at = AppTest.from_function(app_script)
    at.session_state["__cmd"] = "get_or_create_state"
    at.run()
    at.session_state["__cmd"] = "set_user_input"
    at.run()
    assert at.session_state["chat_state"].active_user_input == "foo"


def test_state_set_app_context_params():
    """Test setting app context parameters and selected items."""
    at = AppTest.from_function(app_script)
    at.session_state["__cmd"] = "get_or_create_state"
    at.run()
    at.session_state["__cmd"] = "set_page_name"
    at.run()
    assert at.session_state["app_context"].page_name == "page_name"
    at.session_state["__cmd"] = "set_page_mode"
    at.run()
    assert at.session_state["app_context"].page_mode == "page_mode"
    at.session_state["__cmd"] = "add_selected_1"
    at.run()
    assert at.session_state["app_context"].selected == {"key1": "value1"}
    at.session_state["__cmd"] = "add_selected_2"
    at.run()
    assert at.session_state["app_context"].selected == {"key1": "value1", "key2": "value2"}


def test_update_state_success():
    """Test state update with successful chat response."""
    at = AppTest.from_function(app_script)
    at.session_state["__cmd"] = "get_or_create_state"
    at.run()
    at.session_state["__cmd"] = "add_user_msg"
    at.run()
    at.session_state["__cmd"] = "update_state_success"
    at.run()
    assert at.session_state["chat_state"].msg_history[-1].content == "hello"
    assert at.session_state["chat_state"].msg_history[-1].role == Role.ASSISTANT
    assert "ExperimentDefinition" in at.session_state["chat_state"].supplements


def test_update_state_error():
    """Test state update with error chat response."""
    at = AppTest.from_function(app_script)
    at.session_state["__cmd"] = "get_or_create_state"
    at.run()
    at.session_state["__cmd"] = "update_state_error"
    at.run()
    assert at.session_state["chat_state"].msg_history[-1].content == "error"
    assert at.session_state["chat_state"].msg_history[-1].role == Role.ASSISTANT
