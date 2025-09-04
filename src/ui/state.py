"""Chat state management using Streamlit session state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import streamlit as st

from src.ui.chat.schemas import AppContext, ChatMessage, ChatState, MessageType, Role

if TYPE_CHECKING:
    from src.ui.chat.schemas import ChatResponse


class AppContextManager:
    """
    Manager for app context operations using Streamlit session state.
    Store page name, page mode and selected items in the app context.
    """

    @staticmethod
    def get_or_create_state() -> AppContext:
        """Get existing app context or create a new one."""
        if "app_context" not in st.session_state:
            st.session_state["app_context"] = AppContext()
        return st.session_state.app_context

    @staticmethod
    def set_page_name(page_name: str) -> None:
        """Set the current page name in the app context."""
        app_context = AppContextManager.get_or_create_state()
        app_context.page_name = page_name

    @staticmethod
    def set_page_mode(page_mode: str) -> None:
        """Set the current page mode in the app context."""
        app_context = AppContextManager.get_or_create_state()
        app_context.page_mode = page_mode

    @staticmethod
    def add_selected(key: str, value: Any) -> None:
        """Add a selected item to the app context.

        Args:
            key: Key for the selected item.
            value: Value of the selected item.
        """
        app_context = AppContextManager.get_or_create_state()
        app_context.selected[key] = value


class ChatStateManager:
    """Manager for chat state operations using Streamlit session state."""

    @staticmethod
    def get_or_create_state() -> ChatState:
        """Get existing chat state or create a new one.

        Returns:
            ChatState instance from session state.
        """
        if "chat_state" not in st.session_state:
            st.session_state["chat_state"] = ChatState()
        return st.session_state.chat_state

    @staticmethod
    def add_message(message_type: MessageType, role: Role, content: str) -> None:
        """Add a new message to the chat history.

        Args:
            message_type: Type of the message (MESSAGE or ERROR).
            role: Role of the message sender (USER or ASSISTANT).
            content: Content of the message.
        """
        state = ChatStateManager.get_or_create_state()
        state.msg_history.append(ChatMessage(type=message_type, role=role, content=content))

    @staticmethod
    def clear() -> None:
        """Clear the chat state and create a new one."""
        del st.session_state["chat_state"]
        st.session_state["chat_state"] = ChatState()

    @staticmethod
    def set_user_input(user_input: str | None) -> None:
        """Set the active user input.

        Args:
            user_input: The user input string or None to clear.
        """
        state = ChatStateManager.get_or_create_state()
        state.active_user_input = user_input

    @staticmethod
    def update_state(chat_response: ChatResponse) -> None:
        """Update the chat state with the response from the assistant.

        Args:
            chat_response: The response from the chat controller.
        """
        state = ChatStateManager.get_or_create_state()
        if chat_response.success and chat_response.chat_msg:
            ChatStateManager.add_message(
                message_type=MessageType.MESSAGE, role=Role.ASSISTANT, content=chat_response.chat_msg
            )
            if chat_response.supplement:
                state.supplements[chat_response.supplement.__class__.__name__] = chat_response.supplement
            state.usage = chat_response.usage
        elif not chat_response.success and chat_response.error_msg:
            ChatStateManager.add_message(
                message_type=MessageType.ERROR, role=Role.ASSISTANT, content=chat_response.error_msg
            )
