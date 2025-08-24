"""Main chat interface and application entry point.

This module provides the main Chat class that orchestrates the entire chat
interface, handling initialization of services, rendering UI components,
and managing the chat interaction flow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from src.settings import AssistantServiceCfg
from src.ui.chat.elements import MessageHistoryContainer, TokenUsageBar, UserInputField
from src.ui.chat.schemas import MessageType, Role
from src.ui.chat.services import ChatController, HttpAssistantService
from src.ui.resources import load_assistant_service_cfg
from src.ui.state import ChatStateManager

if TYPE_CHECKING:
    from src.ui.chat.schemas import ChatState


# -------------------- INITIALIZATION -------------------- #


def init() -> tuple[ChatState, ChatController, AssistantServiceCfg]:
    """Initialize chat components and return state and controller.

    Returns:
        Tuple containing chat state and chat controller.
    """
    chat_state = ChatStateManager.get_or_create_state()  # get or create state
    assistant_cfg = load_assistant_service_cfg()  # loaded from cache
    http_service = HttpAssistantService(assistant_cfg)
    controller = ChatController(http_service)
    return chat_state, controller, assistant_cfg


# -------------------- MAIN CHAT COMPONENT -------------------- #


class Chat:
    """Main chat component for the application."""

    @staticmethod
    def render() -> None:
        """Render the complete chat interface."""
        # Initialize config and services
        chat_state, controller, assistant_cfg = init()

        # Render elements
        TokenUsageBar.render(chat_state.usage)
        history_container = MessageHistoryContainer.render(
            chat_state.msg_history,
            is_auto_scroll=assistant_cfg.auto_scroll,
            is_stream_output=assistant_cfg.enable_streaming,
        )
        agent_placeholder = history_container.agent_placeholder
        # TODO: feature - select mode for agent (auto, create, multipurpose)
        UserInputField.render()  # if user input: save to chat state and rerun

        # Handle input and response logic
        if user_input := chat_state.active_user_input:
            agent_placeholder.show_status()
            ChatStateManager.set_user_input(None)
            try:
                response = controller.process_user_input(user_input, chat_state)  # get result
                agent_placeholder.handle_response(response)
                ChatStateManager.update_state(response)
            except Exception as e:
                ChatStateManager.add_message(
                    message_type=MessageType.ERROR, role=Role.ASSISTANT, content=str(e)
                )
            finally:  # prevent eternal loop
                st.rerun()
