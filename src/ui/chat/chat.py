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
from src.ui.resources import load_assistant_service_cfg, get_thread_pool_executor
from src.ui.state import ChatStateManager
from src.ui.chat.elements import chat_scroll

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
    ss = st.session_state
    for k, v in {
        "polling_on": False,
        "fut": None,
        "resp": None,
        "left_cnt": 0,
        "pool": None,
    }.items():
        ss.setdefault(k, v)

    return chat_state, controller, assistant_cfg


# -------------------- MAIN CHAT COMPONENT -------------------- #


class Chat:
    """Main chat component for the application."""

    @staticmethod
    @st.fragment
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
        UserInputField.render(controller)  # if user input: save to chat state and rerun

        # Handle input and response logic
        run_every = 0.5 if chat_state.future_result else None
        st.write(ChatStateManager.get_or_create_state())

        @st.fragment(run_every=run_every)
        def handle_future_response(placeholder):
            state = ChatStateManager.get_or_create_state()
            if not state.future_result:
                return
            if state.future_result.done():
                try:
                    response = state.future_result.result()  # get result
                    state.future_result = None
                    agent_placeholder.handle_response(response)
                    ChatStateManager.update_state(response)
                except Exception as e:
                    ChatStateManager.add_message(
                        message_type=MessageType.ERROR, role=Role.ASSISTANT, content=str(e)
                    )
                finally:  # prevent eternal loop
                    st.rerun()
            else:
                placeholder.show_status()
                chat_scroll()
        handle_future_response(agent_placeholder)

