"""UI elements and components for the chat interface."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import streamlit as st
import streamlit.components.v1 as components

from src.ui.chat.schemas import MessageType, Role
from src.ui.common import enrich_app_ctx
from src.ui.resources import get_thread_pool_executor
from src.ui.state import ChatStateManager

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

    from src.ui.chat.schemas import ChatMessage, ChatResponse, TokenUsage

# -------------------- CONSTANTS -------------------- #

AVATARS = {Role.USER: ":material/cognition:", Role.ASSISTANT: ":material/casino:"}


# -------------------- HELPER FUNCTIONS -------------------- #


def stream_words(text: str, chunk_size: int = 1, delay: float = 0.03):
    """Stream words from text with specified chunk size and delay.

    Args:
        text: The text to stream.
        chunk_size: Number of words per chunk.
        delay: Delay between chunks in seconds.

    Yields:
        Chunks of words with trailing space.
    """
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i : i + chunk_size]) + " "
        time.sleep(delay)


def chat_scroll() -> None:
    """Auto scroll to the end of the chat box."""
    components.html(
        """
        <script>
          const chatBox = window.parent.document.querySelector('.st-key-chat_wrap');
            if (chatBox) {
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        </script>
        """,
        height=1,
    )


# -------------------- UI COMPONENTS -------------------- #


class TokenUsageBadges:
    """Component for displaying token usage badges."""

    @staticmethod
    def render(usage: TokenUsage | None) -> None:
        """Render token usage badges.

        Args:
            usage: Token usage information or None.
        """
        i = usage.request_tokens if usage else 0
        o = usage.response_tokens if usage else 0
        t = usage.total_tokens if usage else 0

        html = f"""
        <style>
          /* general container */
          .token-row {{
              display:inline-flex;
              gap:14px;
              font-family:Menlo,Consolas,monospace;
              font-size:14px;
          }}
    
          /* base badge style */
          .token-badge {{
              padding:6px 14px;
              border-radius:8px;
              font-weight:600;
              line-height:1;
              display:inline-flex;
              align-items:center;
              border:1px solid;
          }}
    
          .token-label {{ margin-right:6px; }}
    
          /* -------------------- LIGHT THEME -------------------- */
          @media (prefers-color-scheme: light) {{
              .token-badge {{ background:#f8fafc; border-color:#e2e8f0; color:#0f172a; }}
              .token-label {{ color:#64748b;}}
          }}
    
          /* -------------------- DARK THEME -------------------- */
          @media (prefers-color-scheme: dark) {{
              .token-badge {{ background:#1e293b; border-color:#334155; color:#f1f5f9; }}
              .token-label {{ color:#94a3b8;}}
          }}
        </style>
    
        <div class="token-row">
            <span class="token-badge"><span class="token-label">input</span>{i}</span>
            <span class="token-badge"><span class="token-label">output</span>{o}</span>
            <span class="token-badge"><span class="token-label">total</span>{t}</span>
        </div>
        """

        st.markdown(html, unsafe_allow_html=True, help="Approx token usage")


class TokenUsageBar:
    """Component for displaying token usage bar with clear chat button."""

    @staticmethod
    def render(usage: TokenUsage | None) -> None:
        """Render token usage bar with clear button.

        Args:
            usage: Token usage information or None.
        """
        header_container = st.container(key="chat_header")
        with header_container:
            col1, col2 = st.columns([1, 7])

            with col1:
                if st.button(":material/cleaning_services:", help="Clear chat", key="clear_chat"):
                    ChatStateManager.clear()
                    st.rerun()
            with col2:
                TokenUsageBadges.render(usage=usage)


class AssistantMessagePlaceholder:
    """Placeholder component for assistant messages."""

    def __init__(self, placeholder: DeltaGenerator, is_stream_output: bool = False) -> None:
        """Initialize the assistant message placeholder.

        Args:
            placeholder: Streamlit delta generator for the placeholder.
            is_stream_output: Whether to stream the output text.
        """
        self.placeholder = placeholder
        self.is_stream_output = is_stream_output

    def handle_response(self, response: ChatResponse) -> None:
        """Handle and display the assistant response.

        Args:
            response: The chat response to display.
        """
        with self.placeholder.chat_message("assistant", avatar=AVATARS[Role.ASSISTANT]):
            if response.success:
                if self.is_stream_output:
                    st.write_stream(stream_words(response.chat_msg)) if response.chat_msg else None
                else:
                    st.markdown(response.chat_msg, unsafe_allow_html=True)
            else:
                self.show_error(response.error_msg)

    def show_status(self) -> None:
        """Show thinking status message."""
        self.placeholder.status("Thinking... Please, don't change the page. It may take a while.")

    def show_error(self, error_msg: str | None) -> None:
        """Show error message.

        Args:
            error_msg: The error message to display.
        """
        self.placeholder.error(f"Something went wrong. {error_msg}")


@dataclass
class MessageHistoryContainer:
    """Container for displaying chat message history."""

    agent_placeholder: AssistantMessagePlaceholder

    @classmethod
    def render(
        cls, message_history: list[ChatMessage], is_auto_scroll: bool = True, is_stream_output: bool = True
    ) -> MessageHistoryContainer:
        """Render the message history container.

        Args:
            message_history: List of chat messages to display.
            is_auto_scroll: Whether to auto-scroll to the bottom.
            is_stream_output: Whether to stream assistant responses.

        Returns:
            MessageHistoryContainer instance with agent placeholder.
        """
        wrap = st.container(key="chat_wrap", border=True, gap="small")

        with wrap:
            chat_container = st.container(key="chat", gap="small")
            with chat_container:
                if not message_history:
                    with st.chat_message("assistant", avatar=":material/casino:"):
                        intro_message = "Hello, my dear friend! How can I help you today?"
                        st.markdown(intro_message)
                        ChatStateManager.add_message(
                            message_type=MessageType.MESSAGE,
                            role=Role.ASSISTANT,
                            content=intro_message,
                            thinking=None,
                        )
                else:
                    for message in message_history:
                        with st.chat_message(message.role, avatar=AVATARS[message.role]):
                            if hasattr(message, "thinking") and message.thinking:
                                st.expander("Thinking Part", expanded=False).markdown(message.thinking)
                            if message.type == MessageType.MESSAGE:
                                st.markdown(message.content)
                            elif message.type == MessageType.ERROR:
                                st.error(message.content)

                if is_auto_scroll:
                    chat_scroll()

                placeholder = st.empty()

        return cls(agent_placeholder=AssistantMessagePlaceholder(placeholder, is_stream_output))


class UserInputField:
    """Component for user input field."""

    @staticmethod
    def render(controller) -> None:
        """Render the user input field and handle input."""
        if user_input := st.chat_input("Write..."):
            pool = get_thread_pool_executor()
            enriched_app_ctx = enrich_app_ctx()
            future_result = pool.submit(
                controller.process_user_input,
                user_input,
                ChatStateManager.get_or_create_state(),
                enriched_app_ctx,
            )
            ChatStateManager.set_future_result(future_result)
            ChatStateManager.add_message(
                message_type=MessageType.MESSAGE,
                role=Role.USER,
                content=user_input,
                thinking=None,
            )
            st.rerun()
