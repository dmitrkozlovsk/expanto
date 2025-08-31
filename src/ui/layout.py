"""Application layout management."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import streamlit as st

from src.ui.chat.chat import Chat

if TYPE_CHECKING:
    from collections.abc import Generator


class AppLayout:
    """High level app organization."""

    @classmethod
    @contextmanager
    def chat(cls) -> Generator[None, None, None]:
        """Create layout with chat (right side) or without chat (full width).

        Usage:
            with AppLayout.chat():
                # Your page content goes here
                st.write("Main page content")
        """
        chat_enabled = st.session_state.get("chat_enabled", True)

        column_weights = [64, 41] if chat_enabled else [999999, 1]
        main_col, chat_col = st.columns(column_weights, gap="small")
        with main_col:
            yield

        with chat_col:
            Chat.render() if chat_enabled else st.write("&nbsp;")