"""Data schemas and models for the chat system."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from concurrent.futures import Future
from pydantic import BaseModel

from assistant.core.schemas import AgentsOutput, ExperimentDefinition

# -------------------- INPUT SCHEMAS -------------------- #


@dataclass
class ChatRequest:
    """Data class for chat request to the assistant service."""

    user_input: str
    chat_uid: str
    app_context: AppContext


# -------------------- OUTPUT SCHEMAS -------------------- #


class TokenUsage(BaseModel):
    """Model for tracking token usage in chat requests."""

    requests: int | None = None
    request_tokens: int | None = None
    response_tokens: int | None = None
    total_tokens: int | None = None
    details: dict[str, int] | None = None


class AssistantResponse(BaseModel):
    """Model for assistant response from the API."""

    output: AgentsOutput
    usage: TokenUsage


@dataclass
class InvokeResult:
    """Result container for assistant service invocation."""

    assistant_response: AssistantResponse | None
    success: bool
    error: str | None = None


class ChatResponse(BaseModel):
    """Model for chat response to be displayed in UI."""

    chat_msg: str | None
    supplement: ExperimentDefinition | None = None
    success: bool
    usage: TokenUsage | None = None
    error_msg: str | None = None


# -------------------- STATE SCHEMAS -------------------- #


class Role(StrEnum):
    """Enumeration for chat message roles."""

    USER = "user"
    ASSISTANT = "assistant"


class MessageType(StrEnum):
    """Enumeration for chat message types."""

    MESSAGE = "message"
    ERROR = "error"


@dataclass
class ChatMessage:
    """Data class representing a single chat message."""

    type: MessageType
    role: Role
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChatState:
    """Data class representing the complete chat state."""

    active_user_input: str | None = None
    future_result: Future[InvokeResult] | None = None
    msg_history: list[ChatMessage] = field(default_factory=list)
    supplements: dict[str, Any] = field(default_factory=dict)
    usage: TokenUsage | None = None
    chat_uid: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class AppContext:
    """Data class for application context information."""

    page_name: str | None = None
    page_mode: str | None = None
    selected: dict[str, Any] = field(default_factory=dict)
