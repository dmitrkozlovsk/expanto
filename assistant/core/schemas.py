"""Schema definitions for the assistant system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai.usage import Usage

from assistant.vdb import VectorDB


class AppContext(BaseModel):
    """Application context information for enriching agent responses.

    Contains current UI state and user selections that help agents
    provide more contextually relevant responses.
    """

    page_name: str | None = None
    page_mode: str | None = None
    selected: dict[str, Any] | None = None


@dataclass
class Deps:
    """Runtime dependencies for agent execution.

    Contains all external resources and services that agents
    need to perform their tasks.
    """

    async_db_engine: Any
    vdb: VectorDB
    app_context: AppContext


class UserData(BaseModel):
    """User request data structure.

    Encapsulates all information from a user request including
    identity, input, and current application context.
    """

    chat_uid: str
    user_input: str
    app_context: AppContext


# ----------------------------- ROUTER SCHEMAS -----------------------------
class RouterOutput(BaseModel):
    """Output schema for routing decisions.

    Contains the selected route, confidence level, and optional
    follow-up questions for clarification.
    """

    route_id: Literal[
        "create_experiment",
        "analyze_experiment",
        "query_internal_db",
        "universal",
        "expanto_assistant",
        "internet_search",
        None,
    ]
    confidence: float = Field(..., ge=0, le=1)
    follow_up_questions: str | None = None


# ------------------------ EXPERIMENT CREATOR SCHEMAS ---------------------
class ExperimentDefinition(BaseModel):
    """Schema for experiment definition and configuration.

    Contains all necessary information to define and track
    an A/B test or experiment.
    """

    name: str
    description: str
    hypotheses: str
    key_metrics: list[str]
    follow_up_message: str
    follow_up_questions: str | None = None


type AgentsOutput = str | ExperimentDefinition


@dataclass
class OrchestrationResult:
    """Result from agent orchestration pipeline.

    Contains the final output, conversation history, and
    resource usage information.
    """

    output: AgentsOutput
    message_history: list[Any]
    usage: Usage


@dataclass
class AssistantResponse:
    """Final response from the assistant service.

    Contains the processed output and cumulative usage statistics.
    """

    output: AgentsOutput
    usage: Usage


@dataclass
class ChatHistory:
    """Chat session history and usage tracking.

    Maintains conversation state and cumulative resource usage
    for a chat session.
    """

    message_history: list[Any]
    usage: Usage
