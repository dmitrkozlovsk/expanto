"""Assistant service for handling user requests and managing conversations.

This module provides the main service class that orchestrates agent interactions,
manages conversation history, and handles caching for improved performance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from cachetools import TTLCache  # type: ignore[import-untyped]
from pydantic_ai.usage import Usage

from assistant.core.agents import AgentOrchestrator
from assistant.core.schemas import AssistantResponse, ChatHistory, Deps, UserData


class AssistantService:
    """Service for handling AI agent interactions and conversation management.

    Provides a high-level interface for processing user requests through
    the agent system while maintaining conversation history and usage tracking.
    """

    def __init__(self, orchestrator: AgentOrchestrator) -> None:
        """Initialize the assistant service.

        Args:
            orchestrator: Agent orchestrator for handling request routing
        """
        self.orchestrator = orchestrator
        self.memory = TTLCache(maxsize=10_000, ttl=43_200)

    async def process_request(self, data: UserData, deps: Deps) -> AssistantResponse:
        """Process a user request through the agent system.

        Args:
            data: User request data including input and context
            deps: Runtime dependencies for agent execution

        Returns:
            Processed response with output and usage statistics
        """

        if data.chat_uid not in self.memory:
            self.memory[data.chat_uid] = ChatHistory(message_history=[], usage=Usage())

        chat_history = self.memory[data.chat_uid]

        result = await self.orchestrator.process(data.user_input, deps, chat_history.message_history)

        chat_history.message_history = result.message_history
        chat_history.usage += result.usage

        return AssistantResponse(
            output=result.output,
            usage=chat_history.usage,
            thinking=result.thinking
        )
