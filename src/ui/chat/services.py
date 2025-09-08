"""HTTP service and chat controller for assistant communication."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import httpx

from src.settings import AssistantServiceCfg
from src.ui.chat.schemas import (
    AssistResponse,
    ChatRequest,
    ChatResponse,
    ChatState,
    InvokeResult,
)
from src.utils import JsonUtils

if TYPE_CHECKING:
    from src.ui.chat.schemas import AppContext


class HttpAssistantService:
    """HTTP service for communicating with the assistant API.

    This service handles HTTP requests to the assistant backend,
    including timeout handling and error management.
    """

    def __init__(self, config: AssistantServiceCfg) -> None:
        """Initialize the HTTP assistant service.

        Args:
            config: Configuration object containing service settings.
        """
        self.config = config
        self.client = httpx.Client(timeout=config.timeout_seconds)

    def invoke(self, request: ChatRequest) -> InvokeResult:
        """Invoke the assistant API with a chat request.

        Args:
            request: The chat request to send to the assistant.

        Returns:
            InvokeResult containing the response or error information.
        """
        try:
            method_url_string = f"{self.config.url}invoke"
            json_message = JsonUtils.jsonify(asdict(request))
            response = self.client.post(
                method_url_string, json=json_message, headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()  # response = AssistResponse
            assistant_response = AssistResponse(**data)
            return InvokeResult(assistant_response=assistant_response, success=True, error=None)
        except httpx.TimeoutException:
            return InvokeResult(
                assistant_response=None,
                success=False,
                error="Agent is thinking too long. Please, reload the chat and try again.",
            )
        except Exception as e:
            # TODO: add proper logging
            return InvokeResult(assistant_response=None, success=False, error=f"Error in agent: {e}")


class ChatController:
    """Controller for managing chat interactions with the assistant service.

    This class handles the flow of chat messages between the user interface
    and the assistant service, including request processing and response handling.
    """

    def __init__(self, assistant_service: HttpAssistantService) -> None:
        """Initialize the chat controller.

        Args:
            assistant_service: The HTTP service for assistant communication.
        """
        self.assistant_service = assistant_service

    @staticmethod
    def decompose_output(invoke_result: AssistResponse) -> tuple[str | None, Any]:
        """Decompose assistant response output into message and supplement data.

        Args:
            invoke_result: The assistant response to decompose.

        Returns:
            A tuple containing the message string and any supplemental data.
        """
        if isinstance(invoke_result.output, str):
            return invoke_result.output, None
        elif (
            hasattr(invoke_result.output, "follow_up_questions")
            and hasattr(invoke_result.output, "follow_up_questions")
            and invoke_result.output.follow_up_questions is not None
        ):
            return (
                invoke_result.output.follow_up_message + "\n" + invoke_result.output.follow_up_questions,
                invoke_result.output,
            )
        elif hasattr(invoke_result.output, "follow_up_questions"):
            return invoke_result.output.follow_up_questions, invoke_result.output
        elif hasattr(invoke_result.output, "follow_up_message"):
            return invoke_result.output.follow_up_message, invoke_result.output
        return f"{invoke_result.output}", invoke_result.output

    def process_user_input(
        self, user_input: str, chat_state: ChatState, app_context: AppContext
    ) -> ChatResponse:
        """Process user input and generate a chat response.

        Args:
            user_input: The user's input message.
            chat_state: The current chat state.

        Returns:
            ChatResponse containing the assistant's response or error information.
        """
        request = ChatRequest(user_input=user_input, chat_uid=chat_state.chat_uid, app_context=app_context)

        invoke_result: InvokeResult = self.assistant_service.invoke(request)

        if invoke_result.success and invoke_result.assistant_response:
            chat_response = self._create_success_response(
                assistant_response=invoke_result.assistant_response
            )
        else:
            chat_response = self._create_error_response(error=invoke_result.error)
        return chat_response

    def _create_success_response(self, assistant_response: AssistResponse) -> ChatResponse:
        """Create a successful chat response."""
        chat_msg, supplement = self.decompose_output(assistant_response)
        return ChatResponse(
            chat_msg=chat_msg,
            supplement=supplement,
            usage=assistant_response.usage,
            success=True,
            thinking=assistant_response.thinking,
        )

    def _create_error_response(self, error: str | None) -> ChatResponse:
        """Create an error chat response."""
        return ChatResponse(
            chat_msg=None,
            supplement=None,
            success=False,
            usage=None,
            error_msg=error or "Unknown error occurred",
            thinking=None,
        )
