"""Tests for assistant utilities."""

from __future__ import annotations

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from assistant.utils import drop_empty_messages


def test_drop_empty_messages_removes_tool_messages():
    """Tool calls and tool results should be stripped from shared history."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content="Hello")]),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="retrieve_internal_db",
                    args={"query": "SELECT 1"},
                    tool_call_id="call_1",
                ),
                TextPart(content="Let me check the database."),
            ]
        ),
        ModelRequest(
            parts=[ToolReturnPart(tool_name="retrieve_internal_db", content="42", tool_call_id="call_1")]
        ),
        ModelResponse(parts=[TextPart(content="The answer is 42.")]),
    ]

    cleaned = drop_empty_messages(messages)

    assert len(cleaned) == 3
    assert isinstance(cleaned[0], ModelRequest)
    assert isinstance(cleaned[0].parts[0], UserPromptPart)
    assert cleaned[0].parts[0].content == "Hello"

    assert isinstance(cleaned[1], ModelResponse)
    assert cleaned[1].parts == [TextPart(content="Let me check the database.")]

    assert isinstance(cleaned[2], ModelResponse)
    assert cleaned[2].parts == [TextPart(content="The answer is 42.")]


def test_drop_empty_messages_keeps_only_textual_turns():
    """Only user prompts and assistant text survive; everything else is dropped."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content="What is Expanto?")]),
        ModelResponse(parts=[TextPart(content="Expanto is an A/B testing platform.")]),
        ModelRequest(parts=[ToolReturnPart(tool_name="some_tool", content="data", tool_call_id="x")]),
    ]

    cleaned = drop_empty_messages(messages)

    assert len(cleaned) == 2
    assert all(isinstance(m, ModelRequest | ModelResponse) for m in cleaned)
    assert cleaned[0].parts[0].content == "What is Expanto?"
    assert cleaned[1].parts[0].content == "Expanto is an A/B testing platform."


def test_drop_empty_messages_empty_input():
    """Empty input returns empty output."""
    assert drop_empty_messages([]) == []
