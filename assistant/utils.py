"""Utility functions for the assistant module."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, Template
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ThinkingPart,
)
from pydantic_ai.usage import Usage

from src.logger_setup import logger


def load_prompt(filename: str) -> Template:
    """Load a prompt from the 'prompts' directory as a Jinja2 template.

    Args:
        filename: Name of the template file to load

    Returns:
        Jinja2 Template object

    Raises:
        Exception: If template loading fails
    """
    try:
        env = Environment(loader=FileSystemLoader(Path(__file__).parent / "prompts"))
        return env.get_template(filename)
    except Exception as e:
        logger.error(f"Error loading prompt {filename}: {e}")
        raise e


def drop_empty_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Remove empty messages and parts from a list of model messages.

    Filters out messages and message parts that don't contain meaningful content.
    For ModelRequest messages, keeps parts that have content. For ModelResponse
    messages, keeps only TextPart instances with content, excluding ThinkingPart.

    Args:
        messages: List of ModelMessage objects to filter

    Returns:
        Filtered list of ModelMessage objects with only non-empty content
    """
    cleaned: list[ModelMessage] = []

    for msg in messages:
        if isinstance(msg, ModelRequest):
            req_parts = [
                p
                for p in msg.parts
                if getattr(p, "has_content", False) or bool(getattr(p, "content", None))
            ]
            if req_parts:
                cleaned.append(ModelRequest(parts=req_parts))

        elif isinstance(msg, ModelResponse):
            # Keep only text responses from assistant, excluding thinking parts
            res_parts: list[ModelResponsePart] = [
                p for p in msg.parts if isinstance(p, TextPart) and p.has_content()
            ]
            res_parts = [p for p in res_parts if not isinstance(p, ThinkingPart)]
            if res_parts:
                cleaned.append(
                    ModelResponse(
                        parts=res_parts,
                        usage=getattr(msg, "usage", None) or Usage(),
                        model_name=getattr(msg, "model_name", None),
                        timestamp=getattr(msg, "timestamp", None) or datetime.now(),
                        vendor_id=getattr(msg, "vendor_id", None),
                    )
                )

    return cleaned
