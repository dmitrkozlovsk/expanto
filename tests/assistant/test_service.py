import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai.usage import Usage

from assistant.core.schemas import Deps, UserData
from assistant.core.service import AssistantResponse, AssistantService

# -------------------------------------- Fixtures --------------------------------------


@pytest.fixture
def orchestrator_mock():
    mock = AsyncMock()
    mock.process.return_value = MagicMock(
        output="test-response",
        message_history=["msg1", "msg2"],
        usage=Usage(requests=1, request_tokens=20, response_tokens=10, total_tokens=30),
    )
    return mock


@pytest.fixture
def assistant_service(orchestrator_mock):
    return AssistantService(orchestrator=orchestrator_mock)


@pytest.fixture
def test_data():
    return UserData(chat_uid="chat-123", user_input="Hello", app_context={})


@pytest.fixture
def test_deps():
    return Deps(
        **{
            "async_db_engine": MagicMock(),
            "vdb": MagicMock(),
            "app_context": MagicMock(),
        }
    )


# -------------------- Happy-path + basic checks (were previously) --------------------


@pytest.mark.asyncio
async def test_process_request_initializes_memory(
    assistant_service, orchestrator_mock, test_data, test_deps
):
    """Test that process_request initializes memory for new chat."""
    assert test_data.chat_uid not in assistant_service.memory

    response = await assistant_service.process_request(test_data, test_deps)

    # response
    assert isinstance(response, AssistantResponse)
    assert response.output == "test-response"
    assert response.usage.request_tokens == 20
    # memory
    assert test_data.chat_uid in assistant_service.memory
    assert assistant_service.memory[test_data.chat_uid].message_history == ["msg1", "msg2"]
    # orchestrator called exactly once
    orchestrator_mock.process.assert_awaited_once_with(test_data.user_input, test_deps, [])


#
@pytest.mark.asyncio
async def test_process_request_accumulates_usage(
    assistant_service, orchestrator_mock, test_data, test_deps
):
    """Test that usage is accumulated across multiple requests."""
    await assistant_service.process_request(test_data, test_deps)

    orchestrator_mock.process.return_value.usage = Usage(
        requests=1, request_tokens=20, response_tokens=10, total_tokens=55
    )

    response = await assistant_service.process_request(test_data, test_deps)
    assert response.usage.total_tokens == 85


@pytest.mark.asyncio
async def test_process_request_handles_orchestrator_exception(
    assistant_service, orchestrator_mock, test_data, test_deps
):
    """Test that orchestrator exceptions are properly handled."""
    orchestrator_mock.process.side_effect = RuntimeError("Orchestrator error")

    with pytest.raises(RuntimeError, match="Orchestrator error"):
        await assistant_service.process_request(test_data, test_deps)

    assert test_data.chat_uid in assistant_service.memory
    chat_history = assistant_service.memory[test_data.chat_uid]

    assert chat_history.message_history == []

    assert chat_history.usage == Usage()


# 3. Parallel chats
@pytest.mark.asyncio
async def test_concurrent_chats(orchestrator_mock, test_deps):
    """Test that concurrent chats work independently."""
    service = AssistantService(orchestrator_mock)

    async def _call(uid: str):
        return await service.process_request(
            UserData(chat_uid=uid, user_input=uid, app_context={}), test_deps
        )

    uids = [f"u{i}" for i in range(100)]
    await asyncio.gather(*[_call(uid) for uid in uids])

    assert set(service.memory.keys()) == set(uids)
