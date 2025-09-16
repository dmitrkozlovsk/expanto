import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# from dirty_equals import IsNow, IsStr
from pydantic_ai import Agent, capture_run_messages, models
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.result import StreamedRunResult

from assistant.core.agents import AgentManager, AgentOrchestrator
from assistant.core.models import ModelFactory
from assistant.core.schemas import ExperimentDefinition, RouterOutput


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def base_agent():
    """Create a BaseAgent wired up with TestModel so no real LLM calls are made."""
    models.ALLOW_MODEL_REQUESTS = False  # Safety guard per docs

    test_model = TestModel(custom_output_text="pong")

    return Agent(
        name="Echo Agent",
        model=test_model,
        system_prompt="Echo system",
        instructions="Echo whatever the user says",
        output_type=str,
    )


@pytest.mark.anyio
async def test_run_returns_expected_output(base_agent):
    """Agent.run should return the custom_output_text defined on TestModel."""
    result = await base_agent.run("ping", deps=None, message_history=[])
    assert result.output == "pong"


@pytest.mark.anyio
async def test_capture_messages(base_agent):
    """Ensure the user prompt was forwarded into the agent message stream."""
    with capture_run_messages() as messages:
        await base_agent.run("ping", deps=None, message_history=[])

    # Expect a ModelRequest that contains a UserPromptPart with "ping"
    assert any(
        isinstance(msg, ModelRequest)
        and any(isinstance(part, UserPromptPart) and part.content == "ping" for part in msg.parts)
        for msg in messages
    )


def test_agent_configuration(base_agent):
    """Verify that Agent correctly forwards config."""
    assert base_agent._system_prompts[0] == "Echo system"
    assert base_agent._instructions == "Echo whatever the user says"
    assert isinstance(base_agent.model, TestModel)
    assert base_agent.output_type is str


@pytest.mark.anyio
@pytest.mark.parametrize("history", [[], None])
async def test_empty_message_history_accepted(base_agent: Agent, history: Any):
    """`message_history` may be an empty list **or** `None` and should work the same."""
    result = await base_agent.run("ping", deps=None, message_history=history)  # type: ignore[arg-type]
    assert result.output == "pong"


@pytest.mark.anyio
async def test_parallel_runs_on_single_instance(base_agent: Agent):
    """Two concurrent `.run()` calls should both succeed and be independent."""

    async def _call() -> str:
        res = await base_agent.run("ping", deps=None, message_history=[])
        return res.output

    out1, out2 = await asyncio.gather(_call(), _call())
    assert out1 == out2 == "pong"


# --------------------- Tests for **AppContextInjectorAgent** ---------------------


@pytest.fixture()
def injector_agent() -> Agent:
    """Return AppContextInjectorAgent with stub model."""
    models.ALLOW_MODEL_REQUESTS = False
    return Agent(
        name="Echo Agent",
        model=TestModel(custom_output_text="pong"),
        system_prompt="Echo system",
        instructions="Echo whatever the user says",
        output_type=str,
    )


@pytest.mark.anyio
async def test_app_context_injected_into_prompt(injector_agent: Agent):
    """Test that agent can run with deps containing app_context."""
    ctx_string = "App Context Mock"
    deps = SimpleNamespace(app_context=ctx_string)

    with capture_run_messages() as messages:
        result = await injector_agent.run("Hello", deps=deps, message_history=[])

    # Verify the agent ran successfully and returned expected output
    assert result.output == "pong"

    # Check that messages were captured
    assert len(messages) > 0

    # Check that UserPromptPart contains the original message
    for msg in messages:
        if not isinstance(msg, ModelRequest):
            continue
        for part in (p for p in msg.parts if isinstance(p, UserPromptPart)):
            assert "Hello" in part.content


@pytest.fixture()
def fake_together_models_class():
    class FakeModelFactory(ModelFactory):
        def __init__(self, *args, **kwargs):
            pass

        def create_router_model(self):
            return TestModel()

        def create_agentic_model(self):
            return TestModel()

        def create_tool_thinker_model(self):
            return TestModel()

    return FakeModelFactory


def test_agent_manager_agent_creation(fake_together_models_class):
    """Test that AgentManager creates agents with correct names."""
    agent_manager = AgentManager(
        model_factory=fake_together_models_class(), tavily_api_key="fake_tavily_api_key"
    )
    assert agent_manager.get_agent("route").name == "Router Agent"
    assert agent_manager.get_agent("create_experiment").name == "Experiment Creator Agent"
    assert agent_manager.get_agent("analyze_experiment").name == "Experiment Analyst Agent"
    assert agent_manager.get_agent("query_internal_db").name == "Internal DB Search Agent"
    assert agent_manager.get_agent("internet_search").name == "Internet Search Agent"
    assert agent_manager.get_agent("universal").name == "Universal Agent"
    assert agent_manager.get_agent("expanto_assistant").name == "Expanto Assistant Agent"


@pytest.fixture()
def fake_agent_manager(fake_together_models_class):
    return AgentManager(model_factory=fake_together_models_class(), tavily_api_key="fake_tavily_api_key")


@pytest.mark.anyio
async def test_agent_output_types(fake_agent_manager):
    """Test that router agent returns correct output type."""
    router_agent = fake_agent_manager._create_router_agent()
    result = await router_agent.run("abrakadabra", deps=None, message_history=[])
    assert isinstance(result.output, RouterOutput)


@pytest.mark.anyio
async def test_agent_orchestrator_low_confidence_returns_follow_up_questions():
    """When confidence < 0.5, follow_up_questions should be returned."""

    # Create mock for RouterOutput with low confidence
    expected_follow_up_question = "Could you please clarify what you mean?"

    # Create RouterOutput with confidence < 0.5
    mock_router_output = RouterOutput(
        route_id="universal", confidence=0.3, follow_up_questions=expected_follow_up_question
    )

    class FakeModelFactory(ModelFactory):
        def __init__(self, *args, **kwargs):
            from types import SimpleNamespace

            # Create mock assistant_models with required attributes
            self.assistant_models = SimpleNamespace(
                fast="test-router",
                agentic="test-agentic",
                tool_thinker="test-tool-thinker",
            )

        def create_router_model(self):
            return TestModel(custom_output_args=mock_router_output.model_dump_json())

        def create_agentic_model(self):
            return TestModel()

        def create_tool_thinker_model(self):
            return TestModel()

    agent_manager = AgentManager(model_factory=FakeModelFactory(), tavily_api_key="fake_tavily_api_key")
    deps = SimpleNamespace(app_context="context")
    orchestrator = AgentOrchestrator(agent_manager)

    result = await orchestrator.process(user_input="unclear question", deps=deps, message_history=[])

    # With low confidence, follow_up_questions should be returned instead of OrchestrationResult
    assert result.output == expected_follow_up_question


@pytest.mark.parametrize(
    "route_id, output",
    [
        ("query_internal_db", "SELECT SELECT"),
        ("internet_search", "SEARCH SEARCH"),
        (
            "create_experiment",
            ExperimentDefinition(
                name="test_name",
                description="test_description",
                hypotheses="test_hypothesis",
                key_metrics=["test_metric1", "test_metric2"],
                follow_up_message="test_message",
                follow_up_questions="test_questions",
            ),
        ),
        ("analyze_experiment", "ANALYZE ANALYZE"),
        ("universal", "UNIVERSAL UNIVERSAL"),
        (None, "UNIVERSAL UNIVERSAL"),
    ],
)
@pytest.mark.anyio
async def test_orchestrator_routes(route_id, output):
    """Test that orchestrator routes to correct agents based on route_id."""
    mock_router_output = RouterOutput(route_id=route_id, confidence=0.9, follow_up_questions=None)

    class FakeModelFactory(ModelFactory):
        def __init__(self, *args, **kwargs):
            from types import SimpleNamespace

            # Create mock assistant_models with required attributes
            self.assistant_models = SimpleNamespace(
                fast="test-router",
                agentic="test-agentic",
                tool_thinker="test-tool-thinker",
            )

        def create_router_model(self):
            return TestModel(custom_output_args=mock_router_output.model_dump_json())

        def create_agentic_model(self):
            return TestModel(call_tools=[], custom_output_text=output)

        def create_tool_thinker_model(self):
            return TestModel(call_tools=[], custom_output_text=output)

    agent_manager = AgentManager(model_factory=FakeModelFactory(), tavily_api_key="fake_tavily_api_key")
    agent_manager.init_agents()

    with (
        agent_manager.agents["create_experiment"].override(
            model=TestModel(
                custom_output_args=output if isinstance(output, str) else output.model_dump_json()
            ),
        ),
        agent_manager.agents["route"].override(
            model=TestModel(
                custom_output_args=mock_router_output.model_dump_json(),
            )
        ),
    ):
        orchestrator = AgentOrchestrator(agent_manager)
        deps = SimpleNamespace(
            app_context="context", vdb=SimpleNamespace(semantic_search=lambda *a, **kw: None)
        )
        result = await orchestrator.process(user_input="clear question", deps=deps, message_history=[])
        assert result.output == output


@pytest.mark.anyio
async def test_agent_orchestrator_extracts_thinking_part():
    """Test that AgentOrchestrator correctly extracts and returns the thinking part."""

    mock_thinking_content = "This is the thinking process."
    mock_output_content = "This is the final answer."
    mock_user_input = "some question"

    mock_run_result = MagicMock(spec=StreamedRunResult)
    mock_run_result.output = mock_output_content
    mock_run_result.usage = MagicMock(return_value=MagicMock())

    mock_new_messages = [
        ModelRequest(
            parts=[UserPromptPart(content=mock_user_input)],
        ),
        ModelResponse(
            parts=[ThinkingPart(content=mock_thinking_content), TextPart(content=mock_output_content)]
        ),
    ]
    mock_run_result.new_messages.return_value = mock_new_messages
    mock_run_result.all_messages.return_value = []

    mock_agent = MagicMock(spec=Agent)
    mock_agent.run = AsyncMock(return_value=mock_run_result)

    mock_router_agent = MagicMock(spec=Agent)
    router_output = RouterOutput(route_id="universal", confidence=0.9, follow_up_questions=None)

    router_run_result = MagicMock(spec=StreamedRunResult)
    router_run_result.output = router_output
    router_run_result.usage = MagicMock(return_value=MagicMock())
    mock_router_agent.run = AsyncMock(return_value=router_run_result)

    mock_agent_manager = MagicMock(spec=AgentManager)

    def get_agent_side_effect(route_id):
        if route_id == "route":
            return mock_router_agent
        return mock_agent

    mock_agent_manager.get_agent.side_effect = get_agent_side_effect

    orchestrator = AgentOrchestrator(agent_manager=mock_agent_manager)
    deps = SimpleNamespace(app_context="context")
    result = await orchestrator.process(user_input=mock_user_input, deps=deps, message_history=[])

    assert result.thinking == mock_thinking_content
    assert result.output == mock_output_content
    mock_agent_manager.get_agent.assert_any_call(route_id="route")
    mock_agent_manager.get_agent.assert_any_call(route_id="universal")
    mock_agent.run.assert_awaited_once_with(mock_user_input, deps=deps, message_history=[])
