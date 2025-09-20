"""Agent management system for the AI assistant."""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent
from pydantic_ai.common_tools.tavily import tavily_search_tool
from pydantic_ai.messages import ModelMessage, ModelResponse, ThinkingPart
from pydantic_ai.settings import ModelSettings

from assistant.core.models import ModelFactory
from assistant.core.schemas import Deps, ExperimentDefinition, OrchestrationResult, RouterOutput
from assistant.core.tools import (
    get_expanto_app_context,
    retrieve_codebase_docs,
    retrieve_internal_db,
    retrieve_metrics_docs,
    retrieve_relevant_docs,
)
from assistant.utils import drop_empty_messages, load_prompt
from src.logger_setup import logger


class AgentManager:
    """Manager for creating and accessing different types of AI agents.

    Handles agent lifecycle including creation, initialization, and retrieval
    of specialized agents for different task types.
    """

    def __init__(self, model_factory: ModelFactory, tavily_api_key: str) -> None:
        """Initialize the agent manager.

        Args:
            model_factory: Factory for creating AI models
            tavily_api_key: API key for Tavily search service
        """
        self.model_factory = model_factory
        self.tavily_api_key = tavily_api_key
        self.system_prompt = load_prompt("expanto_system_prompt.md").render()
        self.agents: dict[str, Any] = {}

    def _create_router_agent(self) -> Agent[Deps, RouterOutput]:
        """Create agent for routing user requests to appropriate handlers.

        Returns:
            Configured router agent
        """
        logger.info("Creating Router agent...")
        return Agent[Deps, RouterOutput](
            name="Router Agent",
            model=self.model_factory.create_router_model(),
            system_prompt=self.system_prompt,
            instructions=load_prompt("router_instructions.md").render(),
            output_type=RouterOutput,
            deps_type=Deps,
            retries=3,
        )

    def _create_experiment_creator_agent(self) -> Agent[Deps, ExperimentDefinition]:
        """Create agent for generating experiment definitions.

        Returns:
            Configured experiment creator agent
        """
        logger.info("Creating Experiment Creator agent...")
        return Agent[Deps, ExperimentDefinition](
            name="Experiment Creator Agent",
            model=self.model_factory.create_agentic_model(),
            system_prompt=self.system_prompt,
            instructions=load_prompt("experiment_creator_instructions.md").render(),
            output_type=ExperimentDefinition,
            deps_type=Deps,
            model_settings=ModelSettings(max_tokens=10_000),
            tools=[
                retrieve_metrics_docs,
            ],
        )

    def _create_internal_database_agent(self) -> Agent[Deps, str]:
        """Create agent for querying internal databases.
        Returns:
            Configured database query agent
        """
        logger.info("Creating Internal Database agent...")
        return Agent[Deps, str](
            name="Internal DB Search Agent",
            model=self.model_factory.create_agentic_model(),
            system_prompt=self.system_prompt,
            instructions=load_prompt("sql_expert_instructions.md").render(),
            output_type=str,
            deps_type=Deps,
            model_settings=ModelSettings(max_tokens=10_000),
            tools=[
                retrieve_internal_db,
            ],
        )

    def _create_experiment_analyst_agent(self) -> Agent[Deps, str]:
        """Create agent for analyzing experimental results.

        Returns:
            Configured experiment analyst agent with context injection
        """
        logger.info("Creating Experiment Analyst agent...")
        return Agent[Deps, str](
            name="Experiment Analyst Agent",
            model=self.model_factory.create_tool_thinker_model(),
            system_prompt=self.system_prompt,
            instructions=load_prompt("experiment_analyst_instructions.md").render(),
            output_type=str,
            deps_type=Deps,
            model_settings=ModelSettings(max_tokens=20_000),
            tools=[
                get_expanto_app_context,
            ],
        )

    def _create_internet_search_agent(self) -> Agent[Deps, str]:
        """Create agent for performing internet searches.

        Returns:
            Configured internet search agent
        """
        logger.info("Creating Internet Search agent...")
        return Agent[Deps, str](
            name="Internet Search Agent",
            model=self.model_factory.create_tool_thinker_model(),
            system_prompt=self.system_prompt,
            instructions=load_prompt("internet_search_instructions.md").render(),
            output_type=str,
            deps_type=Deps,
            model_settings=ModelSettings(max_tokens=10_000),
            tools=[tavily_search_tool(self.tavily_api_key)],
        )

    def _create_expanto_assistant(self) -> Agent[Deps, str]:
        logger.info("Creating Universal agent...")
        return Agent[Deps, str](
            name="Expanto Assistant Agent",
            model=self.model_factory.create_tool_thinker_model(),
            system_prompt=self.system_prompt,
            instructions="Use as many tool call as you needed",
            output_type=str,
            deps_type=Deps,
            model_settings=ModelSettings(max_tokens=10_000),
            tools=[
                retrieve_relevant_docs,
                retrieve_codebase_docs,
            ],
        )

    def _create_universal_agent(self) -> Agent[Deps, str]:
        """Create general-purpose agent for miscellaneous tasks.

        Returns:
            Configured multipurpose agent
        """
        logger.info("Creating Universal agent...")
        return Agent[Deps, str](
            name="Universal Agent",
            model=self.model_factory.create_tool_thinker_model(),
            system_prompt=self.system_prompt,
            instructions="You are the Universal Expanto Agent. "
            "Use any tools if you need to answer user question or execute user task",
            output_type=str,
            deps_type=Deps,
            model_settings=ModelSettings(max_tokens=10_000),
            tools=[
                retrieve_metrics_docs,
                retrieve_relevant_docs,
                retrieve_codebase_docs,
                get_expanto_app_context,
            ],
        )

    def init_agents(self) -> AgentManager:
        """Initialize all available agents.

        Returns:
            Self for method chaining
        """
        logger.info("Initializing agents...")

        self.agents["route"] = self._create_router_agent()
        self.agents["create_experiment"] = self._create_experiment_creator_agent()
        self.agents["analyze_experiment"] = self._create_experiment_analyst_agent()
        self.agents["query_internal_db"] = self._create_internal_database_agent()
        self.agents["internet_search"] = self._create_internet_search_agent()
        self.agents["expanto_assistant"] = self._create_expanto_assistant()
        self.agents["universal"] = self._create_universal_agent()

        for agent in self.agents.values():
            print(agent)
            logger.instrument_pydantic_ai(agent)

        return self

    def get_agent(self, route_id: str) -> Any:
        """Get agent by route identifier.

        Args:
            route_id: Identifier for the desired agent type

        Returns:
            Requested agent or multipurpose agent as fallback
        """
        if not self.agents:
            self.init_agents()
        return self.agents.get(route_id, self.agents["universal"])


class AgentOrchestrator:
    """Orchestrates agent selection and execution pipeline.

    Handles the complete flow from user input through routing to final response,
    including fallback mechanisms and error handling.
    """

    def __init__(self, agent_manager: AgentManager) -> None:
        """Initialize the orchestrator.

        Args:
            agent_manager: Manager instance for accessing agents
        """
        self.agent_manager = agent_manager

    async def process(self, user_input: str, deps: Deps, message_history: list[Any]) -> OrchestrationResult:
        """Process user input through the complete agent pipeline.

        Args:
            user_input: Raw user input/query
            deps: Runtime dependencies
            message_history: Previous conversation context

        Returns:
            Orchestration result with agent output and metadata
        """

        router = self.agent_manager.get_agent(route_id="route")
        route_response = await router.run(user_input, deps=deps, message_history=message_history)
        logger.debug(f"Route response: {route_response}")
        route_output = route_response.output

        if route_output.confidence < 0.5 and route_output.follow_up_questions:
            return OrchestrationResult(
                output=route_response.output.follow_up_questions,
                message_history=message_history,
                usage=route_response.usage(),
                thinking=None,
            )

        selected_agent = self.agent_manager.get_agent(route_id=route_output.route_id)
        logger.info(f"Router decision: {route_output.route_id} → Selected: {selected_agent.name}")
        try:
            response = await selected_agent.run(user_input, deps=deps, message_history=message_history)
        except Exception as e:
            logger.error(f"Agent {selected_agent.name} failed: {e}")
            logger.info("Falling back to Multipurpose agent")
            multipurpose_agent = self.agent_manager.get_agent("multipurpose")
            response = await multipurpose_agent.run(user_input, deps=deps, message_history=message_history)

        # Extract thinking parts
        def extract_thinking_parts(new_messages: list[ModelMessage]) -> str | None:
            model_thinking_parts = []
            try:
                for message in response.new_messages():
                    if isinstance(message, ModelResponse):
                        for part in message.parts:
                            if isinstance(part, ThinkingPart):
                                model_thinking_parts.append(part.content)
            except Exception:
                return None
            return "\n".join(model_thinking_parts) if model_thinking_parts else None

        response_thinking = extract_thinking_parts(response.new_messages())
        cleaned_messages = drop_empty_messages(response.all_messages())

        return OrchestrationResult(
            output=response.output,
            message_history=cleaned_messages,
            thinking=response_thinking,
            usage=response.usage(),
        )
