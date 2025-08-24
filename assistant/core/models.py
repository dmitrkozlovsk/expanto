"""Model factory for creating AI model instances."""

from __future__ import annotations

from pydantic_ai.models import Model
from pydantic_ai.providers import Provider
from pydantic_ai.settings import ModelSettings

from src.settings import AssistantModels


class ModelFactory[P: Provider, M: Model]:
    """Factory for creating AI model instances with consistent configuration."""

    def __init__(
        self, provider_cls: type[P], model_cls: type[M], api_key: str, assistant_models: AssistantModels
    ) -> None:
        """Initialize the model factory.

        Args:
            provider_cls: Provider class (e.g., OpenAI, Anthropic)
            model_cls: Model class for the specific provider
            api_key: Authentication key for the provider
            assistant_models: Configuration containing model names
        """
        self.provider_cls = provider_cls
        self.model_cls = model_cls
        self.api_key = api_key
        self.assistant_models = assistant_models

    def _create_model(self, model_name: str, **kwargs) -> M:
        """Create a model instance with the specified name.

        Args:
            model_name: Name/identifier of the model to create

        Returns:
            Configured model instance
        """
        model = self.model_cls(
            model_name=model_name,  # type: ignore[call-arg]
            provider=self.provider_cls(api_key=self.api_key),  # type: ignore[call-arg]
            **kwargs,
        )
        return model

    def create_router_model(self) -> M:
        """Create model optimized for routing decisions with structured output.

        Returns:
            Model instance configured for routing tasks
        """
        return self._create_model(self.assistant_models.fast, settings=ModelSettings(temperature=0.3))

    def create_tool_thinker_model(self) -> M:
        return self._create_model(self.assistant_models.tool_thinker)

    def create_agentic_model(self) -> M:
        return self._create_model(self.assistant_models.agentic)
