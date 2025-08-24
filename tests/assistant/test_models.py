from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock

import pytest

from assistant.core.models import ModelFactory
from src.settings import AssistantModels


@pytest.fixture
def cfg() -> AssistantModels:
    return AssistantModels(
        fast="fast",
        tool_thinker="tool_thinker",
        agentic="agentic",
    )


@pytest.fixture
def provider_cls() -> Mock:
    provider_instance = Mock(name="ProviderInstance")
    return Mock(name="ProviderClass", return_value=provider_instance)


@pytest.fixture
def model_cls() -> Mock:
    def _factory(*, model_name: str, provider, settings=None):
        instance = Mock(name=f"ModelInstance<{model_name}>")
        instance.model_name = model_name
        instance.provider = provider
        instance.settings = settings
        return instance

    return Mock(name="ModelClass", side_effect=_factory)


@pytest.fixture
def factory(provider_cls, model_cls, cfg) -> ModelFactory:
    return ModelFactory(
        provider_cls=provider_cls,
        model_cls=model_cls,
        api_key="test-key",
        assistant_models=cfg,
    )


@pytest.mark.parametrize(
    "method_name, expected_model_name",
    [
        ("create_router_model", "fast"),
        ("create_tool_thinker_model", "tool_thinker"),
        ("create_agentic_model", "agentic"),
    ],
)
def test_factory_methods_return_correct_model(
    factory, provider_cls, model_cls, method_name, expected_model_name
):
    """Test that factory methods return models with correct names."""
    result = getattr(factory, method_name)()

    provider_cls.assert_called_once_with(api_key="test-key")

    # Verify the call was made with at least the expected parameters
    call_args = model_cls.call_args
    assert call_args.kwargs["model_name"] == expected_model_name
    assert call_args.kwargs["provider"] is provider_cls.return_value
    
    # For router model, verify settings parameter is passed
    if method_name == "create_router_model":
        assert "settings" in call_args.kwargs
        assert call_args.kwargs["settings"]["temperature"] == 0.3

    assert result.model_name == expected_model_name
    assert result.provider is provider_cls.return_value


def test_private_create_model(factory, provider_cls, model_cls):
    """Test private _create_model method with custom name."""
    custom_name = "totally-custom"
    instance = factory._create_model(custom_name)

    provider_cls.assert_called_once_with(api_key="test-key")
    model_cls.assert_called_once_with(
        model_name=custom_name,
        provider=provider_cls.return_value,
    )
    assert instance.model_name == custom_name
    assert instance.provider is provider_cls.return_value


def test_thread_safety(factory):
    """Test that 10 parallel calls create 10 different objects with different names."""
    names = [f"m{i}" for i in range(10)]

    def create(n):
        return factory._create_model(n).model_name

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(create, n) for n in names]

    results = [f.result() for f in as_completed(futures)]
    assert sorted(results) == sorted(names)
