"""Application settings and configuration models."""

from __future__ import annotations

from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class MetricsConfig(BaseModel):
    """Configuration for metrics directory and settings."""

    dir: str


class QueryTemplatesConfig(BaseModel):
    """Configuration for SQL query templates."""

    dir: str
    scenarios: dict[str, str] = Field(default_factory=dict)


class PrecomputeDBConfig(BaseModel):
    """Configuration for precompute database selection."""

    name: Literal["snowflake", "bigquery"]


class AssistantModels(BaseModel):
    """Configuration for AI agent model settings."""

    fast: str
    tool_thinker: str
    agentic: str


class AssistantServiceCfg(BaseModel):
    """Configuration for assistant service connection."""

    url: AnyHttpUrl
    timeout_seconds: int = 600
    enable_streaming: bool = True
    auto_scroll: bool = True


class AssistantConfig(BaseModel):
    """Main assistant configuration container."""

    provider: str
    models: AssistantModels
    service: AssistantServiceCfg


class LogfireConfig(BaseModel):
    """Configuration for Logfire settings."""

    send_to_logfire: bool


class Config(BaseSettings):
    """Main application configuration settings."""

    metrics: MetricsConfig
    queries: QueryTemplatesConfig
    precompute_db: PrecomputeDBConfig
    assistant: AssistantConfig
    logfire: LogfireConfig

    model_config = SettingsConfigDict(toml_file=".streamlit/expanto.toml", env_file=".env")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize settings sources to prioritize TOML configuration."""
        toml_src = TomlConfigSettingsSource(settings_cls)
        return (
            init_settings,
            toml_src,
        )


# ----------------------------- SECRETS ----------------------------- #


class SnowflakeCredentials(BaseModel):
    """Snowflake database connection credentials."""

    account: str
    user: SecretStr
    password: SecretStr
    warehouse: SecretStr
    database: SecretStr
    db_schema: str = Field(alias="schema")


class BigQueryCredentials(BaseModel):
    """BigQuery database connection credentials."""

    file_path: SecretStr
    project_name: SecretStr
    connection_type: Literal["application_default", "service_account"]


class InternalDBConfig(BaseModel):
    """Internal database configuration settings."""

    engine_str: SecretStr
    async_engine_str: SecretStr
    connect_args: dict


class APIKeys(BaseModel):
    """Secret API keys for AI assistant services."""

    provider_api_key: SecretStr = Field(alias="PROVIDER_API_KEY")
    tavily_api_key: SecretStr | None = Field(default=None, alias="TAVILY_API_KEY")
    logfire_token: SecretStr | None = Field(default=None, alias="LOGFIRE_TOKEN")


class Secrets(BaseSettings):
    """Application secrets and sensitive configuration."""

    snowflake: SnowflakeCredentials | None = None
    bigquery: BigQueryCredentials | None = None
    internal_db: InternalDBConfig
    api_keys: APIKeys
    model_config = SettingsConfigDict(toml_file=".streamlit/secrets.toml", env_file=".env")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize settings sources to prioritize TOML secrets configuration."""
        toml_src = TomlConfigSettingsSource(settings_cls)
        return (
            init_settings,
            toml_src,
        )

    @model_validator(mode="after")  # type: ignore[arg-type]
    def at_least_one_db(cls, data: Secrets) -> Secrets:
        """Validate that at least one database configuration is provided."""
        if data.snowflake is None and data.bigquery is None:
            raise ValueError("Credentials for at least one database are required")
        return data
