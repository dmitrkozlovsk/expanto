"""Resource management module for application."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from functools import reduce

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.logger_setup import logger
from src.services.metric_register import Metrics
from src.services.runners.connectors import ConnectorResolver
from src.services.runners.executors import CalculationRunner
from src.services.runners.renderer import QueryRenderer
from src.settings import AssistantServiceCfg, Config, Secrets


@st.cache_resource
def get_thread_pool_executor(max_workers: int | None = None) -> ThreadPoolExecutor:
    if not max_workers:
        cores = os.cpu_count() or 4
        max_workers = min(64, cores * 6)
    return ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="expanto-app")


@st.cache_resource
def load_engine() -> Engine:
    """Create and cache a SQLAlchemy database engine.

    Creates a SQLAlchemy engine using connection parameters from secrets.
    The engine is cached using Streamlit's resource caching mechanism.

    Returns:
        Engine: A configured SQLAlchemy engine for database operations.

    Note:
        This function is decorated with @st.cache_resource to ensure the engine
        is created only once per Streamlit session.
    """
    secrets = Secrets()
    engine = create_engine(
        secrets.internal_db.engine_str.get_secret_value(),
        **secrets.internal_db.connect_args,
        hide_parameters=True,
    )
    logger.instrument_sqlalchemy(engine=engine)
    return engine


@st.cache_resource
def load_metrics_handler() -> Metrics:
    """Create and cache a metrics handler.

    Initializes a Metrics object using the metrics directory path from the
    application configuration. The handler is cached for performance.

    Returns:
        Metrics: A configured metrics handler for managing experiment metrics.

    Note:
        This function is decorated with @st.cache_resource to ensure the handler
        is created only once per Streamlit session.
    """
    expanto_cfg = Config()
    return Metrics(expanto_cfg.metrics.dir)


@st.cache_resource
def load_metric_groups() -> list[str]:
    """Load and cache available metric group names.

    Retrieves all available metric group names from the metrics handler.
    Used for filtering and organizing metrics in the UI.

    Returns:
        list[str]: A list of metric group names.

    Note:
        This function is decorated with @st.cache_resource to ensure the list
        is computed only once per Streamlit session.
    """
    metrics = load_metrics_handler()
    groups = [group.metric_group_name for group in metrics.groups]
    return sorted(set(groups))


@st.cache_resource
def load_metric_tags() -> list[str]:
    """Load and cache all available metric tags.

    Extracts and aggregates all unique tags from all metrics in the system.
    Tags are used for categorizing and filtering metrics in the UI.

    Returns:
        list[str]: A list of unique metric tags across all metrics.

    Note:
        This function is decorated with @st.cache_resource to ensure the tags
        are computed only once per Streamlit session.
    """
    metrics = load_metrics_handler()
    all_tags: set[str] = reduce(
        lambda acc, metric: acc.union(set(metric[1].tags or [])), metrics.flat.items(), set()
    )
    return sorted(list(all_tags))


@st.cache_resource
def load_calculation_runner() -> CalculationRunner:
    """Create and cache a calculation runner.

    Initializes a CalculationRunner with all necessary dependencies including
    connector, database engine, and query renderer. The runner is cached for
    optimal performance across the application.

    Returns:
        CalculationRunner: A fully configured runner for executing calculations
        and queries.

    Note:
        This function is decorated with @st.cache_resource to ensure the runner
        is created only once per Streamlit session.
    """
    secrets = Secrets()
    expanto_cfg = Config()
    metrics = load_metrics_handler()

    connector_ = ConnectorResolver.resolve(
        precompute_db_name=expanto_cfg.precompute_db.name, secrets=secrets
    )
    engine_ = load_engine()
    renderer_ = QueryRenderer(query_config=expanto_cfg.queries, metrics=metrics)
    return CalculationRunner(connector_, engine_, renderer_)


@st.cache_resource
def load_assistant_service_cfg() -> AssistantServiceCfg:
    """Load and cache assistant service configuration.

    Retrieves the assistant service configuration from the application
    configuration. Used for configuring HTTP clients and service endpoints.

    Returns:
        AssistantServiceCfg: Configuration object for the assistant service.

    Note:
        This function is decorated with @st.cache_resource to ensure the
        configuration is loaded only once per Streamlit session.
    """
    return Config().assistant.service


@st.cache_resource
def load_calculation_scenarios() -> list[str]:
    config = Config()
    return list(config.queries.scenarios.keys())


def init_resources() -> tuple[Engine, Metrics, CalculationRunner, AssistantServiceCfg]:
    """Initialize and return all main application resources.

    Convenience function that loads and returns all primary resources needed
    by the Expanto application in a single call.

    Returns:
        tuple[Engine, Metrics, CalculationRunner, AssistantServiceCfg]: A tuple containing:
            - Engine: SQLAlchemy database engine
            - Metrics: Metrics handler for managing experiment metrics
            - CalculationRunner: Runner for executing calculations and queries
            - AssistantServiceCfg: Configuration for the assistant service

    Note:
        All returned resources are cached through their respective load functions.
    """
    return (
        load_engine(),
        load_metrics_handler(),
        load_calculation_runner(),
        load_assistant_service_cfg(),
    )
