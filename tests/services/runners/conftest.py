from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.domain.enums import ExperimentMetricType
from src.domain.models import Observation
from src.domain.results import QueryMetricResult
from src.services.runners.connectors import PrecomputeConnector
from src.services.runners.executors import CalculationRunner
from src.services.runners.renderer import QueryRenderer
from src.settings import (
    BigQueryCredentials,
    SnowflakeCredentials,
)


@pytest.fixture
def fake_creds_snowflake():
    """Fixture providing Snowflake credentials for testing."""
    return SnowflakeCredentials(
        **{"user": "u", "password": "p", "account": "a", "warehouse": "w", "database": "d", "schema": "s"}
    )


@pytest.fixture
def fake_creds_bigquery_service_account():
    """Fixture providing BigQuery service account credentials."""
    return BigQueryCredentials(
        **{"connection_type": "service_account", "file_path": "/fake/path.json", "project_name": "proj"}
    )


@pytest.fixture
def query_renderer(queries_templates_config, metrics):
    """Fixture providing QueryRenderer instance with test configuration."""
    return QueryRenderer(query_config=queries_templates_config, metrics=metrics)


@pytest.fixture(autouse=True)
def patch_bigquery_client():
    """Auto-patch BigQuery client for all tests in this module."""
    with patch("src.services.runners.connectors.bigquery.Client") as MockClient:
        inst = MockClient.from_service_account_json.return_value
        yield inst


@pytest.fixture
def mock_connector():
    """Fixture providing mock database connector."""
    connector = Mock(spec=PrecomputeConnector)
    return connector


@pytest.fixture
def mock_calculation_runner(mock_connector, engine, query_renderer):
    """Fixture providing CalculationRunner instance with mocked dependencies."""
    return CalculationRunner(connector=mock_connector, engine=engine, renderer=query_renderer)


@pytest.fixture
def fake_metric_results():
    """Fixture providing fake query result data and column names."""
    fake_result = [("g", "m", "M", "ratio", 10, 0.1, 1.1, 2.2, 0.3, 0.4, 0.5)]

    cols = [
        "group_name",
        "metric_name",
        "metric_display_name",
        "metric_type",
        "observation_cnt",
        "metric_value",
        "numerator_avg",
        "denominator_avg",
        "numerator_var",
        "denominator_var",
        "covariance",
    ]
    return fake_result, cols


@pytest.fixture
def mock_observation():
    """Fixture providing mock observation object for testing."""
    return Observation(
        id=1,
        experiment_id=1,
        name="Test Observation",
        db_experiment_name="test_experiment_db_name",
        split_id="user_id",
        calculation_scenario="base",
        exposure_start_datetime=datetime(2024, 1, 1),
        exposure_end_datetime=datetime(2024, 1, 31),
        calc_start_datetime=datetime(2024, 1, 1),
        calc_end_datetime=datetime(2024, 1, 31),
        exposure_event="view",
        audience_tables=["active_users"],
        filters=["platform='web'", "(country='US' or country='CA')"],
        custom_test_ids_query=None,
        metric_tags=["main"],
        metric_groups=["core"],
    )


@pytest.fixture
def mock_metric_results():
    """Fixture providing mock QueryMetricResult objects for testing."""
    return [
        QueryMetricResult(
            group_name="control",
            metric_name="conversion_rate",
            metric_display_name="Convertion Rate",
            metric_type=ExperimentMetricType.RATIO,
            observation_cnt=1000,
            metric_value=0.15,
            numerator_avg=150,
            denominator_avg=1000,
            numerator_var=127.5,
            denominator_var=100,
            covariance=50,
        ),
        QueryMetricResult(
            group_name="treatment",
            metric_name="average_session_duration",
            metric_display_name="Average Session Duration",
            metric_type=ExperimentMetricType.RATIO,
            observation_cnt=1000,
            metric_value=0.18,
            numerator_avg=180,
            denominator_avg=1000,
            numerator_var=147.6,
            denominator_var=100,
            covariance=50,
        ),
    ]
