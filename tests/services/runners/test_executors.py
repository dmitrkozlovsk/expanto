from datetime import datetime
from unittest.mock import Mock

import pytest

from src.domain.enums import CalculationPurpose, ExperimentMetricType, JobStatus
from src.domain.models import Observation
from src.domain.results import MetricResult, QueryMetricResult
from src.services.entities.handlers import JobHandler, PrecomputeHandler
from src.services.runners.connectors import PrecomputeConnector
from src.services.runners.executors import CalculationRunner, QueryRenderer


# --------------------------------- QueryRenderer ---------------------------------
@pytest.fixture
def query_renderer(queries_templates_config, metrics):
    """Fixture providing QueryRenderer instance with test configuration."""
    return QueryRenderer(query_config=queries_templates_config, metrics=metrics)


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


def test_render_base_calculation_query_planning(query_renderer, mock_observation):
    """Test rendering base calculation query for planning purpose"""
    query = query_renderer.render_base_calculation_query(
        obs=mock_observation, purpose=CalculationPurpose.PLANNING
    )
    # TODO: fix mock observation and fix fake template according to custom_test_ids_query
    assert isinstance(query, str)
    assert "user_id as split_id" in query
    # assert "test_experiment_db_name" not in query
    assert "planning" in query
    assert "AND (platform='web')" in query
    assert "active_users" in query


def test_render_base_calculation_query_with_metric_names(query_renderer, mock_observation):
    """Test rendering base calculation query with specific metric names"""
    query = query_renderer.render_base_calculation_query(
        obs=mock_observation,
        purpose=CalculationPurpose.PLANNING,
        experiment_metric_names=["click_through_rate"],
    )

    assert isinstance(query, str)
    assert "click_through_rate" in query
    for exp_alias in ["avg_session_duration", "users_who_click_product_ratio", "avg_order_value"]:
        assert exp_alias not in query
    for user_alias in ["product_purchase_cnt", "session_duration"]:
        assert user_alias not in query


# ------------------------------- CalculationRunner -------------------------------


@pytest.fixture
def mock_connector():
    """Fixture providing mock database connector."""
    connector = Mock(spec=PrecomputeConnector)
    return connector


@pytest.fixture
def calculation_runner(mock_connector, engine, query_renderer):
    """Fixture providing CalculationRunner instance with mocked dependencies."""
    return CalculationRunner(connector=mock_connector, engine=engine, renderer=query_renderer)


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


def test_calculation_runner_run_calculations_regular(
    calculation_runner, mock_observation, mock_metric_results, tables, engine
):
    """Test calculation runner executes regular calculations successfully."""
    calculation_runner._connector.fetch_results.return_value = mock_metric_results
    # Execute
    result = calculation_runner.run_calculation(obs=mock_observation, purpose=CalculationPurpose.REGULAR)

    assert result.job_id == 1
    assert result.success is True
    assert len(result.metric_results) == 2
    assert result.metric_results[0].metric_name == "conversion_rate"
    # get result of job
    executed_job = JobHandler(engine).get(1)
    rendered_query = calculation_runner._renderer.render_base_calculation_query(
        obs=mock_observation, purpose=CalculationPurpose.REGULAR
    )
    calculation_runner._connector.fetch_results.assert_called_once_with(rendered_query)
    assert executed_job.status == JobStatus.COMPLETED

    # check the metrics
    stored_metrics = PrecomputeHandler(engine).select(filters={"job_id__eq": executed_job.id})
    metric_names_set = set([row[0].metric_name for row in stored_metrics])
    mock_metric_names_set = set([metric.metric_name for metric in mock_metric_results])
    assert metric_names_set == mock_metric_names_set


def test_calculation_runner_run_calculations_planning(
    calculation_runner, mock_observation, mock_metric_results, tables, engine
):
    """Test calculation runner executes planning calculations without storing."""
    calculation_runner._connector.fetch_results.return_value = mock_metric_results
    # Execute
    result = calculation_runner.run_calculation(
        obs=mock_observation,
        purpose=CalculationPurpose.PLANNING,
    )
    assert isinstance(result.metric_results[0], MetricResult)
    assert set([m.metric_name for m in result.metric_results]) == set(
        [m.metric_name for m in mock_metric_results]
    )

    executed_job = JobHandler(engine).get(1)
    assert executed_job is not None
    stored_metrics = PrecomputeHandler(engine).select(filters={"job_id__eq": executed_job.id})
    assert not stored_metrics


def test_calculation_runner_run_calculations_planning_failed(
    calculation_runner, mock_observation, mock_metric_results, tables, engine
):
    """Test calculation runner handles database errors properly."""
    calculation_runner._connector.fetch_results.side_effect = Exception("Database error")
    calculation_runner.run_calculation(obs=mock_observation, purpose=CalculationPurpose.REGULAR)
    executed_job = JobHandler(engine).get(1)
    assert executed_job.status == JobStatus.FAILED
    assert "Database error" in executed_job.error_message
    stored_metrics = PrecomputeHandler(engine).select(filters={"job_id__eq": executed_job.id})
    assert not stored_metrics
