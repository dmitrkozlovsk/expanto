from unittest.mock import patch

from src.domain.enums import CalculationPurpose, JobStatus
from src.domain.results import MetricResult
from src.services.entities.handlers import JobHandler, PrecomputeHandler

# ------------------------------- CalculationRunner -----------------------------


def test_mock_calculation_runner_run_calculations_regular(
    mock_calculation_runner, mock_observation, mock_metric_results, tables, engine
):
    """Test calculation runner executes regular calculations successfully."""
    mock_calculation_runner.exec._connector.fetch_results.return_value = mock_metric_results
    # Execute
    result = mock_calculation_runner.run_calculation(
        obs=mock_observation, purpose=CalculationPurpose.REGULAR
    )

    assert result.job_id == 1
    assert result.success is True
    assert len(result.metric_results) == 2
    assert result.metric_results[0].metric_name == "conversion_rate"
    # get result of job
    executed_job = JobHandler(engine).get(1)
    rendered_query = mock_calculation_runner.renderer.render(
        obs=mock_observation, purpose=CalculationPurpose.REGULAR
    )
    mock_calculation_runner.exec._connector.fetch_results.assert_called_once_with(rendered_query)
    assert executed_job.status == JobStatus.COMPLETED

    # check the metrics
    stored_metrics = PrecomputeHandler(engine).select(filters={"job_id__eq": executed_job.id})
    metric_names_set = set([row[0].metric_name for row in stored_metrics])
    mock_metric_names_set = set([metric.metric_name for metric in mock_metric_results])
    assert metric_names_set == mock_metric_names_set


def test_mock_calculation_runner_run_calculations_planning(
    mock_calculation_runner, mock_observation, mock_metric_results, tables, engine
):
    """Test calculation runner executes planning calculations without storing."""
    mock_calculation_runner.exec._connector.fetch_results.return_value = mock_metric_results
    # Execute
    result = mock_calculation_runner.run_calculation(
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


def test_mock_calculation_runner_run_calculations_planning_failed(
    mock_calculation_runner, mock_observation, mock_metric_results, tables, engine
):
    """Test calculation runner handles database errors properly."""
    mock_calculation_runner.exec._connector.fetch_results.side_effect = Exception("Database error")
    mock_calculation_runner.run_calculation(obs=mock_observation, purpose=CalculationPurpose.REGULAR)
    executed_job = JobHandler(engine).get(1)
    assert executed_job.status == JobStatus.FAILED
    assert "Database error" in executed_job.error_message
    stored_metrics = PrecomputeHandler(engine).select(filters={"job_id__eq": executed_job.id})
    assert not stored_metrics


def test_calculation_runner_job_creation_failure(mock_calculation_runner, mock_observation):
    """Test that CalculationRunner handles job creation failure."""
    with patch("src.services.runners.executors.JobHandler.create", return_value=None):
        result = mock_calculation_runner.run_calculation(
            obs=mock_observation, purpose=CalculationPurpose.REGULAR
        )

    assert result.success is False
    assert result.job_id is None
    assert "Could not create job for observation #1" in result.error_message


def test_calculation_runner_job_creation_db_exception(mock_calculation_runner, mock_observation):
    """Test that CalculationRunner handles job creation failure due to DB exception."""
    with patch("src.services.runners.executors.JobHandler.create", side_effect=Exception("DB is down")):
        result = mock_calculation_runner.run_calculation(
            obs=mock_observation, purpose=CalculationPurpose.REGULAR
        )

    assert result.success is False
    assert result.job_id is None
    assert "Job create failed for observation #1: DB is down" in result.error_message


def test_calculation_runner_render_failure(mock_calculation_runner, mock_observation, engine, tables):
    """Test that CalculationRunner handles query rendering failure."""
    with patch.object(mock_calculation_runner.renderer, "render", side_effect=Exception("Template error")):
        result = mock_calculation_runner.run_calculation(
            obs=mock_observation, purpose=CalculationPurpose.REGULAR
        )

    assert result.success is False
    assert result.job_id == 1
    assert "Render error: Template error" in result.error_message

    executed_job = JobHandler(engine).get(1)
    assert executed_job.status == JobStatus.FAILED
    assert "Render error: Template error" in executed_job.error_message


def test_calculation_runner_store_metrics_failure(
    mock_calculation_runner, mock_observation, mock_metric_results, engine, tables
):
    """Test that CalculationRunner handles storing metrics failure."""
    mock_calculation_runner.exec._connector.fetch_results.return_value = mock_metric_results
    with patch.object(
        mock_calculation_runner.jobs,
        "store_metrics",
        side_effect=Exception("DB connection failed"),
    ):
        result = mock_calculation_runner.run_calculation(
            obs=mock_observation, purpose=CalculationPurpose.REGULAR
        )

    assert result.success is False
    assert result.job_id == 1
    assert "Store error: DB connection failed" in result.error_message

    executed_job = JobHandler(engine).get(1)
    assert executed_job.status == JobStatus.FAILED
    assert "Store error: DB connection failed" in executed_job.error_message
