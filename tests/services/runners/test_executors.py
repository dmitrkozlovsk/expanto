from unittest.mock import patch

import pytest

from src.domain.enums import CalculationPurpose, JobStatus
from src.domain.models import Observation
from src.domain.results import MetricResult
from src.services.entities.dtos import ObservationDTO
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


# ------------------------ NEW TESTS FOR EDGE CASES ------------------------


@pytest.mark.parametrize(
    "obs_factory, expected_obs_id",
    [
        (
            lambda: Observation(
                id=1,
                experiment_id=1,
                name="SQLAlchemy Test Observation",
                db_experiment_name="test_experiment_db_name",
                split_id="user_id",
                calculation_scenario="base",
                exposure_start_datetime="2024-01-01",
                exposure_end_datetime="2024-01-31",
                calc_start_datetime="2024-01-01",
                calc_end_datetime="2024-01-31",
                exposure_event="view",
                audience_tables=["active_users"],
                filters=["platform='web'"],
                custom_test_ids_query=None,
            ),
            1,
        ),
        (
            lambda: ObservationDTO(
                id=None,
                experiment_id=None,
                name="DTO Test Observation",
                db_experiment_name="test_experiment_db_name",
                split_id="user_id",
                calculation_scenario="base",
                exposure_start_datetime="2024-01-01",
                exposure_end_datetime="2024-01-31",
                calc_start_datetime="2024-01-01",
                calc_end_datetime="2024-01-31",
                exposure_event="view",
                audience_tables=["active_users"],
                filters=["platform='web'"],
                custom_test_ids_query=None,
                _created_at=None,
                _updated_at=None,
            ),
            None,
        ),
        (
            lambda: ObservationDTO(
                id=42,
                experiment_id=1,
                name="DTO Test Observation with ID",
                db_experiment_name="test_experiment_db_name",
                split_id="user_id",
                calculation_scenario="base",
                exposure_start_datetime="2024-01-01",
                exposure_end_datetime="2024-01-31",
                calc_start_datetime="2024-01-01",
                calc_end_datetime="2024-01-31",
                exposure_event="view",
                audience_tables=["active_users"],
                filters=["platform='web'"],
                custom_test_ids_query=None,
                _created_at=None,
                _updated_at=None,
            ),
            42,
        ),
    ],
)
def test_calculation_runner_with_different_observation_types(
    mock_calculation_runner,
    mock_metric_results,
    engine,
    tables,
    obs_factory,
    expected_obs_id,
):
    """Test calculation runner with different observation types and id scenarios."""
    # Create observation using the factory function
    obs = obs_factory()

    # Mock the connector to return metric results
    mock_calculation_runner.exec._connector.fetch_results.return_value = mock_metric_results

    # Determine purpose based on observation type
    purpose = CalculationPurpose.PLANNING if expected_obs_id is None else CalculationPurpose.REGULAR

    # Execute with the observation
    result = mock_calculation_runner.run_calculation(obs=obs, purpose=purpose)

    # Verify that the calculation succeeds
    assert result.success is True
    assert result.job_id == 1
    assert len(result.metric_results) == 2

    # Verify job was created with correct observation_id
    executed_job = JobHandler(engine).get(1)
    assert executed_job.status == JobStatus.COMPLETED
    assert executed_job.observation_id == expected_obs_id


@pytest.mark.parametrize(
    "obs_factory,expected_obs_id_in_message,error_type,expected_error_message",
    [
        (
            lambda: ObservationDTO(
                id=None,
                experiment_id=None,
                name="Test",
                db_experiment_name="test",
                split_id="user_id",
                calculation_scenario="base",
                exposure_start_datetime="2024-01-01",
                exposure_end_datetime="2024-01-31",
                calc_start_datetime="2024-01-01",
                calc_end_datetime="2024-01-31",
                exposure_event="view",
                audience_tables=["active_users"],
                filters=None,
                custom_test_ids_query=None,
                _created_at=None,
                _updated_at=None,
            ),
            "None",
            "job_creation_none",
            "Could not create job for observation #None",
        ),
        (
            lambda: ObservationDTO(
                id=99,
                experiment_id=1,
                name="Test",
                db_experiment_name="test",
                split_id="user_id",
                calculation_scenario="base",
                exposure_start_datetime="2024-01-01",
                exposure_end_datetime="2024-01-31",
                calc_start_datetime="2024-01-01",
                calc_end_datetime="2024-01-31",
                exposure_event="view",
                audience_tables=["active_users"],
                filters=None,
                custom_test_ids_query=None,
                _created_at=None,
                _updated_at=None,
            ),
            "99",
            "job_creation_exception",
            "Job create failed for observation #99: DB connection failed",
        ),
    ],
)
def test_calculation_runner_job_creation_error_handling(
    mock_calculation_runner,
    engine,
    tables,
    obs_factory,
    expected_obs_id_in_message,
    error_type,
    expected_error_message,
):
    """Test job creation error handling with different observation types and error scenarios."""
    # Create observation using the factory function
    obs = obs_factory()

    if error_type == "job_creation_none":
        mock_patch = patch("src.services.runners.executors.JobHandler.create", return_value=None)
    else:  # job_creation_exception
        mock_patch = patch(
            "src.services.runners.executors.JobHandler.create",
            side_effect=Exception("DB connection failed"),
        )

    with mock_patch:
        result = mock_calculation_runner.run_calculation(obs=obs, purpose=CalculationPurpose.PLANNING)

    assert result.success is False
    assert result.job_id is None
    assert expected_error_message in result.error_message
