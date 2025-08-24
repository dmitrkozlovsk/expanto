from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from typing import Any

import pandas as pd  # type: ignore
import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session

from src.domain.models import Base, CalculationJob, Experiment, Observation, Precompute
from src.services.entities.handlers import (
    ExperimentHandler,
    JobHandler,
    ObservationHandler,
    PrecomputeHandler,
)
from src.utils import DatetimeUtils


def test_crud_experiment(engine: Engine, tables: Any) -> None:
    """Test CRUD operations for Experiment entities."""
    exp_handler = ExperimentHandler(engine)
    exp_data = {"name": "Test Experiment handler", "status": "pending"}
    exp_handler.create(**exp_data)
    with Session(engine) as local_session:
        stmt = select(Experiment)
        result = local_session.execute(stmt).all()
        assert len(result) == 1

    exp = exp_handler.get(result[0][0].id)
    assert isinstance(exp, Experiment)
    assert exp.id == result[0][0].id
    assert exp_handler.get(1000000) is None

    exp_handler.update(result[0][0].id, description="new description")
    with Session(engine) as local_session:
        stmt = select(Experiment)
        result = local_session.execute(stmt).all()
        assert result[0][0].description == "new description"

    exp_handler.delete(result[0][0].id)
    with Session(engine) as local_session:
        stmt = select(Experiment)
        result = local_session.execute(stmt).all()
        assert len(result) == 0


def test_crud_observation(engine: Engine, tables: Any) -> None:
    """Test CRUD operations for Observation entities."""
    obs_handler = ObservationHandler(engine)
    obs_data = {
        "experiment_id": 1,
        "name": "Test Observation handler",
        "db_experiment_name": "db_test_experiment",
        "split_id": "split_test",
    }
    obs_handler.create(**obs_data)
    with Session(engine) as local_session:
        stmt = select(Observation)
        result = local_session.execute(stmt).all()
        assert len(result) == 1

    obs = obs_handler.get(result[0][0].id)
    assert isinstance(obs, Observation)
    assert obs_handler.get(1000000) is None

    obs_handler.update(result[0][0].id, name="new name")
    with Session(engine) as local_session:
        stmt = select(Observation)
        result = local_session.execute(stmt).all()
        assert result[0][0].name == "new name"

    obs_handler.delete(result[0][0].id)
    with Session(engine) as local_session:
        stmt = select(Observation)
        result = local_session.execute(stmt).all()
        assert len(result) == 0


def test_crud_calculation_job(engine: Engine, tables: Any) -> None:
    """Test CRUD operations for CalculationJob entities."""
    calc_job_handler = JobHandler(engine)
    calc_job_data = {
        "observation_id": 1,
        "query": "SELECT * FROM test_table",
        "status": "pending",
        "extra": {"execution_time_ms": 2500, "bytes_read": 1536000},
    }
    calc_job_handler.create(**calc_job_data)
    with Session(engine) as local_session:
        stmt = select(CalculationJob)
        result = local_session.execute(stmt).all()
        assert len(result) == 1

    calc_job = calc_job_handler.get(result[0][0].id)
    assert isinstance(calc_job, CalculationJob)
    assert calc_job.extra is not None
    assert calc_job.extra["execution_time_ms"] == 2500
    assert calc_job.extra["bytes_read"] == 1536000
    assert calc_job_handler.get(1000000) is None

    calc_job_handler.update(
        result[0][0].id,
        status="completed",
        extra={"execution_time_ms": 2500, "bytes_read": 1536000, "final_status": "success"},
    )
    with Session(engine) as local_session:
        stmt = select(CalculationJob)
        result = local_session.execute(stmt).all()
        assert result[0][0].status == "completed"
        assert result[0][0].extra["final_status"] == "success"

    calc_job_handler.delete(result[0][0].id)
    with Session(engine) as local_session:
        stmt = select(CalculationJob)
        result = local_session.execute(stmt).all()
        assert len(result) == 0


def test_crud_precompute(engine: Engine, tables: Any) -> None:
    """Test CRUD operations for Precompute entities."""
    precompute_handler = PrecomputeHandler(engine)
    precompute_data = {
        "job_id": 1,
        "group_name": "test_group",
        "metric_name": "test_metric",
        "metric_display_name": "test_metric_display_name",
        "metric_type": "avg",
        "observation_cnt": 100,
        "metric_value": 1_000_000,
        "numerator_avg": 100,
        "denominator_avg": 200,
        "numerator_var": 10,
        "denominator_var": 20,
        "covariance": 0.5,
    }
    precompute_handler.create(**precompute_data)
    with Session(engine) as local_session:
        stmt = select(Precompute)
        result = local_session.execute(stmt).all()
        assert len(result) == 1

    precompute = precompute_handler.get(result[0][0].id)
    assert isinstance(precompute, Precompute)
    assert precompute_handler.get(1000000) is None

    precompute_handler.update(result[0][0].id, metric_value=0.6)
    with Session(engine) as local_session:
        stmt = select(Precompute)
        result = local_session.execute(stmt).all()
        assert result[0][0].metric_value == 0.6

    precompute_handler.delete(result[0][0].id)
    with Session(engine) as local_session:
        stmt = select(Precompute)
        result = local_session.execute(stmt).all()
        assert len(result) == 0


def test_select_experiments(
    engine: Engine,
    mock_experiments: Any,
    mock_observations: Any,
    mock_calculation_jobs: Any,
    mock_precomputes: Any,
) -> None:
    """Test experiment selection with various filters and options."""
    exp_handler = ExperimentHandler(engine)

    result = exp_handler.select()
    assert result is not None
    assert len(result) == 2

    result = exp_handler.select(limit=1)
    assert result is not None
    assert len(result) == 1

    filters: dict[str, Any] = {"status__eq": "completed"}
    result = exp_handler.select(filters=filters)
    assert result is not None
    assert result[0][0].name == "Button Color Test"

    filters = {"key_metrics__contains": "subscription_rate"}
    result = exp_handler.select(filters=filters)
    assert result is not None
    assert result[0][0].name == "Pricing Page Redesign"

    filters = {"status__eq": "completed", "_created_at__gte": DatetimeUtils.utc_now() - timedelta(days=60)}
    result = exp_handler.select(filters=filters)
    assert result is not None
    assert result[0][0].name == "Button Color Test"


def test_select_observations(
    engine: Engine,
    mock_experiments: Any,
    mock_observations: Any,
    mock_calculation_jobs: Any,
    mock_precomputes: Any,
) -> None:
    """Test observation selection with filters, sorting, and return formats."""
    obs_handler = ObservationHandler(engine)

    result = obs_handler.select()
    assert result is not None
    assert len(result) == 3

    result = obs_handler.select(sort_by="_created_at", sort_order="desc")
    assert result is not None
    assert result[0][0].name == "Main Conversion Analysis"

    filters: dict[str, Any] = {"metric_tags__contains__or": ["mobile", "pricing"]}
    result = obs_handler.select(filters=filters)
    assert result is not None
    assert len(result) == 2

    filters = {"name__ilike": "%pricing page%"}
    result = obs_handler.select(filters=filters)
    assert result is not None
    assert result[0][0].name == "Pricing Page Analysis"

    result = obs_handler.select(filters=filters, return_pandas=True)
    assert isinstance(result, pd.DataFrame)

    result = obs_handler.select(filters=filters, columns=["id", "db_experiment_name"])
    assert result is not None
    assert len(result[0]) == 2


def test_select_calculation_jobs(
    engine: Engine,
    mock_experiments: Any,
    mock_observations: Any,
    mock_calculation_jobs: Any,
    mock_precomputes: Any,
) -> None:
    """Test calculation job selection with compound filters."""
    job_handler = JobHandler(engine)

    result = job_handler.select()
    assert result is not None
    assert len(result) == 4

    filters = {
        "observation_id__eq__or": [1, 2],
        "status__eq": "completed",
    }
    result = job_handler.select(filters=filters)
    assert result is not None
    assert set(row[0].observation_id for row in result) == {1, 2}


def test_select_precomputes(
    engine: Engine,
    mock_experiments: Any,
    mock_observations: Any,
    mock_calculation_jobs: Any,
    mock_precomputes: Any,
) -> None:
    """Test precompute selection with complex filtering and pandas output."""
    precompute_handler = PrecomputeHandler(engine)
    filters = {
        "metric_name__eq__or": ["conversion_rate", "mobile_conversion_rate"],
        "metric_type__eq": "ratio",
        "observation_cnt__gte": 500,
        "metric_value__gt": 0.1,
    }

    expected_columns = ["id", "metric_name", "metric_value", "observation_cnt", "_created_at"]

    result = precompute_handler.select(
        filters=filters,
        sort_by="_created_at",
        sort_order="desc",
        limit=2,
        return_pandas=True,
        columns=expected_columns,
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == expected_columns
    assert len(result) == 2
    assert result["_created_at"].is_monotonic_decreasing
    assert all(result["metric_name"].isin(["conversion_rate", "mobile_conversion_rate"]))


def test_bulk_insert_experiments(engine: Engine, tables: Any) -> None:
    """Test bulk insertion of multiple experiments."""
    exp_handler = ExperimentHandler(engine)
    experiments_data = [
        {
            "name": f"Test Experiment {i}",
            "status": "pending",
            "description": f"Description for experiment {i}",
        }
        for i in range(5)
    ]

    created_experiments = exp_handler.bulk_insert(experiments_data)
    assert len(created_experiments) == 5

    with Session(engine) as local_session:
        stmt = select(Experiment)
        result = local_session.execute(stmt).all()
        assert len(result) == 5

        for i, row in enumerate(result):
            exp = row[0]
            assert exp.name == f"Test Experiment {i}"
            assert exp.status == "pending"
            assert exp.description == f"Description for experiment {i}"


def test_bulk_insert_precomputes(engine: Engine, tables: Any) -> None:
    """Test bulk insertion of large number of precomputes."""
    precompute_handler = PrecomputeHandler(engine)
    n_precomputes = 5000
    precomputes_data = [
        {
            "job_id": i + 1,
            "group_name": f"test_group_{i}",
            "metric_name": f"test_metric_{i}",
            "metric_display_name": f"test_display_name_{i}",
            "metric_type": "avg",
            "observation_cnt": 100 + i,
            "metric_value": 1.0 + i * 0.1,
            "numerator_avg": 100 + i,
            "denominator_avg": 200 + i,
            "numerator_var": 10 + i,
            "denominator_var": 20 + i,
            "covariance": 0.5 + i * 0.1,
        }
        for i in range(n_precomputes)
    ]

    created_precomputes = precompute_handler.bulk_insert(precomputes_data)
    assert len(created_precomputes) == n_precomputes

    with Session(engine) as local_session:
        stmt = select(Precompute)
        result = local_session.execute(stmt).all()
        assert len(result) == n_precomputes

        for i, row in enumerate(result):
            precompute = row[0]
            assert precompute.job_id == i + 1
            assert precompute.group_name == f"test_group_{i}"
            assert precompute.metric_name == f"test_metric_{i}"
            assert precompute.observation_cnt == 100 + i


@pytest.fixture
def temp_db_file(tmp_path: Any) -> str:
    """Create a temporary SQLite database file using pytest's tmp_path."""
    db_file = tmp_path / "test_db.sqlite"
    return str(db_file)


@pytest.mark.parametrize("n_tasks", [1000])
def test_concurrent_create_experiments(temp_db_file: str, n_tasks: int) -> None:
    """Test concurrent experiment creation using ThreadPoolExecutor."""
    local_engine = create_engine(
        f"sqlite:///{temp_db_file}",
        echo=True,
        connect_args={"check_same_thread": False},
        pool_size=2,
        max_overflow=1,
        pool_timeout=10,
        pool_recycle=900,
    )
    Base.metadata.create_all(local_engine)

    def create_experiment_task(i):
        # local_session = Session(local_engine)
        local_handler = ExperimentHandler(local_engine)
        experiment = local_handler.create(
            name=f"Exp {i}", status="pending ", description=f"test description {i}"
        )
        experiment_id = experiment.id
        return experiment_id

    # Use ThreadPoolExecutor for simultaneous task execution.
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for i in range(n_tasks):
            futura = executor.submit(create_experiment_task, i)
            futures.append(futura)
            [future.result() for future in as_completed(futures)]

    # check that we added n_tasks experiments
    local_session = Session(local_engine)
    count = local_session.query(Experiment).count()
    local_session.close()
    assert count == n_tasks
