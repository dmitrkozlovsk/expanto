from __future__ import annotations

from typing import TYPE_CHECKING

from src.domain.models import (
    CalculationJob,
    Experiment,
    Observation,
    Precompute,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


def test_cascade_delete_experiment_observations(session: Session) -> None:
    """Test experiment deletion cascades to associated observations."""
    exp = Experiment(name="exp1", status="running")
    obs1 = Observation(
        name="obs1", db_experiment_name="exp1", split_id="user_id", calculation_scenario="default"
    )
    obs2 = Observation(
        name="obs2", db_experiment_name="exp1", split_id="user_id", calculation_scenario="default"
    )
    exp.observations.extend([obs1, obs2])
    session.add(exp)
    session.commit()

    assert session.query(Observation).count() == 2

    session.delete(exp)
    session.commit()

    assert session.query(Experiment).count() == 0
    assert session.query(Observation).count() == 0


def test_cascade_delete_observation_jobs_and_precomputes(session: Session) -> None:
    """Test observation deletion cascades to calculation jobs and precomputes."""
    exp = Experiment(name="exp2", status="planned")
    obs = Observation(
        name="obs", db_experiment_name="exp2", split_id="device_id", calculation_scenario="scenario"
    )
    job = CalculationJob(query="SELECT 1", status="running")
    pre = Precompute(
        group_name="A",
        metric_name="conversion",
        metric_type="cr",
        metric_display_name="Conversion",
        observation_cnt=100,
        metric_value=0.123,
    )
    job.computed_metrics.append(pre)
    obs.metrics_calculation_jobs.append(job)
    exp.observations.append(obs)

    session.add(exp)
    session.commit()

    assert session.query(CalculationJob).count() == 1
    assert session.query(Precompute).count() == 1

    session.delete(obs)
    session.commit()

    assert session.query(Observation).count() == 0
    assert session.query(CalculationJob).count() == 0
    assert session.query(Precompute).count() == 0


def test_json_fields_persistence(session: Session) -> None:
    """Test JSON field serialization and persistence in database models."""
    exp = Experiment(
        name="json_test",
        status="completed",
        key_metrics=["revenue", "retention"],
        hypotheses="test H",
        design="split test",
        conclusion="accepted",
    )
    obs = Observation(
        name="obs_json",
        db_experiment_name="json_test",
        split_id="user_id",
        calculation_scenario="default",
        audience_tables=["seg_1", "seg_2"],
        filters=["device='ios'", "version>'1.0.0'"],
        metric_tags=["tag1"],
        metric_groups=["groupA"],
    )
    job = CalculationJob(
        query="SELECT 1",
        status="completed",
        extra={"execution_time_ms": 1500, "bytes_read": 2048000, "cache_hit": False},
    )
    obs.metrics_calculation_jobs.append(job)
    exp.observations.append(obs)
    session.add(exp)
    session.commit()

    loaded_obs = session.query(Observation).filter_by(name="obs_json").first()
    assert loaded_obs is not None, "Observation should be found in database"
    assert loaded_obs.audience_tables == ["seg_1", "seg_2"]
    assert "device='ios'" in loaded_obs.filters

    loaded_job = loaded_obs.metrics_calculation_jobs[0]
    assert loaded_job.extra is not None, "Job extra field should not be None"
    assert loaded_job.extra["execution_time_ms"] == 1500
    assert loaded_job.extra["bytes_read"] == 2048000
    assert loaded_job.extra["cache_hit"] is False
