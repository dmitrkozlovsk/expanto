from __future__ import annotations

from datetime import UTC, timedelta
from typing import Any

from sqlalchemy.orm import Session

from src.domain.models import CalculationJob, Experiment, Observation, Precompute
from src.utils import DatetimeUtils

UTC = UTC
NOW = DatetimeUtils.utc_now()


def insert_experiment(session: Session, **overrides: Any) -> Experiment:
    exp = Experiment(
        name=overrides.get("name", "Experiment"),
        status=overrides.get("status", "planned"),
        description=overrides.get("description"),
        hypotheses=overrides.get("hypotheses"),
        key_metrics=overrides.get("key_metrics", ["conversion_rate", "click_through_rate"]),
        design=overrides.get("design"),
        conclusion=overrides.get("conclusion"),
        start_datetime=overrides.get("start_datetime", NOW - timedelta(days=30)),
        end_datetime=overrides.get("end_datetime", NOW - timedelta(days=15)),
        _created_at=overrides.get("_created_at", NOW - timedelta(days=45)),
    )
    session.add(exp)
    session.commit()
    session.refresh(exp)
    return exp


def insert_observation(session: Session, experiment_id: int, **overrides: Any) -> Observation:
    obs = Observation(
        experiment_id=experiment_id,
        name=overrides.get("name", "Observation"),
        db_experiment_name=overrides.get("db_experiment_name", "db_experiment"),
        split_id=overrides.get("split_id", "user_id"),
        calculation_scenario=overrides.get("calculation_scenario", "base"),
        exposure_start_datetime=overrides.get("exposure_start_datetime", NOW - timedelta(days=30)),
        exposure_end_datetime=overrides.get("exposure_end_datetime", NOW - timedelta(days=15)),
        calc_start_datetime=overrides.get("calc_start_datetime", NOW - timedelta(days=30)),
        calc_end_datetime=overrides.get("calc_end_datetime", NOW - timedelta(days=15)),
        exposure_event=overrides.get("exposure_event", "page_view"),
        audience_tables=overrides.get("audience_tables", ["active_users"]),
        filters=overrides.get("filters", ["platform='web'", "country in ('USA', 'FR')"]),
        custom_test_ids_query=overrides.get("custom_test_ids_query", "select * from test_ids"),
        metric_tags=overrides.get("metric_tags", ["conversion"]),
        metric_groups=overrides.get("metric_groups", ["engagement"]),
        _created_at=overrides.get("_created_at", NOW - timedelta(days=44)),
    )
    session.add(obs)
    session.commit()
    session.refresh(obs)
    return obs


def insert_job(session: Session, observation_id: int, **overrides: Any) -> CalculationJob:
    job = CalculationJob(
        observation_id=observation_id,
        query=overrides.get("query", "SELECT * FROM metrics WHERE experiment_id = 1"),
        status=overrides.get("status", "completed"),
        error_message=overrides.get("error_message"),
        extra=overrides.get("extra", {"execution_time_ms": 1200, "bytes_read": 1024000}),
        _created_at=overrides.get("_created_at", NOW - timedelta(days=44)),
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def insert_precompute(session: Session, job_id: int, **overrides: Any) -> Precompute:
    pre = Precompute(
        job_id=job_id,
        group_name=overrides.get("group_name", "control"),
        metric_name=overrides.get("metric_name", "conversion_rate"),
        metric_display_name=overrides.get("metric_display_name", "Conversion Rate"),
        metric_type=overrides.get("metric_type", "ratio"),
        observation_cnt=overrides.get("observation_cnt", 1000),
        metric_value=overrides.get("metric_value", 0.15),
        numerator_avg=overrides.get("numerator_avg", 150),
        denominator_avg=overrides.get("denominator_avg", 1000),
        numerator_var=overrides.get("numerator_var", 127.5),
        denominator_var=overrides.get("denominator_var", 0),
        covariance=overrides.get("covariance", 0),
        _created_at=overrides.get("_created_at", NOW - timedelta(days=44)),
    )
    session.add(pre)
    session.commit()
    session.refresh(pre)
    return pre


experiments_content = [
    {
        "name": "Button Color Test",
        "status": "completed",
        "description": "Testing the impact of button color on conversion rates",
        "hypotheses": "Blue buttons will increase conversion rates by 10%",
        "key_metrics": ["conversion_rate", "click_through_rate"],
        "design": "A/B test with 50/50 split",
        "conclusion": "Blue buttons increased conversion by 8%",
        "start_datetime": DatetimeUtils.utc_now() - timedelta(days=30),
        "end_datetime": DatetimeUtils.utc_now() - timedelta(days=15),
        "_created_at": DatetimeUtils.utc_now() - timedelta(days=45),
    },
    {
        "name": "Pricing Page Redesign",
        "status": "running",
        "description": "Testing new pricing page layout",
        "hypotheses": "New layout will increase subscription rates",
        "key_metrics": ["subscription_rate", "time_to_subscribe"],
        "design": "A/B test with 70/30 split",
        "start_datetime": DatetimeUtils.utc_now() - timedelta(days=10),
        "end_datetime": DatetimeUtils.utc_now() - timedelta(days=10),
        "_created_at": DatetimeUtils.utc_now() - timedelta(days=20),
    },
]


# Observations content mirrors the data previously defined in tests/conftest.py
observations_content = [
    {
        "name": "Main Conversion Analysis",
        "db_experiment_name": "button_color_test",
        "split_id": "user_id",
        "calculation_scenario": "base",
        "exposure_start_datetime": DatetimeUtils.utc_now() - timedelta(days=30),
        "exposure_end_datetime": DatetimeUtils.utc_now() - timedelta(days=15),
        "calc_start_datetime": DatetimeUtils.utc_now() - timedelta(days=30),
        "calc_end_datetime": DatetimeUtils.utc_now() - timedelta(days=15),
        "exposure_event": "page_view",
        "audience_tables": ["active_users"],
        "filters": ["platform='web'", "country in ('USA', 'FR')"],
        "custom_test_ids_query": "select * from test_ids",
        "metric_tags": ["conversion"],
        "metric_groups": ["engagement"],
        "_created_at": DatetimeUtils.utc_now() - timedelta(days=44),
    },
    {
        "name": "Mobile Users Analysis",
        "db_experiment_name": "button_color_test",
        "split_id": "device_id",
        "calculation_scenario": "base",
        "exposure_start_datetime": DatetimeUtils.utc_now() - timedelta(days=30),
        "exposure_end_datetime": DatetimeUtils.utc_now() - timedelta(days=15),
        "calc_start_datetime": DatetimeUtils.utc_now() - timedelta(days=30),
        "calc_end_datetime": DatetimeUtils.utc_now() - timedelta(days=15),
        "exposure_event": "mobile_page_view",
        "audience_tables": ["mobile_users"],
        "filters": ["platform='mobile'"],
        "custom_test_ids_query": "select * from test_ids",
        "metric_tags": ["mobile"],
        "metric_groups": ["engagement"],
        "_created_at": DatetimeUtils.utc_now() - timedelta(days=45),
    },
    {
        "name": "Pricing Page Analysis",
        "db_experiment_name": "pricing_page_redesign",
        "split_id": "user_id",
        "calculation_scenario": "base",
        "exposure_start_datetime": DatetimeUtils.utc_now() - timedelta(days=10),
        "exposure_end_datetime": DatetimeUtils.utc_now() - timedelta(days=10),
        "calc_start_datetime": DatetimeUtils.utc_now() - timedelta(days=10),
        "calc_end_datetime": DatetimeUtils.utc_now() - timedelta(days=10),
        "exposure_event": "pricing_page_view",
        "audience_tables": ["potential_subscribers"],
        "filters": ["has_visited_pricing=true"],
        "custom_test_ids_query": "select * from test_ids",
        "metric_tags": ["pricing"],
        "metric_groups": ["conversion"],
        "_created_at": DatetimeUtils.utc_now() - timedelta(days=46),
    },
]

# Calculation jobs content mirrors the data previously defined in tests/conftest.py
calculation_jobs_content = [
    {
        "query": "SELECT * FROM metrics WHERE experiment_id = 1",
        "status": "completed",
        "extra": {"execution_time_ms": 1200, "bytes_read": 1024000},
        "_created_at": DatetimeUtils.utc_now() - timedelta(days=44),
    },
    {
        "query": "SELECT * FROM metrics WHERE experiment_id = 1 AND date > '2024-01-01'",
        "status": "failed",
        "error_message": "Timeout error",
        "extra": {"execution_time_ms": 30000, "error_code": "TIMEOUT"},
        "_created_at": DatetimeUtils.utc_now() - timedelta(days=43),
    },
    {
        "query": "SELECT * FROM mobile_metrics WHERE experiment_id = 1",
        "status": "completed",
        "extra": {"execution_time_ms": 800, "bytes_read": 512000, "cache_hit": True, "foo": "bar"},
        "_created_at": DatetimeUtils.utc_now() - timedelta(days=44),
    },
    {
        "query": "SELECT * FROM pricing_metrics WHERE experiment_id = 2",
        "status": "running",
        "extra": {"start_time": "2024-01-01T10:00:00Z"},
        "_created_at": DatetimeUtils.utc_now() - timedelta(days=19),
    },
]

# Precomputes content mirrors the data previously defined in tests/conftest.py
precomputes_content = [
    {
        "group_name": "control",
        "metric_name": "conversion_rate",
        "metric_display_name": "Conversion Rate",
        "metric_type": "ratio",
        "observation_cnt": 1000,
        "metric_value": 0.15,
        "numerator_avg": 150,
        "denominator_avg": 1000,
        "numerator_var": 127.5,
        "denominator_var": 0,
        "covariance": 0,
        "_created_at": DatetimeUtils.utc_now() - timedelta(days=44),
    },
    {
        "group_name": "treatment",
        "metric_name": "conversion_rate",
        "metric_display_name": "Conversion Rate",
        "metric_type": "ratio",
        "observation_cnt": 1000,
        "metric_value": 0.18,
        "numerator_avg": 180,
        "denominator_avg": 1000,
        "numerator_var": 147.6,
        "denominator_var": 0,
        "covariance": 0,
        "_created_at": DatetimeUtils.utc_now() - timedelta(days=44),
    },
    {
        "group_name": "control",
        "metric_name": "mobile_conversion_rate",
        "metric_display_name": "Conversion Rate in Mobile",
        "metric_type": "ratio",
        "observation_cnt": 500,
        "metric_value": 0.12,
        "numerator_avg": 60,
        "denominator_avg": 500,
        "numerator_var": 52.8,
        "denominator_var": 0,
        "covariance": 0,
        "_created_at": DatetimeUtils.utc_now() - timedelta(days=44),
    },
    {
        "group_name": "control",
        "metric_name": "subscription_rate",
        "metric_display_name": "Subscription Rate",
        "metric_type": "ratio",
        "observation_cnt": 300,
        "metric_value": 0.08,
        "numerator_avg": 24,
        "denominator_avg": 300,
        "numerator_var": 22.08,
        "denominator_var": 0,
        "covariance": 0,
        "_created_at": DatetimeUtils.utc_now() - timedelta(days=19),
    },
]
