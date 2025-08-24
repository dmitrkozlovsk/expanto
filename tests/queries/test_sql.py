from datetime import UTC, datetime

import pytest
from jinja2 import Environment, FileSystemLoader

from src.domain.metrics import ExperimentMetricDefinition, MetricFormula, UserAggregationFormula
from src.domain.models import Observation
from src.services.runners.connectors import ConnectorResolver
from src.settings import Config, Secrets


# ----------------------------------- FIXTURES -----------------------------------
@pytest.fixture(scope="module")
def ref_days():
    """Generate reference datetime range for testing SQL queries."""
    return (
        datetime(2025, 7, 25, 12, 0, 0, tzinfo=UTC),
        datetime(2025, 7, 26, 12, 0, 0, tzinfo=UTC),
    )


@pytest.fixture(scope="module")
def dummy_obs(ref_days):
    """Create dummy observation object with test datetime range."""
    return Observation(
        exposure_start_datetime=ref_days[0],
        exposure_end_datetime=ref_days[1],
        calc_start_datetime=ref_days[0],
        calc_end_datetime=ref_days[1],
        db_experiment_name="",
        split_id="",
        exposure_event="",
        filters=None,
    )


@pytest.fixture(scope="module")
def connector():
    """Initialize database connector for integration tests."""
    config = Config()
    secrets = Secrets()
    _connector = ConnectorResolver.resolve(precompute_db_name=config.precompute_db.name, secrets=secrets)
    return _connector


@pytest.fixture(scope="module")
def jinja2_env():
    """Setup Jinja2 environment for SQL template rendering."""
    jinja2_env = Environment(
        loader=FileSystemLoader(Config().queries.dir),
        trim_blocks=False,
        lstrip_blocks=False,
        keep_trailing_newline=True,
    )
    return jinja2_env


# ------------------------------------ TESTS ------------------------------------
@pytest.mark.integration
def test_dummy(connector, jinja2_env):
    """Basic connectivity test for connector and Jinja2 environment."""
    # Basic connectivity test - check that fixtures are properly initialized
    assert connector is not None
    assert jinja2_env is not None


# to test that events sql is correctly (create dummy obs with dummy values and run)
@pytest.mark.integration
@pytest.mark.slow
def test_events(jinja2_env, connector, dummy_obs, ref_days):
    """Test events.j2 template rendering and data retrieval."""

    template = jinja2_env.get_template("my_base/events.j2")
    query = (
        template.render(
            observation=dummy_obs,
        )
        + "\nLIMIT 100"
    )
    df = connector.run_query_to_df(query)
    assert "event_name" in df.columns
    assert "event_timestamp" in df.columns
    assert ref_days[0] <= df.event_timestamp.min() <= ref_days[1]
    assert ref_days[0] <= df.event_timestamp.max() <= ref_days[1]


@pytest.mark.integration
@pytest.mark.slow
def test_test_ids(jinja2_env, connector, dummy_obs):
    """Test test_ids.j2 template rendering and execution."""
    query = jinja2_env.get_template("my_base/test_ids.j2").render(observation=dummy_obs) + "\n\tLIMIT 500"
    df = connector.run_query_to_df(query)
    assert len(df) >= 0  # Basic validation that query executed


@pytest.mark.integration
@pytest.mark.slow
def test_events_x_test_ids(jinja2_env, connector, dummy_obs):
    """Test events_x_testids.j2 template with combined events and test IDs."""
    entry_tpl = jinja2_env.from_string(
        """
        WITH events AS (
            {% include 'my_base/events.j2' %}
        ),
        test_ids AS (
            {% include 'my_base/test_ids.j2' %}
        )
        {% include 'my_base/events_x_testids.j2' %}
        """
    )
    query = entry_tpl.render(observation=dummy_obs) + "\nLIMIT 500"
    df = connector.run_query_to_df(query)
    assert len(df) >= 0  # Basic validation that query executed


@pytest.mark.integration
@pytest.mark.slow
def test_user_aggregation(jinja2_env, connector, dummy_obs):
    """Test user_aggregation.j2 template with user formula calculations."""
    entry_tpl = jinja2_env.from_string(
        """
        WITH events AS (
        {% include 'my_base/events.j2' %}
        ),
        test_ids AS (
        {% include 'my_base/test_ids.j2' %}
        ),
        events_x_testids AS (
        {% include 'my_base/events_x_testids.j2' %}
        )
        {% include 'my_base/user_aggregation.j2' %}
        """
    )

    user_formula_list = [
        UserAggregationFormula(
            alias="main_page_opened_cnt", sql="""COUNTIF(event_name='main Page Opened')"""
        )
    ]

    query = entry_tpl.render(observation=dummy_obs, user_formula_list=user_formula_list) + "\n\tLIMIT 500"
    df = connector.run_query_to_df(query)
    assert len(df) >= 0  # Basic validation that query executed


@pytest.mark.integration
@pytest.mark.slow
def test_entry(jinja2_env, connector, dummy_obs):
    """Test __entry__.j2 template with experiment metrics execution."""
    user_agg_main_page_metric = UserAggregationFormula(alias="", sql="""""")
    user_formula_list = [user_agg_main_page_metric]
    experiment_metrics_list = [
        ExperimentMetricDefinition(
            alias="",
            type="avg",
            display_name="",
            description="",
            formula=MetricFormula(numerator=user_agg_main_page_metric, denominator=None),
            owner="",
            group_name="",
            tags=None,
        )
    ]
    query = jinja2_env.get_template("my_base/__entry__.j2").render(
        observation=dummy_obs,
        user_formula_list=user_formula_list,
        experiment_metrics_list=experiment_metrics_list,
    )
    results = connector.fetch_results(query)
    assert results is not None  # Basic validation that query executed
