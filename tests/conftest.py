from datetime import timedelta
from unittest.mock import patch

import pytest
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from src.domain.models import Base, CalculationJob, Experiment, Observation, Precompute
from src.services.metric_register import Metrics
from src.settings import Config, QueryTemplatesConfig, Secrets
from src.utils import DatetimeUtils

# ------------------------ INTERNAL DB FIXTURES (SQLALCHEMY) ------------------------


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    """Create temporary SQLite database path for testing."""
    path = tmp_path_factory.mktemp("db") / "test.sqlite"
    return f"sqlite:///{path}"


@pytest.fixture(scope="session")
def engine(db_path):
    """Create SQLAlchemy engine for testing database."""
    return create_engine(db_path, echo=False)


@pytest.fixture(scope="function")
def tables(engine):
    """Create and drop database tables for each test."""
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture(scope="function")
def session(engine: Engine, tables):
    """Create database session with transaction rollback for each test."""
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(engine)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def mock_experiments(session: Session):
    """Create mock experiment data for testing."""
    # Create two experiments
    exp1 = Experiment(
        name="Button Color Test",
        status="completed",
        description="Testing the impact of button color on conversion rates",
        hypotheses="Blue buttons will increase conversion rates by 10%",
        key_metrics=["conversion_rate", "click_through_rate"],
        design="A/B test with 50/50 split",
        conclusion="Blue buttons increased conversion by 8%",
        start_datetime=DatetimeUtils.utc_now() - timedelta(days=30),
        end_datetime=DatetimeUtils.utc_now() - timedelta(days=15),
        _created_at=DatetimeUtils.utc_now() - timedelta(days=45),
    )

    exp2 = Experiment(
        name="Pricing Page Redesign",
        status="running",
        description="Testing new pricing page layout",
        hypotheses="New layout will increase subscription rates",
        key_metrics=["subscription_rate", "time_to_subscribe"],
        design="A/B test with 70/30 split",
        start_datetime=DatetimeUtils.utc_now() - timedelta(days=10),
        end_datetime=DatetimeUtils.utc_now() - timedelta(days=10),
        _created_at=DatetimeUtils.utc_now() - timedelta(days=20),
    )

    session.add_all([exp1, exp2])
    session.commit()
    session.refresh(exp1)
    session.refresh(exp2)
    return [exp1, exp2]


@pytest.fixture(scope="function")
def mock_observations(session, mock_experiments):
    """Create mock observation data for testing."""
    # Create three observations
    obs1 = Observation(
        experiment_id=mock_experiments[0].id,
        name="Main Conversion Analysis",
        db_experiment_name="button_color_test",
        split_id="user_id",
        calculation_scenario="base",
        exposure_start_datetime=DatetimeUtils.utc_now() - timedelta(days=30),
        exposure_end_datetime=DatetimeUtils.utc_now() - timedelta(days=15),
        calc_start_datetime=DatetimeUtils.utc_now() - timedelta(days=30),
        calc_end_datetime=DatetimeUtils.utc_now() - timedelta(days=15),
        exposure_event="page_view",
        audience_tables=["active_users"],
        filters=["platform='web'", "country in ('USA', 'FR')"],
        custom_test_ids_query="select * from test_ids",
        metric_tags=["conversion"],
        metric_groups=["engagement"],
        _created_at=DatetimeUtils.utc_now() - timedelta(days=44),
    )

    obs2 = Observation(
        experiment_id=mock_experiments[0].id,
        name="Mobile Users Analysis",
        db_experiment_name="button_color_test",
        split_id="device_id",
        calculation_scenario="base",
        exposure_start_datetime=DatetimeUtils.utc_now() - timedelta(days=30),
        exposure_end_datetime=DatetimeUtils.utc_now() - timedelta(days=15),
        calc_start_datetime=DatetimeUtils.utc_now() - timedelta(days=30),
        calc_end_datetime=DatetimeUtils.utc_now() - timedelta(days=15),
        exposure_event="mobile_page_view",
        audience_tables=["mobile_users"],
        filters=["platform='mobile'"],
        custom_test_ids_query="select * from test_ids",
        metric_tags=["mobile"],
        metric_groups=["engagement"],
        _created_at=DatetimeUtils.utc_now() - timedelta(days=45),
    )

    obs3 = Observation(
        experiment_id=mock_experiments[1].id,
        name="Pricing Page Analysis",
        db_experiment_name="pricing_page_redesign",
        split_id="user_id",
        calculation_scenario="base",
        exposure_start_datetime=DatetimeUtils.utc_now() - timedelta(days=10),
        exposure_end_datetime=DatetimeUtils.utc_now() - timedelta(days=10),
        calc_start_datetime=DatetimeUtils.utc_now() - timedelta(days=10),
        calc_end_datetime=DatetimeUtils.utc_now() - timedelta(days=10),
        exposure_event="pricing_page_view",
        audience_tables=["potential_subscribers"],
        filters=["has_visited_pricing=true"],
        custom_test_ids_query="select * from test_ids",
        metric_tags=["pricing"],
        metric_groups=["conversion"],
        _created_at=DatetimeUtils.utc_now() - timedelta(days=46),
    )

    session.add_all([obs1, obs2, obs3])
    session.commit()
    return [obs1, obs2, obs3]


@pytest.fixture(scope="function")
def mock_precomputes(session, mock_calculation_jobs):
    """Create mock precompute data for testing."""
    # Create seven precomputes
    precomputes = []

    # Three precomputes for first job
    precomputes.extend(
        [
            Precompute(
                job_id=mock_calculation_jobs[0].id,
                group_name="control",
                metric_name="conversion_rate",
                metric_display_name="Conversion Rate",
                metric_type="ratio",
                observation_cnt=1000,
                metric_value=0.15,
                numerator_avg=150,
                denominator_avg=1000,
                numerator_var=127.5,
                denominator_var=0,
                covariance=0,
                _created_at=DatetimeUtils.utc_now() - timedelta(days=44),
            ),
            Precompute(
                job_id=mock_calculation_jobs[0].id,
                group_name="treatment",
                metric_name="conversion_rate",
                metric_display_name="Conversion Rate",
                metric_type="ratio",
                observation_cnt=1000,
                metric_value=0.18,
                numerator_avg=180,
                denominator_avg=1000,
                numerator_var=147.6,
                denominator_var=0,
                covariance=0,
                _created_at=DatetimeUtils.utc_now() - timedelta(days=44),
            ),
        ]
    )

    # Two precomputes for third job (mobile metrics)
    precomputes.extend(
        [
            Precompute(
                job_id=mock_calculation_jobs[2].id,
                group_name="control",
                metric_name="mobile_conversion_rate",
                metric_display_name="Conversion Rate in Mobile",
                metric_type="ratio",
                observation_cnt=500,
                metric_value=0.12,
                numerator_avg=60,
                denominator_avg=500,
                numerator_var=52.8,
                denominator_var=0,
                covariance=0,
                _created_at=DatetimeUtils.utc_now() - timedelta(days=44),
            ),
        ]
    )

    # Two precomputes for fourth job (pricing metrics)
    precomputes.extend(
        [
            Precompute(
                job_id=mock_calculation_jobs[3].id,
                group_name="control",
                metric_name="subscription_rate",
                metric_display_name="Subscription Rate",
                metric_type="ratio",
                observation_cnt=300,
                metric_value=0.08,
                numerator_avg=24,
                denominator_avg=300,
                numerator_var=22.08,
                denominator_var=0,
                covariance=0,
                _created_at=DatetimeUtils.utc_now() - timedelta(days=19),
            ),
        ]
    )

    session.add_all(precomputes)
    session.commit()
    return precomputes


@pytest.fixture(scope="function")
def mock_calculation_jobs(session, mock_observations):
    """Create mock calculation job data for testing."""
    # Create four calculation jobs
    jobs = [
        CalculationJob(
            observation_id=mock_observations[0].id,
            query="SELECT * FROM metrics WHERE experiment_id = 1",
            status="completed",
            extra={"execution_time_ms": 1200, "bytes_read": 1024000},
            _created_at=DatetimeUtils.utc_now() - timedelta(days=44),
        ),
        CalculationJob(
            observation_id=mock_observations[0].id,
            query="SELECT * FROM metrics WHERE experiment_id = 1 AND date > '2024-01-01'",
            status="failed",
            error_message="Timeout error",
            extra={"execution_time_ms": 30000, "error_code": "TIMEOUT"},
            _created_at=DatetimeUtils.utc_now() - timedelta(days=43),
        ),
        CalculationJob(
            observation_id=mock_observations[1].id,
            query="SELECT * FROM mobile_metrics WHERE experiment_id = 1",
            status="completed",
            extra={"execution_time_ms": 800, "bytes_read": 512000, "cache_hit": True},
            _created_at=DatetimeUtils.utc_now() - timedelta(days=44),
        ),
        CalculationJob(
            observation_id=mock_observations[2].id,
            query="SELECT * FROM pricing_metrics WHERE experiment_id = 2",
            status="running",
            extra={"start_time": "2024-01-01T10:00:00Z"},
            _created_at=DatetimeUtils.utc_now() - timedelta(days=19),
        ),
    ]

    session.add_all(jobs)
    session.commit()
    return jobs


# -------------------------------- METRICS FIXTURES --------------------------------
@pytest.fixture(scope="session")
def metrics_temp_dir(tmp_path_factory):
    """Create temporary directory with mock metrics YAML files."""
    temp_dir = tmp_path_factory.mktemp("metrics_data")

    file1_content = """
    metric_group_name: 'Order Metrics'
    user_aggregations:
      product_purchase_cnt: &product_purchase_cnt
        alias: product_purchase_cnt
        sql: COUNT(CASE event_name WHEN 'product_purchased' THEN event_name ELSE null END)

      product_revenue_sum: &product_revenue_sum
        alias: product_revenue_sum
        sql: SUM(CASE event_name WHEN 'product_purchased' THEN event_value ELSE 0 END)

    metrics:
      - alias: avg_order_value
        type: ratio
        display_name: "Average Order Value"
        description: "Sum of product purchase revenue divided by count"
        formula:
          numerator: *product_revenue_sum
          denominator: *product_purchase_cnt
        owner: me
        tags: ["orders", "sales"]
    """

    file2_content = """
        metric_group_name: 'User Behavior and Engagement Metrics'
        user_aggregations:
          product_view_cnt: &product_view_cnt
            alias: product_view_cnt
            sql: COUNT(CASE event_name WHEN 'product_viewed' THEN event_name ELSE null END)
    
          session_duration: &session_duration
            alias: session_duration
            sql: AVG(session_length)
    
          product_click_flg: &product_click_flg
            alias: product_click_flg
            sql: MAX(CASE event_name WHEN 'product_clicked' THEN 1 ELSE 0 END)
    
          product_purchase_flg: &product_purchase_flg
            alias: product_purchase_flg
            sql: MAX(CASE event_name WHEN 'product_purchased' THEN 1 ELSE 0 END)

          product_view_flg: &product_view_flg
            alias: product_view_flg
            sql: MAX(CASE event_name WHEN 'product_viewed' THEN 1 ELSE 0 END)
    
        metrics:
          - alias: click_through_rate
            type: ratio
            display_name: "CTR"
            description: "Product clicks vs product views"
            formula:
              numerator: *product_click_flg
              denominator: *product_view_cnt
            owner: me
            tags: ["engagement", "ui"]
    
          - alias: conversion_rate
            type: avg
            display_name: "Conversion Rate"
            description: "Product purchases vs product views"
            formula:
              numerator: *product_purchase_flg
              denominator: null
            owner: me
            tags: ["conversion", "sales"]
    
          - alias: avg_session_duration
            type: avg
            display_name: "Avg Session Duration"
            description: "Average session length per user"
            formula:
              numerator: *session_duration
              denominator: null
            owner: me
            tags: ["behavior", "time"]
    
          - alias: users_who_click_product_ratio
            type: proportion
            display_name: "Users with Orders"
            description: "some description"
            formula:
              numerator: *product_click_flg
              denominator: null
            owner: me
            tags: null
        """

    (temp_dir / "order_metrics.yaml").write_text(file1_content)
    (temp_dir / "engagement_metrics.yaml").write_text(file2_content)

    return temp_dir


@pytest.fixture
def metrics(metrics_temp_dir):
    """Create Metrics instance from temporary directory."""
    return Metrics(directory=str(metrics_temp_dir))


# -------------------------------- QUERIES FIXTURES --------------------------------
@pytest.fixture
def queries_templates_config(tmp_path_factory):
    """Create temporary directory with query templates for testing."""
    temp_dir = tmp_path_factory.mktemp("queries")
    base_dir = temp_dir / "base"
    base_dir.mkdir()

    # Create a sample base query template
    base_query = """
        WITH fake_template AS (
        SELECT 
            {{ observation.split_id }} as split_id
            {% if purpose == 'planning' %} ,'planning' as purpose
            {% else %},'result' as purpose {% endif %} 
            {% for user_formula in user_formula_list %}
            , {{ user_formula.sql }} as {{ user_formula.alias }}{% endfor %}
        FROM {% if observation.segment %}{{ observation.segment }}
             {% else %} {{ observation.db_experiment_name }}
             {% endif %}
        WHERE 1 = 1
            AND exposure_timestamp BETWEEN '{{ observation.exposure_start_datetime }}' AND '{{ observation.exposure_end_datetime }}' 
            AND event_timestamp BETWEEN '{{ observation.calc_start_datetime }}' AND '{{ observation.calc_end_datetime }}'
            {% if observation.filters %}
            {% for filter in observation.filters %}
            AND ({{ filter }}){% if not loop.last %} {% endif %}
            {% endfor %}{% endif %}
            {% if observation.audience_tables %}
            AND EXISTS (
                SELECT 1 FROM {% for table in observation.audience_tables %}
                {{ table }}{% if not loop.last %} UNION ALL {% endif %}
                {% endfor %}
            ){% endif %}
        GROUP BY 1
        {% for metric in experiment_metrics_list %}
        SELECT
              GROUP_NAME as group_name
             , '{{ metric.alias }}' as metric_name
             , '{{ metric.type }}' as metric_type
             , '{{ metric.display_name }}' as metric_display_name
             , COUNT(distinct fake_template.{{ observation.split_id }}) as observations_cnt
             , {{ metric.sql }} as metric_value
             , AVG({{ metric.formula.numerator.alias }}) as numerator_avg
             , VAR_SAMP({{ metric.formula.numerator.alias }}) as numerator_var
             {% if metric.formula.denominator -%}
             , AVG({{ metric.formula.denominator.alias }}) as denominator_avg
             , VAR_SAMP({{ metric.formula.denominator.alias }}) as denominator_var
             , COVAR_SAMP({{ metric.formula.denominator.alias }}, {{ metric.formula.numerator.alias }}) as covariance
             {% else -%}
             , CAST(null as float) as denominator_avg
             , CAST(null as float) as denominator_var
             , CAST(null as float) as covariance
             {% endif -%}
        FROM fake_template
        GROUP BY group_name
            {% if not loop.last %}
        UNION ALL{% endif %}
        {% endfor %}
        )
    """  # noqa: E501
    (base_dir / "base.sql").write_text(base_query)

    return QueryTemplatesConfig(dir=str(temp_dir), scenarios= {'base': "base/base.sql"})


# --------------------------- OTHER FIXTURES ---------------------------


@pytest.fixture
def disable_files_in_settings(monkeypatch):
    """Force Config() and Secrets() to use only __init__ and env sources."""
    """Force Config() and Secrets() to use only __init__ and env sources."""

    def only_init_and_env(
        cls,
        settings_cls: BaseSettings,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, dotenv_settings)

    # Convert to classmethod and patch
    monkeypatch.setattr(
        Config,
        "settings_customise_sources",
        classmethod(only_init_and_env),
        raising=True,
    )
    monkeypatch.setattr(
        Secrets,
        "settings_customise_sources",
        classmethod(only_init_and_env),
        raising=True,
    )


@pytest.fixture
def fake_load_expanto_cfg(metrics_temp_dir, queries_templates_config):
    """Create fake Expanto configuration for testing."""
    return Config.model_validate(
        {
            "metrics": {"dir": str(metrics_temp_dir)},
            "queries": {
                "dir": queries_templates_config.dir,
                "scenarios": {
                    "base": "base/base.sql",
                    "base2": "base2",
                }
            },
            "precompute_db": {"name": "snowflake"},
            "assistant": {
                "provider": "super_druper_provider",
                "models": {
                    "fast": "Q",
                    "agentic": "Q",
                    "tool_thinker": "Q",
                },
                "service": {
                    "url": "http://127.0.0.1:8000",
                    "timeout_seconds": 600,
                    "enable_streaming": True,
                    "auto_scroll": True,
                },
            },
            "logfire":{"send_to_logfire": False}
        }
    )


@pytest.fixture
def fake_load_secrets_cfg(db_path):
    """Create fake secrets configuration for testing."""
    return Secrets.model_validate(
        {
            "internal_db": {
                "engine_str": db_path,
                "async_engine_str": "fake",
                "connect_args": {},
            },
            "bigquery": {
                "file_path": "fake",
                "project_name": "fake",
                "connection_type": "application_default",
            },
            "snowflake": None,
            "api_keys": {
                "PROVIDER_API_KEY": "TOGETHER_API_KEY",
                 "TAVILY_API_KEY": "TAVILY_API_KEY",
                 "LOGFIRE_TOKEN": "LOGFIRE_TOKEN"
            },
        },
    )


@pytest.fixture
def patch_configs(
    monkeypatch, disable_files_in_settings, fake_load_secrets_cfg, fake_load_expanto_cfg, engine, metrics
):
    """Patch configuration loading for UI testing."""
    monkeypatch.setattr(
        "src.ui.resources.Secrets",
        lambda *a, **kw: fake_load_secrets_cfg,
        raising=True,
    )

    # Same for Config()
    monkeypatch.setattr(
        "src.ui.resources.Config",
        lambda *a, **kw: fake_load_expanto_cfg,
        raising=True,
    )


@pytest.fixture
def patch_client_and_resolver():
    """Mock BigQuery client and connector resolver for testing."""
    with patch("src.services.runners.connectors.bigquery.Client") as MockClient:
        inst = MockClient.from_service_account_json.return_value

        with patch("src.services.runners.connectors.ConnectorResolver.resolve", return_value=inst):
            yield inst


@pytest.fixture(scope="session")
def docs_temp_dir(tmp_path_factory):
    """Create temporary directory with mock documentation files."""
    temp_dir = tmp_path_factory.mktemp("docs")
    (temp_dir / "integration.md").write_text("Integration documentation. how to start with expanto")
    (temp_dir / "queries.md").write_text("Queries documentation. how to create queries")
    (temp_dir / "precompute.md").write_text("Precompute documentation. how to create precompute")
    (temp_dir / "metrics.md").write_text("Metrics documentation. how to create metrics")
    return temp_dir


@pytest.fixture(scope="session")
def root_dir(tmp_path_factory):
    """Create temporary root directory structure for testing."""
    temp_dir = tmp_path_factory.mktemp("root")
    (temp_dir / "src").mkdir()
    (temp_dir / ".expanto").mkdir()
    (temp_dir / "src" / "agent").mkdir()
    (temp_dir / "src" / "services").mkdir()
    (temp_dir / "src" / ".env").write_text("ENV_VAR=!!!")
    (temp_dir / "src" / "agent" / "vdb.py").write_text("This is a vdb code")
    (temp_dir / "src" / "services" / "metric_register.py").write_text("This is a metric register code")
    (temp_dir / "src" / "services" / "bigquery.py").write_text("This is a bigquery code")
    (temp_dir / ".expanto" / "secrets.toml").write_text("SECRET_KEY=000111")
    return temp_dir
