
import pytest
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from src.domain.models import Base, CalculationJob, Experiment, Observation, Precompute
from src.services.metric_register import Metrics
from src.settings import Config, QueryTemplatesConfig, Secrets
from tests.assets.configs import dir_fake_metrics, dir_fake_query, fake_configs_dict, fake_secrets_dict
from tests.assets.entities import (
    calculation_jobs_content,
    experiments_content,
    insert_experiment,
    insert_job,
    insert_observation,
    insert_precompute,
    observations_content,
    precomputes_content,
)

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
def mock_experiments(session: Session) -> list[Experiment]:
    """Create mock experiment data for testing."""
    exp1 = insert_experiment(session, **experiments_content[0])
    exp2 = insert_experiment(session, **experiments_content[1])
    return [exp1, exp2]


@pytest.fixture(scope="function")
def mock_observations(session, mock_experiments) -> list[Observation]:
    """Create mock observation data for testing."""
    obs1 = insert_observation(session, int(mock_experiments[0].id), **observations_content[0])
    obs2 = insert_observation(session, int(mock_experiments[0].id), **observations_content[1])
    obs3 = insert_observation(session, int(mock_experiments[1].id), **observations_content[2])
    return [obs1, obs2, obs3]


@pytest.fixture(scope="function")
def mock_calculation_jobs(session, mock_observations) -> list[CalculationJob]:
    """Create mock calculation job data for testing."""
    job1 = insert_job(session, int(mock_observations[0].id), **calculation_jobs_content[0])
    job2 = insert_job(session, int(mock_observations[1].id), **calculation_jobs_content[1])
    job3 = insert_job(session, int(mock_observations[1].id), **calculation_jobs_content[2])
    job4 = insert_job(session, int(mock_observations[2].id), **calculation_jobs_content[3])
    return [job1, job2, job3, job4]


@pytest.fixture(scope="function")
def mock_precomputes(session, mock_calculation_jobs) -> list[Precompute]:
    """Create mock precompute data for testing."""
    precompute1 = insert_precompute(session, int(mock_calculation_jobs[0].id), **precomputes_content[0])
    precompute2 = insert_precompute(session, int(mock_calculation_jobs[0].id), **precomputes_content[1])
    precompute3 = insert_precompute(session, int(mock_calculation_jobs[2].id), **precomputes_content[2])
    precompute4 = insert_precompute(session, int(mock_calculation_jobs[3].id), **precomputes_content[3])
    return [precompute1, precompute2, precompute3, precompute4]


# -------------------------------- METRICS FIXTURES --------------------------------
@pytest.fixture(scope="session")
def metrics_temp_dir():
    """Return path to bundled test metrics directory under tests/assets/metrics."""
    return dir_fake_metrics


@pytest.fixture(scope="session")
def metrics(metrics_temp_dir) -> Metrics:
    """Create Metrics instance from temporary directory."""
    return Metrics(directory=str(metrics_temp_dir))


# -------------------------------- QUERIES FIXTURES --------------------------------
@pytest.fixture(scope="session")
def queries_templates_config() -> QueryTemplatesConfig:
    """Return path to bundled test query template under tests/assets/query.j2."""
    return QueryTemplatesConfig(dir=str(dir_fake_query), scenarios={"base": "query.j2"})


# --------------------------- CONFIG FIXTURES ---------------------------


@pytest.fixture
def disable_files_in_settings(monkeypatch):
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
def fake_load_expanto_cfg(metrics_temp_dir, queries_templates_config) -> Config:
    """Create fake Expanto configuration for testing."""
    return Config.model_validate(fake_configs_dict)


@pytest.fixture
def fake_load_secrets_cfg(db_path) -> Secrets:
    """Create fake secrets configuration for testing."""
    fake_secrets_dict["internal_db"]["engine_str"] = db_path
    return Secrets.model_validate(fake_secrets_dict)
