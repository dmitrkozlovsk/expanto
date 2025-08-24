from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.domain.results import QueryMetricResult
from src.services.runners.connectors import (
    BigQueryConnector,
    ConnectorResolver,
    SnowflakeConnector,
)
from src.settings import (
    BigQueryCredentials,
    InternalDBConfig,
    Secrets,
    SnowflakeCredentials,
)


@pytest.fixture
def fake_data():
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


# ------------------------------ SnowflakeConnector ------------------------------
@pytest.fixture
def creds_snowflake():
    """Fixture providing Snowflake credentials for testing."""
    return SnowflakeCredentials(
        **{"user": "u", "password": "p", "account": "a", "warehouse": "w", "database": "d", "schema": "s"}
    )


@pytest.fixture
def creds_bigquery_service_account():
    """Fixture providing BigQuery service account credentials."""
    return BigQueryCredentials(
        **{"connection_type": "service_account", "file_path": "/fake/path.json", "project_name": "proj"}
    )


def test_get_connection_snowflake_unwraps_secrets(monkeypatch, creds_snowflake):
    """Test Snowflake connector unwraps credentials properly."""
    captured: dict = {}

    # 2. Mock Snowflake connection call
    def fake_connect(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()  # fake connection

    monkeypatch.setattr("snowflake.connector.connect", fake_connect)

    SnowflakeConnector(creds_snowflake).get_connection()

    assert all(isinstance(v, str) for v in captured.values()), f"Not str in connect(): {captured}"
    assert "schema" in captured and "db_schema" not in captured


def test_get_client_bigquery_unwraps_secrets(monkeypatch, creds_bigquery_service_account):
    """Test BigQuery connector unwraps service account credentials."""
    captured = []

    def fake_connect(*args):
        captured.extend(args)
        return SimpleNamespace()

    monkeypatch.setattr("google.cloud.bigquery.Client.from_service_account_json", fake_connect)

    BigQueryConnector(creds_bigquery_service_account)._get_client()

    assert all(isinstance(v, str) for v in captured), f"Not str in client(): {captured}"


def test_fetch_results_success_bigquery(
    creds_snowflake: SnowflakeCredentials, fake_data: tuple[tuple, list[str]]
):
    """Test successful result fetching from Snowflake connector."""
    connector = SnowflakeConnector(creds_snowflake)
    fake_result, cols = fake_data

    fake_metadata = []
    for col in cols:
        m = MagicMock()
        m.configure_mock(name=col)  # returns None but mutates m
        fake_metadata.append(m)

    with patch.object(SnowflakeConnector, "run_query", return_value=(fake_result, fake_metadata)):
        qmrs = connector.fetch_results("dummy")

    fake_dict = {col: val for col, val in zip(cols, fake_result[0], strict=False)}
    assert qmrs == [QueryMetricResult(**fake_dict)]


def test_fetch_results_missing_column_bigquery(
    creds_snowflake: SnowflakeCredentials, fake_data: tuple[tuple, list[str]]
):
    """Test error handling when required columns are missing."""
    connector = SnowflakeConnector(creds_snowflake)
    fake_metadata = []
    for col in ["group_name", "metric_name"]:
        m = MagicMock()
        m.configure_mock(name=col)  # returns None but mutates m
        fake_metadata.append(m)

    with (
        patch.object(SnowflakeConnector, "run_query", return_value=([], fake_metadata)),
        pytest.raises(ValueError, match="Missing expected columns"),
    ):
        connector.fetch_results("dummy")


# ------------------------------ BigQueryConnector ------------------------------


@pytest.fixture(autouse=True)
def patch_client():
    """Auto-patch BigQuery client for all tests in this module."""
    with patch("src.services.runners.connectors.bigquery.Client") as MockClient:
        inst = MockClient.from_service_account_json.return_value
        yield inst


def test_fetch_results_success_snowflake(
    creds_bigquery_service_account: BigQueryCredentials,
    patch_client: MagicMock,
    fake_data: tuple[tuple, list[str]],
):
    """Test successful result fetching from BigQuery connector."""
    # fake row iterator
    fake_result, cols = fake_data
    mock_row = {k: v for k, v in zip(cols, fake_result[0], strict=False)}
    patch_client.fetch.return_value = mock_row

    patch_client.query.return_value.result.return_value = [mock_row]

    connector = BigQueryConnector(creds_bigquery_service_account)
    res = connector.fetch_results("dummy")

    assert res == [QueryMetricResult(**mock_row)]


def test_run_query_error_snowflake(creds_bigquery_service_account):
    """Test BigQuery connector handles query errors properly."""
    connector = BigQueryConnector(creds_bigquery_service_account)
    patch_client = connector.client
    patch_client.query.side_effect = RuntimeError("fail")
    with pytest.raises(RuntimeError):
        connector.run_query("dummy")


def test_resolve_snowflake(creds_snowflake, fake_load_expanto_cfg):
    """Test resolver returns Snowflake connector for snowflake config."""
    config = fake_load_expanto_cfg
    config.precompute_db.name = "snowflake"
    secrets = Secrets(
        snowflake=creds_snowflake,
        internal_db=InternalDBConfig(
            **{
                "engine_str": "sqlite:///:memory:",
                "async_engine_str": "sqlite+aiosqlite:///:memory:",
                "connect_args": {"check_same_thread": False},
            }
        ),
    )
    connector = ConnectorResolver.resolve(precompute_db_name=config.precompute_db.name, secrets=secrets)
    assert isinstance(connector, SnowflakeConnector)


def test_resolve_bigquery(creds_bigquery_service_account, fake_load_expanto_cfg):
    """Test resolver returns BigQuery connector for bigquery config."""
    config = fake_load_expanto_cfg
    config.precompute_db.name = "bigquery"
    secrets = Secrets(
        bigquery=creds_bigquery_service_account,
        internal_db=InternalDBConfig(
            **{
                "engine_str": "sqlite:///:memory:",
                "async_engine_str": "sqlite+aiosqlite:///:memory:",
                "connect_args": {"check_same_thread": False},
            }
        ),
    )
    connector = ConnectorResolver.resolve(precompute_db_name=config.precompute_db.name, secrets=secrets)

    assert isinstance(connector, BigQueryConnector)
