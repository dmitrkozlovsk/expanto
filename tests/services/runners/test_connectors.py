from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.domain.results import QueryMetricResult
from src.services.runners.connectors import (
    BigQueryConnector,
    ConnectorResolver,
    SnowflakeConnector,
)
from src.settings import (
    BigQueryCredentials,
    SnowflakeCredentials,
)

if TYPE_CHECKING:
    from src.settings import Secrets


def test_get_connection_snowflake_unwraps_secrets(monkeypatch, fake_creds_snowflake):
    """Test Snowflake connector unwraps credentials properly."""
    captured: dict = {}

    # 2. Mock Snowflake connection call
    def fake_connect(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()  # fake connection

    monkeypatch.setattr("snowflake.connector.connect", fake_connect)

    SnowflakeConnector(fake_creds_snowflake).get_connection()

    assert all(isinstance(v, str) for v in captured.values()), f"Not str in connect(): {captured}"
    assert "schema" in captured and "db_schema" not in captured


def test_get_connection_snowflake_fails(monkeypatch, fake_creds_snowflake):
    """Test Snowflake connector handles connection failure."""

    def fake_connect_fails(**kwargs):
        raise RuntimeError("Connection failed")

    monkeypatch.setattr("snowflake.connector.connect", fake_connect_fails)

    connector = SnowflakeConnector(fake_creds_snowflake)

    with pytest.raises(RuntimeError, match="Connection failed"):
        connector.get_connection()


def test_run_query_snowflake_fails(fake_creds_snowflake):
    """Test Snowflake connector handles query execution failure."""
    connector = SnowflakeConnector(fake_creds_snowflake)

    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = RuntimeError("Query failed")

    mock_connection = MagicMock()
    mock_connection.cursor.return_value = mock_cursor
    mock_connection.__enter__.return_value = mock_connection  # for `with` statement

    with patch.object(SnowflakeConnector, "get_connection", return_value=mock_connection), pytest.raises(
        RuntimeError, match="Query failed"
    ):
        connector.run_query("dummy query")


def test_get_client_snowflake_unwraps_secrets(monkeypatch, fake_creds_bigquery_service_account):
    """Test BigQuery connector unwraps service account credentials."""
    captured = []

    def fake_connect(*args):
        captured.extend(args)
        return SimpleNamespace()

    monkeypatch.setattr("google.cloud.bigquery.Client.from_service_account_json", fake_connect)

    BigQueryConnector(fake_creds_bigquery_service_account)._get_client()

    assert all(isinstance(v, str) for v in captured), f"Not str in client(): {captured}"


def test_fetch_results_success_snowflake(
    fake_creds_snowflake: SnowflakeCredentials, fake_metric_results: tuple[tuple, list[str]]
):
    """Test successful result fetching from Snowflake connector."""
    connector = SnowflakeConnector(fake_creds_snowflake)
    fake_result, cols = fake_metric_results

    fake_metadata = []
    for col in cols:
        m = MagicMock()
        m.configure_mock(name=col)  # returns None but mutates m
        fake_metadata.append(m)

    with patch.object(SnowflakeConnector, "run_query", return_value=(fake_result, fake_metadata)):
        qmrs = connector.fetch_results("dummy")

    fake_dict = {col: val for col, val in zip(cols, fake_result[0], strict=False)}
    assert qmrs == [QueryMetricResult(**fake_dict)]


def test_fetch_results_missing_column_snowflake(
    fake_creds_snowflake: SnowflakeCredentials, fake_metric_results: tuple[tuple, list[str]]
):
    """Test error handling when required columns are missing."""
    connector = SnowflakeConnector(fake_creds_snowflake)
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


def test_run_query_to_df_snowflake(
    fake_creds_snowflake: SnowflakeCredentials, fake_metric_results: tuple[tuple, list[str]]
):
    """Test run_query_to_df for Snowflake connector."""
    connector = SnowflakeConnector(fake_creds_snowflake)
    fake_result, cols = fake_metric_results

    fake_metadata = []
    for col in cols:
        m = MagicMock()
        m.configure_mock(name=col.upper())
        fake_metadata.append(m)

    with patch.object(SnowflakeConnector, "run_query", return_value=(fake_result, fake_metadata)):
        df = connector.run_query_to_df("dummy")

    expected_df = pd.DataFrame(fake_result, columns=cols)
    pd.testing.assert_frame_equal(df, expected_df)


# ------------------------------ BigQueryConnector ------------------------------


def test_get_client_bigquery_fails(monkeypatch, fake_creds_bigquery_service_account):
    """Test BigQuery connector handles client creation failure."""

    def fake_connect_fails(*args, **kwargs):
        raise RuntimeError("Connection failed")

    monkeypatch.setattr("google.cloud.bigquery.Client.from_service_account_json", fake_connect_fails)

    with pytest.raises(RuntimeError, match="Connection failed"):
        BigQueryConnector(fake_creds_bigquery_service_account)


def test_fetch_results_success_bigquery(
    fake_creds_bigquery_service_account: BigQueryCredentials,
    patch_bigquery_client: MagicMock,
    fake_metric_results: tuple[tuple, list[str]],
):
    """Test successful result fetching from BigQuery connector."""
    # fake row iterator
    fake_result, cols = fake_metric_results
    mock_row = {k: v for k, v in zip(cols, fake_result[0], strict=False)}
    patch_bigquery_client.fetch.return_value = mock_row

    patch_bigquery_client.query.return_value.result.return_value = [mock_row]

    connector = BigQueryConnector(fake_creds_bigquery_service_account)
    res = connector.fetch_results("dummy")

    assert res == [QueryMetricResult(**mock_row)]


def test_run_query_to_df_bigquery(
    fake_creds_bigquery_service_account: BigQueryCredentials,
    patch_bigquery_client: MagicMock,
    fake_metric_results: tuple[tuple, list[str]],
):
    """Test run_query_to_df for BigQuery connector."""
    connector = BigQueryConnector(fake_creds_bigquery_service_account)
    fake_result, cols = fake_metric_results
    expected_df = pd.DataFrame(fake_result, columns=cols)

    mock_iterator = MagicMock()
    mock_iterator.to_dataframe.return_value = expected_df
    patch_bigquery_client.query.return_value.result.return_value = mock_iterator

    df = connector.run_query_to_df("dummy")

    pd.testing.assert_frame_equal(df, expected_df)
    patch_bigquery_client.query.assert_called_once_with("dummy")
    mock_iterator.to_dataframe.assert_called_once()


def test_run_query_error_bigquery(fake_creds_bigquery_service_account):
    """Test BigQuery connector handles query errors properly."""
    connector = BigQueryConnector(fake_creds_bigquery_service_account)
    patch_bigquery_client = connector.client
    patch_bigquery_client.query.side_effect = RuntimeError("fail")
    with pytest.raises(RuntimeError):
        connector.run_query("dummy")


def test_fetch_results_missing_column_bigquery(
    fake_creds_bigquery_service_account: BigQueryCredentials,
    patch_bigquery_client: MagicMock,
):
    """Test error handling for missing columns in BigQuery results."""
    connector = BigQueryConnector(fake_creds_bigquery_service_account)
    mock_row = {"group_name": "control", "metric_name": "conversion"}  # Missing keys
    patch_bigquery_client.query.return_value.result.return_value = [mock_row]

    with pytest.raises(ValueError, match="Missing expected columns"):
        connector.fetch_results("dummy")


# ------------------------------ Resolver ------------------------------


def test_resolve_snowflake(fake_creds_snowflake, fake_load_expanto_cfg, fake_load_secrets_cfg):
    """Test resolver returns Snowflake connector for snowflake config."""
    config = fake_load_expanto_cfg
    config.precompute_db.name = "snowflake"
    secrets = fake_load_secrets_cfg
    connector = ConnectorResolver.resolve(precompute_db_name=config.precompute_db.name, secrets=secrets)
    assert isinstance(connector, SnowflakeConnector)


def test_resolve_bigquery(
    fake_creds_bigquery_service_account, fake_load_expanto_cfg, fake_load_secrets_cfg
):
    """Test resolver returns BigQuery connector for bigquery config."""
    config = fake_load_expanto_cfg
    config.precompute_db.name = "bigquery"
    secrets = fake_load_secrets_cfg
    connector = ConnectorResolver.resolve(precompute_db_name=config.precompute_db.name, secrets=secrets)

    assert isinstance(connector, BigQueryConnector)


def test_resolve_no_credentials(fake_load_secrets_cfg: Secrets):
    """Test resolver raises AttributeError when no credentials are provided."""
    secrets = fake_load_secrets_cfg
    secrets.snowflake = None
    secrets.bigquery = None

    with pytest.raises(AttributeError, match="Could not find any credentials"):
        ConnectorResolver.resolve(precompute_db_name="snowflake", secrets=secrets)
