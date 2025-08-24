"""Database connectors for precompute operations.

This module provides standardized database connectors for Snowflake and BigQuery,
implementing a common interface for query execution and result fetching. It handles
connection management, query execution, and result conversion to standardized
QueryMetricResult objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import pandas as pd  # type: ignore
import snowflake.connector
from google.cloud import bigquery
from pydantic import SecretStr

from src.domain.results import QueryMetricResult
from src.logger_setup import logger

if TYPE_CHECKING:
    from src.settings import (
        BigQueryCredentials,
        Secrets,
        SnowflakeCredentials,
    )


class PrecomputeConnector(ABC):
    """Abstract base class for database connectors used in precompute operations."""

    required_columns = list(QueryMetricResult.model_fields.keys())

    @abstractmethod
    def run_query(self, query: str):
        """Execute a database query."""
        ...

    @abstractmethod
    def run_query_to_df(self, query: str) -> pd.DataFrame:
        """Execute a database query and return results as a pandas DataFrame."""
        ...

    @abstractmethod
    def fetch_results(self, query: str) -> list[QueryMetricResult]:
        """Execute a query and return results as QueryMetricResult objects."""
        ...


class SnowflakeConnector(PrecomputeConnector):
    """Snowflake database connector implementation."""

    def __init__(self, credentials: SnowflakeCredentials) -> None:
        """Initialize the Snowflake connector."""
        self.credentials = credentials

    def get_connection(self):
        """Create and return a Snowflake database connection.

        Returns:
            snowflake.connector.SnowflakeConnection: Active Snowflake connection.

        Raises:
            Exception: If connection fails.
        """
        try:
            cred_dict = {
                k: v.get_secret_value() if isinstance(v, SecretStr) else v
                for k, v in self.credentials.model_dump(by_alias=True).items()
            }
            return snowflake.connector.connect(**cred_dict)
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake, error: {e}")
            raise e

    def run_query(self, query: str) -> tuple[list[tuple], list[snowflake.connector.cursor.ResultMetadata]]:
        """Execute a query on Snowflake.

        Args:
            query (str): The SQL query to execute.

        Returns:
            tuple[list[tuple], list[ResultMetadata]]: Query results and metadata.

        Raises:
            Exception: If query execution fails.
        """
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                result = cursor.execute(query).fetchall()
                metadata = cursor.description
            return result, metadata
        except Exception as e:
            logger.error(f"Failed to run query {query}, error: {e}")
            raise e

    def run_query_to_df(self, query: str) -> pd.DataFrame:
        """Execute a query on Snowflake and return results as a pandas DataFrame.

        Args:
            query (str): The SQL query to execute.

        Returns:
            pd.DataFrame: Query results as a pandas DataFrame.

        Raises:
            Exception: If query execution fails.
        """

        results, metadata = self.run_query(query)
        columns = [meta.name.lower() for meta in metadata]
        return pd.DataFrame(results, columns=columns)

    def fetch_results(self, query: str) -> list[QueryMetricResult]:
        """Execute a query and return results as QueryMetricResult objects.

        Args:
            query (str): The SQL query to execute.

        Returns:
            list[QueryMetricResult]: List of QueryMetricResult objects.

        Raises:
            ValueError: If query results are missing required columns.
        """

        def _results_to_qmr(
            results: list[tuple], metadata: list[snowflake.connector.cursor.ResultMetadata]
        ) -> list[QueryMetricResult]:
            """Convert Snowflake results to QueryMetricResult objects.

            Args:
                results (list[tuple]): Query results.
                metadata (list[ResultMetadata]): Result metadata.

            Returns:
                list[QueryMetricResult]: Converted results.

            Raises:
                ValueError: If required columns are missing.
            """
            column_map = {meta.name.lower(): idx for idx, meta in enumerate(metadata)}

            missing = [col for col in self.required_columns if col not in column_map]
            if missing:
                logger.error(f"Missing expected columns in query results: {missing}")
                raise ValueError(f"Missing expected columns in query results: {missing}")

            result_list = []
            for record in results:
                record_dict = {key: record[idx] for key, idx in column_map.items()}
                result_list.append(QueryMetricResult(**record_dict))
            return result_list

        results, metadata = self.run_query(query)
        return _results_to_qmr(results, metadata)


class BigQueryConnector(PrecomputeConnector):
    """BigQuery database connector implementation."""

    def __init__(self, credentials: BigQueryCredentials) -> None:
        """Initialize the BigQuery connector."""
        self.credentials = credentials
        self.client = self._get_client()

    def _get_client(self):
        """Create and return a BigQuery client.

        Returns:
            bigquery.Client: Configured BigQuery client.

        Raises:
            Exception: If client creation fails.
        """
        try:
            if self.credentials.connection_type == "service_account":
                return bigquery.Client.from_service_account_json(
                    self.credentials.file_path.get_secret_value()
                )
            elif self.credentials.connection_type == "application_default":
                from google.auth import load_credentials_from_file

                creds, project_id = load_credentials_from_file(
                    self.credentials.file_path.get_secret_value(),
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                return bigquery.Client(
                    credentials=creds, project=self.credentials.project_name.get_secret_value()
                )
        except Exception as e:
            logger.exception(f"Failed to connect to BigQuery, error: {e}")
            raise e

    def run_query(self, query: str) -> bigquery.table.RowIterator:
        """Execute a query on BigQuery.

        Args:
            query (str): The SQL query to execute.

        Returns:
            bigquery.table.RowIterator: Query results iterator.
        """
        try:
            result = self.client.query(query).result()
        except Exception as e:
            logger.exception(f"Failed to run query {query}, error: {e}")
            raise e

        return result

    def run_query_to_df(self, query: str) -> pd.DataFrame:
        """Execute a query on BigQuery and return results as a pandas DataFrame.

        Args:
            query (str): The SQL query to execute.

        Returns:
            pd.DataFrame: Query results as a pandas DataFrame.

        Raises:
            Exception: If query execution fails.
        """
        return self.run_query(query).to_dataframe()

    def fetch_results(self, query: str) -> list[QueryMetricResult]:
        """Execute a query and return results as QueryMetricResult objects.

        Args:
            query (str): The SQL query to execute.

        Returns:
            list[QueryMetricResult]: List of QueryMetricResult objects.

        Raises:
            ValueError: If query results are missing required columns.
        """
        row_iterator = self.run_query(query)
        result_list = []

        columns_check = False
        for row in row_iterator:
            if not columns_check:
                missing = [col for col in self.required_columns if col not in row.keys()]  # noqa: SIM118
                if missing:
                    logger.error(f"Missing expected columns in query results: {missing}")
                    raise ValueError(f"Missing expected columns in query results: {missing}")
                columns_check = True
            result_dict = {key: row.get(key) for key in self.required_columns}
            result_list.append(QueryMetricResult(**result_dict))
        return result_list


class ConnectorResolver:
    """Class for creating appropriate database connectors."""

    @staticmethod
    def resolve(
        precompute_db_name: Literal["snowflake", "bigquery"],
        secrets: Secrets,
    ) -> BigQueryConnector | SnowflakeConnector:
        """Create and return the appropriate database connector.

        Args:
            config (ExpantoConfig): Application configuration.
            secrets (Secrets): Database credentials.

        Returns:
            BigQueryConnector | SnowflakeConnector: Configured database connector.

        Raises:
            AttributeError: If no valid credentials are found for either database.
        """

        if precompute_db_name == "snowflake" and secrets.snowflake:
            return SnowflakeConnector(secrets.snowflake)
        elif precompute_db_name == "bigquery" and secrets.bigquery:
            return BigQueryConnector(secrets.bigquery)
        else:
            raise AttributeError("Could not find any credentials to connect to Snowflake or BigQuery")
