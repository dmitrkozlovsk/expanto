"""Tools and utilities for AI agent interactions."""

from __future__ import annotations

import asyncio

import pandas as pd  # type: ignore
from pydantic_ai import RunContext
from sqlalchemy import text

from assistant.core.schemas import Deps
from src.logger_setup import logger
from src.utils import ValidationUtils


async def retrieve_internal_db(ctx: RunContext[Deps], query: str) -> str:
    """Retrieve data from the internal SQL database.

    Args:
        ctx: The run context containing the agent dependencies
        query: The SQL query to execute

    Returns:
        A markdown representation of the query result, or an error message
    """

    logger.info(f"Executing internal DB query: `{query}`")
    try:
        ValidationUtils.validate_sql_query(query)
        async with ctx.deps.async_db_engine.connect() as conn:
            result = await conn.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            logger.info(f"Query returned {len(df)} rows")
            if df.empty:
                return "Query executed successfully, but returned no results."
            return df.to_markdown() + "\n\n" + f"# Query: {query}"
    except Exception as e:
        logger.error(f"Error executing query: {e}", exc_info=True)
        return f"An error occurred while executing the query: {e}"


async def retrieve_relevant_docs(ctx: RunContext[Deps], queries: list[str]) -> dict[str, list[dict]] | None:
    # todo: add more info about the project/docs
    """Retrieve relevant documentation about Expanto project from the vector database.

    Documentation is stored in the "documentation" collection. Returns docs from /docs folder.

    Args:
        queries: List of queries to search for in the vector database (English only)

    Returns:
        A dictionary mapping queries to search results, or None if no results
    """

    vdb = ctx.deps.vdb
    results = await asyncio.to_thread(
        vdb.semantic_search, collection_name="documentation", queries=queries, n_results=2
    )
    return results


async def retrieve_codebase_docs(ctx: RunContext[Deps], queries: list[str]) -> dict[str, list[dict]] | None:
    """Retrieve relevant code documentation about Expanto project from the vector database.

    Documentation is stored in the "codebase" collection. Returns code files.

    Args:
        queries: List of queries to search for in the vector database (English only)

    Returns:
        A dictionary mapping queries to search results, or None if no results
    """
    vdb = ctx.deps.vdb
    results = await asyncio.to_thread(
        vdb.semantic_search, collection_name="codebase", queries=queries, n_results=2
    )
    return results


async def retrieve_metrics_docs(ctx: RunContext[Deps], queries: list[str]) -> dict[str, list[dict]] | None:
    """Retrieve relevant metric descriptions from repository.

    Based on the query, returns the most relevant metric descriptions.

    Args:
        queries: List of queries to search for in the vector database (English only)

    Returns:
        A dictionary mapping queries to search results, or None if no results
    """
    vdb = ctx.deps.vdb
    results = await asyncio.to_thread(
        vdb.semantic_search, collection_name="metrics", queries=queries, n_results=2
    )
    return results


async def get_expanto_app_context(ctx: RunContext[Deps]) -> str:
    """Get detailed information about the current Expanto application context.

    Retrieves and formats the current application state including UI context,
    user selections, and active data for enhanced agent awareness.

    Context may include:
        - Current page (experiments, results, observations, planner)
        - Page mode (create, edit, view, analyze)
        - Selected experiment definitions
        - Selected experiment results data
        - Selected observation definitions
        - Active filters and configurations
        - etc.

    Args:
        ctx: The run context containing application dependencies

    Returns:
        Formatted string with current application context details,
        or error message if context retrieval fails
    """
    try:
        app_ctx_str = f"{ctx.deps.app_context}"
        return app_ctx_str

    except Exception as e:
        logger.error(f"Error retrieving application context: {e}", exc_info=True)
        return f"AppContext: Error retrieving context - {e}"
