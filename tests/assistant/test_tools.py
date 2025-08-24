from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from assistant.core.tools import (
    retrieve_codebase_docs,
    retrieve_internal_db,
    retrieve_metrics_docs,
    retrieve_relevant_docs,
)


@asynccontextmanager
async def mock_connect():
    mock_conn = AsyncMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [(1, "test")]
    mock_result.keys.return_value = ["id", "value"]
    mock_conn.execute.return_value = mock_result
    yield mock_conn


class MockAppContext:
    def __init__(self, data):
        self.data = data


class MockDeps:
    def __init__(self):
        self.async_db_engine = AsyncMock()
        self.vdb = AsyncMock()
        self.app_context = MockAppContext(data={"run_id": 123, "status": "done"})


class MockCtx:
    def __init__(self):
        self.deps = MockDeps()


@pytest.mark.asyncio
async def test_retrieve_internal_db_success():
    """Test successful retrieval from internal database."""
    ctx = MockCtx()
    ctx.deps.async_db_engine.connect = mock_connect

    result = await retrieve_internal_db(ctx, "SELECT * FROM test")
    assert "1 |" in result and "test" in result


@pytest.mark.asyncio
async def test_retrieve_internal_db_sql_not_valid():
    """Test that forbidden SQL keywords are detected."""
    ctx = MockCtx()
    ctx.deps.async_db_engine.connect = mock_connect
    result = await retrieve_internal_db(
        ctx, "WITH u AS (UPDATE users SET name='x' RETURNING *) SELECT * FROM u"
    )
    assert "Forbidden SQL keywords detected" in result


@pytest.mark.asyncio
async def test_retrieve_relevant_docs_success():
    """Test successful retrieval of relevant documents."""
    ctx = MockCtx()
    expected_result = {
        "search query": [
            {"id": "doc1", "document": "This is document 1"},
            {"id": "doc2", "document": "This is document 2"},
        ]
    }
    ctx.deps.vdb.semantic_search = MagicMock(return_value=expected_result)

    result = await retrieve_relevant_docs(ctx, ["search query"])
    assert result == expected_result


@pytest.mark.asyncio
async def test_retrieve_codebase_docs_empty():
    """Test retrieval of codebase docs when no results found."""
    ctx = MockCtx()
    expected_result = {"anything": []}
    ctx.deps.vdb.semantic_search = MagicMock(return_value=expected_result)

    result = await retrieve_codebase_docs(ctx, ["anything"])
    assert result == expected_result


@pytest.mark.asyncio
async def test_retrieve_metrics_docs_error():
    """Test retrieval of metrics docs when no results found."""
    ctx = MockCtx()
    expected_result = {"metric": []}
    ctx.deps.vdb.semantic_search = MagicMock(return_value=expected_result)

    result = await retrieve_metrics_docs(ctx, ["metric"])
    assert result == expected_result


@pytest.mark.asyncio
async def test_retrieve_relevant_docs_none():
    """Test retrieval of relevant docs when search returns None."""
    ctx = MockCtx()
    ctx.deps.vdb.semantic_search = MagicMock(return_value=None)

    result = await retrieve_relevant_docs(ctx, ["no results"])
    assert result is None


@pytest.mark.asyncio
async def test_retrieve_codebase_docs_multiple_queries():
    """Test retrieval of codebase docs with multiple queries."""
    ctx = MockCtx()
    expected_result = {
        "query1": [{"id": "file1.py", "document": "scode content 1"}],
        "query2": [{"id": "file2.py", "document": "code content 2"}],
    }
    ctx.deps.vdb.semantic_search = MagicMock(return_value=expected_result)

    result = await retrieve_codebase_docs(ctx, ["query1", "query2"])
    assert result == expected_result


@pytest.mark.asyncio
async def test_retrieve_metrics_docs_with_results():
    """Test retrieval of metrics docs with actual results."""
    ctx = MockCtx()
    expected_result = {
        "conversion rate": [
            {"id": "conversion_rate", "document": "Alias: conversion_rate\nType: ratio\n..."}
        ]
    }
    ctx.deps.vdb.semantic_search = MagicMock(return_value=expected_result)

    result = await retrieve_metrics_docs(ctx, ["conversion rate"])
    assert result == expected_result
