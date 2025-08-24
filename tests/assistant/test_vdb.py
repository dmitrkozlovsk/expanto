import pytest

from assistant.vdb import VectorDB


def test_metrics_collection_without_metrics(tmp_path_factory):
    """Test creating metrics collection without metrics files."""
    vdb = VectorDB(metrics_directory=tmp_path_factory.mktemp("fake"))
    vdb.create_metric_collection()
    assert vdb.metric_collection is not None
    assert vdb.metric_collection.count() == 0


def test_create_metric_collection_with_metrics(metrics_temp_dir):
    """Test creating metrics collection with existing metrics files."""
    vdb = VectorDB(metrics_directory=metrics_temp_dir)

    vdb.create_metric_collection()

    assert vdb.metric_collection is not None
    assert vdb.metric_collection.count() == 5

    all_docs = vdb.metric_collection.get()
    assert len(all_docs["ids"]) == 5

    expected_metric_names = {
        "avg_order_value",
        "click_through_rate",
        "conversion_rate",
        "avg_session_duration",
        "users_who_click_product_ratio",
    }
    assert set(all_docs["ids"]) == expected_metric_names


def test_search_metrics_collection(metrics_temp_dir):
    """Test semantic search in metrics collection."""
    vdb = VectorDB(metrics_directory=metrics_temp_dir)
    vdb.create_metric_collection()
    queries = ["order value", "C T R"]
    results = vdb.semantic_search(collection_name="metrics", queries=queries, n_results=2)
    assert "avg_order_value" in [r["id"] for r in results["order value"]]
    assert "click_through_rate" in [r["id"] for r in results["C T R"]]


def test_create_docs_collection(docs_temp_dir, metrics_temp_dir):
    """Test creating documentation collection."""
    vdb = VectorDB(docs_directory=docs_temp_dir, metrics_directory=metrics_temp_dir)
    vdb.create_docs_collection()
    assert vdb.docs_collection is not None
    assert vdb.docs_collection.count() == 4
    all_docs = vdb.docs_collection.get()
    assert len(all_docs["ids"]) == 4


def test_semantic_search_docs_collection(docs_temp_dir, metrics_temp_dir):
    """Test semantic search in documentation collection."""
    vdb = VectorDB(docs_directory=docs_temp_dir, metrics_directory=metrics_temp_dir)
    vdb.create_docs_collection()
    results = vdb.semantic_search(collection_name="documentation", queries=["integration"], n_results=2)
    assert "doc_integration.md" in [r["id"] for r in results["integration"]]


def test_create_code_collection(root_dir, metrics_temp_dir):
    """Test creating code collection."""
    vdb = VectorDB(root_directory=root_dir, metrics_directory=metrics_temp_dir)
    vdb.create_code_collection()
    assert vdb.code_collection is not None
    all_docs = vdb.code_collection.get()
    assert len(all_docs["ids"]) == 3

    for id_name in all_docs["ids"]:
        assert ".expanto" not in id_name
        assert ".src" not in id_name


def test_semantic_search_code_collection(root_dir, metrics_temp_dir):
    """Test semantic search in code collection."""
    vdb = VectorDB(root_directory=root_dir, metrics_directory=metrics_temp_dir)
    vdb.create_code_collection()
    results = vdb.semantic_search(collection_name="codebase", queries=["bigquery"], n_results=1)
    assert "src/services/bigquery.py" in [r["id"] for r in results["bigquery"]]
    results = vdb.semantic_search(
        collection_name="codebase", queries=["how metric register works"], n_results=2
    )
    assert "src/services/metric_register.py" in [r["id"] for r in results["how metric register works"]]


@pytest.mark.parametrize(
    "factory_kwargs, create_method, collection_attr",
    [
        # metrics
        (
            lambda paths: {"metrics_directory": paths["metrics"]},
            "create_metric_collection",
            "metric_collection",
        ),
        # docs
        (
            lambda paths: {"docs_directory": paths["docs"], "metrics_directory": paths["metrics"]},
            "create_docs_collection",
            "docs_collection",
        ),
        # code
        (
            lambda paths: {"root_directory": paths["root"], "metrics_directory": paths["metrics"]},
            "create_code_collection",
            "code_collection",
        ),
    ],
    ids=["metrics", "docs", "code"],
)
def test_collections_are_idempotent(
    factory_kwargs,
    create_method,
    collection_attr,
    metrics_temp_dir,
    docs_temp_dir,
    root_dir,
):
    """Test that collection creation is idempotent."""
    paths = {
        "metrics": metrics_temp_dir,
        "docs": docs_temp_dir,
        "root": root_dir,
    }

    vdb = VectorDB(**factory_kwargs(paths))

    getattr(vdb, create_method)()
    collection = getattr(vdb, collection_attr)
    first_count = collection.count()
    first_ids = set(collection.get()["ids"])

    getattr(vdb, create_method)()
    collection = getattr(vdb, collection_attr)
    second_count = collection.count()
    second_ids = set(collection.get()["ids"])

    assert first_count == second_count, "Collection count changed"
    assert first_ids == second_ids, "Collection ids changed"
