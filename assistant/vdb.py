"""Vector database management for metrics, documentation, and codebase."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import chromadb

from src.logger_setup import logger
from src.services.metric_register import Metrics

if TYPE_CHECKING:
    from chromadb.api.client import ClientAPI


metric_doc_template = """
Alias: {alias}
Type: {type}
Display Name: {display_name}
Description: {description}
SQL: {sql}
Formula: 
    Numerator: {numerator}
    Denominator: {denominator}
Tags: {tags}
Owner: {owner}
Group Name: {group_name}
"""

Scalar = str | int | float | bool | None
Metadata = Mapping[str, Scalar]


class VectorDB:
    """Universal vector database for managing and querying metrics, documentation, and codebase.

    Uses different ChromaDB collections for different content types:
    - metrics: Metric definitions from YAML files
    - documentation: Project documentation files (*.md) - one file = one document
    - codebase: Source code files (*.py, *.yaml, *.json, etc.) - one file = one document

    Attributes:
        metrics_directory: Path to the directory containing metric definitions
        docs_directory: Path to the directory containing documentation
        root_directory: Path to the project root for codebase indexing
        chroma_client: ChromaDB client instance
        metric_collection: Collection storing metric embeddings
        docs_collection: Collection storing documentation
        code_collection: Collection storing codebase
        metrics: Metrics service instance
        indexable_extensions: File extensions to index for codebase
        skip_directories: Directories to skip when indexing codebase
    """

    def __init__(
        self,
        metrics_directory: str | None = None,
        docs_directory: str = "docs",
        root_directory: str = ".",
    ) -> None:
        """Initialize the VectorDB instance.

        Args:
            metrics_directory: Path to the directory containing metric definitions
            docs_directory: Path to the directory containing documentation
            root_directory: Path to the project root directory
        """
        self.metrics_directory: Path = Path(metrics_directory) if metrics_directory else Path(".")
        self.docs_directory: Path = Path(docs_directory)
        self.root_directory: Path = Path(root_directory)
        self.chroma_client: ClientAPI = chromadb.Client()
        self.metric_collection: chromadb.Collection | None = None
        self.docs_collection: chromadb.Collection | None = None
        self.code_collection: chromadb.Collection | None = None

        # Initialize metrics service
        self.metrics = Metrics(self.metrics_directory)

        # File types to index for codebase
        self.indexable_extensions = {".py", ".md", ".txt", ".yaml", ".yml", ".sql"}

        # Directories to skip when indexing codebase
        self.skip_directories = {
            "__pycache__",
            ".git",
            ".pytest_cache",
            "node_modules",
            ".venv",
            "venv",
            ".env",
            "dist",
            "build",
            ".uv",
            ".streamlit",
        }

    def create_metric_collection(self) -> None:
        """Create and populate the metrics collection in ChromaDB.

        This method:
        1. Creates a new collection in ChromaDB
        2. Retrieves all metrics from the metrics manager
        3. Formats each metric into a document using the metric_doc_template
        4. Creates metadata for each metric
        5. Adds all documents to the collection with their respective metadata and IDs
        """
        self.metric_collection = self.chroma_client.get_or_create_collection(name="metrics")

        # Get all metrics from the metrics manager
        flat_metrics = self.metrics.flat

        # Prepare documents and metadata for each metric
        documents: list[str] = []
        metadatas: list[Metadata] = []
        ids: list[str] = []

        for alias, metric in flat_metrics.items():
            metric_doc = metric_doc_template.format(
                alias=alias,
                type=metric.type,
                display_name=metric.display_name if metric.display_name else "",
                description=metric.description if metric.description else "",
                sql=metric.sql if metric.sql else "",
                numerator=metric.formula.numerator if metric.formula.numerator else "",
                denominator=metric.formula.denominator if metric.formula.denominator else "",
                tags=f"{', '.join(metric.tags)}" if metric.tags else "",
                owner=metric.owner or "",
                group_name=metric.group_name or "",
            )
            logger.debug("Metric doc created", metric_doc=metric_doc)
            # Create metadata
            metadata = {"group_name": metric.group_name, "owner": metric.owner or ""}

            documents.append(metric_doc)
            metadatas.append(metadata)
            ids.append(alias)

        # Add all documents to the collection
        if documents:
            self.metric_collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

    def create_docs_collection(self) -> None:
        """Create and populate the documentation collection. One file = one document."""
        self.docs_collection = self.chroma_client.get_or_create_collection(name="documentation")

        documents: list[str] = []
        metadatas: list[Metadata] = []
        ids: list[str] = []

        # Index all markdown and text files in docs directory
        if self.docs_directory.exists():
            for file_path in self.docs_directory.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in [".md", ".txt", ".rst"]:
                    try:
                        content = file_path.read_text(encoding="utf-8")

                        # One file = one document
                        doc_id = f"doc_{file_path.relative_to(self.docs_directory)}"
                        documents.append(content)
                        metadatas.append(
                            {
                                "source": str(file_path.relative_to(self.docs_directory)),
                                "file_type": file_path.suffix,
                                "file_size": len(content),
                                "content_type": "documentation",
                            }
                        )
                        ids.append(doc_id)

                    except Exception as e:
                        logger.warning(f"Failed to index {file_path}: {e}")

        if documents:
            self.docs_collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
            logger.info(f"Indexed {len(documents)} documentation files")

    def create_code_collection(self) -> None:
        """Create and populate the codebase collection. One file = one document."""
        self.code_collection = self.chroma_client.get_or_create_collection(name="codebase")

        documents: list[str] = []
        metadatas: list[Metadata] = []
        ids: list[str] = []

        # Index code files
        for file_path in self._get_indexable_files():
            content = file_path.read_text(encoding="utf-8")

            # Skip very large files
            if len(content) > 100000:
                logger.info(f"Skipping large file: {file_path}")
                continue

            # One file = one document
            doc_id = f"{file_path.relative_to(self.root_directory)}"
            documents.append(content)
            metadatas.append(
                {
                    "file_path": str(file_path.relative_to(self.root_directory)),
                    "file_size": len(content),
                    "content_type": "code",
                }
            )
            ids.append(doc_id)

        if documents:
            self.code_collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
            logger.info(f"Indexed {len(documents)} code files")

    def create_all_collections(self) -> None:
        """Create and populate all collections."""
        logger.info("Creating all vector database collections...")

        try:
            self.create_metric_collection()
            self.create_docs_collection()
            self.create_code_collection()
            logger.info("All collections created successfully")
        except Exception as e:
            logger.error(f"Error creating collections: {e}", exc_info=True)
            raise

    def semantic_search(
        self,
        collection_name: Literal["metrics", "documentation", "codebase"],
        queries: list[str],
        n_results: int = 3,
        deduplicate: bool = True,
    ) -> dict[str, list[dict]] | None:
        """Perform semantic search across the specified collection.

        Args:
            collection_name: Name of the collection to search
            queries: List of search queries
            n_results: Maximum number of results per query
            deduplicate: Whether to remove duplicate documents across all queries

        Returns:
            Dictionary mapping each query to list of results, or None if no results.
            Each result is a dict with "id" and "document" keys.
        """
        logger.debug(f"VectorDB queries: {queries}")
        collection = self.chroma_client.get_or_create_collection(name=collection_name)
        row_results = collection.query(query_texts=queries, n_results=n_results)
        ids_nested = row_results.get("ids")
        docs_nested = row_results.get("documents")

        if ids_nested is None or docs_nested is None:
            logger.warning("No ids or documents in row_results")
            return None

        logger.debug(f"VectorDB query results: {row_results}")

        results: dict[str, list[dict]] = {}
        if not deduplicate:
            for index, query in enumerate(queries):
                results[query] = []
                for i in range(len(ids_nested[index])):
                    results[query].append({"id": ids_nested[index][i], "document": docs_nested[index][i]})
            return results

        # Deduplicate documents by ID
        seen_ids: set[str] = set()

        for index, query in enumerate(queries):
            results[query] = []
            for i in range(len(ids_nested[index])):
                doc_id = ids_nested[index][i]
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    results[query].append({"id": doc_id, "document": docs_nested[index][i]})

        return results

    def _get_indexable_files(self) -> Generator[Path, None, None]:
        """Get all files that should be indexed from codebase."""
        for file_path in self.root_directory.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.indexable_extensions
                and not any(skip_dir in file_path.parts for skip_dir in self.skip_directories)
            ):
                yield file_path
