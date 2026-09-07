"""Entity handlers for database operations.

This module contains handlers for performing CRUD operations on database entities.
It provides a generic base handler and specific handlers for each entity type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd  # type: ignore
from sqlalchemy import insert, or_, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from src.domain.models import CalculationJob, Experiment, Observation, Precompute
from src.services.entities.dtos import DTORegistry

if TYPE_CHECKING:
    from collections.abc import Sequence

    import sqlalchemy
    from sqlalchemy import Column


class GenericEntityHandler[E]:
    """Base handler class for database entities with generic CRUD operations.

    This class provides a foundation for entity-specific handlers with common database
    operations including create, read, update, delete, and advanced querying capabilities.

    Attributes:
        entity: The SQLAlchemy model class this handler manages.
        engine: SQLAlchemy database engine instance.

    Args:
        db_engine: SQLAlchemy database engine instance.
    """

    entity: type[E]

    def __init__(self, db_engine: Engine):
        """Initialize the handler with a database engine."""
        self.engine = db_engine

    def _get_session(self) -> Session:
        """Get a new database session.

        Returns:
            A new SQLAlchemy session instance.
        """
        return Session(self.engine)

    def create(self, **kwargs) -> E | None:
        """Create a new entity instance.

        Args:
            **kwargs: Entity attributes to set on the new instance.

        Returns:
            The created entity instance or None if creation failed.
        """
        with self._get_session() as local_session:
            obj = self.entity(**kwargs)
            local_session.add(obj)
            local_session.commit()
            local_session.refresh(obj)
            return obj

    def bulk_insert(self, records: list[dict[str, Any]]) -> Sequence[E]:
        """Insert multiple entity records in a single database operation.

        Args:
            records: List of dictionaries containing entity attributes for each record to insert.

        Returns:
            List of created entity instances.
        """
        with self._get_session() as local_session:
            created_records = local_session.scalars(
                insert(self.entity).returning(self.entity), records
            ).all()
            local_session.commit()
            return created_records

    # ----------------------------- QUERY METHODS ---------------------------------
    def _get_by_id_stmt(self, id_: int | Column[int]) -> sqlalchemy.sql.selectable.Select:  # type: ignore[attr-defined]
        """Create a select statement for finding an entity by ID.

        Args:
            id_: The ID of the entity to find.

        Returns:
            A SQLAlchemy select statement.
        """
        return select(self.entity).where(self.entity.id == id_)  # type: ignore[attr-defined]

    def get(self, id_: int | Column[int], as_dto: bool = False) -> E | Any | None:
        """Fetch an entity by its ID from the database.

        Args:
            id_: The ID of the entity to fetch.
            as_dto: Whether to return the result as a DTO. Defaults to False.

        Returns:
            The entity instance, DTO, or None if not found.
        """
        stmt = self._get_by_id_stmt(id_)
        with self._get_session() as local_session:
            obj = local_session.execute(stmt).scalar_one_or_none()
        if as_dto and obj:
            return DTORegistry.to_dto(obj)
        return obj

    def update(self, id_: int | Column[int], **kwargs) -> E | None:
        """Update an entity's attributes.

        Args:
            id_: The ID of the entity to update.
            **kwargs: Entity attributes to update.

        Returns:
            The updated entity instance or None if not found.
        """
        stmt = self._get_by_id_stmt(id_)
        with self._get_session() as local_session:
            obj = local_session.execute(stmt).scalar_one_or_none()
            if obj:
                for key, value in kwargs.items():
                    if hasattr(obj, key):
                        setattr(obj, key, value)
                local_session.commit()
                local_session.refresh(obj)
            return obj

    def delete(self, id_: int | Column[int]) -> E | None:
        """Delete an entity by its ID.

        Args:
            id_: The ID of the entity to delete.

        Returns:
            The deleted entity instance or None if not found.
        """
        stmt = self._get_by_id_stmt(id_)
        with self._get_session() as local_session:
            obj = local_session.execute(stmt).scalar_one_or_none()
            if obj:
                local_session.delete(obj)
                local_session.commit()
            return obj

    def select(
        self,
        filters: dict[str, Any] | None = None,
        sort_by: str | None = None,
        sort_order: str | None = None,
        limit: int | None = None,
        return_pandas: bool = False,
        columns: list[str] | None = None,
    ) -> list[sqlalchemy.engine.row.Row] | pd.DataFrame | None:
        """Query entities with filtering, sorting, and pagination.

        This method supports complex querying with the following features:
        - Filtering with multiple operators (eq, ilike, in, gte, lte, gt, lt, contains)
        - OR conditions using "__or" suffix
        - Sorting by any column
        - Result limiting
        - Optional pandas DataFrame output
        - Column selection

        Filter operators:
            - eq: Equal to
            - ilike: Case-insensitive LIKE
            - in: Value in list
            - gte: Greater than or equal
            - lte: Less than or equal
            - gt: Greater than
            - lt: Less than
            - contains: Contains substring

        Args:
            filters: Dictionary of field names and values to filter by.
                    Use "__" suffix for operators (e.g., "name__ilike").
                    Use "__or" suffix for OR conditions.
            sort_by: Column name to sort by.
            sort_order: Sort direction ("asc" or "desc").
            limit: Maximum number of results to return.
            return_pandas: Whether to return results as a pandas DataFrame.
            columns: List of column names to select.

        Returns:
            List of SQLAlchemy rows or pandas DataFrame containing the results.
        """
        stmt = select(*[getattr(self.entity, col) for col in columns]) if columns else select(self.entity)

        if filters:
            for field, value in filters.items():
                is_or = False
                if "__or" in field:
                    field = field.replace("__or", "")
                    is_or = True

                if "__" in field:
                    field_name, op = field.split("__", 1)
                else:
                    field_name, op = field, "eq"

                column = getattr(self.entity, field_name)

                def build_clause(val_, column_, op_):
                    if op_ == "eq":
                        return column_ == val_
                    elif op_ == "ilike":
                        return column_.ilike(val_)
                    elif op_ == "in":
                        return column_.in_(val_)
                    elif op_ == "gte":
                        return column_ >= val_
                    elif op_ == "lte":
                        return column_ <= val_
                    elif op_ == "gt":
                        return column_ > val_
                    elif op_ == "lt":
                        return column_ < val_
                    elif op_ == "contains":
                        return column_.contains(val_)

                if is_or and isinstance(value, list):
                    stmt = stmt.where(or_(*[build_clause(v, column, op) for v in value]))
                else:
                    stmt = stmt.where(build_clause(value, column, op))
        if sort_by:
            column = getattr(self.entity, sort_by)
            stmt = stmt.order_by(column.desc() if sort_order == "desc" else column.asc())
        if limit:
            stmt = stmt.limit(limit)
        with self._get_session() as session:
            if return_pandas:
                return pd.read_sql(stmt, session.bind)
            return session.execute(stmt).fetchall()


# ------------------------------ ENTITY HANDLERS --------------------------------


class ExperimentHandler(GenericEntityHandler[Experiment]):
    """Handler for Experiment entities.

    Attributes(Available columns for filtering):
        - id: Integer (primary key)
        - name: String (unique)
        - status: String
        - description: Text
        - hypotheses: Text
        - key_metrics: List
        - design: Text
        - conclusion: Text
        - start_datetime: DateTime
        - end_datetime: DateTime
        - _created_at: DateTime
    """

    entity = Experiment


class ObservationHandler(GenericEntityHandler[Observation]):
    """Handler for Observation entities.

    Attributes(Available columns for filtering):
        - id: Integer (primary key)
        - experiment_id: Integer (foreign key)
        - name: String
        - db_experiment_name: String
        - split_id: Text
        - calculation_scenario: String
        - exposure_start_datetime: DateTime
        - exposure_end_datetime: DateTime
        - calc_start_datetime: DateTime
        - calc_end_datetime: DateTime
        - exposure_event: Text
        - audience_tables: List
        - filters: List
        - custom_test_ids_query: Text
        - metric_tags: List
        - metric_groups: List
        - _created_at: DateTime
    """

    entity = Observation


class JobHandler(GenericEntityHandler[CalculationJob]):
    """Handler for CalculationJob entities.

    Attributes(Available columns for filtering):
        - id: Integer (primary key)
        - observation_id: Integer (foreign key)
        - query: Text
        - status: String
        - error_message: Text
        - extra: JSON (query execution metadata)
        - _created_at: DateTime
    """

    entity = CalculationJob


class PrecomputeHandler(GenericEntityHandler[Precompute]):
    """Handler for Precompute entities.

    Attributes(Available columns for filtering):
        - id: Integer (primary key)
        - job_id: Integer (foreign key)
        - group_name: Text
        - metric_name: Text
        - metric_type: Text
        - observation_cnt: Integer
        - metric_value: Float
        - numerator_avg: Float
        - denominator_avg: Float
        - numerator_var: Float
        - denominator_var: Float
        - covariance: Float
        - _created_at: DateTime
    """

    entity = Precompute


# ----------------------------- MAIN HANDLER FACADE ------------------------------


class EntitiesHandler:
    """Main handler class that provides access to all entity handlers.

    This class serves as a facade for accessing all entity handlers in the system.
    It initializes individual handlers for each entity type.

    Args:
        db_engine: SQLAlchemy database engine instance.
    """

    def __init__(self, db_engine: Engine):
        """Initialize the handler with a database engine.

        Args:
            db_engine: SQLAlchemy database engine instance.
        """
        self.engine = db_engine

        # Initialize individual entity handlers
        self.experiments = ExperimentHandler(db_engine)
        self.observations = ObservationHandler(db_engine)
        self.jobs = JobHandler(db_engine)
        self.precomputes = PrecomputeHandler(db_engine)

    # TODO: Implement complex queries for joined operations across entities
