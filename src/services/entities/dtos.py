"""Data Transfer Objects (DTOs) for database entities.

This module provides automatic DTO generation for SQLAlchemy ORM models.
DTOs are used for serializing database entities and providing immutable data containers.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import MISSING, make_dataclass
from dataclasses import field as dc_field
from typing import TYPE_CHECKING, Any

from sqlalchemy.inspection import inspect as sa_inspect

from src.domain.models import CalculationJob, Experiment, Observation, Precompute

if TYPE_CHECKING:
    from sqlalchemy.orm.mapper import Mapper


# ------------------------------- DTO REGISTRY ---------------------------------


class DTORegistry:
    """Registry for generating and caching DTO classes for ORM models.

    This class automatically generates immutable dataclass DTOs based on SQLAlchemy
    model metadata. DTOs are cached to avoid regeneration.
    """

    _cache: dict[type, type] = {}

    @classmethod
    def dto_class(cls, model_cls: type) -> type:
        """Get or generate a DTO class for the specified model.

        Args:
            model_cls: The SQLAlchemy model class to create a DTO for.

        Returns:
            A dataclass type representing the DTO for the model.
        """
        if model_cls in cls._cache:
            return cls._cache[model_cls]

        mapper: Mapper = sa_inspect(model_cls)
        required: list = []
        optional: list = []

        for col in mapper.columns:
            name = col.key
            try:
                py_type = col.type.python_type  # May raise exception for unsupported types
            except Exception:
                py_type = Any
            typ = py_type | None if col.nullable else py_type
            (optional if col.nullable else required).append(
                (name, typ, dc_field(default=None if col.nullable else MISSING))
            )

        fields = required + optional
        dto_cls = make_dataclass(
            cls_name=f"{model_cls.__name__}DTO",
            fields=fields,
            frozen=True,
            slots=True,
            repr=True,
            eq=True,
            kw_only=True,  # Avoids "non-default after default" issues
        )
        cls._cache[model_cls] = dto_cls
        return dto_cls

    @classmethod
    def to_dto(cls, obj: Any) -> Any:
        """Convert an ORM model instance to its corresponding DTO.

        Args:
            obj: The ORM model instance to convert.

        Returns:
            A DTO instance with data from the ORM model.
        """
        model_cls = type(obj)
        dto_cls = cls.dto_class(model_cls)
        cols = [c.key for c in sa_inspect(model_cls).columns]
        data = {k: getattr(obj, k) for k in cols}
        return dto_cls(**data)

    @classmethod
    def to_dtos(cls, objs: Iterable[Any]) -> list[Any]:
        """Convert multiple ORM model instances to their corresponding DTOs.

        Args:
            objs: An iterable of ORM model instances to convert.

        Returns:
            A list of DTO instances.
        """
        objs = list(objs)
        if not objs:
            return []
        model_cls = type(objs[0])
        dto_cls = cls.dto_class(model_cls)
        cols = [c.key for c in sa_inspect(model_cls).columns]
        return [dto_cls(**{k: getattr(o, k) for k in cols}) for o in objs]


# ------------------------------ DTO INSTANCES --------------------------------

# Pre-generated DTO classes for common entity types
ObservationDTO: type[Any] = DTORegistry.dto_class(Observation)
ExperimentDTO: type[Any] = DTORegistry.dto_class(Experiment)
CalculationJobDTO: type[Any] = DTORegistry.dto_class(CalculationJob)
PrecomputeDTO: type[Any] = DTORegistry.dto_class(Precompute)
