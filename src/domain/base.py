from typing import Any

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all database models using SQLAlchemy's DeclarativeBase."""

    def to_dict(self) -> dict[str, Any]:
        """Convert the model instance to a dictionary.

        Returns:
            dict[str, Any]: A dictionary with column names as keys and their values.
        """
        return {key: getattr(self, key) for key in self.__table__.columns.keys()}  # noqa: SIM118
