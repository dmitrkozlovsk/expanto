"""Database Models for the Expanto experiment platform."""

from uuid import UUID, uuid4

from sqlalchemy.orm import Mapped, mapped_column

from expanto.domain.models.base import Base


class User(Base):
    __tablename__ = "users"
    __table_args__ = {"comment": ""}  # TODO: to be filled in

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
        comment="Unique identifier for the user",
    )
    name: Mapped[str] = mapped_column(
        nullable=False,
        comment="Name of the user",
    )
