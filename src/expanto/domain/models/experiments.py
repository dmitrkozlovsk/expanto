"""Database Models for the Expanto experiment platform."""

from uuid import UUID

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from expanto.domain.models.base import Base


class Experiment(Base):
    __tablename__ = "experiments"
    __table_args__ = {"comment": ""}  # TODO: to be filled in

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True, comment="Unique identifier for the experiment"
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        comment="Unique name of the experiment used for identification and querying",
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        comment="Status of the experiment (e.g., 'running', 'completed', 'failed')",
    )

    # current_revision_id: Mapped[int] = mapped_column(
    #     ForeignKey("experiment_revision.id"),
    #     nullable=False,
    #     comment="ID of the current revision of the experiment",
    # )

    created_by: Mapped[UUID] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        comment="ID of the user who created the experiment",
    )
