"""Module for handling experiment-related input components in the Streamlit UI.

This module provides various input components and form handlers for managing experiments
in the Streamlit interface. It includes filters, form inputs, and selection components
for experiment management.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd  # type: ignore
import streamlit as st

from src.domain.enums import ExperimentStatus
from src.domain.models import Experiment
from src.ui.data_loaders import get_experiments_for_filter
from src.ui.resources import load_metrics_handler
from src.ui.state import ChatStateManager
from src.utils import DatetimeUtils as dt_utils

if TYPE_CHECKING:
    from datetime import date

    from sqlalchemy.engine.row import Row


@dataclass
class ExperimentsFilter:
    """Filter component for experiment list view.

    This class provides filtering controls for the experiments list, including
    name search, date range, status selection, and result limit.

    Attributes:
        experiment_name: Filter string for experiment names.
        created_at_start: Start date for creation date filter.
        created_at_end: End date for creation date filter.
        status_list_filter: List of selected experiment statuses.
        limit_filter: Maximum number of results to display.
    """

    experiment_name: str
    created_at_start: date
    created_at_end: date
    status_list_filter: list
    limit_filter: int

    @classmethod
    def render(cls) -> ExperimentsFilter:
        """Renders the experiment filter controls.

        Returns:
            ExperimentsFilter: Instance with the current filter values.
        """
        col1, col2, col3, col4 = st.columns([10, 10, 20, 7])
        with col1:
            experiment_name_filter = st.text_input(
                "Experiment Name",
                value="",
                help="Filter experiments by name (case sensitive). Works with sql %Like% operator.",
            )

        with col2, st.container():
            container_columns = st.columns(2)
            with container_columns[0]:
                created_at_start = st.date_input("Created from", dt_utils.utc_today() - timedelta(days=180))
            with container_columns[1]:
                created_at_end = st.date_input("Created to", dt_utils.utc_today() + timedelta(days=2))
        with col3:
            status_list_filter: list[str] = st.multiselect(
                "Status",
                ExperimentStatus.list(),
                [ExperimentStatus.PLANNED.value, ExperimentStatus.RUNNING.value],
            )
        with col4:
            limit_filter = st.number_input("Limit", min_value=2, max_value=500, value=50, step=25)

        return cls(
            experiment_name=experiment_name_filter,
            created_at_start=created_at_start,
            created_at_end=created_at_end,
            status_list_filter=status_list_filter,
            limit_filter=limit_filter,
        )


@dataclass
class ExpDfColumnsFilter:
    """Column filter for experiment dataframe display.

    This class manages the selection of columns to display in the experiment
    dataframe view.

    Attributes:
        selected_columns: List of column names to display.
    """

    selected_columns: list[str]

    @classmethod
    def render(cls, exp_df: pd.DataFrame) -> ExpDfColumnsFilter:
        """Renders the column selection interface.

        Args:
            exp_df: DataFrame containing experiment data.

        Returns:
            ExpDfColumnsFilter: Instance with selected columns.
        """
        selected_columns = st.multiselect(
            label="Display columns",
            options=list(exp_df.columns),
            default=[
                "id",
                "see_observations",
                "update",
                "name",
                "status",
                "start_datetime",
                "end_datetime",
                "key_metrics",
            ],
        )
        return cls(selected_columns=selected_columns)


@dataclass
class ExperimentFormData:
    """Data container for experiment form fields.

    This class holds all the input fields for creating or editing an experiment.

    Attributes:
        name: Name of the experiment.
        status: Current status of the experiment.
        description: Detailed description of the experiment.
        hypotheses: Experimental hypotheses.
        key_metrics: List of metrics to track.
        design: Experimental design details.
        start_datetime: Planned start time.
        end_datetime: Planned end time.
        conclusion: Experiment conclusion or results.
    """

    name: str | None
    status: str | None
    description: str | None
    hypotheses: str | None
    key_metrics: list[str] | None
    design: str | None
    start_datetime: datetime | None
    end_datetime: datetime | None
    conclusion: str | None


@dataclass
class ExperimentFormInputs:
    """Form input handler for experiment creation/editing.

    This class manages the form interface for creating or editing experiments,
    including form submission state.

    Attributes:
        experiment: Form data container.
        experiment_id: ID of the experiment being edited, if any.
        submitted: Whether the form has been submitted.
    """

    experiment: ExperimentFormData
    experiment_id: int | None
    submitted: bool

    @classmethod
    def render(cls, predefined: Experiment | None = None) -> ExperimentFormInputs | None:
        """Renders the experiment form interface.

        Args:
            predefined: Optional experiment instance to pre-fill the form.

        Returns:
            ExperimentFormInputs: Instance with form data and submission state.
        """
        state = ChatStateManager.get_or_create_state()
        if (predefined is None) and (exp_definition := state.supplements.get("ExperimentDefinition")):
            predefined = Experiment(
                **{
                    "id": None,
                    "name": exp_definition.name,
                    "status": ExperimentStatus.PLANNED,
                    "description": exp_definition.description,
                    "hypotheses": exp_definition.hypotheses,
                    "key_metrics": exp_definition.key_metrics,
                    "design": None,
                    "start_datetime": None,
                    "end_datetime": None,
                }
            )
        elif predefined is None:
            predefined = Experiment(
                **{
                    "id": None,
                    "name": None,
                    "status": ExperimentStatus.PLANNED,
                    "description": None,
                    "hypotheses": None,
                    "key_metrics": [],
                    "design": None,
                    "start_datetime": None,
                    "end_datetime": None,
                }
            )

        with st.form("experiment_form"):
            statis_options = ExperimentStatus.list()
            name = st.text_input("Experiment Name (required)", value=predefined.name)
            status: str | None = st.selectbox(
                label="Status",
                options=statis_options,
                index=statis_options.index(str(predefined.status)),
            )

            description = st.text_area("Description", value=predefined.description)
            hypotheses = st.text_area("Hypotheses", value=predefined.hypotheses)

            base_options = list(load_metrics_handler().flat.keys())
            default_metrics: list[str] = list(predefined.key_metrics) if predefined.key_metrics else []
            options_unique = sorted(set(base_options) | set(default_metrics))
            key_metrics: list[str] = st.multiselect(
                "Key Metrics",
                options=options_unique,
                default=[m for m in default_metrics if m in options_unique],
            )

            design = st.text_area("Design", value=predefined.design)
            utc_now_dt = dt_utils.utc_now().date()

            col1, col2 = st.columns([5, 2])
            with col1:
                start_date = st.date_input(
                    "Start Date (UTC)",
                    value=predefined.start_datetime.date() if predefined.start_datetime else utc_now_dt,
                )
                end_date = st.date_input(
                    "End Date (UTC)",
                    value=predefined.end_datetime.date() if predefined.end_datetime else None,
                )
            with col2:
                start_time = st.time_input(
                    "Start Time (UTC)",
                    value=predefined.start_datetime.time() if predefined.start_datetime else "00:00",
                )
                end_time = st.time_input(
                    "End Time (UTC)",
                    value=predefined.end_datetime.time() if predefined.end_datetime else "00:00",
                )

            conclusion = st.text_area("Conclusion", value=predefined.conclusion)

            start_datetime = datetime.combine(start_date, start_time)
            end_datetime = datetime.combine(end_date, end_time) if end_date and end_time else None

            submitted = st.form_submit_button("Submit Form")
            if (not name or name.strip() == "") and submitted:
                st.toast("⚠️The required field `Name` must be filled.")
                return None

            return cls(
                experiment=ExperimentFormData(
                    name=name,
                    status=status,
                    description=description,
                    hypotheses=hypotheses,
                    key_metrics=key_metrics,
                    design=design,
                    start_datetime=start_datetime,
                    end_datetime=end_datetime,
                    conclusion=conclusion,
                ),
                experiment_id=int(predefined.id) if predefined.id else None,
                submitted=submitted,
            )


@dataclass
class ExperimentSelectBox:
    """Selection component for choosing an experiment.

    This class provides a dropdown interface for selecting an experiment
    from the available options.

    Attributes:
        id: Selected experiment ID.
        name: Selected experiment name.
    """

    id: int
    name: str

    @classmethod
    def render(cls, predefined_exp_id: int | None) -> ExperimentSelectBox | None:
        """Renders the experiment selection dropdown.

        Args:
            predefined_exp_id: Optional experiment ID from URL to pre-select.

        Returns:
            ExperimentSelectBox | None: Instance with selected experiment data,
                or None if no selection made.
        """
        exp_options = get_experiments_for_filter() or []
        filter_ = filter(lambda r: r[1][0] == predefined_exp_id, enumerate(exp_options))
        generator = map(lambda r: r[0], filter_)
        select_box_index: int | None = next(generator, None)

        selected: Row | None = st.selectbox(
            label="Select experiment",
            options=exp_options,
            format_func=lambda x: x[1],
            index=select_box_index,
        )
        if selected is None:
            return None
        return cls(id=selected[0], name=selected[1])
