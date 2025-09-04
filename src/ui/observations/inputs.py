"""This module provides dataclasses for handling observation-related inputs in the UI.

It includes classes for:
- Filtering observations.
- Managing DataFrame column selections.
- Handling observation form data and inputs.
- Creating select boxes for observations and experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd  # type: ignore
import streamlit as st

from src.domain.models import Observation
from src.ui.data_loaders import get_observations_for_filter
from src.ui.experiments.inputs import ExperimentSelectBox
from src.ui.resources import load_calculation_scenarios
from src.utils import DatetimeUtils as dt_utils
from src.utils import ValidationUtils

if TYPE_CHECKING:
    from sqlalchemy.engine.row import Row

    from src.ui.common import URLParams


@dataclass
class ObservationsFilter:
    """Dataclass to hold observation filter values."""

    experiment_id: int | None
    experiment_name: str | None
    observation_name: str
    created_at_start: datetime
    created_at_end: datetime
    limit_filter: int

    @classmethod
    def render(cls, url_exp_id: int | None = None) -> ObservationsFilter:
        """Render filters for observations page.

        Args:
            url_exp_id (int | None): The experiment ID from URL parameters.
                Defaults to None.

        Returns:
            ObservationsFilter: An instance of the class with filter values.
        """

        col1, col2, col3, col4 = st.columns([10, 10, 15, 7])

        with col1:
            experiment_selectbox = ExperimentSelectBox.render(url_exp_id)

        with col2:
            observation_name_filter = st.text_input("Observation Name", "")

        with col3:  # noqa: SIM117
            with st.container():  # noqa: SIM117
                container_columns = st.columns(2)
                with container_columns[0]:
                    created_at_start_date = st.date_input(
                        "Created from", dt_utils.utc_today() - timedelta(days=180)
                    )
                with container_columns[1]:
                    created_at_end_date = st.date_input(
                        "Created to", dt_utils.utc_today() + timedelta(days=1)
                    )

        with col4:
            limit_filter = st.number_input("Limit", min_value=2, max_value=500, value=50, step=25)

        created_at_start = datetime.combine(created_at_start_date, datetime.min.time())
        created_at_end = datetime.combine(created_at_end_date, datetime.min.time())

        return cls(
            experiment_id=experiment_selectbox.id if experiment_selectbox else None,
            experiment_name=experiment_selectbox.name if experiment_selectbox else None,
            observation_name=observation_name_filter,
            created_at_start=created_at_start,
            created_at_end=created_at_end,
            limit_filter=limit_filter,
        )


@dataclass
class ObsDfColumnsFilter:
    """Dataclass to hold the selected columns for the observation DataFrame."""

    selected_columns: list[str]

    @classmethod
    def render(cls, obs_df: pd.DataFrame) -> ObsDfColumnsFilter:
        """Render a multiselect widget to filter DataFrame columns.

        Args:
            obs_df (pd.DataFrame): The observation DataFrame to get columns from.

        Returns:
            ObsDfColumnsFilter: An instance of the class with selected columns.
        """
        display_columns = st.multiselect(
            "Display columns",
            list(obs_df.columns),
            [
                "see_results",
                "name",
                "x",
                "split_id",
                "filters",
                "audience_tables",
                "metric_tags",
                "metric_groups",
            ],
        )
        return cls(display_columns)


@dataclass
class ObservationFormData:
    """Dataclass to hold data from the observation form."""

    experiment_id: int
    name: str
    db_experiment_name: str
    split_id: str
    calculation_scenario: str
    exposure_start_datetime: datetime
    exposure_end_datetime: datetime
    calc_start_datetime: datetime
    calc_end_datetime: datetime
    exposure_event: str | None
    audience_tables: list[str] | None
    filters: list[str] | None
    custom_test_ids_query: str | None
    metric_tags: list[str] | None
    metric_groups: list[str] | None


@dataclass
class ObservationFormInputs:
    """Dataclass to hold the observation form inputs and state."""

    observation: ObservationFormData
    observation_id: int | None
    submitted: bool

    @classmethod
    def render(
        cls,
        experiment_id: int | None = None,
        experiment_name: str | None = None,
        predefined: Observation | None = None,
    ) -> ObservationFormInputs | None:
        """Render the observation form.

        This method displays a form for creating or updating an observation. It handles
        pre-filling form fields if a predefined observation is provided. It also
        performs validation on form submission.

        Args:
            experiment_id (int | None, optional): The ID of the experiment.
                Defaults to None.
            experiment_name (str | None, optional): The name of the experiment.
                Defaults to None.
            predefined (Observation | None, optional): A predefined Observation object
                to populate the form. Defaults to None.

        Returns:
            ObservationFormInputs | None: An instance of ObservationFormInputs if the form
                is valid and submitted, otherwise None.
        """
        if experiment_id is None or experiment_name is None:
            return None

        if predefined is None:
            lower_exp_name = "_".join(experiment_name.lower().split())
            predefined_obs_name = f"{lower_exp_name}_obs ()"
            predefined = Observation(
                experiment_id=experiment_id,
                name=predefined_obs_name,
                db_experiment_name=None,
                split_id=None,
                calculation_scenario=None,
                exposure_start_datetime=None,
                exposure_end_datetime=None,
                calc_start_datetime=None,
                calc_end_datetime=None,
                exposure_event=None,
                audience_tables=[],
                filters=[],
                custom_test_ids_query=None,
                metric_tags=[],
                metric_groups=[],
            )

        with st.form("observation_form"):
            col1l1, col2l1 = st.columns(2)
            with col1l1:
                name = st.text_input("Observation Name (required)", value=predefined.name)
            with col2l1:
                db_experiment_name = st.text_input(
                    "Database Experiment Name (required)", value=predefined.db_experiment_name
                )
            col1l2, col2l2 = st.columns(2)
            with col1l2:
                split_id = st.text_input("Split ID (required)", value=predefined.split_id)
            with col2l2:
                scenario_list = load_calculation_scenarios()
                scenario_value = str(predefined.calculation_scenario)
                default_index = (
                    scenario_list.index(scenario_value) if scenario_value in scenario_list else 0
                )
                calculation_scenario = st.selectbox(
                    "Calculation Scenario",
                    scenario_list,
                    index=default_index,
                )
            col1l3, col2l3 = st.columns(2)
            # dates and times
            utc_now_dt = dt_utils.utc_now().date()
            with col1l3:
                # Exposure period
                st.subheader("Exposure Period")
                col1, col2 = st.columns([5, 2])
                with col1:
                    exposure_start_date = st.date_input(
                        "Start Date",
                        value=predefined.exposure_start_datetime.date()
                        if predefined.exposure_start_datetime
                        else utc_now_dt,
                        key="exposure_start_date_input_key",
                    )
                with col2:
                    exposure_start_time = st.time_input(
                        "Start Time",
                        value=predefined.exposure_start_datetime.time()
                        if predefined.exposure_start_datetime
                        else "00:00",
                        key="exposure_start_time_input_key",
                    )
                col1, col2 = st.columns([5, 2])
                with col1:
                    exposure_end_date = st.date_input(
                        "End Date",
                        value=predefined.exposure_end_datetime.date()
                        if predefined.exposure_end_datetime
                        else utc_now_dt + timedelta(days=14),
                        key="exposure_end_date_input_key",
                    )
                with col2:
                    exposure_end_time = st.time_input(
                        "End Time",
                        value=predefined.exposure_end_datetime.time()
                        if predefined.exposure_end_datetime
                        else "00:00",
                        key="exposure_end_time_input_key",
                    )

            with col2l3:
                # Calculation period
                st.subheader("Calculation Period")
                col1, col2 = st.columns([5, 2])
                with col1:
                    calc_start_date = st.date_input(
                        "Start Date",
                        value=predefined.calc_start_datetime.date()
                        if predefined.calc_start_datetime
                        else utc_now_dt,
                        key="calc_start_date_input_key",
                    )
                with col2:
                    calc_start_time = st.time_input(
                        "Start Time",
                        value=predefined.calc_start_datetime.time()
                        if predefined.calc_start_datetime
                        else "00:00",
                        key="calc_start_time_input_key",
                    )
                col1, col2 = st.columns([5, 2])
                with col1:
                    calc_end_date = st.date_input(
                        "End Date",
                        value=predefined.calc_end_datetime.date()
                        if predefined.calc_end_datetime
                        else utc_now_dt + timedelta(days=14),
                        key="calc_end_date_input_key",
                    )
                with col2:
                    calc_end_time = st.time_input(
                        "End Time",
                        value=predefined.calc_end_datetime.time()
                        if predefined.calc_end_datetime
                        else "00:00",
                        key="calc_end_time_input_key",
                    )

            # Additional settings
            st.subheader("Additional Settings")
            exposure_event = st.text_input(
                "Exposure Event", value=predefined.exposure_event, key="exposure_event_input_key"
            )
            filters = st.text_area(
                "Filters (one per line)",
                value="\n".join(predefined.filters) if predefined.filters else "",
                key="filters_input_key",
            )
            audience_tables = st.text_area(
                "Audience Tables (one per line)",
                value="\n".join(predefined.audience_tables) if predefined.audience_tables else "",
                key="audience_tables_input_key",
            )
            custom_test_ids_query = st.text_area(
                "Custom Test Ids Query",
                value=predefined.custom_test_ids_query,
                key="custom_test_ids_input_key",
                help="Enter a read-only SQL query (SELECT, WITH, VALUES). "
                "Supports JOINs, CTEs, subqueries, and complex aggregations. "
                "Modifying operations (INSERT, UPDATE, DELETE) are not allowed.",
            )

            metric_tags: list[str] | None = st.multiselect(
                "Metric Tags",
                options=predefined.metric_tags,
                default=predefined.metric_tags,
                key="metric_tags_input_key",
            )
            metric_groups: list[str] | None = st.multiselect(
                "Metric Groups", options=predefined.metric_groups, default=predefined.metric_groups
            )

            submitted = st.form_submit_button("Submit Observation Form")

            if (
                not name
                or not db_experiment_name
                or not split_id
                or name == ""
                or db_experiment_name == " "
                or split_id == ""
            ):
                if submitted:
                    st.toast(
                        "Please fill all required text fields: Name, DB Experiment Name, and Split ID.",
                        icon="⚠️",
                    )
                return None

            if (
                exposure_start_date is None
                or exposure_start_time is None
                or exposure_end_date is None
                or exposure_end_time is None
                or calc_start_date is None
                or calc_start_time is None
                or calc_end_date is None
                or calc_end_time is None
            ):
                if submitted:
                    st.toast("All date and time fields must be filled.", icon="⚠️")
                return None

            for field in [
                db_experiment_name,
                split_id,
                exposure_event,
                filters,
                audience_tables,
            ]:
                try:
                    ValidationUtils.check_for_sql_injection(field)
                except Exception as e:
                    st.toast(f"Invalid field: {field} 'Potential SQL injection detected'", icon="⚠️")
                    st.warning(e)
                    return None

            # Validate custom test IDs query separately with SQL query validation
            if custom_test_ids_query and custom_test_ids_query.strip():
                try:
                    ValidationUtils.validate_sql_query(custom_test_ids_query)
                except Exception as e:
                    st.toast(f"Invalid SQL query: {str(e)}", icon="⚠️")
                    st.warning(f"Custom Test IDs Query validation failed: {e}")
                    return None

            return cls(
                observation=ObservationFormData(
                    experiment_id=experiment_id,
                    name=name,
                    db_experiment_name=db_experiment_name,
                    split_id=split_id,
                    calculation_scenario=calculation_scenario,
                    exposure_start_datetime=datetime.combine(exposure_start_date, exposure_start_time),
                    exposure_end_datetime=datetime.combine(exposure_end_date, exposure_end_time),
                    calc_start_datetime=datetime.combine(calc_start_date, calc_start_time),
                    calc_end_datetime=datetime.combine(calc_end_date, calc_end_time),
                    exposure_event=exposure_event,
                    audience_tables=[t.strip() for t in audience_tables.split("\n") if t.strip()],
                    filters=[f.strip() for f in filters.split("\n") if f.strip()],
                    custom_test_ids_query=custom_test_ids_query,
                    metric_tags=metric_tags,
                    metric_groups=metric_groups,
                ),
                observation_id=int(predefined.id) if predefined.id else None,
                submitted=submitted,
            )


@dataclass
class ObservationSelectBox:
    """Dataclass to hold the selected observation from a select box."""

    id: int
    experiment_id: int
    name: str

    @classmethod
    def render(
        cls, predefined_obs_id: int | None, selected_exp_id: int | None
    ) -> ObservationSelectBox | None:
        """Render a select box for observations.

        The select box can be filtered by a selected experiment ID.

        Args:
            predefined_obs_id (int | None): The ID of a predefined observation to select.
            selected_exp_id (int | None): The ID of the experiment to filter
                observations by.

        Returns:
            ObservationSelectBox | None: An instance of the class with selected
                observation details, or None if no observation is selected.
        """
        obs_options = get_observations_for_filter() or []
        if selected_exp_id:
            obs_options = list(filter(lambda r: r[1] == selected_exp_id, obs_options))  # type: ignore[arg-type]
        filter_: filter = filter(lambda r: r[1][0] == predefined_obs_id, enumerate(obs_options))  # type: ignore[arg-type]
        generator = map(lambda r: r[0], filter_)
        select_box_index: int | None = next(generator, None)
        selected: Row | None = st.selectbox(
            label="Select observation",
            options=obs_options,
            format_func=lambda x: x[2],
            index=select_box_index,
            key="observations_select_box",
        )
        if selected is None:
            return None
        return cls(id=selected[0], experiment_id=selected[1], name=selected[2])


class ObservationSelectBoxAfterExp:
    """Renders an experiment select box followed by an observation select box."""

    @staticmethod
    def render(url_params: URLParams) -> ObservationSelectBox | None:
        """Render experiment and observation select boxes.

        The observation select box is filtered based on the selected experiment.

        Args:
            url_params (URLParams): URL parameters which may contain experiment and
                observation IDs.

        Returns:
            ObservationSelectBox | None: An instance of ObservationSelectBox with the
                selected observation, or None.
        """
        col1, col2 = st.columns(2)
        with col1:
            exp_select_box = ExperimentSelectBox.render(url_params.experiment_id)
            selected_exp_id = exp_select_box.id if exp_select_box else None
        with col2:
            obs_select_box = ObservationSelectBox.render(url_params.observation_id, selected_exp_id)
        return obs_select_box
