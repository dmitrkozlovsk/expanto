"""This module contains Streamlit UI elements for the results page."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd  # type: ignore
import streamlit as st

from src.services.analytics.calculators import SignificanceCalculator
from src.ui.resources import load_metrics_handler
from src.ui.results.inputs import ExperimentGroupsFilters, MetricsFilters, PValueThresholdFilter
from src.ui.results.styler import SignificanceTableStyler, StColumnConfig

if TYPE_CHECKING:
    from pandas.io.formats.style import Styler  # type: ignore
    from streamlit.elements.lib.column_types import ColumnConfig

    from src.services.metric_register import ExperimentMetricDefinition

from functools import partial

TABLE_ROW_PXLS = 31  # pixel height of a table row
INDENT_TABLE_PXLS = 33  # indentation for a table
MAX_TABLE_HEIGHT_PXLS = 560  # Maximum pixel height for a table to prevent it from exceeding the viewport


# --------------------------------HELP FUNC-----------------------------------
def is_metric_in_metric_filters(
    metric_name: str, flat_metrics: dict[str, Any], metric_filters: MetricsFilters
) -> bool:
    """Checks if a metric is in the metric filters."""
    metric = flat_metrics.get(metric_name)
    if not metric:
        return True

    if not (metric_filters.groups or metric_filters.tags):
        return True

    in_group = metric_filters.groups and metric.group_name in metric_filters.groups
    has_tag = metric_filters.tags and bool(set(metric.tags) & set(metric_filters.tags))

    return bool(in_group or has_tag)


def define_metric_group(metric_name: str, flat_metrics: dict[str, ExperimentMetricDefinition]) -> str:
    """Defines the group name for a metric."""
    exp_definition = flat_metrics.get(metric_name)
    return exp_definition.group_name or "Other" if exp_definition else "Other"


def put_selected_metric_in_state(styled_df: Styler, dataframe_key: str) -> None:
    """Puts the selected metric in the Streamlit session state."""
    rows = st.session_state.get(dataframe_key, {}).get("selection", {}).get("rows", [])
    if rows:
        flat_metrics = load_metrics_handler().flat
        index = rows[0]
        metric_id = styled_df.data.iloc[index].name
        metric_name = styled_df.data.loc[metric_id]["metric_name"]
        metric_info = flat_metrics.get(metric_name)
        st.session_state["metric_info_to_show"] = metric_info


# -------------------------------- Elements -----------------------------------


class RunJobDialog:
    """Renders a dialog to choose how to run a calculation job."""

    @staticmethod
    def render(observation_id: int) -> None:
        """Renders the 'Run Job' dialog.

        This dialog presents the user with two options for running a calculation:
        - "Run and wait": Runs the calculation in the foreground.
        - "Run in background": Runs the calculation in the background.

        Based on the user's choice, it sets the `obs_to_be_calculated`
        session state variable and triggers a rerun of the Streamlit app.

        Args:
            observation_id: The ID of the observation to be calculated.
        """

        @st.dialog("Choose option and confirm")
        def dialog(observation_id_: int) -> None:
            st.markdown("""
        - Run and wait: Stay on this page while calculation completes
        - Run in background: Continue using the app while calculation runs""")
            col1, col2 = st.columns(2)
            with col1:
                run_and_wait_clicked = st.button("Run and wait", width="stretch")
                if run_and_wait_clicked:
                    st.session_state["obs_to_be_calculated"] = {
                        "observation_id": observation_id_,
                        "calc_type": "foreground",
                    }
                    st.rerun()
            with col2:
                run_in_back_clicked = st.button("Run in background", width="stretch")
                if run_in_back_clicked:
                    st.session_state["obs_to_be_calculated"] = {
                        "observation_id": observation_id_,
                        "calc_type": "background",
                    }
                    st.rerun()

        dialog(observation_id)


class RunJobButton:
    """Renders a button to initiate a calculation job."""

    @staticmethod
    def render(observation_id: int | None) -> None:
        """Renders the 'Run calculation job' button.

        If an observation ID is provided, this method displays a button that,
        when clicked, opens the `RunJobDialog`. If no observation ID is
        provided, it displays a warning message.

        Args:
            observation_id: The ID of the selected observation.
        """
        if not observation_id:
            st.warning("You need to select an observation")
        else:
            run_calculation_button = st.button("Run new job", type="primary", width="stretch")
            if run_calculation_button:
                st.session_state["show_dialog"] = True
                RunJobDialog.render(observation_id)


class MetricInfoDialog:
    """Renders a dialog with detailed information about a selected metric."""

    @staticmethod
    @st.dialog("Metric Detailed Information", width="large")
    def render(metric: ExperimentMetricDefinition) -> None:
        st.write(metric)


class SampleRatioMismatchCheckExpander:
    """Renders an expander for the Sample Ratio Mismatch (SRM) check."""

    @staticmethod
    @st.fragment
    def render(observation_cnt: dict[str, Any]) -> None:
        """Renders the content of the SRM check expander.

        Args:
            observation_cnt: A dictionary with observation counts per group.
        """
        with st.expander("SRM Check", expanded=False):
            st.markdown("place for srm")


class ResultsDataframes:
    """Renders the results table for metrics."""

    @staticmethod
    def _one_table_view(
        df: pd.DataFrame,
        styler: SignificanceTableStyler,
        column_config: dict[str, ColumnConfig | None],
        compared_group: str,
        metric_group_name: str = "",
    ):
        """Renders a single results table.

        Args:
            df: DataFrame with results to display.
            styler: An instance of SignificanceTableStyler for styling the table.
            column_config: A dictionary with column configurations.
            compared_group: The name of the compared group.
            metric_group_name: The name of the metric group (used for the dataframe key).
        """
        styled_df = styler.apply_styles(df)

        column_order = [
            "metric_display_name",
            "metric_value_control",
            "metric_value_compared",
            "diff_ratio",
            "p_value",
        ]
        table_rows_cnt = styled_df.data.shape[0]
        table_height = min(TABLE_ROW_PXLS * table_rows_cnt + INDENT_TABLE_PXLS, MAX_TABLE_HEIGHT_PXLS)
        dataframe_key = f"styled_df_{compared_group}_{metric_group_name}"

        st.dataframe(
            data=styled_df,
            width="stretch",
            height=table_height,
            key=dataframe_key,
            column_config=column_config,
            column_order=column_order,
            hide_index=True,
            on_select=partial(put_selected_metric_in_state, styled_df, dataframe_key),
            selection_mode="single-row",
        )

    @staticmethod
    def _grouped_view(
        df: pd.DataFrame,
        styler: SignificanceTableStyler,
        column_config: dict[str, ColumnConfig | None],
        compared_group: str,
    ):
        """Renders the results table, grouping metrics by their metric group.

        This view creates a separate table for each metric group.

        Args:
            df: DataFrame with results to display.
            styler: An instance of SignificanceTableStyler for styling the table.
            column_config: A dictionary with column configurations.
            compared_group: The name of the compared group.
        """
        metric_groups = list(df.metric_group.unique())
        for metric_group_name in metric_groups:
            group_filter = df.metric_group == metric_group_name
            metric_group_df = df[group_filter]
            if metric_group_df.empty:
                continue
            st.caption(f"###### {metric_group_name}")
            ResultsDataframes._one_table_view(
                metric_group_df, styler, column_config, compared_group, metric_group_name
            )

    @staticmethod
    @st.fragment
    def render(
        compare_group: str,
        df: pd.DataFrame,
        styler: SignificanceTableStyler,
        grouped_view_flg: bool = False,
    ) -> None:
        """Renders the main results dataframes.

        This method can render the results in a single table or in multiple
        tables grouped by metric group, depending on the `grouped_view_flg`.
        It also handles the display of the MetricInfoDialog when a row is selected.

        Args:
            compare_group: The name of the compared group.
            df: The DataFrame with the results to display.
            styler: An instance of SignificanceTableStyler for styling the table.
            grouped_view_flg: If True, renders the tables in a grouped view.
        """
        column_config = {col: StColumnConfig.get(col) for col in df.columns}

        if grouped_view_flg:
            ResultsDataframes._grouped_view(df, styler, column_config, compare_group)
        else:
            ResultsDataframes._one_table_view(df, styler, column_config, compare_group)

        if metric := st.session_state.get("metric_info_to_show"):
            MetricInfoDialog.render(metric)
            st.session_state["metric_info_to_show"] = None


class TablesLayout:
    """Renders the results tables for experiment groups."""

    @staticmethod
    @st.fragment
    def render(
        prepared_tables: list[PreparedTable],
        observations_cnt: dict[str, Any],
        styler: SignificanceTableStyler,
    ) -> None:
        """Renders the results tables for experiment groups."""

        tab_names = [f"Test Group: {table.compared_group}" for table in prepared_tables]

        for index, tab in enumerate(st.tabs(tab_names)):
            with tab:
                prepared_table = prepared_tables[index]

                compared_group = prepared_table.compared_group
                control_group = prepared_table.control_group

                # define header for tab
                col1, col2 = st.columns([30, 15], vertical_alignment="center")
                with col1:
                    header_str = (
                        f"#### `{control_group} {observations_cnt.get(control_group)}` "
                        f"vs. `{compared_group} {observations_cnt.get(compared_group)}`"
                    )
                    st.markdown(header_str)
                with col2:
                    container = st.container(vertical_alignment="center", horizontal_alignment="right")
                    with container:
                        grouped_view_flg = st.toggle("Grouped View", key=f"grouped_view_toggle_{index}")

                ResultsDataframes.render(compared_group, prepared_table.df, styler, grouped_view_flg)

        return None


@dataclass
class SettingsColumnLayout:
    """Renders the settings column for the results page."""

    p_value_threshold: float | None
    metric_filters: MetricsFilters
    group_filters: ExperimentGroupsFilters
    observations_cnt: dict[str, Any]

    @classmethod
    def render(cls, precomputes: pd.DataFrame | None) -> SettingsColumnLayout | None:
        """Renders the settings column and returns the selected settings.

        This method displays filters for experiment groups, p-value threshold,
        and metrics. It also shows a warning or error if precomputes are
        missing or empty.

        Args:
            precomputes: A DataFrame containing the precomputed metrics.

        Returns:
            An instance of SettingsColumnLayout with the selected settings,
            or None if the settings are incomplete or precomputes are invalid.
        """
        with st.container(vertical_alignment="bottom", horizontal_alignment="center"):
            st.markdown("#### Settings")
        if isinstance(precomputes, pd.DataFrame) and precomputes.empty:
            st.warning("Precomputes are empty")
            return None
        elif not isinstance(precomputes, pd.DataFrame) and not precomputes:
            st.error("Precomputes for selected job not found.")
            return None

        group_filters = ExperimentGroupsFilters.render(precomputes.reset_index())

        if not group_filters.compared_groups:
            st.info("Please select at least one compared group")
            return None
        if not group_filters.control_group:
            st.info("Please select a control group")
            return None

        p_value_threshold_filter = PValueThresholdFilter.render().threshold
        metric_filters = MetricsFilters.render()

        observations_cnt = precomputes.groupby("group_name")["observation_cnt"].unique().to_dict()
        SampleRatioMismatchCheckExpander.render(observations_cnt)

        return cls(
            p_value_threshold=p_value_threshold_filter,
            metric_filters=metric_filters,
            group_filters=group_filters,
            observations_cnt=observations_cnt,
        )


@dataclass
class PreparedTable:
    """Prepares a table for display."""

    df: pd.DataFrame
    control_group: str
    compared_group: str


class ResultPageLayout:
    """Renders the results page layout."""

    @staticmethod
    def calc_and_prepare_results_table(
        precomputes: pd.DataFrame, settings: SettingsColumnLayout
    ) -> list[PreparedTable] | None:
        """Filters precomputes, calculates significance, and prepares result tables.

        This method takes the precomputed data and user-defined settings,
        calculates statistical significance, filters the metrics based on
        the settings, and prepares a list of tables for display, one for
        each compared group.

        Args:
            precomputes: The DataFrame with precomputed metrics.
            settings: An instance of SettingsColumnLayout with user-defined settings.

        Returns:
            A list of PreparedTable objects, or None if significance calculation fails.
        """
        assert settings.group_filters.control_group is not None
        assert settings.group_filters.compared_groups is not None
        try:
            calculator = SignificanceCalculator(
                precomputed_metrics_df=precomputes, control_group=settings.group_filters.control_group
            )
            raw_significance_table = calculator.get_metrics_significance_df()
        except Exception as e:
            error_message = f"Could not get significance table: {e}"
            st.error(error_message)
            return None

        # filling missing display names
        unknown_names_filter = raw_significance_table.loc[:, ("", "metric_display_name")].isna()
        raw_significance_table.loc[unknown_names_filter, ("", "metric_display_name")] = (
            raw_significance_table.loc[unknown_names_filter, ("", "metric_name")]
        )

        flat_metrics = load_metrics_handler().flat

        # get metric's group name
        raw_significance_table.loc[:, ("", "metric_group")] = raw_significance_table.loc[
            :, ("", "metric_name")
        ].apply(define_metric_group, flat_metrics=flat_metrics)

        metric_filter = raw_significance_table.loc[:, ("", "metric_name")].apply(
            is_metric_in_metric_filters,
            flat_metrics=flat_metrics,
            metric_filters=settings.metric_filters,
        )

        filtered_significance_table = raw_significance_table[metric_filter]

        prepared_tables: list[PreparedTable] = []

        for compared_group in settings.group_filters.compared_groups:
            compared_group_columns = [
                col
                for col in filtered_significance_table.columns
                if col[0] == compared_group
                or col[1] in ("metric_display_name", "metric_type", "metric_name", "metric_group")
            ]
            filtered_compared_groups_df = filtered_significance_table[compared_group_columns]
            filtered_compared_groups_df.columns = [col[1] for col in filtered_compared_groups_df.columns]
            prepared_table = PreparedTable(
                df=filtered_compared_groups_df,
                control_group=settings.group_filters.control_group,
                compared_group=compared_group,
            )
            prepared_tables.append(prepared_table)

        return prepared_tables

    @staticmethod
    def render(precomputes: pd.DataFrame | None) -> list[PreparedTable] | None:
        """Renders the main layout of the results page.

        This method sets up the two-column layout, with settings on the left
        and results tables on the right.

        Args:
            precomputes: A DataFrame containing the precomputed metrics.

        Returns:
            A list of PreparedTable objects if the tables are successfully
            generated, otherwise None.
        """
        layout_col1, layout_col2 = st.columns(
            [12, 30],
            vertical_alignment="top",
        )

        with layout_col1:  # left col with settings
            settings = SettingsColumnLayout.render(precomputes)
        if settings is None:
            return None

        with layout_col2:  # right column with metric results
            prepared_tables = ResultPageLayout.calc_and_prepare_results_table(precomputes, settings)
            if not prepared_tables:
                st.warning("No metrics to display")
                return None
            styler = SignificanceTableStyler(settings.p_value_threshold)
            TablesLayout.render(prepared_tables, observations_cnt=settings.observations_cnt, styler=styler)
        return prepared_tables
