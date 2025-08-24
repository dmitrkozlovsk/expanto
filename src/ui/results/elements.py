"""This module contains Streamlit UI elements for the results page."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import streamlit as st

from src.ui.resources import load_metrics_handler
from src.ui.results.inputs import ExperimentGroupsFilters
from src.ui.results.styler import SignificanceTableStyler, StColumnConfig

if TYPE_CHECKING:
    import pandas as pd  # type: ignore
    from pandas.io.formats.style import Styler  # type: ignore

    from src.ui.results.inputs import MetricsFilters


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
    def render(rows: list[int], styled_significance_table_compared_group: Styler) -> None:
        """Renders the metric information dialog.

        This dialog displays detailed information for a metric selected from
        the results table.

        Args:
            rows: A list of selected row indices from the dataframe. It is
                expected to contain a single index.
            styled_significance_table_compared_group: The styled Pandas Styler
                object containing the significance data for the compared group.
        """
        flat_metrics = load_metrics_handler().flat
        index = rows[0]
        metric_id = styled_significance_table_compared_group.data.iloc[index].name
        metric_name = styled_significance_table_compared_group.data.loc[metric_id]["metric_name"]
        metric_info = flat_metrics.get(metric_name)
        st.write(metric_info)


@dataclass
class ResultsTables:
    """Renders the results tables for experiment groups."""

    groups_results: Any

    @classmethod
    def render(
        cls,
        significance_df: pd.DataFrame,
        group_filters: ExperimentGroupsFilters,
        selected_pvalue_threshold: float | None,
        metric_filters: MetricsFilters,
        observation_cnt: dict[str, Any],
    ) -> ResultsTables | None:
        """Renders the significance results in separate tabs for each compared group.

        For each group being compared against the control group, this method
        creates a tab. Inside each tab, it displays a formatted and styled
        table of significance metrics. It handles filtering of columns and
        enables row selection to view detailed metric information via
        `MetricInfoDialog`.

        Args:
            significance_df: DataFrame containing the full significance results.
            group_filters: An object containing the selected control and
                compared groups.
            result_table_column_filter: An object containing the selected
                columns to display in the table.
            selected_pvalue_threshold: The p-value threshold for styling
                significant results.
        """
        unknown_names_filter = significance_df.loc[:, ("", "metric_display_name")].isna()
        significance_df.loc[unknown_names_filter, ("", "metric_display_name")] = significance_df.loc[
            unknown_names_filter, ("", "metric_name")
        ]
        if not group_filters.compared_groups:
            st.info("Please select at least one group")
            return None
        if not group_filters.control_group:
            st.info("Please select a control group")
            return None

        groups_results = []

        tab_names = [f"Test Group: {group}" for group in group_filters.compared_groups]
        flat_metrics = load_metrics_handler().flat

        for index, tab in enumerate(st.tabs(tab_names)):
            with tab:
                compared_group = group_filters.compared_groups[index]

                control_group = group_filters.control_group
                header_str = (
                    f"`{control_group} {observation_cnt.get(control_group)}` "
                    f"vs. `{compared_group} {observation_cnt.get(compared_group)}`"
                )

                tab.header(header_str)
                compared_group_columns = [
                    col
                    for col in significance_df.columns
                    if col[0] == compared_group
                    or col[1] in ("metric_display_name", "metric_type", "metric_name")
                ]
                significance_table_compared_group = significance_df[compared_group_columns]
                significance_table_compared_group.columns = [
                    col[1] for col in significance_table_compared_group.columns
                ]

                def filter_metric(metric_name: str, _flat_metrics: dict[str, Any]) -> bool:
                    metric = _flat_metrics.get(metric_name)
                    if not metric:
                        return True

                    if not (metric_filters.groups or metric_filters.tags):
                        return True

                    in_group = metric_filters.groups and metric.group_name in metric_filters.groups
                    has_tag = metric_filters.tags and bool(set(metric.tags) & set(metric_filters.tags))

                    return bool(in_group or has_tag)

                filter_ = significance_table_compared_group.metric_name.apply(
                    filter_metric, _flat_metrics=flat_metrics
                )
                filtered_significance_table_compared_group = significance_table_compared_group[filter_]

                column_config = {
                    col: StColumnConfig.get(col)
                    for col in filtered_significance_table_compared_group.columns
                }

                styler = SignificanceTableStyler(
                    p_value_threshold=selected_pvalue_threshold, selected_metric_groups=None
                )

                styled_significance_table_compared_group = styler.apply_styles(
                    filtered_significance_table_compared_group
                )

                column_order = [
                    "metric_display_name",
                    "metric_value_control",
                    "metric_value_compared",
                    "diff_ratio",
                    "p_value",
                ]

                table_height = min(
                    30 * styled_significance_table_compared_group.data.shape[0] + 36, 560
                )  # todo fix magic numbers
                selection = st.dataframe(
                    data=styled_significance_table_compared_group,
                    use_container_width=True,
                    height=table_height,
                    key=f"styled_significance_table_compared_group_table_{index}",
                    column_config=column_config,
                    column_order=column_order,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                )
                rows = selection.selection["rows"]  # type: ignore[attr-defined]
                if len(rows) > 0:
                    MetricInfoDialog.render(rows, styled_significance_table_compared_group)

                groups_results.append(
                    {
                        "table": styled_significance_table_compared_group.data.round(5).to_csv(index=False),
                        "control_group": group_filters.control_group,
                        "compared_group": compared_group,
                    }
                )

        return cls(groups_results=groups_results)
