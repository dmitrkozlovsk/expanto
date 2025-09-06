"""This module defines the layout and functionality of the results page
in the Streamlit application.

It orchestrates the display of experiment results, including job selection,
group filtering, significance calculations, and the presentation of results in tables.
"""

from concurrent.futures import Future
from typing import Any

import pandas as pd  # type: ignore
import streamlit as st

from src.services.analytics.calculators import SignificanceCalculator
from src.ui.actions import run_observation_calculation
from src.ui.common import URLParams, put_return_in_app_ctx
from src.ui.data_loaders import (
    get_observation_by_id,
    get_precomputes_by_job_id,
)
from src.ui.layout import AppLayout
from src.ui.results.elements import ResultsTables, RunJobButton
from src.ui.results.inputs import (
    ExperimentGroupsFilters,
    MetricsFilters,
    PValueThresholdFilter,
    SelectJobFilters,
)
from src.ui.state import AppContextManager


class ResultsPage:
    """Encapsulates the logic and layout for the results page."""

    @classmethod
    @st.fragment
    def render(cls, url_params: URLParams) -> None:
        """Renders the entire results page with chat layout.

        This method handles the complete workflow of the results page, from user inputs
        for job and group selection to the final display of significance test results.
        It uses URL parameters to allow for direct links to specific results.

        Args:
            url_params: An instance of URLParams containing query parameters from the URL.
        """
        AppContextManager.set_page_name("results")
        AppContextManager.set_page_mode("")
        with AppLayout.chat():
            cls._render_content(url_params)

    @classmethod
    @put_return_in_app_ctx
    def _render_content(cls, url_params: URLParams) -> dict[str, Any] | None:
        """Renders the main content of the results page."""

        cls._handle_obs_to_be_calculated()

        layout_col1, layout_col2 = st.columns([12, 30])

        with layout_col1:
            job_filters = SelectJobFilters.render(url_params.job_id, url_params.observation_id)
            if job_filters.selected_observation_id:
                RunJobButton.render(job_filters.selected_observation_id)
            st.divider()
            if job_filters.selected_job_id is None:
                st.info("No jobs selected. Please select a job.")
                return None

            metric_precomputes = get_precomputes_by_job_id(job_filters.selected_job_id)
            if isinstance(metric_precomputes, pd.DataFrame) and metric_precomputes.empty:
                st.warning("Precomputes are empty")
                return None
            elif not isinstance(metric_precomputes, pd.DataFrame) and not metric_precomputes:
                st.error("Precomputes for selected job not found.")
                return None
            group_filters = ExperimentGroupsFilters.render(metric_precomputes.reset_index())

            observation_cnt = metric_precomputes.groupby("group_name")["observation_cnt"].unique().to_dict()

            if not group_filters.control_group or not group_filters.compared_groups:
                return None

            try:
                calculator = SignificanceCalculator(
                    precomputed_metrics_df=metric_precomputes, control_group=group_filters.control_group
                )

                raw_significance_table = calculator.get_metrics_significance_df()
            except Exception as e:
                error_message = f"Could not get significance table: {e}"
                st.error(error_message)
                return None

            pvalue_threshold_filter = PValueThresholdFilter.render()
            metric_filters = MetricsFilters.render()

        with layout_col2:
            try:
                result_tables_rendered = ResultsTables.render(
                    raw_significance_table,
                    group_filters,
                    pvalue_threshold_filter.threshold,
                    metric_filters,
                    observation_cnt,
                )
                return cls._unite_result_page_return(job_filters, result_tables_rendered)
            except Exception as e:
                error_message = f"Could not render results table. Error: {e}"
                st.error(error_message)
                return None

    @staticmethod
    def _handle_obs_to_be_calculated() -> None:
        if st.session_state.get("obs_calculation_status_success"):
            job_id = st.query_params.get("job_id", "Unknown")
            st.toast(f"✅️ Job {job_id} has finished successfully!")
            del st.session_state["obs_calculation_status_success"]
            return None

        if "obs_to_be_calculated" not in st.session_state:
            return None

        try:
            job_result = run_observation_calculation(**st.session_state.obs_to_be_calculated)
        except Exception as e:
            job_result = None
            error_message = f"❌ Failed to run calculation: `{e}`"
            st.error(error_message)
        finally:
            del st.session_state["obs_to_be_calculated"]
        if isinstance(job_result, Future):
            st.rerun()
            return None
        if job_result and not job_result.success and job_result.error_message:
            error_message = f"❌ Job calculation ended with error: `{job_result.error_message}`"
            st.error(error_message)
            return None
        if job_result and job_result.success:
            st.query_params["job_id"] = str(job_result.job_id)
            st.session_state["obs_calculation_status_success"] = True
            st.rerun()

        return None

    @staticmethod
    def _unite_result_page_return(
        job_filters: SelectJobFilters, result_table_render: ResultsTables | None
    ) -> dict[str, Any]:
        """Unite page return data."""
        selected: dict[str, Any] = {}
        if job_filters.selected_observation_id:
            observation = get_observation_by_id(job_filters.selected_observation_id)
            selected["experiment_id"] = int(observation.experiment_id) if observation else None
            selected["observation_id"] = int(job_filters.selected_observation_id)

        if job_filters.selected_job_id:
            selected["job_id"] = job_filters.selected_job_id

        if result_table_render and result_table_render.groups_results:
            selected["results"] = result_table_render.groups_results

        return selected


# ---------------------------- Page ----------------------------

parsed_url_params = URLParams.parse()
ResultsPage.render(parsed_url_params)
