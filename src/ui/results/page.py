"""This module defines the layout and functionality of the results page
in the Streamlit application.

It orchestrates the display of experiment results, including job selection,
group filtering, significance calculations, and the presentation of results in tables.
"""

from concurrent.futures import Future
from typing import Any

import streamlit as st

from src.ui.actions import run_observation_calculation
from src.ui.common import URLParams, put_return_in_app_ctx
from src.ui.data_loaders import (
    get_observation_by_id,
    get_precomputes_by_job_id,
)
from src.ui.layout import AppLayout
from src.ui.results.elements import TablesLayout, RunJobButton, ResultPageLayout
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

        #First, trying to run job calculation if needed
        cls._handle_obs_to_be_calculated()

        #Render main filters and Run Button
        top_col1, top_col2 = st.columns([30, 7], vertical_alignment='bottom')
        with top_col1:
            job_filters = SelectJobFilters.render(url_params.job_id, url_params.observation_id)
        with top_col2:
            if job_filters.selected_observation_id:
                RunJobButton.render(job_filters.selected_observation_id)
        if job_filters.selected_job_id is None:
            st.info("No jobs selected. Please select a job.")
            return None

        #Download Precomputes
        metric_precomputes = get_precomputes_by_job_id(job_filters.selected_job_id)

        #Render Main Layout with results
        result_tables_rendered = ResultPageLayout.render(metric_precomputes)

        return cls._unite_result_page_return(job_filters, result_tables_rendered)

    @staticmethod
    def _handle_obs_to_be_calculated() -> None:
        #todo: add full docstring
        """Handle the calculation of observations."""
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
        job_filters: SelectJobFilters, result_table_render: TablesLayout | None
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
