"""Experiment planner page."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

import streamlit as st

from src.domain.results import MetricResult
from src.ui.common import URLParams, put_return_in_app_ctx
from src.ui.data_loaders import get_precomputes_for_planning
from src.ui.experiments.inputs import ExperimentSelectBox
from src.ui.layout import AppLayout
from src.ui.planner.elements import EffectSizePanel, ExperimentDetailsExpander
from src.ui.planner.inputs import GetPrecomputesForm
from src.ui.state import AppContextManager

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

    from src.ui.planner.inputs import DummyObsSelectedParams


class ExperimentPlannerPage:
    """Main page component for experiment planning and analysis."""

    @classmethod
    @st.fragment
    def render(cls, url_params_: URLParams) -> None:
        """Render the experiment planner page.

        Args:
            url_params_: URL parameters containing experiment ID and other settings.
        """
        AppContextManager.set_page_name("planner")
        AppContextManager.set_page_mode("")
        with AppLayout.chat():
            cls._render_content(url_params_)

    @classmethod
    @put_return_in_app_ctx
    def _render_content(cls, url_params_: URLParams) -> dict | None:
        """Render the main content of the planner page.

        Args:
            url_params_: URL parameters containing experiment ID and other settings.

        Returns:
            Dictionary with experiment ID and plan info if successful, None otherwise.
        """
        # -------------------------------- Initialize State --------------------------------
        if not st.session_state.get("job_results_cache", None):
            st.session_state["job_results_cache"] = {}

        # ------------------------------ Error Messages Field ------------------------------
        status_field = st.empty()

        # ------------------------------------ Layouts ------------------------------------
        layout_col1, layout_col2 = st.columns([1, 3])
        with layout_col1:
            experiments_select_box = ExperimentSelectBox.render(url_params_.experiment_id)
            selected_experiment_id: int | None = (
                experiments_select_box.id if experiments_select_box else None
            )
            cls._render_inputs_layout(status_field, selected_experiment_id)

        with layout_col2:
            return cls._render_graph_layout(selected_experiment_id)

    @classmethod
    def _render_inputs_layout(
        cls, status_field: DeltaGenerator, selected_experiment_id: int | None
    ) -> None:
        """Render the inputs layout for experiment planning.

        Args:
            status_field: Streamlit component for displaying status messages.
            selected_experiment_id: ID of the selected experiment.
        """
        ExperimentDetailsExpander.render(selected_experiment_id)
        precomputes_form = GetPrecomputesForm.render(selected_experiment_id)
        if precomputes_form.is_submitted and precomputes_form.selected_obs_params:
            metric_names = precomputes_form.selected_obs_params.experiment_metric_names
            if not metric_names:
                st.error("Please select at least one metric")
                return None
            with st.spinner("Metrics is being calculated..."):
                job_result = cls._handle_planning_job(status_field, precomputes_form.selected_obs_params)
                st.session_state.job_results_cache[selected_experiment_id] = job_result
                st.session_state.precomputes_form = precomputes_form

    @staticmethod
    def _handle_planning_job(status_field: DeltaGenerator, params: DummyObsSelectedParams) -> dict | None:
        """Handle the planning job execution.

        Args:
            status_field: Streamlit component for displaying status messages.
            params: Parameters for the planning calculation.

        Returns:
            Job result dictionary if successful, None otherwise.
        """
        try:
            job_result = get_precomputes_for_planning(**asdict(params))
        except Exception as e:
            status_field.error(f"❌ Failed to run calculation: {e}")
            return None
        if not job_result.get("success", False) and (error_message := job_result.get("error_message")):
            error_message = f"❌ Job calculation ended with error: {error_message}"
            status_field.error(error_message)
        elif job_result.get("success", False) and (job_id := job_result.get("job_id", None)):
            st.toast(f"✅️ Job {job_id} has finished successfully!")
            return job_result
        return None

    @staticmethod
    def _render_graph_layout(selected_experiment_id: int | None) -> dict | None:
        """Render the graph layout for experiment results.

        Args:
            selected_experiment_id: ID of the selected experiment.

        Returns:
            Dictionary with experiment ID and plan info if results are available, None otherwise.
        """
        if not (precomputes_form := st.session_state.get("precomputes_form")):
            st.info("Fill in the fields and start observations")
            return None

        if not (job_result := st.session_state.job_results_cache.get(selected_experiment_id, None)):
            st.info("Fill in the fields and start observations")
            return None

        if metric_results_dict := job_result.get("metric_results"):
            metric_results = [MetricResult(**mr) for mr in metric_results_dict]
        else:
            metric_results = []

        if metric_results:
            selection_info = EffectSizePanel.render(metric_results, precomputes_form).info
            return {"experiment_id": selected_experiment_id, "plan_info": selection_info}

        else:
            st.warning("Metrics Precomputes are empty")
            return None


url_params = URLParams.parse()
ExperimentPlannerPage.render(url_params)
