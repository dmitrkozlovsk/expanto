"""UI elements for the planner page in Streamlit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import plotly.graph_objects as go  # type: ignore
import streamlit as st
from streamlit.elements.plotly_chart import PlotlyState

from src.domain.results import MetricResult
from src.services.analytics.calculators import SampleSizeCalculator
from src.ui.data_loaders import get_experiment_by_id
from src.ui.planner.inputs import EffectRangeSlider, ErrorsSelectors, GetPrecomputesForm

if TYPE_CHECKING:
    from src.services.analytics.calculators import SampleSizeCalculation


class ExperimentDetailsExpander:
    """Renders a Streamlit expander to display experiment details."""

    @classmethod
    def render(cls, experiment_id_: int | None) -> None:
        """Renders a Streamlit expander with details of a given experiment.

        If the experiment ID is provided, it fetches the experiment data and
        displays its status, dates, description, and hypotheses.

        Args:
            experiment_id_: The ID of the experiment to display details for.
        """
        if experiment_id_ is None:
            return
        experiment = get_experiment_by_id(experiment_id_)
        if experiment is None:
            return
        with st.expander("Experiment Details", expanded=False):
            st.markdown(f"""
            **Status:** {experiment.status}  
            **Start Date:** {experiment.start_datetime}  
            **End Date:** {experiment.end_datetime if experiment.end_datetime else "Not set"}  
            
            **Description:**  
            {experiment.description}
            
            **Hypotheses:**  
            {experiment.hypotheses}
            """)


class EffectSizeChart:
    """Represents a Plotly chart for visualizing effect size versus sample size.

    This class encapsulates a Plotly figure object for displaying sample size
    calculation results.

    Attributes:
        fig: A Plotly figure object.
    """

    @staticmethod
    def create(sample_sizes_calc_results: list[SampleSizeCalculation]) -> go.Figure:
        """Creates a Plotly chart for effect size vs. sample size.

        This method generates a line chart showing the relationship between
        detectable effect size, sample size, and observation days for different metrics.

        Args:
            sample_sizes_calc_results: A list of `SampleSizeCalculation` objects,
                each containing the data for one metric trace on the chart.

        Returns:
            An instance of `EffectSizeChart` containing the configured Plotly figure.
        """
        fig = go.Figure()
        for calc in sample_sizes_calc_results:
            fig.add_trace(
                go.Scatter(
                    x=calc.sample_sizes,
                    y=calc.effect_sizes,
                    mode="lines+markers",
                    name=calc.metric_name,
                    customdata=calc.sample_sizes / calc.observations_per_day,
                    hovertemplate=(
                        f"<b>{calc.metric_name}</b><br>"
                        "Sample size: %{x:.0f}<br>"
                        "Δ: %{y:.1%}<br>"
                        "Days: %{customdata:.1f}<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            title="Detectable effect vs. Sample Size",
            xaxis_title="Sample size per group (log scale)",
            yaxis_title="Relative effect",
            xaxis_type="log",
            template="plotly_white",
            legend_title_text="Metric",
        )
        fig.update_yaxes(tickformat=".0%")
        return fig


@dataclass
class SizeChartSelection:
    """Represents selection information from the effect size chart.

    Attributes:
        info: Formatted information about the selected design point.
    """

    info: str | None

    @classmethod
    def render(
        cls,
        calculations: list[SampleSizeCalculation],
        selection: PlotlyState,
    ) -> SizeChartSelection:
        """Generate information about selected sample size design.

        Args:
            calculations: List of sample size calculations.
            selection: Selection data from the plot.

        Returns:
            Formatted string with selection info or None if no selection.
        """
        if selection and selection.get("selection", {}).get("points"):
            point = selection["selection"]["points"][0]
            metric_name = calculations[point["curve_number"]].metric_name
            sample_size = int(point["x"])
            effect_size = point["y"]
            days = point["customdata"]

            return cls(
                info=f"### Selected Design\n"
                f"To detect a **{effect_size:.1%}** effect for metric **{metric_name}**, "
                f"you'll need:\n"
                f"- **{sample_size:,}** observations per group\n"
                f"- **{days:.1f}** days of experiment"
            )
        return cls(info=None)


@dataclass
class EffectSizePanel:
    """Panel component for effect size analysis and visualization.

    Attributes:
        info: Information about the selected effect size design.
    """

    info: str | None

    @classmethod
    def render(
        cls,
        metric_results: list[MetricResult],
        precomputes_form: GetPrecomputesForm,
    ) -> EffectSizePanel:
        """Render the effect size analysis panel.

        Args:
            metric_results: List of metric calculation results.
            precomputes_form: Form containing experiment parameters.

        Returns:
            EffectSizePanel instance with selection information.
        """
        col1, col2 = st.columns([2, 1])
        with col1:
            error_selectors = ErrorsSelectors.render()
        with col2:
            effect_range_slider = EffectRangeSlider.render()

        try:
            calculations = SampleSizeCalculator.calculate(
                metric_results=metric_results,
                effect_sizes=effect_range_slider.selected_effect_range,
                reference_period_days=precomputes_form.exposure_delta_days,
                alpha=error_selectors.selected_alpha,
                beta=error_selectors.selected_beta,
            )
        except Exception as error:
            error_message = str(error)
            st.error(error_message)
            return cls(info=None)

        fig = EffectSizeChart.create(calculations)
        event = st.plotly_chart(fig, key="effect_size_chart", on_select="rerun", selection_mode="points")
        placeholder = st.empty()
        if event:
            info = SizeChartSelection.render(calculations, event).info
            if info:
                placeholder.markdown(info)
            else:
                placeholder.markdown("&nbsp;")
            return cls(info=info)
        else:
            placeholder.markdown("&nbsp;")
        return cls(info=None)
