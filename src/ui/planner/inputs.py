"""Defines the user input components for the Experiment Planner page."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import streamlit as st

from src.ui.data_loaders import (
    get_available_exposure_events,
    get_experiment_by_id,
)
from src.ui.observations.inputs import CalculationScenarioSelectBox
from src.ui.resources import load_metrics_handler


@dataclass
class KeyMetricsSelectBox:
    """A select box for choosing key metrics for an experiment.

    Attributes:
        selected_metrics: A list of metric names that have been selected.
    """

    selected_metrics: list[str]

    @classmethod
    def render(
        cls,
        experiment_id: int | None = None,
    ) -> KeyMetricsSelectBox:
        """Renders a multiselect box for key metrics.

        If an experiment_id is provided, it attempts to pre-select metrics
        associated with that experiment.

        Args:
            experiment_id: The optional ID of an experiment to load default metrics from.

        Returns:
            An instance of KeyMetricsSelectBox with the selected metrics.
        """
        # Display experiment key metrics
        experiment = None
        if experiment_id is not None:
            experiment = get_experiment_by_id(experiment_id)
        key_metrics: list[str] = list(experiment.key_metrics) if experiment else []

        metrics = load_metrics_handler()

        metric_options = list(metrics.flat.keys())
        metric_intersection = list(set(key_metrics).intersection(set(metric_options)))
        default_metrics = metric_intersection if len(metric_intersection) > 0 else None
        # Metrics selection
        selected_metrics = st.multiselect(
            "Select metrics to plan (choose at least one metric)",
            options=metric_options,
            default=default_metrics,
        )
        return cls(selected_metrics=selected_metrics)


@dataclass
class TimeInputSection:
    """A section for selecting exposure and calculation date ranges.

    Attributes:
        selected_exposure_start_datetime: The start date for the exposure period.
        selected_exposure_end_datetime: The end date for the exposure period.
        selected_calc_start_datetime: The start date for the calculation period.
        selected_calc_end_datetime: The end date for the calculation period.
        exposure_delta_days: The number of days in the exposure period.
    """

    selected_exposure_start_datetime: date
    selected_exposure_end_datetime: date
    selected_calc_start_datetime: date
    selected_calc_end_datetime: date
    exposure_delta_days: int

    @classmethod
    def render(cls, experiment_id: int | None) -> TimeInputSection:
        """Renders date input fields for exposure and calculation periods.

        It displays start and end date inputs for both exposure and calculation periods,
        pre-filled with default values (e.g., last 30 days).

        Returns:
            An instance of TimeInputSection with the selected date ranges.
        """
        experiment = None
        if experiment_id is not None:
            experiment = get_experiment_by_id(experiment_id)

        predefined_start_dt = (
            experiment.start_datetime.date()
            if experiment and experiment.start_datetime
            else datetime.now().date()
        )
        predefined_end_dt = (
            experiment.end_datetime.date()
            if experiment and experiment.end_datetime
            else datetime.now().date()
        )

        col_level_1_l, col_level_1_r = st.columns(2)
        with col_level_1_l:
            st.markdown("###### Exposure")
            selected_exposure_start_datetime = st.date_input(
                "Start", value=predefined_start_dt - timedelta(days=30)
            )
            selected_exposure_end_datetime = st.date_input(
                "End", value=predefined_end_dt - timedelta(days=1)
            )
        with col_level_1_r:
            st.markdown("###### Calculation")
            selected_calc_start_datetime = st.date_input(
                "Start ", value=predefined_start_dt - timedelta(days=30)
            )
            selected_calc_end_datetime = st.date_input("End ", value=predefined_end_dt - timedelta(days=1))
        exposure_delta_days = (selected_exposure_end_datetime - selected_exposure_start_datetime).days
        return cls(
            selected_exposure_start_datetime=selected_exposure_start_datetime,
            selected_exposure_end_datetime=selected_exposure_end_datetime,
            selected_calc_start_datetime=selected_calc_start_datetime,
            selected_calc_end_datetime=selected_calc_end_datetime,
            exposure_delta_days=exposure_delta_days,
        )


@dataclass
class ExposureEventSelector:
    """A select box for choosing an exposure event.

    Attributes:
        selected_exposure_event: The name of the selected exposure event.
    """

    selected_exposure_event: str | None

    @classmethod
    def render(cls) -> ExposureEventSelector:
        """Renders a select box for choosing an exposure event.

        The options are populated from available exposure events.

        Returns:
            An instance of ExposureEventSelector with the selected event.
        """
        available_exposure_events = get_available_exposure_events()
        index = (
            available_exposure_events.index(None)
            if available_exposure_events and None in available_exposure_events
            else 0
        )
        selected_exposure_event = st.selectbox(
            "Exposure Event", options=available_exposure_events, index=index
        )
        return cls(selected_exposure_event=selected_exposure_event)


@dataclass
class SplitIdSelectBox:
    """A select box for choosing a split ID.

    Attributes:
        selected_split_id: The selected split ID ('user_id' or 'device_id').
    """

    selected_split_id: str

    @classmethod
    def render(cls) -> SplitIdSelectBox:
        """Renders a select box for choosing a split ID.

        The options are hardcoded to 'user_id' and 'device_id'.

        Returns:
            An instance of SplitIdSelectBox with the selected split ID.
        """
        selected_split_id = st.selectbox("Split ID", ["user_id"])
        return cls(selected_split_id=selected_split_id)


@dataclass
class ErrorsSelectors:
    """A pair of selectors for statistical error rates (Alpha and Beta).

    Attributes:
        selected_alpha: The selected Type I error rate (alpha).
        selected_beta: The selected Type II error rate (beta).
    """

    selected_alpha: float
    selected_beta: float

    @classmethod
    def render(cls) -> ErrorsSelectors:
        """Renders sliders for selecting alpha and beta values.

        Provides a user-friendly way to select statistical error thresholds
        from a predefined list of options.

        Returns:
            An instance of ErrorsSelectors with the chosen alpha and beta values.
        """
        col1, col2 = st.columns(2)
        with col1:
            select_box_alpha = st.select_slider(
                "Alpha (Type I Error)",
                options=["0.1%", "1%", "5%", "10%"],
                value="5%",  # Default value
            )
            alpha_mapping = {"0.1%": 0.001, "1%": 0.01, "5%": 0.05, "10%": 0.1}
            selected_alpha = alpha_mapping[select_box_alpha]

        with col2:
            select_box_beta = st.select_slider(
                "Beta (Type II Error)",
                options=["5%", "10%", "20%"],
                value="20%",  # Default value
            )
            beta_mapping = {"5%": 0.05, "10%": 0.1, "20%": 0.2}
            selected_beta = beta_mapping[select_box_beta]
        return cls(selected_alpha=selected_alpha, selected_beta=selected_beta)


@dataclass
class EffectRangeSlider:
    """A slider for selecting a range of relative effect sizes.

    Attributes:
        selected_effect_range: A list of floats representing the selected
            range of relative effect sizes.
    """

    selected_effect_range: list[float]

    @classmethod
    def render(cls) -> EffectRangeSlider:
        """Renders a slider for selecting a range of relative effect sizes.

        The user can select a minimum and maximum effect size in percentages.

        Returns:
            An instance of EffectRangeSlider with the selected effect range.
        """
        values = st.slider(
            label="Relative Effect Size (%)",
            min_value=0.5,
            max_value=30.0,
            value=(1.0, 10.0),
            step=0.5,
            format="%f%%",
        )
        min_int = int(values[0] * 10)
        max_int = int(values[1] * 10)
        values_range = [v / 1000 for v in range(min_int, max_int)]
        return cls(selected_effect_range=values_range)


@dataclass
class DummyObsSelectedParams:
    """Data structure to hold parameters for a dummy observation.

    These parameters are collected from the user inputs in the 'Get Precomputes' form.

    Attributes:
        calc_start_datetime: Start date for the calculation period.
        calc_end_datetime: End date for the calculation period.
        exposure_start_datetime: Start date for the exposure period.
        exposure_end_datetime: End date for the exposure period.
        exposure_event: The name of the exposure event.
        split_id: The ID to split users by (e.g., 'user_id').
        experiment_metric_names: A list of metric names for the experiment.
    """

    calc_start_datetime: date | None = None
    calc_end_datetime: date | None = None
    exposure_start_datetime: date | None = None
    exposure_end_datetime: date | None = None
    exposure_event: str | None = None
    split_id: str | None = None
    calculation_scenario: str | None = None
    experiment_metric_names: list[str] | None = None


@dataclass
class GetPrecomputesForm:
    """A form to collect parameters for fetching precomputed results.

    This form gathers all necessary inputs from the user to run a power analysis
    or plan an experiment based on historical data.

    Attributes:
        exposure_delta_days: The number of days in the exposure period.
        is_submitted: A boolean flag indicating if the form has been submitted.
        selected_obs_params: An object containing the parameters selected in the form.
    """

    exposure_delta_days: int
    is_submitted: bool = False
    selected_obs_params: DummyObsSelectedParams | None = None

    @classmethod
    def render(
        cls,
        experiment_id_: int | None = None,
    ) -> GetPrecomputesForm:
        """Renders the 'Get Precomputes' form and handles its submission.

        This method encapsulates all the input elements for experiment planning
        within a Streamlit form. It handles input validation (e.g., checking dates)
        and submission logic.

        Args:
            experiment_id_: An optional experiment ID to pre-fill some of the form fields.

        Returns:
            An instance of GetPrecomputesForm containing the form's state and
            the selected parameters upon submission.
        """
        with st.form(f"Dummy Observation Params for {experiment_id_}"):
            calculation_scenario = CalculationScenarioSelectBox.render().selected_scenario
            key_metrics_selectbox = KeyMetricsSelectBox.render(experiment_id_)
            time_input_section = TimeInputSection.render(experiment_id_)
            col1, col2 = st.columns(2)
            with col1:
                exposure_event_selectbox = ExposureEventSelector.render()
            with col2:
                split_id_selectbox = SplitIdSelectBox.render()

            submitted = st.form_submit_button("Get Precomputes")

            def check_dates() -> None:
                if (
                    time_input_section.selected_exposure_end_datetime
                    <= time_input_section.selected_exposure_start_datetime
                ):
                    raise ValueError("Exposure end date must be after start date")
                if (
                    time_input_section.selected_calc_end_datetime
                    <= time_input_section.selected_calc_start_datetime
                ):
                    raise ValueError("Calculation end date must be after start date")

            try:
                check_dates()
            except Exception as e:
                st.error(f"Dates are not valid: {e}")
                return cls(exposure_delta_days=0)

            if submitted:
                return cls(
                    is_submitted=submitted,
                    exposure_delta_days=time_input_section.exposure_delta_days,
                    selected_obs_params=DummyObsSelectedParams(
                        calc_start_datetime=time_input_section.selected_calc_start_datetime,
                        calc_end_datetime=time_input_section.selected_calc_end_datetime,
                        exposure_start_datetime=time_input_section.selected_exposure_start_datetime,
                        exposure_end_datetime=time_input_section.selected_exposure_end_datetime,
                        exposure_event=exposure_event_selectbox.selected_exposure_event,
                        split_id=split_id_selectbox.selected_split_id,
                        calculation_scenario=calculation_scenario,
                        experiment_metric_names=key_metrics_selectbox.selected_metrics,
                    ),
                )
        return cls(exposure_delta_days=0)
