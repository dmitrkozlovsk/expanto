"""This module defines dataclasses for UI input elements on the results page.

These components are used to filter and select data for display, including:
- Job and observation selection.
- Experiment group filtering.
- P-value threshold selection.
- Table column selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import streamlit as st

from src.ui.data_loaders import get_job_by_id, get_jobs_by_observation_id
from src.ui.observations.inputs import ObservationSelectBox
from src.ui.resources import load_metric_groups, load_metric_tags

if TYPE_CHECKING:
    import pandas as pd  # type: ignore


@dataclass
class JobSelectBox:
    """A select box for choosing a job."""

    job_id: int | None = None
    observation_id: int | None = None

    @classmethod
    def render(
        cls,
        selected_observation_id: int | None,
        predefined_job_id: int | None,
    ) -> JobSelectBox | None:
        """Renders a select box to choose a job for a given observation.

        Args:
            selected_observation_id: The ID of the observation to show jobs for.
            predefined_job_id: The job ID to pre-select.

        Returns:
            An instance of JobSelectBox with the selected job's info, or None if no job is selected.
        """
        # one job row: 'id', 'observation_id','_created_at
        jobs = get_jobs_by_observation_id(selected_observation_id) if selected_observation_id else []
        if jobs:
            jobs.sort(key=lambda job: job, reverse=True)
        job_options = jobs or []

        filter_result = filter(lambda r: r[1][0] == predefined_job_id, enumerate(job_options))  # type: ignore[arg-type]
        generator = map(lambda r: r[0], filter_result)
        select_box_index: int | None = next(generator, 0)

        selectbox_job = st.selectbox(
            "Select job",
            options=job_options,
            format_func=lambda job: f"{job[-1].strftime('%Y-%m-%d %H:%M')} (№{job[0]})",
            index=select_box_index,
            key=f"job_select_box_{selected_observation_id}",
        )
        if selectbox_job is None:
            return None

        return cls(job_id=selectbox_job[0], observation_id=selectbox_job[1])


@dataclass
class SelectJobFilters:
    """A component for selecting an observation and a job."""

    selected_observation_name: str | None = None
    selected_observation_id: int | None = None
    selected_job_id: int | None = None

    @classmethod
    def render(cls, job_id: int | None, observation_id: int | None) -> SelectJobFilters:
        """Renders filters to select an observation and a job.

        It can be initialized with a job ID or an observation ID to pre-select
        the corresponding filters.

        Args:
            job_id: The job ID to pre-select.
            observation_id: The observation ID to pre-select.

        Returns:
            An instance of SelectJobFilters with the selected filter values.
        """
        if job_id:
            predefined_job_id = job_id
            predefined_job = get_job_by_id(predefined_job_id)
            predefined_observation_id = int(predefined_job.observation_id) if predefined_job else None
        elif not job_id and observation_id:
            predefined_job_id = None
            predefined_observation_id = observation_id
        else:
            predefined_job_id = None
            predefined_observation_id = None

        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            selected_observation = ObservationSelectBox.render(predefined_observation_id, None)
        with filter_col2:
            selected_job = JobSelectBox.render(
                selected_observation_id=selected_observation.id if selected_observation else None,
                predefined_job_id=predefined_job_id,
            )

        return cls(
            selected_observation_name=selected_observation.name if selected_observation else None,
            selected_observation_id=selected_observation.id if selected_observation else None,
            selected_job_id=selected_job.job_id if selected_job else None,
        )


@dataclass
class ExperimentGroupsFilters:
    """Filters for selecting control and comparison groups in an experiment."""

    control_group: str | None = None
    compared_groups: list[str] | None = None

    @classmethod
    def render(cls, precomputes: pd.DataFrame) -> ExperimentGroupsFilters:
        """Renders select boxes for choosing control and comparison groups.

        Args:
            precomputes: DataFrame containing precomputed results
                with a 'group_name' column.

        Returns:
            An instance with the selected groups.
        """
        # Get unique group names from precomputes
        available_groups = precomputes.group_name.unique().tolist()
        if len(available_groups) <= 1:
            st.warning(
                f"Insufficient groups for comparison. "
                f"Found only {len(available_groups)} group(s): {available_groups}"
            )
            return cls(control_group=None, compared_groups=None)

        selected_control_group = st.selectbox(
            "Select control group",
            options=available_groups,
            index=available_groups.index("control") if "control" in available_groups else 0,
        )

        # Filter out control group from comparison options
        comparison_groups = [g for g in available_groups if g != selected_control_group]

        # Select groups to compare against control
        selected_compared_groups = st.multiselect(
            "Select groups to compare", options=comparison_groups, default=comparison_groups
        )

        return cls(control_group=selected_control_group, compared_groups=selected_compared_groups)


@dataclass
class PValueThresholdFilter:
    """A filter for setting the p-value threshold."""

    threshold: float | None = 0.05

    @classmethod
    def set_default_if_none(cls, value: float | None) -> float:
        """Returns a default p-value threshold if the given value is None.

        Args:
            value: The p-value to check.

        Returns:
            The original value or the default (0.05).
        """
        return 0.05 if value is None else value

    @classmethod
    def render(cls, key: str | None = None) -> PValueThresholdFilter:
        """Renders pills to select a p-value threshold.

        Returns:
            An instance with the selected threshold.
        """
        # Select p-value threshold for highlighting
        option_map = {0.05: "5%", 0.01: "1%", 0.001: "0.1%"}
        selected_p_value_threshold = st.pills(
            "P-value threshold",
            options=option_map.keys(),
            default=0.05,
            format_func=lambda option: option_map[option],
            selection_mode="single",
            help='P-Value threshold for highlighting. Default is 5% (0.05).',
            key=f"p_value_threshold_pill_{key}",
        )

        return cls(threshold=selected_p_value_threshold)


@dataclass
class MetricsFilters:
    """A filter for selecting which columns to display in a table."""

    groups: set[str]
    tags: set[str]

    @classmethod
    def render(cls) -> MetricsFilters:
        """Renders multiselect boxes for filtering metrics by groups and tags.

        Returns:
            An instance with the selected filter values.
        """
        available_tags = load_metric_tags()
        available_groups = load_metric_groups()
        groups = st.multiselect(
            "Metric groups",
            options=available_groups,
        )
        tags = st.multiselect(
            "Metric tags",
            options=available_tags,
        )
        return cls(groups=set(groups), tags=set(tags))
