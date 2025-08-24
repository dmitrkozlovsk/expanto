"""Data loading utilities.

This module provides cached data loading functions for experiments, observations,
calculation jobs, precomputes, and other entities used throughout the Streamlit UI.
All functions use Streamlit's caching mechanism to improve performance.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import streamlit as st

from src.logger_setup import logger
from src.services.entities.dtos import CalculationJobDTO, ExperimentDTO, ObservationDTO
from src.services.entities.handlers import (
    ExperimentHandler,
    JobHandler,
    ObservationHandler,
    PrecomputeHandler,
)
from src.ui.resources import load_calculation_runner, load_engine

if TYPE_CHECKING:
    from datetime import datetime

    import pandas as pd  # type: ignore
    from sqlalchemy.engine.row import Row  # type: ignore[attr-defined]

    from src.ui.experiments.inputs import ExperimentsFilter
    from src.ui.observations.inputs import ObservationsFilter

from src.domain.enums import CalculationPurpose

# ------------------------------- EXPERIMENTS -------------------------------


@st.cache_data(ttl=1, max_entries=200)
def get_experiment_by_id(experiment_id: int) -> ExperimentDTO | None:  # type: ignore[valid-type]
    """Get experiment details by ID.

    Args:
        experiment_id: The unique identifier of the experiment to retrieve.

    Returns:
        The experiment object if found, None if not found or on error.

    Raises:
        Exception: Any database or connection errors are caught and logged.
    """
    try:
        engine = load_engine()
        handler = ExperimentHandler(engine)
        experiment = handler.get(experiment_id, as_dto=True)
        return experiment
    except Exception as error:
        error_message = f"Could not get experiment with ID {experiment_id}: {error}"
        logger.exception(error_message)
        st.error(error_message)
        return None


@st.cache_data(ttl=300, max_entries=1)
def get_experiments_for_filter() -> list[Row] | None:
    """Fetch all experiments for selection dropdown.

    Returns:
        A list of database rows containing experiment ID and name columns,
        or None if an error occurs.

    Raises:
        Exception: Any database or connection errors are caught and logged.
    """
    try:
        engine = load_engine()
        handler = ExperimentHandler(engine)
        exp_list = handler.select(columns=["id", "name"])
        return exp_list
    except Exception as error:
        error_message = f"Could not get experiments: {error}"
        logger.exception(error_message)
        st.error(error_message)
        return None


@st.cache_data(ttl=300, max_entries=5)
def get_experiments_full_df(filters: ExperimentsFilter) -> pd.DataFrame | None:
    """Get filtered experiments as a pandas DataFrame.ObservationDT

    Args:
        filters: An ExperimentsFilter object containing:
            - experiment_name: Optional string to filter by experiment name (case-insensitive).
            - created_at_start: Start datetime for filtering by creation date.
            - created_at_end: End datetime for filtering by creation date.
            - status_list_filter: List of statuses to filter by.
            - limit_filter: Maximum number of records to return.

    Returns:
        A pandas DataFrame containing the filtered experiments,
        or None if an error occurs.

    Raises:
        Exception: Any database or connection errors are caught and logged.
    """
    try:
        engine = load_engine()
        handler = ExperimentHandler(engine)

        handler_filters = {
            "_created_at__gte": filters.created_at_start,
            "_created_at__lt": filters.created_at_end,
            "status__eq__or": filters.status_list_filter,
        }
        if filters.experiment_name:
            handler_filters["name__ilike"] = f"%{filters.experiment_name}%"

        exp_df = handler.select(filters=handler_filters, limit=filters.limit_filter, return_pandas=True)
        return exp_df
    except Exception as error:
        error_message = f"Could not get experiments: {error}"
        logger.exception(error_message)
        st.error(error_message)
        return None


# ------------------------------ OBSERVATIONS ------------------------------


@st.cache_data(ttl=300, max_entries=5)
def get_observations_full_df(filters: ObservationsFilter) -> pd.DataFrame | None:
    """Get filtered observations as a pandas DataFrame.

    Args:
        filters: An ObservationsFilter object containing:
            - observation_name: Optional string to filter by observation name (case-insensitive).
            - created_at_start: Start datetime for filtering by creation date.
            - created_at_end: End datetime for filtering by creation date.
            - experiment_id: Optional experiment ID to filter observations by.
            - limit_filter: Maximum number of records to return.

    Returns:
        A pandas DataFrame containing the filtered observations,
        or None if an error occurs.

    Raises:
        Exception: Any database or connection errors are caught and logged.
    """
    try:
        engine = load_engine()
        handler = ObservationHandler(engine)
        handler_filters: dict[str, Any] = {
            "_created_at__gte": filters.created_at_start,
            "_created_at__lt": filters.created_at_end,
        }
        if filters.observation_name:
            handler_filters["name__ilike"] = f"%{filters.observation_name}%"

        if filters.experiment_id:
            handler_filters["experiment_id__eq"] = filters.experiment_id

        exp_df = handler.select(filters=handler_filters, limit=filters.limit_filter, return_pandas=True)
        return exp_df

    except Exception as error:
        error_message = f"Could not get observations: {error}"
        logger.exception(error_message)
        st.error(error_message)
        return None


@st.cache_data(ttl=1200, max_entries=200)
def get_observation_by_id(observation_id: int) -> ObservationDTO | None:  # type: ignore[valid-type]
    """Get observation details by ID.

    Args:
        observation_id: The unique identifier of the observation to retrieve.

    Returns:
        The observation object if found, None if not found or on error.

    Raises:
        Exception: Any database or connection errors are caught and logged.
    """
    try:
        engine = load_engine()
        handler = ObservationHandler(engine)
        observation = handler.get(observation_id, as_dto=True)
        return observation
    except Exception as error:
        error_message = f"Could not get observation with ID {observation_id}: {error}"
        logger.exception(error_message)
        st.error(error_message)
        return None


@st.cache_data(ttl=1800, max_entries=1)
def get_observations_for_filter() -> list[Row] | None:
    """Get observations for selection dropdown.

    Returns:
        A list of database rows containing observation ID, experiment_id,
        and name columns, or None if an error occurs.

    Raises:
        Exception: Any database or connection errors are caught and logged.
    """
    try:
        engine = load_engine()
        handler = ObservationHandler(engine)
        obs_list = handler.select(columns=["id", "experiment_id", "name"])
        return obs_list
    except Exception as error:
        error_message = f"Could not get observations: {error}"
        logger.exception(error_message)
        st.error(error_message)
        return None


# ---------------------------------- JOBS ----------------------------------


@st.cache_data(ttl=1200, max_entries=200)
def get_jobs_by_observation_id(observation_id: int) -> list[Row] | None:
    """Get all completed calculation jobs for a specific observation.

    Args:
        observation_id: The unique identifier of the observation.

    Returns:
        A list of database rows containing job information (id, observation_id, _created_at)
        for completed jobs, or None if an error occurs.

    Raises:
        Exception: Any database or connection errors are caught and logged.
    """
    try:
        engine = load_engine()
        handler = JobHandler(engine)
        jobs = handler.select(
            columns=["id", "observation_id", "_created_at"],
            filters={
                "observation_id__eq": observation_id,
                "status__eq": "completed",
            },
        )
        return jobs
    except Exception as error:
        error_message = f"Could not get jobs: {error}"
        logger.exception(error_message)
        st.error(error_message)
        return None


@st.cache_data(ttl=1200, max_entries=200)
def get_observation_id_by_job_id(job_id: int) -> int | None:
    """Get the observation ID associated with a calculation job.

    Args:
        job_id: The unique identifier of the calculation job.

    Returns:
        The observation ID if the job is found, None if not found or on error.

    Raises:
        Exception: Any database or connection errors are caught and logged.
    """
    try:
        engine = load_engine()
        handler = JobHandler(engine)
        job = handler.get(job_id)
        return int(job.observation_id) if job else None
    except Exception as error:
        error_message = f"Could not get job: {error}"
        logger.exception(error_message)
        st.error(error_message)
        return None


@st.cache_data(ttl=1200, max_entries=200)
def get_job_by_id(job_id: int) -> CalculationJobDTO | None:  # type: ignore[valid-type]
    """Get calculation job details by ID.

    Args:
        job_id: The unique identifier of the calculation job to retrieve.

    Returns:
        The calculation job object if found, None if not found or on error.

    Raises:
        Exception: Any database or connection errors are caught and logged.
    """
    try:
        engine = load_engine()
        handler = JobHandler(engine)
        job = handler.get(job_id, as_dto=True)
        return job
    except Exception as error:
        error_message = f"Could not get job: {error}"
        logger.exception(error_message)
        st.error(error_message)
        return None


# ------------------------------- PRECOMPUTES -------------------------------


@st.cache_data(ttl=1800, max_entries=200)
def get_precomputes_by_job_id(job_id: int) -> pd.DataFrame | None:
    """Get all precomputed metrics for a specific calculation job.

    Args:
        job_id: The unique identifier of the calculation job.

    Returns:
        A pandas DataFrame containing precomputed metric results,
        or None if an error occurs.

    Raises:
        Exception: Any database or connection errors are caught and logged.
    """
    try:
        engine = load_engine()
        handler = PrecomputeHandler(engine)
        precomputes = handler.select(filters={"job_id__eq": job_id}, return_pandas=True)
        return precomputes
    except Exception as error:
        error_message = f"Could not get precomputes: {error}"
        logger.exception(error_message)
        st.error(error_message)
        return None


@st.cache_data(ttl=1800, max_entries=200, show_spinner=False)
def get_precomputes_for_planning(
    calc_start_datetime: datetime,
    calc_end_datetime: datetime,
    exposure_start_datetime: datetime,
    exposure_end_datetime: datetime,
    exposure_event: str,
    split_id: str,
    experiment_metric_names: list[str],
) -> dict:
    """Run calculation and get metric precomputes for experiment planning.

    Creates a dummy observation with the provided parameters and runs calculations
    for planning purposes to estimate metric results.

    Args:
        calc_start_datetime: Start datetime for the calculation window.
        calc_end_datetime: End datetime for the calculation window.
        exposure_start_datetime: Start datetime for the exposure window.
        exposure_end_datetime: End datetime for the exposure window.
        exposure_event: The event that triggers user exposure to the experiment.
        split_id: Identifier for the experiment split/variant.
        experiment_metric_names: List of metric names to calculate.

    Returns:
        A dictionary containing metric calculation results,
        or empty dict if calculation fails or returns no results.

    Raises:
        Exception: Any calculation or processing errors are caught and logged.
    """
    dummy_observation = ObservationDTO(
        id=None,
        calc_start_datetime=calc_start_datetime,
        calc_end_datetime=calc_end_datetime,
        exposure_start_datetime=exposure_start_datetime,
        exposure_end_datetime=exposure_end_datetime,
        exposure_event=exposure_event,
        split_id=split_id,
        experiment_id=None,
        name=None,
        db_experiment_name=None,
        calculation_scenario=None,
        _created_at=None,
        _updated_at=None,
    )

    runner = load_calculation_runner()
    job_result = runner.run_calculation(
        dummy_observation,
        CalculationPurpose.PLANNING,
        experiment_metric_names_=experiment_metric_names,
    )
    return json.loads(job_result.model_dump_json()) if job_result else {}


# ---------------------------------- OTHER ----------------------------------
@st.cache_data(ttl=24600, max_entries=1)
def get_available_exposure_events() -> list[str | None]:
    """Get list of all available exposure events.

    Returns a predefined list of exposure events that can be used to trigger
    user exposure to experiments. This is currently mock data that should be
    replaced with real database queries in production.

    Returns:
        A list of exposure event names as strings, including None for no event.

    Note:
        This is currently mock data and should be replaced with real data
        fetched from the database in production.
    """
    # Mock data - in production, would fetch from database
    # todo replace by real data
    # todo transfer it to resources.py
    # todo SELECT DISTINCT event_name from events where timestamp between two days ago
    return [
        None,
    ]
