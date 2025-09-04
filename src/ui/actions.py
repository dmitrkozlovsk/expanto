"""This module contains all actions for the UI.

Actions are grouped by functionality:
- Planner actions: for sample size calculations and experiment planning
- Experiment actions: for managing experiments (CRUD operations)
- Observation actions: for managing observations (CRUD operations)
- Results actions: for running calculations and handling results
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Literal

import streamlit as st

from src.domain.enums import CalculationPurpose, PageMode
from src.domain.results import JobResult
from src.logger_setup import logger
from src.services.entities.handlers import ExperimentHandler, ObservationHandler
from src.ui.data_loaders import (
    get_experiment_by_id,
    get_experiments_for_filter,
    get_experiments_full_df,
    get_jobs_by_observation_id,
    get_observation_by_id,
    get_observations_for_filter,
    get_observations_full_df,
)
from src.ui.resources import get_thread_pool_executor, load_calculation_runner, load_engine

if TYPE_CHECKING:
    from concurrent.futures import Future

    import pandas as pd  # type: ignore

    from src.domain.models import Observation
    from src.ui.experiments.inputs import ExperimentFormData
    from src.ui.observations.inputs import ObservationFormData


# --------------------------- EXPERIMENT ACTIONS ---------------------------


def create_experiment(experiment: ExperimentFormData) -> None:
    """Creates a new experiment with the provided data.

    Args:
        experiment: The experiment data to create a new experiment with.

    Note:
        This function will clear relevant caches after a successful creation.
    """
    try:
        handler = ExperimentHandler(load_engine())
        exp = handler.create(**asdict(experiment))
        get_experiments_for_filter.clear()  # type: ignore[attr-defined]
        get_experiments_full_df.clear()  # type: ignore[attr-defined]
        if exp:
            st.toast(f"✅ Experiment <name:{exp.name}, id:{exp.id}> successfully created!")
        else:
            st.toast("⚠️ Something went wrong, please try again")
    except Exception as e:
        st.toast(f"❌ Error creating experiment: <Experiment name :{experiment.name}>\nError:  {e}")
        logger.exception(str(e))


def update_experiment(experiment_id: int, experiment: ExperimentFormData) -> None:
    """Updates an existing experiment with new data.

    Args:
        experiment_id: The ID of the experiment to update.
        experiment: The new experiment data to apply.

    Note:
        This function will clear relevant caches after a successful update.
    """
    try:
        handler = ExperimentHandler(load_engine())
        # Check if experiment exists before attempting update
        existing_exp = handler.get(experiment_id)
        if not existing_exp:
            st.toast(f"Experiment with ID {experiment_id} not found", icon="⚠️")
            return

        exp = handler.update(id_=experiment_id, **asdict(experiment))
        get_experiment_by_id.clear(experiment_id=experiment_id)  # type: ignore[attr-defined]
        get_experiments_for_filter.clear()  # type: ignore[attr-defined]
        get_experiments_full_df.clear()  # type: ignore[attr-defined]
        if exp:
            st.toast(f"Experiment <name:{exp.name}, id:{exp.id}> successfully updated!", icon="✅")
        else:
            st.toast("Something went wrong, please try again", icon="⚠️")
    except Exception as e:
        st.toast(f"Error updating experiment: <Experiment name :{experiment.name}>\nError:  {e}", icon="❌")
        logger.exception(str(e))


def delete_experiment(experiment_id: int) -> None:
    """Deletes an experiment with the provided ID.

    Args:
        experiment_id: The ID of the experiment to delete.

    Note:
        This function will clear relevant caches after a successful deletion.
    """
    try:
        handler = ExperimentHandler(load_engine())
        exp = handler.get(experiment_id)
        if exp:
            handler.delete(experiment_id)
            get_experiment_by_id.clear(experiment_id=experiment_id)  # type: ignore[attr-defined]
            get_experiments_for_filter.clear()  # type: ignore[attr-defined]
            get_experiments_full_df.clear()  # type: ignore[attr-defined]
            st.toast(f"Experiment <name:{exp.name}, id:{exp.id}> successfully deleted!", icon="✅")
        else:
            st.toast("Experiment not found", icon="⚠️")
    except Exception as e:
        st.toast(f"Error deleting experiment with ID {experiment_id}\nError: {e}", icon="❌")
        logger.exception(str(e))


def generate_hyperlinks_for_experiment_df(pandas_df: pd.DataFrame) -> pd.DataFrame:
    """Generates hyperlinks for experiment dataframe rows.

    Args:
        pandas_df: Input dataframe containing experiment data.

    Returns:
        DataFrame with added hyperlink columns for updating and viewing observations.
    """
    pandas_df["update"] = pandas_df["id"].apply(lambda x: f"/?mode={PageMode.UPDATE}&experiment_id={x}")
    pandas_df["see_observations"] = pandas_df["id"].apply(
        lambda x: f"/observations?mode={PageMode.LIST}&experiment_id={x}"
    )
    return pandas_df


# -------------------------- OBSERVATION ACTIONS ---------------------------


def create_observation(observation: ObservationFormData) -> None:
    """Creates a new observation and handles UI feedback.

    This function takes observation data, uses the ObservationHandler to create
    a new observation in the database, clears relevant caches, and displays
    a success or error message in the Streamlit UI.

    Args:
        observation: An object containing the new observation's data.
    """
    try:
        handler = ObservationHandler(load_engine())
        obs = handler.create(**asdict(observation))
        get_observations_for_filter.clear()  # type: ignore[attr-defined]
        get_observations_full_df.clear()  # type: ignore[attr-defined]
        if obs:
            st.toast(f"Observation <name:{obs.name}, id:{obs.id}> successfully created!", icon="✅")
        else:
            st.toast("Something went wrong, please try again", icon="⚠️")
    except Exception as e:
        st.toast(f"Error creating observation: {e}", icon="❌")
        logger.exception(str(e))


def update_observation(observation_id: int, observation: ObservationFormData) -> None:
    """Updates an existing observation and handles UI feedback.

    This function takes an observation ID and new data, uses the
    ObservationHandler to update the observation in the database, clears
    relevant caches, and displays a success or error message in the
    Streamlit UI.

    Args:
        observation_id: The ID of the observation to update.
        observation: An object containing the updated observation's data.
    """
    try:
        handler = ObservationHandler(load_engine())
        # Check if observation exists before attempting update
        existing_obs = handler.get(observation_id)
        if not existing_obs:
            st.toast(f"Observation with ID {observation_id} not found", icon="⚠️")
            return

        obs = handler.update(observation_id, **asdict(observation))
        get_observation_by_id.clear(observation_id=observation_id)  # type: ignore[attr-defined]
        get_observations_for_filter.clear()  # type: ignore[attr-defined]
        get_observations_full_df.clear()  # type: ignore[attr-defined]
        if obs:
            st.toast(f"Observation <name:{obs.name}, id:{obs.id}> successfully updated!", icon="✅")
        else:
            st.toast("Something went wrong, please try again", icon="⚠️")
    except Exception as e:
        st.toast(f"Error updating observation: {e}", icon="❌")
        logger.exception(str(e))


def delete_observation(observation_id: int) -> None:
    """Deletes an observation and handles UI feedback.

    This function deletes an observation with the provided ID using the
    ObservationHandler, clears relevant caches, and displays a success or error
    message in the Streamlit UI.

    Args:
        observation_id: The ID of the observation to delete.
    """
    try:
        handler = ObservationHandler(load_engine())
        obs = handler.get(observation_id)
        if obs:
            handler.delete(observation_id)
            get_observation_by_id.clear(observation_id=observation_id)  # type: ignore[attr-defined]
            get_observations_for_filter.clear()  # type: ignore[attr-defined]
            get_observations_full_df.clear()  # type: ignore[attr-defined]
            st.toast(f"Observation <name:{obs.name}, id:{obs.id}> successfully deleted!", icon="✅")
        else:
            st.toast("Observation not found", icon="⚠️")
    except Exception as e:
        st.toast(f"Error deleting observation with ID {observation_id}\nError: {e}", icon="❌")
        logger.exception(str(e))


def generate_hyperlinks_for_observations_df(pandas_df: pd.DataFrame) -> pd.DataFrame:
    """Adds hyperlinks to a DataFrame of observations.

    This function adds 'see_results' and 'update' columns to the given
    DataFrame. These columns contain URL strings for viewing results and
    updating each observation, respectively.

    Args:
        pandas_df: The DataFrame of observations to modify. It must
            contain an 'id' column.

    Returns:
        The DataFrame with added hyperlink columns.
    """
    pandas_df["see_results"] = pandas_df["id"].apply(lambda x: f"/results?observation_id={x}")
    pandas_df["update"] = pandas_df["id"].apply(lambda x: f"/observations?mode=Update&observation_id={x}")
    return pandas_df


# ----------------------------- RESULTS ACTIONS ----------------------------


def run_regular_calculation(observation: Observation) -> JobResult | None:
    """Runs a regular calculation for a given observation.

    Args:
        observation: The observation to run the calculation for.

    Returns:
        The job ID of the calculation if it was started, otherwise None.
    """
    runner = load_calculation_runner()
    job_result = runner.run_calculation(observation, CalculationPurpose.REGULAR)
    return job_result


def run_regular_calculation_in_background(observation: Observation) -> Future:
    """Runs a regular calculation for a given observation in the background.

    Args:
        observation: The observation to run the calculation for.
    """
    runner = load_calculation_runner()
    executor = get_thread_pool_executor()
    future = executor.submit(runner.run_calculation, observation, CalculationPurpose.REGULAR)
    return future


def run_observation_calculation(
    observation_id: int, calc_type: Literal["foreground", "background"]
) -> JobResult | None:
    """Handles running a calculation for an observation based on the calculation type.

    This function retrieves an observation by its ID and then runs a calculation
    either in the foreground or background.

    Args:
        observation_id: The ID of the observation to run the calculation for.
        calc_type: The type of calculation to run.
            Can be "foreground" or "background".

    Returns:
        The job ID if the calculation is run in the foreground, otherwise None.

    Raises:
        ValueError: If an invalid calculation type is provided.
    """
    observation = get_observation_by_id(observation_id)
    if not observation:
        raise ValueError(f"Could not find observation with ID: {observation_id}")

    if calc_type == "foreground":
        job_result = run_regular_calculation(observation)
        get_jobs_by_observation_id.clear()  # type: ignore[attr-defined]
        return job_result
    elif calc_type == "background":
        run_regular_calculation_in_background(observation)
    else:
        raise ValueError(f"Invalid calculation type: {calc_type}. Must be 'foreground' or 'background'.")
