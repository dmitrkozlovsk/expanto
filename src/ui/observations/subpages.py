"""Module containing observation-related page components for the UI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from src.ui.actions import create_observation, delete_observation, update_observation
from src.ui.common import put_return_in_app_ctx
from src.ui.data_loaders import get_observation_by_id, get_observations_full_df
from src.ui.experiments.inputs import ExperimentSelectBox
from src.ui.observations.elements import ObservationDetailedInfo, ObservationsDataframe
from src.ui.observations.inputs import (
    ObservationFormInputs,
    ObservationSelectBoxAfterExp,
    ObservationsFilter,
)

if TYPE_CHECKING:
    from src.ui.common import URLParams


# ------------------------- Observations List ---------------------------------
class ObservationListPage:
    """A page for listing and viewing observations."""

    @classmethod
    @put_return_in_app_ctx
    def render(cls, url_exp_id: int | None) -> dict | None:
        """Render the observation list page.

        This method displays filters for observations and shows them in a dataframe.
        It also handles deletion of observations.

        Args:
            url_exp_id: The experiment ID from the URL, if any.
        """
        # Render filters
        filters = ObservationsFilter.render(url_exp_id)
        if (obs_df := get_observations_full_df(filters)) is not None:
            if (obs_to_delete := st.session_state.pop("obs_to_delete", None)) is not None:
                delete_observation(int(obs_to_delete.id))
                obs_df = get_observations_full_df(filters)
            observation_dataframe = ObservationsDataframe.render(obs_df)
            ObservationDetailedInfo.render(observation_dataframe.selected_observation)
        else:
            return None
        if (selected_observation := observation_dataframe.selected_observation) is not None:
            return {
                "selected_experiment_id": selected_observation.to_dict().get("experiment_id"),
                "selected_observation_id": selected_observation.to_dict().get("id"),
            }
        else:
            return None


# -------------------------- Create Observation --------------------------------


class CreateObservationPage:
    """A page for creating a new observation."""

    @classmethod
    def render(cls, url_params: URLParams) -> None:
        """Render the create observation page.

        This method displays a form for creating a new observation. It can
        pre-fill the form if an observation is being copied.

        Args:
            url_params: The URL parameters.
        """
        if "obs_to_copy" in st.session_state:
            predefined_exp_id = st.session_state.obs_to_copy.experiment_id
            predefined_observation = st.session_state.obs_to_copy
            if "ts" in st.query_params:
                del st.query_params["ts"]
        else:
            predefined_exp_id = None
            predefined_observation = None
        exp_select_box = ExperimentSelectBox.render(predefined_exp_id)
        if exp_select_box is None:
            st.info("Please select an experiment.")
            return None

        obs_form_input = ObservationFormInputs.render(
            exp_select_box.id, exp_select_box.name, predefined_observation
        )
        if obs_form_input and obs_form_input.submitted:
            create_observation(obs_form_input.observation)
            if "obs_to_copy" in st.session_state:
                del st.session_state["obs_to_copy"]


# -------------------------- Update Observation --------------------------------


class UpdateObservationPage:
    """A page for updating an existing observation."""

    @classmethod
    @put_return_in_app_ctx
    def render(cls, url_params: URLParams) -> dict | None:
        """Render the update observation page.

        This method displays a form for updating an existing observation,
        which is selected via a select box.

        Args:
            url_params: The URL parameters.
        """
        obs_select_box = ObservationSelectBoxAfterExp.render(url_params)
        if obs_select_box is None:
            st.info("Please select an observation.")
            return None
        selected_observation = get_observation_by_id(obs_select_box.id)
        obs_form_input = ObservationFormInputs.render(
            obs_select_box.experiment_id, obs_select_box.name, selected_observation
        )
        if obs_form_input and obs_form_input.submitted and obs_form_input.observation_id:
            update_observation(obs_form_input.observation_id, obs_form_input.observation)
        if selected_observation:
            return {
                "experiment_id": selected_observation.experiment_id,
                "observation_id": selected_observation.id,
            }
        else:
            return None
