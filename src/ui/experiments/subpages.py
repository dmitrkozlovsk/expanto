"""Module containing experiment-related page components for the UI.

This module provides classes for rendering different experiment-related subpages including
experiment listing, creation, and updating functionality.
"""

from __future__ import annotations

import streamlit as st

from src.ui.actions import create_experiment, delete_experiment, update_experiment
from src.ui.common import put_return_in_app_ctx
from src.ui.data_loaders import get_experiment_by_id, get_experiments_full_df
from src.ui.experiments.elements import ExperimentDetailedInfo, ExperimentsDataframe
from src.ui.experiments.inputs import (
    ExperimentFormInputs,
    ExperimentSelectBox,
    ExperimentsFilter,
)


# ---------------------------------------- Experiment List ----------------------------------------
class ExperimentListPage:
    """A class representing the experiment list page component.

    This class handles the rendering of the experiment list page, including filters,
    data table, and detailed experiment information.
    """

    @classmethod
    @put_return_in_app_ctx
    def render(cls) -> dict[str, int] | None:
        """Renders the experiment list page.

        This method:
        1. Renders experiment filters
        2. Fetches experiment data based on filters
        3. Displays the main experiments table
        4. Shows detailed information for selected experiment

        Returns:
            Dictionary with experiment_id if an experiment is selected, None otherwise.
        """
        filters = ExperimentsFilter.render()  # render filters
        exp_df = get_experiments_full_df(filters=filters)  # download the data
        if exp_df is None:
            st.info("No experiments found.")
            return None

        if (exp_to_delete := st.session_state.pop("exp_to_delete", None)) is not None:
            delete_experiment(int(exp_to_delete.id))
            exp_df = get_experiments_full_df(filters=filters)
        experiments_dataframe = ExperimentsDataframe.render(exp_df)
        ExperimentDetailedInfo.render(experiments_dataframe.selected_experiment)  # show detailed info

        if (selected_experiment := experiments_dataframe.selected_experiment) is not None:
            return {"experiment_id": selected_experiment.to_dict()["id"]}
        else:
            return None


# --------------------------------------- Create Experiment ---------------------------------------


class CreateExperimentPage:
    """A class representing the experiment creation page component.

    This class handles the rendering of the experiment creation form and
    processing of new experiment submissions.
    """

    @classmethod
    @put_return_in_app_ctx
    def render(cls) -> None:
        """Renders the experiment creation page.

        This method:
        1. Renders the experiment creation form
        2. Processes form submission
        3. Creates new experiment if form is submitted
        """
        exp_form_input = ExperimentFormInputs.render()
        if exp_form_input and exp_form_input.submitted:
            create_experiment(exp_form_input.experiment)


# --------------------------------------- Update Experiment ---------------------------------------


class UpdateExperimentPage:
    """A class representing the experiment update page component.

    This class handles the rendering of the experiment update form and
    processing of experiment updates.
    """

    @classmethod
    @put_return_in_app_ctx
    def render(cls, url_exp_id: int | None = None) -> dict[str, int] | None:
        """Renders the experiment update page.

        Args:
            url_exp_id: The ID of the experiment to update.
                If provided, this experiment will be pre-selected.

        Returns:
            Dictionary with experiment_id if an experiment is selected, None otherwise.

        This method:
        1. Renders experiment selection box
        2. If experiment is selected:
           - Fetches experiment data
           - Renders update form with pre-filled data
           - Processes form submission
           - Updates experiment if form is submitted
        """
        selected_exp = ExperimentSelectBox.render(predefined_exp_id=url_exp_id)
        if selected_exp is None:
            st.info("Please select an experiment.")
            return None
        experiment = get_experiment_by_id(selected_exp.id)
        exp_form_input = ExperimentFormInputs.render(predefined=experiment)
        if exp_form_input and exp_form_input.submitted and exp_form_input.experiment_id:
            update_experiment(exp_form_input.experiment_id, exp_form_input.experiment)
        if experiment:
            return {"experiment_id": int(experiment.id)}
        return None
