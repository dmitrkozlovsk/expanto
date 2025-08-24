"""Module for rendering experiment-related UI elements in Streamlit.

This module provides classes and functions for displaying experiment data in various formats,
including dataframes and detailed information views.
"""

import uuid
from dataclasses import dataclass

import pandas as pd  # type: ignore
import streamlit as st

from src.ui.actions import generate_hyperlinks_for_experiment_df


@dataclass
class ExperimentsDataframe:
    """A class for rendering and managing experiment data in a Streamlit dataframe.

    This class handles the display of experiment data in a tabular format with interactive
    selection capabilities and hyperlinks for related actions.

    Attributes:
        selected_experiment (pd.Series | None): The currently selected experiment row, if any.
    """

    selected_experiment: pd.Series | None = None

    @classmethod
    def render(cls, exp_df: pd.DataFrame) -> "ExperimentsDataframe":
        """Renders the experiment dataframe in Streamlit.

        Args:
            exp_df: DataFrame containing experiment data.

        Returns:
            Instance with selected experiment data if any row is selected.
        """
        if exp_df.empty:
            st.warning("Experiments table is empty")
            return cls()

        exp_df = generate_hyperlinks_for_experiment_df(exp_df)  # create update  / see_observation
        display_columns = [
            "id",
            "see_observations",
            "update",
            "name",
            "status",
            "start_datetime",
            "end_datetime",
            "key_metrics",
        ]
        displayed_dataframe = st.dataframe(
            data=exp_df,
            hide_index=True,
            column_order=display_columns,
            selection_mode="single-row",
            on_select="rerun",
            column_config={
                "update": st.column_config.LinkColumn(
                    "update", help="Update the experiment", display_text="Update"
                ),
                "see_observations": st.column_config.LinkColumn(
                    "see_observations",
                    help="Explore the observations of the experiment",
                    display_text="See Observations",
                ),
            },
            key=st.session_state.get("exp_df_table_key", "experiments_table"),
        )
        if not displayed_dataframe.selection["rows"]:  # type: ignore[attr-defined]
            return cls()
        selected_idx = displayed_dataframe.selection["rows"][0]  # type: ignore[attr-defined]
        selected_exp_series = exp_df.iloc[selected_idx, :]
        return cls(selected_experiment=selected_exp_series)


class DeleteExperimentButton:
    """A class for rendering a button to delete an experiment."""

    @classmethod
    def render(cls, exp_row: pd.Series) -> None:
        """Renders a button to delete an experiment.

        Args:
            exp_row: Series containing experiment data to delete.
        """
        if st.button("Delete Experiment", type="primary"):

            @st.dialog("Delete Experiment?")
            def delete_experiment_dialog():
                st.write(exp_row)
                st.markdown(
                    f"If you delete this experiment, all observations will be deleted as well."
                    f"Are you sure you want to delete experiment **{exp_row['name']}**?"
                )
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("Delete", width="stretch", help="Delete this experiment", type="primary"):
                        st.session_state["exp_to_delete"] = exp_row
                        st.session_state["exp_df_table_key"] = str(uuid.uuid4())
                        st.rerun()
                with col2:
                    if st.button("Cancel", width="stretch", help="Cancel the deletion operation"):
                        st.rerun()

            delete_experiment_dialog()


class ExperimentDetailedInfo:
    """A class for rendering detailed information about a single experiment.

    This class provides a formatted display of all experiment details including settings,
    metrics, description, design, hypotheses, and conclusions.
    """

    @classmethod
    def render(cls, exp_row: pd.Series | None) -> None:
        """Renders detailed experiment information in a formatted container.

        Args:
            exp_row: Series containing experiment data to display.
        """
        if exp_row is None:
            return None
        with st.container(border=True):
            st.subheader("Experiment Details")
            markdown = f"""
#### Settings
| Exp Settings Type           | Value                             |
|-----------------------------|----------------------------------|
| **Experiment ID**           | {exp_row["id"]}              |
| **Name**                    | {exp_row["name"]}          |
| **Status**                  | {exp_row["status"]}        |
| **Start Date**              | `{exp_row["start_datetime"]}` |
| **End Date**                | `{exp_row["end_datetime"]}`  |
###### Key Metrics
{"".join([f"- `{metric}`\n" for metric in exp_row["key_metrics"]])}
#### Description
{exp_row["description"]}
#### Design
{exp_row["design"]}
#### Hypotheses
{exp_row["hypotheses"]}
#### Conclusion
{exp_row["conclusion"] if exp_row["conclusion"] else "_No conclusion yet._"}
                """
            st.markdown(markdown)
            DeleteExperimentButton.render(exp_row)
