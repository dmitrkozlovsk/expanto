"""This module contains UI elements for displaying observations in Streamlit."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import pandas as pd  # type: ignore
import streamlit as st

from src.domain.models import Observation
from src.ui.actions import generate_hyperlinks_for_observations_df


@dataclass
class ObservationsDataframe:
    """A dataclass to manage and display a dataframe of observations.

    This class handles the rendering of an observations table in Streamlit,
    allowing for single-row selection.

    Attributes:
        selected_observation: A pandas Series representing the selected row in the
            observation dataframe, or None if no row is selected.
    """

    selected_observation: pd.Series | None = None

    @classmethod
    def render(cls, obs_df: pd.DataFrame) -> ObservationsDataframe:
        """Renders the observation table and handles row selection.

        If the provided DataFrame is empty, a warning is displayed. Otherwise,
        it generates hyperlinks for the dataframe, displays it with specific
        columns and configurations, and captures user selection.

        Args:
            obs_df: A pandas DataFrame containing the observations to display.

        Returns:
            An instance of `ObservationsDataframe` containing the selected
            observation data as a pandas Series. If no row is selected,
            `selected_observation` will be None.
        """
        if obs_df.empty:
            st.warning("Observation table is empty")
            return cls()

        obs_df = generate_hyperlinks_for_observations_df(obs_df)
        display_columns = [
            "see_results",
            "update",
            "name",
            "calculation_scenario",
            "split_id",
            "filters",
            "audience_tables",
            "metric_tags",
            "metric_groups",
        ]

        # Display the dataframe with selection enabled
        displayed_dataframe = st.dataframe(
            obs_df,
            hide_index=True,
            column_order=display_columns,
            selection_mode="single-row",
            on_select="rerun",
            column_config={
                "update": st.column_config.LinkColumn(
                    "update", help="Update the observation", display_text="Update"
                ),
                "see_results": st.column_config.LinkColumn(
                    "see_results",
                    help="View metrics for this observation",
                    display_text="See Results",
                ),
            },
            key=st.session_state.get("obs_df_table_key", "observations_table"),
        )
        if not displayed_dataframe.selection["rows"]:  # type: ignore[attr-defined]
            return cls()
        selected_idx = displayed_dataframe.selection["rows"][0]  # type: ignore[attr-defined]
        selected_obs_series = obs_df.loc[
            obs_df.index[selected_idx],
            [col for col in obs_df.columns if col not in ["update", "see_results"]],
        ]
        return cls(selected_observation=selected_obs_series)


@dataclass
class DeleteObservationButton:
    """A class for rendering a button to delete an observation."""

    @classmethod
    def render(cls, obs_row: pd.Series) -> None:
        """Renders a button to delete an observation.

        When the button is clicked, a confirmation dialog appears. If the user
        confirms the deletion, the observation's data is stored in the session
        state to be handled by the page logic, and the UI is updated.

        Args:
            obs_row: A pandas Series representing the observation to be deleted.
                It must contain an 'id' and 'name' field.
        """
        if st.button("Delete Observation", type="primary", key=f"delete_obs_{obs_row['id']}"):

            @st.dialog("Delete Observation?")
            def delete_observation_dialog() -> None:
                st.write(obs_row)
                st.markdown(f"Are you sure you want to delete observation **{obs_row['name']}**?")
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("Delete", width="stretch", help="Delete this observation", type="primary"):
                        st.session_state["obs_to_delete"] = obs_row
                        st.session_state["obs_df_table_key"] = str(uuid.uuid4())
                        st.rerun()
                with col2:
                    if st.button("Cancel", width="stretch", help="Cancel the deletion operation"):
                        st.rerun()

            delete_observation_dialog()


@dataclass
class ObservationDetailedInfo:
    """A class to display detailed information about a single observation."""

    @classmethod
    def render(cls, obs_row: pd.Series) -> None:
        """Renders a container with detailed information about an observation.

        This view includes action buttons (e.g., "See Results", "Copy"), and
        a formatted display of the observation's properties such as its ID, name,
        dates, and configuration. It also includes a button to delete the
        observation.

        Args:
            obs_row: A pandas Series containing the data of the observation to
                display. If None, nothing is rendered.
        """
        if obs_row is None:
            return None
        with st.container(border=True):
            # Add buttons in a row at the top
            col1, col2, col3 = st.columns([1, 1, 1], gap="small", vertical_alignment="bottom")
            with col1:
                st.link_button(
                    "See Results",
                    f"/results?observation_id={obs_row['id']}",
                    width="stretch",
                    help="See results for this observation",
                )
            with col2:
                if st.button(
                    "Copy",
                    key=f"copy_obs_{obs_row['id']}",
                    width="stretch",
                    help="Copy this observation and create new one",
                ):
                    st.session_state["obs_to_copy"] = Observation(**obs_row.to_dict())
                    st.query_params["mode"] = "Create"
                    st.query_params["submode"] = "Copy"
                    st.query_params["ts"] = str(time.time())
                    st.rerun()
            with col3:
                pass
                # TODO: Implement precompute calculation logic

            st.subheader("Observation Details")

            md = f"""
#### Settings
| Observation Settings Type | Value                             |
|---------------------------|----------------------------------|
| **Observation ID**        | {obs_row["id"]}         |
| **Name**                  | {obs_row["name"]}       |
| **Experiment ID**         | {obs_row["experiment_id"]} |
| **Calculation Scenario**  | {obs_row["calculation_scenario"]} |
| **Split ID**              | {obs_row["split_id"]}   |
| **Created At**            | `{obs_row["_created_at"]}` |

#### Exposure Period
- **Start**: `{obs_row["exposure_start_datetime"]}`
- **End**: `{obs_row["exposure_end_datetime"]}`

#### Calculation Period
- **Start**: `{obs_row["calc_start_datetime"]}`
- **End**: `{obs_row["calc_end_datetime"]}`

#### Filters and Segments
- **Filters**: {obs_row["filters"] if obs_row["filters"] else "_None_"}
- **Audience Tables**: {obs_row["audience_tables"] if obs_row["audience_tables"] else "_None_"}
- **Metric Tags**: {obs_row["metric_tags"] if obs_row["metric_tags"] else "_None_"}
- **Metric Groups**: {obs_row["metric_groups"] if obs_row["metric_groups"] else "_None_"}

#### Custom Test IDs Query
```sql
{obs_row["custom_test_ids_query"] if obs_row["custom_test_ids_query"] else "_None_"}
```
                """
            st.markdown(md)
            DeleteObservationButton.render(obs_row)
