from __future__ import annotations

from datetime import UTC, datetime, timedelta

import streamlit as st
from sqlalchemy import select
from streamlit.testing.v1 import AppTest

from src.domain.models import Observation
from src.ui.common import URLParams


def list_page(url_exp_id: int | None = None):
    """Render observation list page for testing."""
    from src.ui.observations.subpages import ObservationListPage

    return ObservationListPage.render(url_exp_id)


def create_page(url_params):
    """Render observation creation page for testing."""
    from src.ui.observations.subpages import CreateObservationPage

    return CreateObservationPage.render(url_params)


def update_page(url_params):
    """Render observation update page for testing."""
    from src.ui.observations.subpages import UpdateObservationPage

    return UpdateObservationPage.render(url_params)


def test_list_observation_no_obs(patch_configs):
    """Test observation list page when no observations exist."""
    at = AppTest.from_function(list_page)
    at.run()
    # elements check
    assert "experiment" in at.selectbox[0].label
    assert "Observation Name" in at.text_input[0].label
    assert "Limit" in at.number_input[0].label
    today_utc = datetime.now(UTC).date()
    assert at.date_input[1].value == today_utc + timedelta(days=1)

    # errors check
    assert "Could not get experiments:" in at.error[0].value
    assert "Could not get observations:" in at.error[1].value


def test_create_observation_no_obs(patch_configs):
    """Test observation creation page when no experiments exist."""
    at = AppTest.from_function(create_page, kwargs={"url_params": URLParams()})
    at.run()
    # elements check
    assert "experiment" in at.selectbox[0].label
    # errors check
    assert "Could not get experiments:" in at.error[0].value
    assert "Please select an experiment" in at.info[0].value


def test_update_observation_no_obs(patch_configs):
    """Test observation update page when no data exists."""
    at = AppTest.from_function(update_page, kwargs={"url_params": URLParams()})
    at.run()
    # elements check
    assert "experiment" in at.selectbox[0].label
    assert "observation" in at.selectbox[1].label
    # errors check
    assert "Could not get experiments:" in at.error[0].value
    assert "Could not get observations:" in at.error[1].value
    assert "Please select an observation" in at.info[0].value


def test_list_observation(tables, mock_experiments, mock_observations, session, patch_configs):
    """Test observation list page with existing observations."""
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_function(list_page)
    at.run()
    assert at.dataframe[0].value.shape[0] == 3


def test_delete_observation(tables, mock_experiments, mock_observations, session, patch_configs):
    """Test observation deletion functionality."""
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_function(list_page)
    at.run()
    df = at.dataframe[0].value
    row = df.iloc[0]
    at.session_state["obs_to_delete"] = row
    at.run()
    session.expire_all()
    observations = session.execute(select(Observation)).fetchall()
    assert len(observations) == 2
    assert at.dataframe[0].value.shape[0] == 2


def test_copy_observation(tables, mock_experiments, mock_observations, session, patch_configs):
    """Test observation copying functionality."""
    st.cache_resource.clear()
    st.cache_data.clear()

    at_list = AppTest.from_function(list_page)
    at_list.run()
    row = at_list.dataframe[0].value.iloc[0]
    obs_to_copy = Observation(
        **row[
            [
                "name",
                "id",
                "experiment_id",
                "db_experiment_name",
                "split_id",
                "calculation_scenario",
                "exposure_start_datetime",
                "exposure_end_datetime",
                "calc_start_datetime",
                "calc_end_datetime",
            ]
        ].to_dict()
    )

    params = URLParams(mode="Create", submode="Copy")
    at_create = AppTest.from_function(create_page, kwargs={"url_params": params})
    at_create.session_state["obs_to_copy"] = obs_to_copy
    at_create.run()
    assert at_create.text_input[0].value == obs_to_copy.name
    assert at_create.text_input[1].value == obs_to_copy.db_experiment_name
    assert at_create.text_input[2].value == obs_to_copy.split_id
    assert at_create.selectbox[1].value == obs_to_copy.calculation_scenario
    assert at_create.date_input[0].value == obs_to_copy.exposure_start_datetime.date()
    assert at_create.date_input[1].value == obs_to_copy.exposure_end_datetime.date()
    assert at_create.date_input[2].value == obs_to_copy.calc_start_datetime.date()
    assert at_create.date_input[3].value == obs_to_copy.calc_end_datetime.date()
    new_name = obs_to_copy.name + "copy"
    at_create.text_input[0].set_value(new_name).run()
    at_create.button[0].click().run()
    session.expire_all()
    observations = session.execute(select(Observation)).fetchall()
    assert len(observations) == 4
    assert observations[-1][-1].name == new_name


def test_create_observation(tables, mock_experiments, mock_observations, session, patch_configs):
    """Test successful observation creation."""
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_function(create_page, kwargs={"url_params": URLParams()})
    at.run()
    at.selectbox[0].select((1, "Pricing Page Redesign")).run()

    obs_name = "test_create_observation"
    at.text_input[0].set_value(obs_name).run()
    at.text_input[1].set_value("db_exp_name").run()
    at.text_input[2].set_value("split_id").run()

    at.button[0].click().run()
    observations = session.execute(select(Observation)).fetchall()
    assert len(observations) == 4
    assert observations[-1][-1].name == obs_name


def test_update_observation(tables, mock_experiments, mock_observations, session, patch_configs):
    """Test successful observation update."""
    st.cache_resource.clear()  # need to cache to reload configs
    st.cache_data.clear()
    at = AppTest.from_function(update_page, kwargs={"url_params": URLParams()})
    at.run()

    at.selectbox[0].select((1, "Pricing Page Redesign")).run()  # select experiment
    at.selectbox[1].select((1, 1, "Pricing Page Analysis")).run()  # select observation

    # updating fields
    at.text_input[0].set_value("updated_obs").run()
    at.button[0].click().run()

    session.expire_all()
    observations = session.execute(select(Observation)).fetchall()
    obs = next(obs[0] for obs in observations if obs[0].name == "updated_obs")
    assert obs.name == "updated_obs"


def test_update_observation_with_args(tables, mock_experiments, mock_observations, session, patch_configs):
    """Test observation update with URL parameters."""
    st.cache_resource.clear()
    st.cache_data.clear()
    params = URLParams(mode="Update", observation_id=3)
    at = AppTest.from_function(update_page, kwargs={"url_params": params})
    at.run()
    at.selectbox[0].select((1, "Pricing Page Redesign")).run()  # select experiment

    assert at.selectbox[1].value == (3, 2, "Pricing Page Analysis")
    new_name = "updated_obs"
    at.text_input[0].set_value(new_name).run()
    at.button[0].click().run()
    session.expire_all()
    observations = session.execute(select(Observation).where(Observation.id == 3)).fetchall()
    assert observations[0][0].name == new_name
