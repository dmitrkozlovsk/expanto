from __future__ import annotations

from datetime import UTC, datetime, timedelta

import streamlit as st
from sqlalchemy import select
from streamlit.testing.v1 import AppTest

from src.domain.models import Experiment


def list_page():
    """Render experiment list page for testing."""
    from src.ui.experiments.subpages import ExperimentListPage

    return ExperimentListPage.render()


def create_page():
    """Render experiment creation page for testing."""
    from src.ui.experiments.subpages import CreateExperimentPage

    return CreateExperimentPage.render()


def update_page(url_exp_id: int | None = None):
    """Render experiment update page for testing."""
    from src.ui.experiments.subpages import UpdateExperimentPage

    return UpdateExperimentPage.render(url_exp_id)


def test_list_experiment_without_experiments(patch_configs):
    """Test experiment list page behavior when no experiments exist."""
    at = AppTest.from_function(list_page)
    at.run()
    assert "Filter experiment" in at.text_input[0].help
    today_utc = datetime.now(UTC).date()
    assert at.date_input[1].value == today_utc + timedelta(days=2)
    assert set(at.multiselect[0].value) == {"planned", "running"}
    assert at.number_input[0]
    assert "Could not get experiments:" in at.error[0].value


def test_list_experiment(tables, mock_experiments, session, patch_configs):
    """Test experiment list page with existing experiments and filtering."""
    st.cache_resource.clear()  # need to cache to reload configs
    st.cache_data.clear()
    at = AppTest.from_function(list_page)
    at.run()
    assert at.dataframe[0].value.shape[0] == 1
    assert at.multiselect[0].select("completed")
    at.run()
    assert at.dataframe[0].value.shape[0] == 2


def test_create_experiment_errors(patch_configs):
    """Test experiment creation validation errors."""
    st.cache_resource.clear()  # need to cache to reload configs
    st.cache_data.clear()
    at = AppTest.from_function(create_page)
    at.run()
    assert "Experiment Name" in at.text_input[0].label
    assert at.selectbox[0].value == "planned"
    assert "Description" in at.text_area[0].label
    assert "Hypotheses" in at.text_area[1].label
    assert at.multiselect[0].label == "Key Metrics"
    today_utc = datetime.now(UTC).date()
    assert today_utc == at.date_input[0].value
    assert at.button[0].click().run()
    assert "⚠️The required field `Name`" in at.toast[0].value


def test_create_experiment(tables, mock_experiments, session, patch_configs):
    """Test successful experiment creation."""
    st.cache_resource.clear()  # need to cache to reload configs
    st.cache_data.clear()
    at = AppTest.from_function(create_page)
    at.run()
    exp_name = "test_create_experiment"
    at.text_input[0].set_value(exp_name)
    at.button[0].click().run()
    experiments = session.execute(select(Experiment)).fetchall()
    assert len(experiments) == 3
    assert experiments[-1][-1].name == exp_name


def test_update_experiment_error(patch_configs):
    """Test experiment update page error handling when no experiments exist."""
    at = AppTest.from_function(update_page)
    at.run()
    assert "Could not get experiments" in at.error[0].value
    assert "Select experiment" in at.selectbox[0].label


def test_update_experiment(tables, mock_experiments, session, patch_configs):
    """Test successful experiment update with field modifications."""
    st.cache_resource.clear()  # need to cache to reload configs
    st.cache_data.clear()
    at = AppTest.from_function(update_page)
    at.run()
    at.selectbox[0].select((1, "Pricing Page Redesign")).run()

    # updating fields
    at.text_input[0].set_value("0").run()  # name
    at.selectbox[1].select("paused").run()  # status
    at.text_area[0].set_value("0").run()  # Description
    at.text_area[1].set_value("0").run()  # Hypothesis
    today_utc_plus_1 = datetime.now(UTC).date() + timedelta(days=1)
    at.date_input[0].set_value(today_utc_plus_1).run()  # start
    at.button[0].click().run()
    session.expire_all()
    experiments = session.execute(select(Experiment)).fetchall()
    exp = next(exp[0] for exp in experiments if exp[0].name == "0")
    assert exp.name == "0"
    assert exp.status == "paused"
    assert exp.description == "0"
    assert exp.hypotheses == "0"
    assert today_utc_plus_1 == exp.start_datetime.date()


def test_update_experiment_query_param(tables, mock_experiments, session, patch_configs):
    """Test experiment update page with URL query parameters."""
    st.cache_resource.clear()  # need to cache to reload configs
    st.cache_data.clear()
    at = AppTest.from_function(update_page, kwargs={"url_exp_id": 2})
    at.run()
    assert at.selectbox[0].value[0] == 2
    at = AppTest.from_function(update_page, kwargs={"url_exp_id": 1})
    at.run()
    assert at.selectbox[0].value[0] == 1
    at = AppTest.from_function(update_page, kwargs={"url_exp_id": 9})
    at.run()
    assert at.selectbox[0].value is None


def test_delete_experiment(tables, mock_experiments, session, patch_configs):
    """Test experiment deletion functionality."""
    st.cache_resource.clear()
    st.cache_data.clear()
    at = AppTest.from_function(list_page)
    at.run()
    at.multiselect[0].select("completed")
    at.multiselect[0].select("running")
    at.run()

    # Verify initial state
    assert at.dataframe[0].value.shape[0] == 2
    df = at.dataframe[0].value
    row = df.iloc[0]

    at.session_state["exp_to_delete"] = row
    at.run()

    session.expire_all()
    experiments = session.execute(select(Experiment)).fetchall()
    assert len(experiments) == 1  # One experiment should remain

    at.run()
    assert at.dataframe[0].value.shape[0] == 1  # Table should show one less experiment
