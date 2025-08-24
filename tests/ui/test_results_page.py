from __future__ import annotations

import pytest
import streamlit as st
from sqlalchemy import select
from streamlit.testing.v1 import AppTest

from src.domain.models import CalculationJob


@pytest.fixture(autouse=True)
def _clear_streamlit_caches():
    """Clear Streamlit caches before each test."""
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.fixture
def results_page_app_test():
    from src.ui.state import AppContextManager
    """Create AppTest instance for results page testing."""
    at = AppTest.from_file("src/ui/results/page.py")
    AppContextManager.get_or_create_state()
    return at


def test_results_page_no_data(results_page_app_test, patch_configs):
    """Test results page behavior when no data is available."""
    at = results_page_app_test
    at.run()
    assert at.selectbox[0].label == "Select observation"
    assert at.selectbox[1].label == "Select job"
    assert "Could not get observations:" in at.error[0].value


def test_result_page(
    results_page_app_test,
    patch_configs,
    tables,
    mock_experiments,
    mock_observations,
    mock_calculation_jobs,
    mock_precomputes,
):
    """Test results page with complete data setup."""
    at = results_page_app_test
    at.run()
    at.selectbox[0].select((1, 1, "Main Conversion Analysis")).run()
    assert at.selectbox[2].value == "control"
    assert at.multiselect[0].value == ["treatment"]
    assert at.button_group[0].options[0].content == "5%"
    assert "Metric groups" in at.multiselect[1].label
    assert "Metric tags" in at.multiselect[2].label


def test_observation_id_in_query_params(
    results_page_app_test,
    patch_configs,
    tables,
    mock_experiments,
    mock_observations,
    mock_calculation_jobs,
):
    """Test results page with observation_id in URL query parameters."""
    at = results_page_app_test
    at.query_params["observation_id"] = "2"
    at.run()

    from src.ui.data_loaders import get_jobs_by_observation_id

    jobs = get_jobs_by_observation_id(2)

    assert at.selectbox[0].label == "Select observation"
    assert at.selectbox[0].value == (2, 1, "Mobile Users Analysis")
    assert at.selectbox[1].label == "Select job"
    assert at.selectbox[1].options[0][-2] == str(jobs[0][0])  # job number as string


def test_job_id_in_query_params(
    results_page_app_test,
    patch_configs,
    tables,
    mock_experiments,
    mock_observations,
    mock_calculation_jobs,
):
    """Test results page with job_id in URL query parameters."""
    at = results_page_app_test
    at.query_params["job_id"] = "4"
    at.run()

    assert at.selectbox[0].label == "Select observation"
    assert at.selectbox[0].value == (3, 2, "Pricing Page Analysis")
    assert at.selectbox[1].label == "Select job"
    assert at.selectbox[1].value is None
    assert at.info[0].value == "No jobs selected. Please select a job."

    at.query_params["job_id"] = "3"
    at.run()

    from src.ui.data_loaders import get_jobs_by_observation_id

    jobs = get_jobs_by_observation_id(2)

    assert at.selectbox[0].label == "Select observation"
    assert at.selectbox[0].value == (2, 1, "Mobile Users Analysis")
    assert at.selectbox[1].label == "Select job"
    assert at.selectbox[1].options[0][-2] == str(jobs[0][0])  # job number as string


def test_run_calculation(
    results_page_app_test,
    patch_configs,
    tables,
    mock_experiments,
    mock_observations,
    mock_calculation_jobs,
    queries_templates_config,
    patch_client_and_resolver,
    session,
):
    """Test calculation execution from results page."""
    len_jobs_before = len(session.execute(select(CalculationJob)).fetchall())
    at = results_page_app_test
    at.session_state["obs_to_be_calculated"] = {"observation_id": 1, "calc_type": "foreground"}
    at.run()
    session.expire_all()
    jobs = session.execute(select(CalculationJob)).fetchall()
    assert len(jobs) == len_jobs_before + 1
