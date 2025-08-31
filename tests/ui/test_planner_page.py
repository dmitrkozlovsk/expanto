from __future__ import annotations

from unittest.mock import patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def _clear_streamlit_caches():
    """Clear Streamlit caches before each test."""
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.fixture
def planner_page_app_test():
    from src.ui.state import AppContextManager

    """Create AppTest instance for planner page testing."""
    at = AppTest.from_file("src/ui/planner/page.py")
    AppContextManager.get_or_create_state()
    return at


def test_planner_page_no_db(planner_page_app_test, patch_configs, metrics):
    """Test planner page behavior when no database data is available."""
    at = planner_page_app_test
    at.run()
    assert "Could not get experiments" in at.error[0].value


def test_planner_page_initial_display(
    planner_page_app_test, patch_configs, tables, mock_experiments, metrics
):
    """Test planner page initial display and element setup."""
    at = planner_page_app_test
    at.run()
    assert "Select metrics to plan" in at.multiselect[0].label
    assert at.date_input[0].label == "Start"
    assert at.date_input[1].label == "End"
    assert at.date_input[2].label == "Start "
    assert at.date_input[3].label == "End "
    assert at.selectbox[1].label == "Exposure Event"
    assert at.selectbox[2].label == "Split ID"

    assert at.button[0].label == "Get Precomputes"
    assert "Fill in the fields and start observations" in at.info[0].value
    assert set(at.selectbox[0].options) == set([exp.name for exp in mock_experiments])
    at.selectbox[0].select((1, "Button Color Test")).run()
    exp_metrics = next(filter(lambda exp: exp.name == "Button Color Test", mock_experiments)).key_metrics
    assert set(at.multiselect[0].value) == set(exp_metrics)
    assert at.selectbox[1].value is None


def test_planner_page_get_precomputes_runner_error(
    planner_page_app_test, patch_configs, tables, mock_experiments
):
    """Test planner page error handling when getting precomputes fails."""
    at = planner_page_app_test
    at.run()

    at.selectbox[0].select((2, "Pricing Page Redesign")).run()
    at.button[0].click().run()
    assert "Please select at least one metric" in at.error[0].value


def test_planner_page_with_precomputes(planner_page_app_test, patch_configs, tables, mock_experiments):
    """Test planner page functionality with mocked precompute data."""
    at = planner_page_app_test

    mock_metric_results_data = [
        {
            "job_id": 1,
            "group_name": "control",
            "metric_name": "conversion_rate",
            "metric_display_name": "Conversion Rate",
            "metric_type": "proportion",
            "observation_cnt": 10000,
            "metric_value": 0.1,
            "numerator_avg": 0.1,
            "numerator_var": 0.09,
            "denominator_avg": 1,
            "denominator_var": 1,
            "covariance": 1,
        }
    ]

    # Mock the JobResult structure that get_precomputes_for_planning should return
    mock_job_result = {
        "job_id": 1,
        "success": True,
        "metric_results": mock_metric_results_data,
        "error_message": None,
    }

    mock_selection = {
        "selection": {
            "points": [
                {
                    "curve_number": 0,
                    "point_number": 5,
                    "x": 12589.254117941673,
                    "y": 0.05,
                    "customdata": 1.2589254117941674,
                }
            ]
        }
    }
    with patch(
        "src.ui.data_loaders.get_precomputes_for_planning",
        return_value=mock_job_result,
    ) as mock_get_precomputes:
        at.session_state["chat_enabled"] = False
        at.run()

        at.multiselect[0].select("conversion_rate").run()
        at.button[0].click().run()

        mock_get_precomputes.assert_called_once()

        assert not at.warning
        assert at.select_slider[0].label == "Alpha (Type I Error)"
        assert at.select_slider[1].label == "Beta (Type II Error)"
        assert at.slider[0].label == "Relative Effect Size (%)"

        # Find the plotly chart session state key
        chart_key = None
        for key in at.session_state._state._keys():
            if key.endswith("-effect_size_chart"):
                chart_key = key
                break
        if chart_key:
            at.session_state["effect_size_chart"] = mock_selection
        at.run()
        print(at)
        assert "### Selected Design" in at.markdown[-2].value
