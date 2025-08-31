"""Tests for UI common utilities."""

import streamlit as st
from streamlit.testing.v1 import AppTest

from src.domain.enums import PageMode
from src.ui.common import URLParams
import pytest

def url_params_app():
    """Test Streamlit app for URLParams parsing."""
    import streamlit as st
    from src.ui.common import URLParams
    
    result = URLParams.parse()
    st.session_state["parsed_url_params"] = result


@pytest.mark.parametrize(
    "query_params, expected_params",
    [
        ( # Happy path - all valid parameters
            {"observation_id": "123", "job_id": "456", "experiment_id": "789", "mode": "List", "submode": "details"}, 
            {"observation_id": 123, "job_id": 456, "experiment_id": 789, "mode": PageMode.LIST, "submode": "details"}
        ),
        ( # Partial parameters
            {"experiment_id": "42", "mode": "Create"}, 
            {"observation_id": None, "job_id": None, "experiment_id": 42, "mode": PageMode.CREATE, "submode": None}
        ),
        ( # Empty parameters
            {}, 
            {"observation_id": None, "job_id": None, "experiment_id": None, "mode": None, "submode": None}
        ),
        ( # Invalid integer parameters
            {"observation_id": "not_a_number", "job_id": "abc", "experiment_id": "12.5"}, 
            {"observation_id": None, "job_id": None, "experiment_id": None, "mode": None, "submode": None}
        ),
        ( # Mixed valid and invalid parameters
            {"observation_id": "123", "job_id": "invalid", "experiment_id": "456", "mode": "BadMode", "submode": "valid_sub"}, 
            {"observation_id": 123, "job_id": None, "experiment_id": 456, "mode": None, "submode": "valid_sub"}
        ),
        ( # Very long submode string
            {"submode": "a" * 1000}, 
            {"observation_id": None, "job_id": None, "experiment_id": None, "mode": None, "submode": "a" * 1000}
        ),
        ( # Minimum and maximum integer values
            {"observation_id": str(-2**63), "job_id": str(2**63 - 1)}, 
            {"observation_id": -2**63, "job_id": 2**63 - 1, "experiment_id": None, "mode": None, "submode": None}
        ),
        ( # Edge case: all fields with None-inducing values
            {"observation_id": "NaN", "job_id": "infinity", "experiment_id": "null", "mode": "undefined", "submode": None}, 
            {"observation_id": None, "job_id": None, "experiment_id": None, "mode": None, "submode": "None"}
        ),
    ]
)
def test_url_params_parse(query_params, expected_params):
    """Test URLParams.parse() with various parameter combinations and edge cases."""
    at = AppTest.from_function(url_params_app)
    at.query_params = query_params
    at.run()
    parsed_url_params = at.session_state["parsed_url_params"]
    assert parsed_url_params == URLParams(**expected_params)