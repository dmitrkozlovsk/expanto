import pytest

from src.services.analytics.stat_functions import sample_ratio_mismatch_test


@pytest.mark.parametrize(
    "observed_counts, expected_ratios, expected_is_srm, p_value_threshold",
    [
        ([1000, 1010], None, False, 0.001),
        ([1000, 1200], None, True, 0.001),
        ([100, 205], [1 / 3, 2 / 3], False, 0.001),
        ([300, 700], [0.25, 0.75], True, 0.001),
    ],
)
def test_sample_ratio_mismatch_happy_path(
    observed_counts, expected_ratios, expected_is_srm, p_value_threshold
):
    result = sample_ratio_mismatch_test(observed_counts, expected_ratios, alpha=p_value_threshold)
    assert result.is_srm is expected_is_srm
    if expected_is_srm:
        assert result.p_value < p_value_threshold
    else:
        assert result.p_value >= p_value_threshold


@pytest.mark.parametrize(
    "observed_counts, expected_ratios, error_message",
    [
        ([100], None, "observed_counts must contain at least 2 groups"),
        ([100, -10], None, "All observed counts must be non-negative"),
        ([100, 100], [0.5, 0.4, 0.1], "expected_ratios must have the same length as observed_counts"),
        ([100, 100], [0.6, 0.6], "expected_ratios must sum to 1"),
        ([0, 0], None, "Total count must be positive"),
    ],
)
def test_sample_ratio_mismatch_corner_cases(observed_counts, expected_ratios, error_message):
    """Tests corner cases and invalid inputs for sample_ratio_mismatch_test."""
    with pytest.raises(ValueError, match=error_message):
        sample_ratio_mismatch_test(observed_counts, expected_ratios)
