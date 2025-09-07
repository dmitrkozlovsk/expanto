import math
from collections.abc import Callable

import numpy as np
import pytest
from scipy.stats import norm, ttest_ind  # type: ignore
from statsmodels.stats.power import TTestIndPower  # type: ignore
from statsmodels.stats.proportion import (  # type: ignore
    confint_proportions_2indep,  # type: ignore
    proportions_ztest,  # type: ignore
    samplesize_proportions_2indep_onetail,  # type: ignore
)  # type: ignore

from src.services.analytics.stat_functions import (
    ztest_proportion,
    sample_size_proportion_z_test
)


@pytest.fixture(
    params=[
        # Parameters: (mean1, var1, n1, mean2, var2, n2), expected p-value, comparison function
        [(0.5, 1000, 0.5, 1000), 1, lambda pv_calc, pv_expected: pv_calc == pv_expected],
        [(0.5, 1000, 0.6, 1000), 0.05, lambda pv_calc, pv_expected: pv_calc < pv_expected],
    ]
)
def from_stats_ztest(request):
    """Fixture providing parameters for Welch's t-test and expected outcomes."""
    return request.param


@pytest.fixture(
    params=[
        # Parameters:p_1, n_1, p_2, n_2, seed
        (0.5, 100, 0.55, 100, 42),
        (0.001, 2000, 0.0011, 3000, 1111),
        (0.15, 5000, 0.159, 25000, None),
    ]
)
def generate_two_samples_ztest(request) -> Callable:
    """Fixture generating two random samples based on provided statistics."""
    p_1, n_1, p_2, n_2, seed = request.param

    def _generate_two_samples():
        if seed is not None:
            np.random.seed(seed)
        data1 = np.random.binomial(n=1, p=p_1, size=n_1)
        data2 = np.random.binomial(n=1, p=p_2, size=n_2)
        return data1, data2

    return _generate_two_samples


def test_ztest_proportion_simple(from_stats_ztest):
    """Test basic z-test proportion functionality."""
    samples_params, pv_expected, func = from_stats_ztest
    ztest_p_value = ztest_proportion(*samples_params).p_value
    assert func(ztest_p_value, pv_expected)


def test_ztest_proportion_vs_proportions_ztest(generate_two_samples_ztest):
    """Compare ztest results with statsmodels.stats.proportion.proportions_ztest."""

    data_1, data_2 = generate_two_samples_ztest()

    p_1, n_1, p_2, n_2 = np.mean(data_1), len(data_1), np.mean(data_2), len(data_2)
    count, nobs = [np.sum(data_1), np.sum(data_2)], [len(data_1), len(data_2)]

    # test p_value
    my_ztest_proportion_result = ztest_proportion(p_1, n_1, p_2, n_2)
    stat, pval = proportions_ztest(count, nobs)
    assert math.isclose(my_ztest_proportion_result.p_value, pval, abs_tol=0.00001)

    # test conf int
    sttsmdls_ci_lower, sttsmdls_ci_upper = confint_proportions_2indep(
        count[1], nobs[1], count[0], nobs[0], method="wald", compare="diff"
    )
    assert math.isclose(
        sttsmdls_ci_lower, my_ztest_proportion_result.ci.lower, abs_tol=0.0001, rel_tol=0.0001
    )

#------------------------------CORNER CASE---------------------------------



#------------------------------SAMPLE SIZE---------------------------------
@pytest.mark.parametrize(
    "p1, effect_size, alpha, beta",
    [
        (0.5, 0.1, 0.05, 0.2),
        (0.3, 0.2, 0.1, 0.2),
        (0.1, 0.5, 0.01, 0.05),
        (0.05, 0.4, 0.1, 0.2),
        (0.8, 0.05, 0.05, 0.1),
        (0.179, 0.02, 0.01, 0.05),
    ],
)
def test_sample_size_proportion_z_test_basic(p1, effect_size, alpha, beta):
    """Test sample_size_proportion_z_test function with basic test cases."""
    # Calculate sample size using our function
    n_required = sample_size_proportion_z_test(p1, effect_size, alpha, beta)

    # Calculate expected difference
    p2 = p1 + (p1 * effect_size)
    diff = p2 - p1

    # Calculate sample size using statsmodels
    power = 1 - beta
    n_expected = samplesize_proportions_2indep_onetail(
        diff=diff, prop2=p1, power=power, alpha=alpha, alternative="two-sided"
    )

    # Allow for small differences due to rounding and different calculation methods
    # Sample size calculations can vary slightly between implementations
    assert abs(n_required - n_expected) <= max(3, 0.005 * n_expected)

def test_sample_size_proportion_z_test_value_errors():
    """Test that sample_size_proportion_z_test raises appropriate ValueErrors for invalid inputs."""

    with pytest.raises(ValueError, match="p1 values must be in range"):
        sample_size_proportion_z_test(1, 0.1, 0.05, 0.2)  # p1 = 1

    # Test p2 value error - effect_size must not make p2 outside (0, 1)
    with pytest.raises(ValueError, match="p2 values must be in range"):
        sample_size_proportion_z_test(0.5, 1.1, 0.05, 0.2)  # p2 > 1

def test_sample_size_proportion_z_test_with_arrays():
    """Test that sample_size_proportion_z_test works correctly with array inputs."""
    # Create test arrays
    p1_array = np.array([0.1, 0.3, 0.5, 0.7])
    effect_size_array = np.array([0.2, 0.1, 0.15, 0.05])
    alpha = 0.05
    beta = 0.2
    power = 1 - beta

    # Calculate p2 and diff for statsmodels
    p2_array = p1_array + (p1_array * effect_size_array)
    diff_array = p2_array - p1_array

    # Run our function with array inputs
    our_sample_sizes = sample_size_proportion_z_test(p1_array, effect_size_array, alpha, beta)

    # Calculate expected sample sizes using statsmodels
    expected_sample_sizes = []
    for p1, diff in zip(p1_array, diff_array, strict=False):
        n_expected = samplesize_proportions_2indep_onetail(
            diff=diff, prop2=p1, power=power, alpha=alpha, alternative="two-sided"
        )
        expected_sample_sizes.append(n_expected)

    expected_sample_sizes = np.array(expected_sample_sizes)

    # Compare array calculation results with expected values
    for i in range(len(p1_array)):
        sample_size_diff = abs(our_sample_sizes[i] - expected_sample_sizes[i])
        assert sample_size_diff <= max(3, 0.005 * expected_sample_sizes[i]), (
            f"Values differ at index {i}: ours={our_sample_sizes[i]}, expected={expected_sample_sizes[i]}"
        )


