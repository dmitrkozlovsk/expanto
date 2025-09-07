import pytest
from src.services.analytics.stat_functions import ttest_welch, sample_size_t_test
import numpy as np
import math
from collections.abc import Callable

from scipy.stats import norm, ttest_ind  # type: ignore
from statsmodels.stats.power import TTestIndPower  # type: ignore
from statsmodels.stats.proportion import (  # type: ignore
    confint_proportions_2indep,  # type: ignore
    proportions_ztest,  # type: ignore
    samplesize_proportions_2indep_onetail,  # type: ignore
)  # type: ignore


# ----------------------------- TESTING T-TEST FUNCTION -----------------------------
@pytest.fixture(
    params=[
        # Parameters: (mean1, var1, n1, mean2, var2, n2), expected p-value, comparison function
        [(5, 2, 10, 5, 2, 10), 1, lambda pv_calc, pv_expected: pv_calc == pv_expected],
        [(2, 1, 50, 3, 2, 40), 0.05, lambda pv_calc, pv_expected: pv_calc < pv_expected],
    ]
)
def from_stats_ttest(request):
    """Fixture providing parameters for Welch's t-test and expected outcomes."""
    return request.param


@pytest.fixture(
    params=[
        # Parameters: mean1, var1, n1, mean2, var2, n2, seed
        (10, 20, 30, 10, 20, 30, 42),
        (0.01, 18, 55, 0.05, 19, 60, 1111),
        (0.84, 2, 10000, 0.89, 3, 10000, None),
    ]
)
def generate_two_samples_ttest(request) -> Callable:
    """Fixture generating two random samples based on provided statistics."""
    mean_1, var_1, n_1, mean_2, var_2, n_2, seed = request.param

    def _generate_two_samples():
        if seed is not None:
            np.random.seed(seed)
        data1 = np.random.normal(loc=mean_1, scale=np.sqrt(var_1), size=n_1)
        data2 = np.random.normal(loc=mean_2, scale=np.sqrt(var_2), size=n_2)
        return data1, data2

    return _generate_two_samples


def test_ttest_welch_simple(from_stats_ttest):
    """Test ttest_welch against precomputed p-values with defined comparison core."""
    samples_params, pv_expected, func = from_stats_ttest
    pv_calc = ttest_welch(*samples_params).p_value
    assert func(pv_calc, pv_expected)


def test_ttest_welch_vs_ttest_ind(generate_two_samples_ttest):
    """Compare ttest_welch results with scipy.stats.ttest_ind."""

    data_1, data_2 = generate_two_samples_ttest()

    mean_1, var_1, n_1, mean_2, var_2, n_2 = (
        np.mean(data_1),
        np.var(data_1, ddof=1),
        len(data_1),
        np.mean(data_2),
        np.var(data_2, ddof=1),
        len(data_2),
    )

    # test p_value
    ttest_welch_p_value = ttest_welch(mean_1, var_1, n_1, mean_2, var_2, n_2).p_value
    ttest_ind_p_value = ttest_ind(data_1, data_2, equal_var=False).pvalue
    assert math.isclose(ttest_welch_p_value, ttest_ind_p_value, abs_tol=0.00001)

    # test conf int
    scipy_ci_lower, scipy_ci_upper = ttest_ind(data_2, data_1, equal_var=False).confidence_interval()
    welch_result = ttest_welch(mean_1, var_1, n_1, mean_2, var_2, n_2)
    welch_ci_lower, welch_ci_upper = welch_result.ci.lower, welch_result.ci.upper
    assert math.isclose(scipy_ci_lower, welch_ci_lower, abs_tol=0.0001, rel_tol=0.0001)
    assert math.isclose(scipy_ci_upper, welch_ci_upper, abs_tol=0.0001, rel_tol=0.0001)

#------------------------------CORNER CASE---------------------------------

def test_t_test_zero_variance_no_warnings(recwarn):
    """Test that ttest_welch does not raise warnings when the variance is zero."""

    test = ttest_welch(
        mean_1=np.array([5]),
        var_1=np.array([0]),
        n_1=np.array([100]),
        mean_2=np.array([5]),
        var_2=np.array([0]),
        n_2=np.array([100]),
    )
    assert len(recwarn) == 0
    assert np.all(np.isnan(test.p_value))
    assert np.all(np.isnan(test.statistic))




#------------------------------SAMPLE SIZE---------------------------------
@pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.ConvergenceWarning")
@pytest.mark.parametrize(
    "avg1, var, effect_size, alpha, beta",
    [
        (100.0, 1.0, 0.2, 0.05, 0.2),  # Basic case
        (50.0, 2.0, 0.5, 0.01, 0.1),  # Larger effect size, more stringent significance
        (10.0, 0.5, 0.1, 0.1, 0.2),  # Smaller values
        (1000.0, 100.0, 0.01, 0.05, 0.1),  # Large values, small effect
        (0.1, 0.01, 0.5, 0.05, 0.2),  # Small values, large effect
    ],
)
def test_sample_size_t_test_basic(avg1, var, effect_size, alpha, beta):
    """Test sample_size_t_test function with basic test cases."""
    # Calculate sample size using our function
    n_required = sample_size_t_test(avg1=avg1, var=var, effect_size=effect_size, alpha=alpha, beta=beta)

    # Calculate expected sample size using statsmodels
    analysis = TTestIndPower()
    effect_size_cohen = effect_size * avg1 / np.sqrt(var)  # Convert to Cohen's d
    power = 1 - beta
    n_expected = analysis.solve_power(effect_size=effect_size_cohen, power=power, alpha=alpha, ratio=1.0)

    # Allow for small differences due to different calculation methods
    # Sample size calculations can vary slightly between implementations
    assert abs(n_required - n_expected) <= max(3, 0.05 * n_expected)

@pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.ConvergenceWarning")
def test_sample_size_t_test_with_arrays():
    """Test that sample_size_t_test works correctly with array inputs."""
    # Create test arrays
    avg1_array = np.array([100.0, 50.0, 10.0])
    var_array = np.array([1.0, 2.0, 0.5])
    effect_size_array = np.array([0.2, 0.5, 0.1])
    alpha = 0.05
    beta = 0.2

    # Run our function with array inputs
    our_sample_sizes = sample_size_t_test(
        avg1=avg1_array, var=var_array, effect_size=effect_size_array, alpha=alpha, beta=beta
    )

    # Calculate expected sample sizes using statsmodels
    from statsmodels.stats.power import TTestIndPower

    analysis = TTestIndPower()
    power = 1 - beta

    expected_sample_sizes = []
    for avg1, var, effect_size in zip(avg1_array, var_array, effect_size_array, strict=True):
        effect_size_cohen = effect_size * avg1 / np.sqrt(var)
        n_expected = analysis.solve_power(
            effect_size=effect_size_cohen, power=power, alpha=alpha, ratio=1.0
        )
        if isinstance(n_expected, np.ndarray):
            n_expected = float(n_expected[0])
        expected_sample_sizes.append(n_expected)

    expected_sample_sizes = np.array(expected_sample_sizes)

    # Compare array calculation results with expected values
    for i in range(len(avg1_array)):
        sample_size_diff = abs(our_sample_sizes[i] - expected_sample_sizes[i])
        assert sample_size_diff <= max(3, 0.05 * expected_sample_sizes[i]), (
            f"Values differ at index {i}: ours={our_sample_sizes[i]}, expected={expected_sample_sizes[i]}"
        )


def test_sample_size_t_test_edge_cases():
    """Test sample_size_t_test with edge cases."""
    # Test edge case when effect size is 0
    with pytest.raises(ValueError, match="Effect size cannot be zero"):
        sample_size_t_test(avg1=100.0, var=1.0, effect_size=0.0, alpha=0.05, beta=0.2)
    with pytest.raises(ValueError, match="Effect size cannot be zero"):
        sample_size_t_test(
            avg1=np.array([100.0, 200]),
            var=np.array([100.0, 200]),
            effect_size=np.array([0, 2]),
            alpha=0.05,
            beta=0.2,
        )