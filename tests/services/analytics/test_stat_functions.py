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
    delta_test_ratio_metric,
    ratio_metric_sample_variance,
    ratio_metric_test,
    sample_size_proportion_z_test,
    sample_size_ratio_metric,
    sample_size_t_test,
    ttest_welch,
    ztest_proportion,
)


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


# -------------------------------- TESTING ZTEST FUNCTION --------------------------------
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


@pytest.fixture
def generate_samples_ratio() -> Callable:
    """Fixture generating two random samples based on provided statistics."""

    def _generate_samples(mean_n, std_n, mean_d, std_d, correlation, sample_size, seed=None):
        cov_nd = correlation * std_n * std_d
        cov_matrix = [[std_n**2, cov_nd], [cov_nd, std_d**2]]
        data = np.random.multivariate_normal([mean_n, mean_d], cov_matrix, size=sample_size)
        return data[:, 0], data[:, 1]

    return _generate_samples


@pytest.fixture
def generate_samples_ratio_expon() -> Callable:
    """Fixture generating two random samples based on provided statistics."""

    def _generate_samples(lambda_n, lambda_d, correlation, sample_size, seed=None):
        cov_nd = correlation * 1 * 1
        cov_matrix = [[1, cov_nd], [cov_nd, 1]]
        data = np.random.multivariate_normal([0, 0], cov_matrix, size=sample_size)
        U_n, U_d = norm.cdf(data[:, 0]), norm.cdf(data[:, 1])

        def f(lambda_, U):
            return -1 / lambda_ * np.log(1 - U)

        sample_n, sample_d = f(lambda_n, U_n), f(lambda_d, U_d)
        return sample_n, sample_d

    return _generate_samples


def generate_stats_by_ratio_samples(numerator_1, denominator_1, numerator_2, denominator_2):
    n_mean_1, n_var_1, d_mean_1, d_var_1, cov_1 = (
        numerator_1.mean(),
        numerator_1.var(ddof=1),
        denominator_1.mean(),
        denominator_1.var(ddof=1),
        np.cov(numerator_1, denominator_1, ddof=1)[0, 1],
    )
    n_mean_2, n_var_2, d_mean_2, d_var_2, cov_2 = (
        numerator_2.mean(),
        numerator_2.var(ddof=1),
        denominator_2.mean(),
        denominator_2.var(ddof=1),
        np.cov(numerator_2, denominator_2, ddof=1)[0, 1],
    )

    ratio_1 = n_mean_1 / d_mean_1
    ratio_2 = n_mean_2 / d_mean_2

    ratio_var_1 = ratio_metric_sample_variance(n_mean_1, n_var_1, d_mean_1, d_var_1, cov_1)
    ratio_var_2 = ratio_metric_sample_variance(n_mean_2, n_var_2, d_mean_2, d_var_2, cov_2)

    return ratio_1, ratio_var_1, ratio_2, ratio_var_2


@pytest.mark.parametrize(
    "samples_params_list, sample_size, seed, pval_check",
    [
        ([(100, 20, 50, 30, 0.5), (100, 20, 50, 30, 0.5)], 10000, 43, lambda pval: pval > 0.05),
        ([(100, 20, 50, 30, 0.1), (120, 20, 50, 30, 0.1)], 100000, 42, lambda pval: pval < 0.01),
        ([(100, 20, 50, 30, 0.5), (120, 20, 50, 30, 0.5)], 1000, None, lambda pval: pval < 0.05),
    ],
)
def test_ratio_metric_test_trivial(
    generate_samples_ratio, samples_params_list, sample_size, seed, pval_check
):
    sample_1_params, sample_2_params = samples_params_list

    numerator_1, denominator_1 = generate_samples_ratio(*sample_1_params, sample_size, seed)
    numerator_2, denominator_2 = generate_samples_ratio(*sample_2_params, sample_size, seed)

    ratio_1, ratio_var_1, ratio_2, ratio_var_2 = generate_stats_by_ratio_samples(
        numerator_1, denominator_1, numerator_2, denominator_2
    )

    ratio_test_result = ratio_metric_test(
        ratio_1, ratio_var_1 / sample_size, ratio_2, ratio_var_2 / sample_size
    )
    assert pval_check(ratio_test_result.p_value)


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_simulations, samples_params, sample_size, seed",
    [(5000, (120, 20, 50, 30, 0.1), 2000, 122), (10000, (120, 20, 50, 30, 0.1), 20000, None)],
)
def test_ratio_metric_test_reliability_norm(
    generate_samples_ratio, n_simulations, samples_params, sample_size, seed
):
    pvals = []
    for _ in range(n_simulations):
        numerator_1, denominator_1 = generate_samples_ratio(*samples_params, sample_size, seed)
        numerator_2, denominator_2 = generate_samples_ratio(*samples_params, sample_size, seed)

        ratio_1, ratio_var_1, ratio_2, ratio_var_2 = generate_stats_by_ratio_samples(
            numerator_1, denominator_1, numerator_2, denominator_2
        )

        ratio_test_result = ratio_metric_test(
            ratio_1, ratio_var_1 / sample_size, ratio_2, ratio_var_2 / sample_size
        )

        pvals.append(ratio_test_result.p_value)
    pvals = np.array(pvals)

    for threshhold in [0.1, 0.05]:
        share = sum(pvals < threshhold) / len(pvals)
        assert threshhold * 0.85 <= share <= threshhold * 1.15


@pytest.mark.slow
@pytest.mark.parametrize(
    "n_simulations, samples_params, sample_size, seed",
    # samples_params(lambda_n, lambda_d, correlation)
    [(200000, (1 / 2, 1 / 20, 0.85), 5000, 120), (10000, (1 / 2, 1 / 10, 0.4), 20000, None)],
)
def test_ratio_metric_test_reliability_expon(
    generate_samples_ratio_expon, n_simulations, samples_params, sample_size, seed
):
    pvals = []
    for _ in range(n_simulations):
        numerator_1, denominator_1 = generate_samples_ratio_expon(*samples_params, sample_size, seed)
        numerator_2, denominator_2 = generate_samples_ratio_expon(*samples_params, sample_size, seed)

        ratio_1, ratio_var_1, ratio_2, ratio_var_2 = generate_stats_by_ratio_samples(
            numerator_1, denominator_1, numerator_2, denominator_2
        )

        ratio_test_result = ratio_metric_test(
            ratio_1, ratio_var_1 / sample_size, ratio_2, ratio_var_2 / sample_size
        )

        pvals.append(ratio_test_result.p_value)

    pvals = np.array(pvals)

    for threshold in [0.1, 0.05, 0.01]:
        share = sum(pvals < threshold) / len(pvals)
        assert threshold * 0.85 <= share <= threshold * 1.15


# ------------------------- TESTING SAMPLE SIZE PROPORTION TEST -------------------------
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


#### TESTING SAMPLE SIZE T-TEST ####
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


@pytest.mark.parametrize(
    "mean_n, var_n, mean_d, var_d, cov, beta",
    [(100.0, 100.0, 20, 200, 0.5, 0.2), (1, 1, 2, 2, 0.9999, 0.05), (1, 1, 2, 2, 0, 0.05)],
)
def test_delta_test_ratio_metric_reliability(mean_n, var_n, mean_d, var_d, cov, beta):
    # Test that the sample size is correct for a given effect size and alpha
    # I.E. that the p-value is less than alpha if we use the computed sample size.
    # WARNING: this test bases on delta_test_ratio_metric (it's reliability is tested above)
    for effect_size in [0.001, 0.01, 0.03, 0.1, 0.2, 0.5]:
        for alpha in [0.01, 0.05, 0.1]:
            our_sample_size = sample_size_ratio_metric(
                mean_n, var_n, mean_d, var_d, cov, effect_size, alpha, beta
            )

            pval = delta_test_ratio_metric(
                mean_n / mean_d,
                mean_n,
                var_n,
                mean_d,
                var_d,
                cov,
                our_sample_size,
                mean_n * (1 + effect_size) / mean_d,
                mean_n * (1 + effect_size),
                var_n,
                mean_d,
                var_d,
                cov,
                our_sample_size,
            ).p_value
            assert pval < alpha


def test_sample_size_ratio_metric_value_errors():
    """Test array inputs with zero effect size"""
    with pytest.raises(ValueError, match="Effect size cannot be zero"):
        sample_size_ratio_metric(
            mean_n=np.array([100.0, 200.0]),
            var_n=np.array([100.0, 200.0]),
            mean_d=np.array([20.0, 40.0]),
            var_d=np.array([200.0, 400.0]),
            cov=np.array([0.5, 0.5]),
            effect_size=np.array([0.1, 0.0]),
            alpha=0.05,
            beta=0.2,
        )

    # Test array inputs with zero denominator mean
    with pytest.raises(ValueError, match="Mean of denominator cannot be zero"):
        sample_size_ratio_metric(
            mean_n=np.array([100.0, 200.0]),
            var_n=np.array([100.0, 200.0]),
            mean_d=np.array([20.0, 0.0]),
            var_d=np.array([200.0, 400.0]),
            cov=np.array([0.5, 0.5]),
            effect_size=np.array([0.1, 0.1]),
            alpha=0.05,
            beta=0.2,
        )
