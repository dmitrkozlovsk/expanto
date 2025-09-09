from collections.abc import Callable

import numpy as np
import pytest
from scipy.stats import norm  # type: ignore

from src.services.analytics.stat_functions import (
    delta_test_ratio_metric,
    ratio_metric_sample_variance,
    ratio_metric_test,
    sample_size_ratio_metric,
)


# -------------------------------- TESTING RATIO FUNCTION --------------------------------
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
        ([(100, 20, 50, 30, 0.5), (120, 20, 50, 30, 0.5)], 1000, 100, lambda pval: pval < 0.05),
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
    [(5000, (120, 20, 50, 30, 0.1), 2000, 122), (10000, (120, 20, 50, 30, 0.1), 20000, 55)],
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
    [(200000, (1 / 2, 1 / 20, 0.85), 5000, 120), (10000, (1 / 2, 1 / 10, 0.4), 20000, 1010)],
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


# ------------------------------CORNER CASE---------------------------------


def test_ratio_metric_mean_d_zero_no_warnings(recwarn):
    """
    Test of warnings in the ratio_metric_sample_variance function.
    """
    var = ratio_metric_sample_variance(
        mean_n=np.array([50]),
        var_n=np.array([10]),
        mean_d=np.array([0]),
        var_d=np.array([5]),
        cov=np.array([2]),
    )
    assert np.isnan(var)
    assert len(recwarn) == 0


def test_ratio_metric_test_zero_variances_no_warnings(recwarn):
    test_result = ratio_metric_test(
        metric_value_1=np.array([10]),
        variance_1=np.array([0]),
        metric_value_2=np.array([10]),
        variance_2=np.array([0]),
    )

    assert len(recwarn) == 0
    assert test_result.p_value == 1.0
    assert test_result.diff_ratio == 0.0


def test_ratio_metric_test_zero_values_no_warnings(recwarn):
    test_result = ratio_metric_test(
        metric_value_1=np.array([0]),
        variance_1=np.array([5]),
        metric_value_2=np.array([10]),
        variance_2=np.array([5]),
    )
    assert len(recwarn) == 0
    assert test_result.p_value < 0.01
    assert np.isinf(test_result.diff_ratio)


# ------------------------------SAMPLE SIZE---------------------------------
@pytest.mark.parametrize(
    "mean_n, var_n, mean_d, var_d, cov, beta",
    [(100.0, 100.0, 20, 200, 0.5, 0.2), (1, 1, 2, 2, 0.9999, 0.05), (1, 1, 2, 2, 0, 0.05)],
)
def test_sample_size_delta_test_ratio_metric_reliability(mean_n, var_n, mean_d, var_d, cov, beta):
    """
    Test of reliability of the sample_size_ratio_metric function.
    """
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
