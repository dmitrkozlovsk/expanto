"""
Statistical functions for A/B testing and experimental analysis.

This module provides a collection of statistical functions for analyzing experimental data,
including t-tests, z-tests, ratio metric analysis, and sample size calculations. The functions
support both scalar and vectorized (numpy array) operations.

Key features:
- Welch's t-test for comparing means with unequal variances
- Z-test for comparing proportions
- Ratio metric analysis using delta method
- Sample size calculations for various test types
- Support for both scalar and vectorized operations
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
from scipy.optimize import root_scalar  # type: ignore
from scipy.special import ndtr, stdtr, stdtrit  # type: ignore
from scipy.stats import norm, t  # type: ignore
from statsmodels.stats.power import TTestIndPower  # type: ignore

from src.services.analytics._constants import NORM_PPF_ALPHA_TWO_SIDED, NORM_PPF_BETA
from src.utils import ValidationUtils

# Define types for result
ConfidenceInterval = namedtuple("ConfidenceInterval", ["lower", "upper"])
TestResult = namedtuple("TestResult", ["statistic", "p_value", "ci", "diff_abs", "diff_ratio"])


def ttest_welch(
    mean_1: float | np.ndarray,
    var_1: float | np.ndarray,
    n_1: float | int | np.ndarray,
    mean_2: float | np.ndarray,
    var_2: float | np.ndarray,
    n_2: float | int | np.ndarray,
) -> TestResult:
    """
    Perform Welch's t-test to compare two population means with unequal variances and sizes.

    Args:
        mean_1 (float | np.ndarray): Sample mean of the first population.
        var_1 (float | np.ndarray): Sample variance of the first population.
        n_1 (float | int | np.ndarray): Sample size of the first population. Must be positive.
        mean_2 (float | np.ndarray): Sample mean of the second population.
        var_2 (float | np.ndarray): Sample variance of the second population.
        n_2 (float | int | np.ndarray): Sample size of the second population. Must be positive.

    Returns:
        TestResult: Named tuple containing:
            - statistic: The t-statistic
            - p_value: Two-tailed p-value
            - ci: Confidence interval (ConfidenceInterval named tuple with lower and upper bounds)
            - diff_abs: Absolute difference between means (mean_2 - mean_1)
            - diff_ratio: Relative difference between means ((mean_2 - mean_1) / mean_1)

    Raises:
        ValueError: If any variance is negative or any sample size is not positive
        TypeError: If arguments are not all of the same type (all float or all np.ndarray)
        ValueError: If np.ndarray arguments have different shapes
    """
    # Check if all arguments are either float or np.ndarray
    args_list = [mean_1, var_1, n_1, mean_2, var_2, n_2]
    ValidationUtils.check_all_args_is_digit_or_ndarray(args_list)
    if all(isinstance(x, np.ndarray) for x in args_list):
        ValidationUtils.check_all_ndarray_has_same_shape([np.asarray(arr) for arr in args_list])

    if np.any(var_1 < 0) or np.any(var_2 < 0):
        raise ValueError("Variances must be non-negative")

    if np.any(n_1 <= 0) or np.any(n_2 <= 0):
        raise ValueError("Sample sizes must be positive")

    t_diff = mean_2 - mean_1
    t_se = (var_2 / n_2 + var_1 / n_1) ** 0.5
    t_stat = t_diff / t_se

    df_numerator = (var_2 / n_2 + var_1 / n_1) ** 2
    df_denominator = (var_2**2) / (n_2**2 * (n_2 - 1)) + (var_1**2) / (n_1**2 * (n_1 - 1))

    degrees_of_freedom = np.round(df_numerator / df_denominator)

    p_value = 2 * (1 - stdtr(degrees_of_freedom, abs(t_stat)))

    # calculating stat_ppf
    stat_ppf = np.where(
        degrees_of_freedom > 100000, NORM_PPF_ALPHA_TWO_SIDED[0.05], t.ppf(1 - 0.05 / 2, degrees_of_freedom)
    )

    ci_lower = t_diff - t_se * stat_ppf
    ci_upper = t_diff + t_se * stat_ppf
    ci = ConfidenceInterval(ci_lower, ci_upper)

    # Handle division by zero for diff_ratio
    diff_ratio = np.where(mean_1 != 0, t_diff / mean_1, np.inf)

    return TestResult(statistic=t_stat, p_value=p_value, ci=ci, diff_abs=t_diff, diff_ratio=diff_ratio)


def ztest_proportion(
    p_1: float | np.ndarray,
    n_1: float | int | np.ndarray,
    p_2: float | np.ndarray,
    n_2: float | int | np.ndarray,
) -> TestResult:
    """
    Z-test for comparing two proportions.

    Args:
        p_1 (float | np.ndarray): Proportion in the first sample. Must be between 0 and 1.
        n_1 (float | int | np.ndarray): Size of the first sample. Must be positive.
        p_2 (float | np.ndarray): Proportion in the second sample. Must be between 0 and 1.
        n_2 (float | int | np.ndarray): Size of the second sample. Must be positive.

    Returns:
        TestResult: Named tuple containing:
            - statistic: The z-statistic
            - p_value: Two-tailed p-value
            - ci: Confidence interval (ConfidenceInterval named tuple with lower and upper bounds)
            - diff_abs: Absolute difference between proportions (p_2 - p_1)
            - diff_ratio: Relative difference between proportions ((p_2 - p_1) / p_1)

    Raises:
        ValueError: If any proportion is not between 0 and 1 or any sample size is not positive
        TypeError: If arguments are not all of the same type (all float or all np.ndarray)
        ValueError: If np.ndarray arguments have different shapes
    """
    ValidationUtils.check_all_args_is_digit_or_ndarray([p_1, n_1, p_2, n_2])
    if all(isinstance(x, np.ndarray) for x in [p_1, n_1, p_2, n_2]):
        ValidationUtils.check_all_ndarray_has_same_shape([np.asarray(arr) for arr in [p_1, n_1, p_2, n_2]])

    if np.any((p_1 < 0) | (p_1 > 1)) or np.any((p_2 < 0) | (p_2 > 1)):
        raise ValueError("Proportions must be between 0 and 1")

    if np.any(n_1 <= 0) or np.any(n_2 <= 0):
        raise ValueError("Sample sizes must be positive")

    z_diff = p_2 - p_1
    p_pooled = (p_1 * n_1 + p_2 * n_2) / (n_1 + n_2)
    z_denominator = (p_pooled * (1 - p_pooled) * (1 / n_1 + 1 / n_2)) ** (1 / 2)
    z_stat = z_diff / z_denominator
    p_value = 2 * (1 - ndtr(abs(z_stat)))
    se = (p_1 * (1 - p_1) / n_1 + p_2 * (1 - p_2) / n_2) ** 0.5

    ci_lower = z_diff - se * NORM_PPF_ALPHA_TWO_SIDED[0.05]
    ci_upper = z_diff + se * NORM_PPF_ALPHA_TWO_SIDED[0.05]
    ci = ConfidenceInterval(ci_lower, ci_upper)

    # Handle division by zero for diff_ratio
    diff_ratio = np.where(p_1 != 0, z_diff / p_1, np.inf)

    return TestResult(statistic=z_stat, p_value=p_value, ci=ci, diff_abs=z_diff, diff_ratio=diff_ratio)


def ratio_metric_sample_variance(
    mean_n: float | np.ndarray,
    var_n: float | np.ndarray,
    mean_d: float | np.ndarray,
    var_d: float | np.ndarray,
    cov: float | np.ndarray,
) -> float | np.ndarray:
    """
    Calculate the sample variance of a ratio metric using the delta method.

    This function calculates the variance of a ratio metric where the ratio is defined
        as mean_n / mean_d.

    Args:
        mean_n (float | np.ndarray): Mean of the numerator.
        var_n (float | np.ndarray): Variance of the numerator. Must be non-negative.
        mean_d (float | np.ndarray): Mean of the denominator. Cannot be zero.
        var_d (float | np.ndarray): Variance of the denominator. Must be non-negative.
        cov (float | np.ndarray): Covariance between numerator and denominator.

    Returns:
        float | np.ndarray: The estimated variance of the ratio metric.

    Raises:
        ValueError: If any variance is negative or if mean_d is zero
        TypeError: If arguments are not all of the same type (all float or all np.ndarray)
        ValueError: If np.ndarray arguments have different shapes
    """
    ValidationUtils.check_all_args_is_digit_or_ndarray([mean_n, var_n, mean_d, var_d, cov])
    if all(isinstance(x, np.ndarray) for x in [mean_n, var_n, mean_d, var_d, cov]):
        ValidationUtils.check_all_ndarray_has_same_shape(
            [np.asarray(arr) for arr in [mean_n, var_n, mean_d, var_d, cov]]
        )
    if np.any(var_n < 0) or np.any(var_d < 0):
        raise ValueError("Variances must be non-negative")

    term1 = var_n / (mean_d**2)
    term2 = (mean_n**2 / mean_d**4) * var_d
    term3 = 2 * (mean_n / mean_d**3) * cov

    res_var_raw = term1 + term2 - term3
    res_var = np.where((res_var_raw > -1e-9) & (res_var_raw < 0), 0, res_var_raw)
    return res_var


def ratio_metric_test(
    metric_value_1: float | np.ndarray,
    variance_1: float | np.ndarray,
    metric_value_2: float | np.ndarray,
    variance_2: float | np.ndarray,
) -> TestResult:
    """
    Statistical test for comparing two ratio metrics.

    This function performs a z-test to compare two ratio metrics with known variances.

    Args:
        metric_value_1 (float | np.ndarray): Value of the first ratio metric. Cannot be zero.
        variance_1 (float | np.ndarray): Variance of the first ratio metric. Must be non-negative.
        metric_value_2 (float | np.ndarray): Value of the second ratio metric.
        variance_2 (float | np.ndarray): Variance of the second ratio metric. Must be non-negative.

    Returns:
        TestResult: Named tuple containing:
            - statistic: The z-statistic
            - p_value: Two-tailed p-value
            - ci: Confidence interval (ConfidenceInterval named tuple with lower and upper bounds)
            - diff_abs: Absolute difference between metrics (metric_2 - metric_1)
            - diff_ratio: Relative difference between metrics ((metric_2 - metric_1) / metric_1)

    Raises:
        ValueError: If any variance is negative or if metric_value_1 is zero
        TypeError: If arguments are not all of the same type (all float or all np.ndarray)
        ValueError: If np.ndarray arguments have different shapes
    """
    args_list = [metric_value_1, variance_1, metric_value_2, variance_2]
    ValidationUtils.check_all_args_is_digit_or_ndarray(args_list)
    if all(isinstance(x, np.ndarray) for x in args_list):
        ValidationUtils.check_all_ndarray_has_same_shape([np.asarray(arr) for arr in args_list])
    if np.any(variance_1 < 0) or np.any(variance_2 < 0):
        raise ValueError("Variances must be non-negative")

    diff_metric = metric_value_2 - metric_value_1
    diff_var = variance_2 + variance_1
    diff_se = np.sqrt(diff_var)

    # Use np.isclose for robust floating point comparison to avoid division by zero
    is_se_zero = np.isclose(diff_se, 0)
    is_diff_zero = np.isclose(diff_metric, 0)

    # Calculate z_stat safely
    # Set z_stat to 0 if standard error is zero and difference is zero,
    # to +/- infinity if standard error is zero but difference is not,
    # and to the calculated value otherwise.
    z_stat = np.divide(diff_metric, diff_se, out=np.zeros_like(diff_metric, dtype=float), where=~is_se_zero)
    z_stat_if_se_zero = np.where(is_diff_zero, 0.0, np.inf * np.sign(diff_metric))
    z_stat = np.where(is_se_zero, z_stat_if_se_zero, z_stat)

    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    ci_lower = diff_metric - diff_se * NORM_PPF_ALPHA_TWO_SIDED[0.05]
    ci_upper = diff_metric + diff_se * NORM_PPF_ALPHA_TWO_SIDED[0.05]
    ci = ConfidenceInterval(ci_lower, ci_upper)

    return TestResult(
        statistic=z_stat,
        p_value=p_value,
        ci=ci,
        diff_abs=diff_metric,
        diff_ratio=diff_metric / metric_value_1,
    )


def delta_test_ratio_metric(
    metric_1: float | np.ndarray,
    mean_n_1: float | np.ndarray,
    var_n_1: float | np.ndarray,
    mean_d_1: float | np.ndarray,
    var_d_1: float | np.ndarray,
    cov_1: float | np.ndarray,
    n_1: float | int | np.ndarray,
    metric_2: float | np.ndarray,
    mean_n_2: float | np.ndarray,
    var_n_2: float | np.ndarray,
    mean_d_2: float | np.ndarray,
    var_d_2: float | np.ndarray,
    cov_2: float | np.ndarray,
    n_2: float | int | np.ndarray,
) -> TestResult:
    """
    Perform a statistical test to compare two ratio metrics using the delta method.

    This function first calculates the variance of two ratio metrics using the delta method,
    then performs a statistical test to compare them.

    Args:
        metric_1 (float | np.ndarray): Value of the first ratio metric.
        mean_n_1 (float | np.ndarray): Mean of the numerator for the first ratio.
        var_n_1 (float | np.ndarray): Variance of the numerator for the first ratio.
            Must be non-negative.
        mean_d_1 (float | np.ndarray): Mean of the denominator for the first ratio. Cannot be zero.
        var_d_1 (float | np.ndarray): Variance of the denominator for the first ratio.
            Must be non-negative.
        cov_1 (float | np.ndarray): Covariance between numerator and denominator for the ratio 1.
        n_1 (float | int | np.ndarray): Sample size for the first ratio. Must be positive.
        metric_2 (float | np.ndarray): Value of the second ratio metric.
        mean_n_2 (float | np.ndarray): Mean of the numerator for the second ratio.
        var_n_2 (float | np.ndarray): Variance of the numerator for the second ratio.
            Must be non-negative.
        mean_d_2 (float | np.ndarray): Mean of the denominator for the second ratio. Cannot be zero.
        var_d_2 (float | np.ndarray): Variance of the denominator for the second ratio.
            Must be non-negative.
        cov_2 (float | np.ndarray): Covariance between numerator and denominator for the ratio 2.
        n_2 (float | int | np.ndarray): Sample size for the second ratio. Must be positive.

    Returns:
        TestResult: Named tuple containing:
            - statistic: The z-statistic
            - p_value: Two-tailed p-value
            - ci: Confidence interval (ConfidenceInterval named tuple with lower and upper bounds)
            - diff_abs: Absolute difference between metrics (metric_2 - metric_1)
            - diff_ratio: Relative difference between metrics ((metric_2 - metric_1) / metric_1)

    Raises:
        ValueError: If any variance is negative, any sample size is not positive,
            or any denominator mean is zero
        TypeError: If arguments are not all of the same type (all float or all np.ndarray)
        ValueError: If np.ndarray arguments have different shapes
    """
    args_list = [
        metric_1,
        mean_n_1,
        var_n_1,
        mean_d_1,
        var_d_1,
        cov_1,
        n_1,
        metric_2,
        mean_n_2,
        var_n_2,
        mean_d_2,
        var_d_2,
        cov_2,
        n_2,
    ]
    ValidationUtils.check_all_args_is_digit_or_ndarray(args_list)
    if all(isinstance(x, np.ndarray) for x in args_list):
        ValidationUtils.check_all_ndarray_has_same_shape([np.asarray(arr) for arr in args_list])
    delta_var_1 = (
        ratio_metric_sample_variance(
            mean_n=mean_n_1, var_n=var_n_1, mean_d=mean_d_1, var_d=var_d_1, cov=cov_1
        )
        / n_1
    )

    delta_var_2 = (
        ratio_metric_sample_variance(
            mean_n=mean_n_2,
            var_n=var_n_2,
            mean_d=mean_d_2,
            var_d=var_d_2,
            cov=cov_2,
        )
        / n_2
    )

    return ratio_metric_test(metric_1, delta_var_1, metric_2, delta_var_2)


# -------------------------- SAMPLE SIZE FUNCTIONS -------------------------- #


def sample_size_proportion_z_test(
    p1: float | np.ndarray,
    effect_size: float | np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray | float:
    """
    Calculate the sample size for a proportion z-test.

    Args:
        p1 (float | np.ndarray): Reference proportion. Must be in range (0, 1).
        effect_size (float | np.ndarray): Relative difference between proportions.
            For example, effect_size=0.2 means detecting a 20% difference from p1.
        alpha (float): Significance level. Must be one of the predefined values in
            NORM_PPF_ALPHA_TWO_SIDED.
        beta (float): Type 2 error rate. Must be one of the predefined values in
            NORM_PPF_BETA.

    Returns:
        np.ndarray | float: Required sample size for each group. Returns a float for scalar inputs
            and a numpy array for array inputs.

    Raises:
        ValueError: If alpha or beta is not one of the predefined values
        ValueError: If p1 is not in range (0, 1)
        ValueError: If the resulting p2 (p1 + p1*effect_size) is not in range (0, 1)
    """
    if alpha not in NORM_PPF_ALPHA_TWO_SIDED:
        raise ValueError(
            f"alpha must be one of the following values: {list(NORM_PPF_ALPHA_TWO_SIDED.keys())}"
        )
    if beta not in NORM_PPF_BETA:
        raise ValueError(f"beta must be one of the following values: {list(NORM_PPF_BETA.keys())}")

    if np.any((p1 <= 0) | (p1 >= 1)):
        raise ValueError("p1 values must be in range (0, 1)")

    p2 = p1 + (p1 * effect_size)
    if np.any((p2 <= 0) | (p2 >= 1)):
        raise ValueError(
            f"p2 values must be in range (0, 1). Current falue is  {p2}, effect_size is {effect_size}"
        )
    z_alpha = NORM_PPF_ALPHA_TWO_SIDED[alpha]
    z_beta = NORM_PPF_BETA[beta]

    n_required = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p1 - p2) ** 2

    return np.ceil(n_required)


def sample_size_t_test(
    avg1: float | np.ndarray,
    var: float | np.ndarray,
    effect_size: float | np.ndarray,
    alpha: float = 0.05,
    beta: float = 0.2,
) -> np.ndarray | float:
    """
    Calculate the required sample size for a two-sample t-test with equal variances.

    Args:
        avg1 (float | np.ndarray): Mean of the first group (control group).
        var (float | np.ndarray): Variance of the data (assumed equal in both groups).
            Must be non-negative.
        effect_size (float | np.ndarray): Relative difference to detect,
            expressed as a proportion of avg1.
            For example, effect_size=0.2 means detecting a 20% difference from avg1.
            Cannot be zero.
        alpha (float, optional): Significance level. Must be one of the predefined values in
            NORM_PPF_ALPHA_TWO_SIDED.
            Defaults to 0.05.
        beta (float, optional): Type II error rate. Must be one of the predefined values in
            NORM_PPF_BETA.
            Defaults to 0.2 (meaning power = 0.8).

    Returns:
        np.ndarray | float: Required sample size for each group. Returns a float for scalar inputs
            and a numpy array for array inputs.

    Raises:
        ValueError: If effect_size is zero
        ValueError: If alpha or beta is not one of the predefined values
        ValueError: If var is negative
        ValueError: If input types are inconsistent (mixing float and np.ndarray)
    """
    if np.any(effect_size == 0):
        raise ValueError("Effect size cannot be zero")

    if alpha not in NORM_PPF_ALPHA_TWO_SIDED:
        raise ValueError(
            f"alpha must be one of the following values: {list(NORM_PPF_ALPHA_TWO_SIDED.keys())}"
        )
    if beta not in NORM_PPF_BETA:
        raise ValueError(f"beta must be one of the following values: {list(NORM_PPF_BETA.keys())}")

    z_alpha = NORM_PPF_ALPHA_TWO_SIDED[alpha]
    z_beta = NORM_PPF_BETA[beta]

    avg2 = avg1 + (avg1 * effect_size)
    delta = avg2 - avg1
    n_required_norm_approx = 2 * (z_alpha + z_beta) ** 2 * var / (avg2 - avg1) ** 2

    def solve_sample_size(n: float, d: float, v: float) -> float:
        """Solve for sample size using t-distribution for small sample sizes."""
        degrees_of_freedom = 2 * n - 2
        t_alpha = stdtrit(degrees_of_freedom, 1 - alpha / 2)
        se = (2 * v / n) ** 0.5
        lambda_ = d / se
        left_side = stdtr(degrees_of_freedom, lambda_ - t_alpha)
        return left_side - (1 - beta)

    def find_root_one_case(d: float, v: float, n_approx: float) -> float:
        """Find the root of the sample size equation using Brent's method."""
        bracket = [2, n_approx * 4]
        root = root_scalar(
            lambda n: solve_sample_size(n, d, v),
            bracket=bracket,
            method="brentq",
            xtol=1e-2,
        ).root
        return root

    def find_s_size_one_case(
        d: float,
        v: float,
        n_approx: float,
        r_alpha: float,
        r_beta: float,
    ) -> float:
        """
        Calculate sample size using either normal approximation or exact t-distribution method.
        """
        if n_approx < 150:
            s_size = TTestIndPower().solve_power(
                effect_size=d / np.sqrt(v), power=1 - r_beta, alpha=r_alpha, ratio=1.0
            )
        else:
            s_size = find_root_one_case(d, v, n_approx)
        if isinstance(s_size, np.ndarray):
            s_size = s_size[0]
        return float(s_size)

    if (
        isinstance(delta, np.ndarray)
        and isinstance(var, np.ndarray)
        and isinstance(effect_size, np.ndarray)
    ):
        sample_sizes_list = []
        for d, v, n_approx in zip(
            np.atleast_1d(delta), np.atleast_1d(var), np.atleast_1d(n_required_norm_approx), strict=True
        ):
            root = find_s_size_one_case(d, v, n_approx, alpha, beta)
            sample_sizes_list.append(root)
        return np.ceil(sample_sizes_list)
    elif isinstance(delta, float) and isinstance(var, float) and isinstance(effect_size, float):
        sample_size = find_s_size_one_case(
            float(delta), float(var), float(n_required_norm_approx), float(alpha), float(beta)
        )
        return np.ceil(sample_size)
    else:
        raise ValueError("Invalid input types of `delta` or `var` or `effect_size`")


def sample_size_ratio_metric(
    mean_n: float | np.ndarray,
    var_n: float | np.ndarray,
    mean_d: float | np.ndarray,
    var_d: float | np.ndarray,
    cov: float | np.ndarray,
    effect_size: float | np.ndarray,
    alpha: float = 0.05,
    beta: float = 0.2,
) -> np.ndarray | float:
    """
    Calculate the required sample size for a ratio metric test using the delta method.

    This function calculates the sample size needed to detect a given effect size in a ratio metric
    with specified statistical power and significance level.

    Args:
        mean_n (float | np.ndarray): Mean of the numerator.
        var_n (float | np.ndarray): Variance of the numerator. Must be non-negative.
        mean_d (float | np.ndarray): Mean of the denominator. Cannot be zero.
        var_d (float | np.ndarray): Variance of the denominator. Must be non-negative.
        cov (float | np.ndarray): Covariance between numerator and denominator.
        effect_size (float | np.ndarray): Relative difference to detect,
            expressed as a proportion of the ratio.
            For example, effect_size=0.2 means detecting a 20% difference in the ratio.
            Cannot be zero.
        alpha (float, optional): Significance level. Must be one of the predefined values
            in NORM_PPF_ALPHA_TWO_SIDED.
            Defaults to 0.05.
        beta (float, optional): Type II error rate. Must be one of the predefined values
        in NORM_PPF_BETA.
            Defaults to 0.2 (meaning power = 0.8).

    Returns:
        np.ndarray | float: Required sample size. Returns a float for scalar inputs
            and a numpy array for array inputs.

    Raises:
        ValueError: If effect_size is zero
        ValueError: If mean_d is zero
        ValueError: If any variance is negative
        ValueError: If alpha or beta is not one of the predefined values
    """
    if np.any(effect_size == 0):
        raise ValueError("Effect size cannot be zero")

    if np.any(mean_d == 0):
        raise ValueError("Mean of denominator cannot be zero")

    if alpha not in NORM_PPF_ALPHA_TWO_SIDED:
        raise ValueError(
            f"alpha must be one of the following values: {list(NORM_PPF_ALPHA_TWO_SIDED.keys())}"
        )
    if beta not in NORM_PPF_BETA:
        raise ValueError(f"beta must be one of the following values: {list(NORM_PPF_BETA.keys())}")

    ratio = mean_n / mean_d
    ratio_var = ratio_metric_sample_variance(
        mean_n=mean_n, var_n=var_n, mean_d=mean_d, var_d=var_d, cov=cov
    )

    target_ratio = ratio * (1 + effect_size)
    diff = target_ratio - ratio

    z_alpha = NORM_PPF_ALPHA_TWO_SIDED[alpha]
    z_beta = NORM_PPF_BETA[beta]

    n_required = 2 * (z_alpha + z_beta) ** 2 * ratio_var / (diff**2)
    return np.ceil(n_required)
