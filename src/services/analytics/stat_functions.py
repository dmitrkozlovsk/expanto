"""Statistical functions for experimental analysis.

This module provides a collection of statistical functions for analyzing experimental
data, including t-tests, z-tests, ratio metric analysis, and sample size
calculations. The functions support both scalar and vectorized (NumPy array)
operations.

Key features:
- Welch's t-test for comparing means with unequal variances.
- Z-test for comparing proportions.
- Ratio metric analysis using the delta method.
- Sample size calculations for various test types.
- Support for both scalar and vectorized operations.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
from scipy.optimize import root_scalar  # type: ignore
from scipy.special import ndtr, stdtr, stdtrit  # type: ignore
from scipy.stats import chi2, norm, t  # type: ignore
from statsmodels.stats.power import TTestIndPower  # type: ignore

from src.services.analytics._constants import NORM_PPF_ALPHA_TWO_SIDED, NORM_PPF_BETA
from src.utils import ValidationUtils

# Define types for result
ConfidenceInterval = namedtuple("ConfidenceInterval", ["lower", "upper"])
TestResult = namedtuple("TestResult", ["statistic", "p_value", "ci", "diff_abs", "diff_ratio"])
SRMResult = namedtuple(
    "SRMResult", ["statistic", "p_value", "df", "expected", "observed", "allocation", "is_srm"]
)


def ttest_welch(
    mean_1: float | np.ndarray,
    var_1: float | np.ndarray,
    n_1: float | int | np.ndarray,
    mean_2: float | np.ndarray,
    var_2: float | np.ndarray,
    n_2: float | int | np.ndarray,
) -> TestResult:
    """Performs Welch's t-test for two independent samples.

    This test compares the means of two populations with unequal variances and
    sample sizes. It supports both scalar and vectorized inputs.

    Args:
        mean_1: Sample mean of the first population.
        var_1: Sample variance of the first population.
        n_1: Sample size of the first population. Must be positive.
        mean_2: Sample mean of the second population.
        var_2: Sample variance of the second population.
        n_2: Sample size of the second population. Must be positive.

    Returns:
        A `TestResult` named tuple with the following fields:
        - statistic: The calculated t-statistic.
        - p_value: The two-tailed p-value.
        - ci: A `ConfidenceInterval` with lower and upper bounds.
        - diff_abs: The absolute difference between means (mean_2 - mean_1).
        - diff_ratio: The relative difference between means.

    Raises:
        ValueError: If any variance is negative or sample size is not positive.
        TypeError: If arguments are not of the same type (all float or all
            np.ndarray).
        ValueError: If np.ndarray arguments have different shapes.
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
    t_stat = np.divide(t_diff, t_se, out=np.full_like(t_diff, np.nan, dtype=float), where=t_se != 0)

    df_numerator = (var_2 / n_2 + var_1 / n_1) ** 2
    df_denominator = (var_2**2) / (n_2**2 * (n_2 - 1)) + (var_1**2) / (n_1**2 * (n_1 - 1))

    degrees_of_freedom = np.divide(
        df_numerator,
        df_denominator,
        out=np.full_like(df_numerator, np.nan, dtype=float),
        where=df_denominator != 0,
    )

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
    """Performs a Z-test for comparing two proportions.

    This test is used to determine if there is a significant difference between
    the proportions of two independent samples.

    Args:
        p_1: Proportion in the first sample. Must be between 0 and 1.
        n_1: Size of the first sample. Must be positive.
        p_2: Proportion in the second sample. Must be between 0 and 1.
        n_2: Size of the second sample. Must be positive.

    Returns:
        A `TestResult` named tuple with the following fields:
        - statistic: The calculated z-statistic.
        - p_value: The two-tailed p-value.
        - ci: A `ConfidenceInterval` with lower and upper bounds.
        - diff_abs: The absolute difference between proportions (p_2 - p_1).
        - diff_ratio: The relative difference between proportions.

    Raises:
        ValueError: If any proportion is not between 0 and 1 or any sample
            size is not positive.
        TypeError: If arguments are not of the same type (all float or all
            np.ndarray).
        ValueError: If np.ndarray arguments have different shapes.
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

    z_stat = np.zeros_like(z_diff, dtype=float)
    np.divide(z_diff, z_denominator, out=z_stat, where=(z_denominator != 0))

    p_value = 2 * (1 - ndtr(abs(z_stat)))
    se = (p_1 * (1 - p_1) / n_1 + p_2 * (1 - p_2) / n_2) ** 0.5

    ci_lower = z_diff - se * NORM_PPF_ALPHA_TWO_SIDED[0.05]
    ci_upper = z_diff + se * NORM_PPF_ALPHA_TWO_SIDED[0.05]
    ci = ConfidenceInterval(ci_lower, ci_upper)

    # Handle division by zero for diff_ratio
    diff_ratio = np.full_like(z_diff, np.inf, dtype=float)
    np.divide(z_diff, p_1, out=diff_ratio, where=(p_1 != 0))

    return TestResult(statistic=z_stat, p_value=p_value, ci=ci, diff_abs=z_diff, diff_ratio=diff_ratio)


def ratio_metric_sample_variance(
    mean_n: float | np.ndarray,
    var_n: float | np.ndarray,
    mean_d: float | np.ndarray,
    var_d: float | np.ndarray,
    cov: float | np.ndarray,
) -> float | np.ndarray:
    """Calculates the sample variance of a ratio metric using the delta method.

    The variance of a ratio metric (mean_n / mean_d) is estimated based on the
    means, variances, and covariance of its numerator and denominator.

    Args:
        mean_n: Mean of the numerator.
        var_n: Variance of the numerator. Must be non-negative.
        mean_d: Mean of the denominator. Cannot be zero.
        var_d: Variance of the denominator. Must be non-negative.
        cov: Covariance between the numerator and denominator.

    Returns:
        The estimated variance of the ratio metric.

    Raises:
        ValueError: If any variance is negative or if mean_d is zero.
        TypeError: If arguments are not of the same type (all float or all
            np.ndarray).
        ValueError: If np.ndarray arguments have different shapes.
    """
    ValidationUtils.check_all_args_is_digit_or_ndarray([mean_n, var_n, mean_d, var_d, cov])
    if all(isinstance(x, np.ndarray) for x in [mean_n, var_n, mean_d, var_d, cov]):
        ValidationUtils.check_all_ndarray_has_same_shape(
            [np.asarray(arr) for arr in [mean_n, var_n, mean_d, var_d, cov]]
        )
    if np.any(var_n < 0) or np.any(var_d < 0):
        raise ValueError("Variances must be non-negative")

    mean_ne_zero_mask = mean_d != 0

    term1 = np.full_like(mean_d, np.nan, dtype=float)
    np.divide(var_n, mean_d**2, out=term1, where=mean_ne_zero_mask)

    term2 = np.full_like(mean_d, np.nan, dtype=float)
    np.divide(mean_n**2 * var_d, mean_d**4, out=term2, where=mean_ne_zero_mask)

    term3 = np.full_like(mean_d, np.nan, dtype=float)
    np.divide(2 * mean_n * cov, mean_d**3, out=term3, where=mean_ne_zero_mask)

    res_var_raw = term1 + term2 - term3
    res_var = np.where((res_var_raw > -1e-9) & (res_var_raw < 0), 0, res_var_raw)
    return res_var


def ratio_metric_test(
    metric_value_1: float | np.ndarray,
    variance_1: float | np.ndarray,
    metric_value_2: float | np.ndarray,
    variance_2: float | np.ndarray,
) -> TestResult:
    """Performs a statistical test for comparing two ratio metrics.

    This function conducts a z-test to compare two ratio metrics with known
    variances, which is suitable for large sample sizes.

    Args:
        metric_value_1: Value of the first ratio metric.
        variance_1: Variance of the first ratio metric. Must be non-negative.
        metric_value_2: Value of the second ratio metric.
        variance_2: Variance of the second ratio metric. Must be non-negative.

    Returns:
        A `TestResult` named tuple with the following fields:
        - statistic: The calculated z-statistic.
        - p_value: The two-tailed p-value.
        - ci: A `ConfidenceInterval` with lower and upper bounds.
        - diff_abs: The absolute difference between metrics.
        - diff_ratio: The relative difference between metrics.

    Raises:
        ValueError: If any variance is negative.
        TypeError: If arguments are not of the same type (all float or all
            np.ndarray).
        ValueError: If np.ndarray arguments have different shapes.
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

    # avoiding division by zero
    eps = 1e-12
    is_se_zero = np.isclose(diff_se, 0.0, atol=eps)
    is_diff_zero = np.isclose(diff_metric, 0.0, atol=eps)

    z_stat = np.divide(diff_metric, diff_se, out=np.zeros_like(diff_metric, dtype=float), where=~is_se_zero)
    need_inf = is_se_zero & ~is_diff_zero
    if np.any(need_inf):
        inf_val = np.sign(diff_metric) * np.inf
        z_stat = np.where(need_inf, inf_val, z_stat)

    p_value = 2 * (1 - norm.cdf(abs(z_stat)))

    ci_lower = diff_metric - diff_se * NORM_PPF_ALPHA_TWO_SIDED[0.05]
    ci_upper = diff_metric + diff_se * NORM_PPF_ALPHA_TWO_SIDED[0.05]
    ci = ConfidenceInterval(ci_lower, ci_upper)

    diff_ratio = np.empty_like(diff_metric, dtype=float)

    m1_zero = np.isclose(metric_value_1, 0.0, atol=eps)
    diff_zero = np.isclose(diff_metric, 0.0, atol=eps)

    np.divide(diff_metric, metric_value_1, out=diff_ratio, where=~m1_zero)

    mask_zero = m1_zero & diff_zero
    diff_ratio[mask_zero] = 0.0

    # zero baseline and non-zero difference -> ±inf (without inf*0, because mask excludes diff==0)
    mask_inf = m1_zero & ~diff_zero
    if np.any(mask_inf):
        inf_val = np.sign(diff_metric) * np.inf
        diff_ratio = np.where(mask_inf, inf_val, diff_ratio)

    return TestResult(
        statistic=z_stat,
        p_value=p_value,
        ci=ci,
        diff_abs=diff_metric,
        diff_ratio=diff_ratio,
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
    """Performs a test to compare two ratio metrics using the delta method.

    This function first calculates the variance of each ratio metric using the
    delta method and then performs a z-test to compare them.

    Args:
        metric_1: Value of the first ratio metric.
        mean_n_1: Mean of the numerator for the first ratio.
        var_n_1: Variance of the numerator for the first ratio.
        mean_d_1: Mean of the denominator for the first ratio.
        var_d_1: Variance of the denominator for the first ratio.
        cov_1: Covariance for the first ratio.
        n_1: Sample size for the first ratio.
        metric_2: Value of the second ratio metric.
        mean_n_2: Mean of the numerator for the second ratio.
        var_n_2: Variance of the numerator for the second ratio.
        mean_d_2: Mean of the denominator for the second ratio.
        var_d_2: Variance of the denominator for the second ratio.
        cov_2: Covariance for the second ratio.
        n_2: Sample size for the second ratio.

    Returns:
        A `TestResult` named tuple with the test results.
        - statistic: The calculated z-statistic.
        - p_value: The two-tailed p-value.
        - ci: A `ConfidenceInterval` with lower and upper bounds.
        - diff_abs: The absolute difference between metrics.
        - diff_ratio: The relative difference between metrics.

    Raises:
        ValueError: If variances are negative, sample sizes are not positive,
            or denominator means are zero.
        TypeError: If arguments are not of the same type.
        ValueError: If np.ndarray arguments have different shapes.
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
    """Calculates the required sample size for a two-sample proportion z-test.

    Args:
        p1: Baseline proportion in the control group. Must be in (0, 1).
        effect_size: The relative effect size to detect (e.g., 0.1 for a 10%
            increase).
        alpha: The significance level (Type I error rate).
        beta: The Type II error rate (1 - power).

    Returns:
        The required sample size for each group.

    Raises:
        ValueError: If alpha or beta are not among predefined values, or if p1
            or the resulting p2 are not in the (0, 1) range.
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
    """Calculates the required sample size for a two-sample t-test.

    This function assumes equal variances between the two groups.

    Args:
        avg1: Mean of the control group.
        var: Pooled variance of the two groups. Must be non-negative.
        effect_size: The relative effect size to detect (e.g., 0.1 for a 10%
            increase from avg1). Cannot be zero.
        alpha: The significance level (Type I error rate).
        beta: The Type II error rate (1 - power).

    Returns:
        The required sample size for each group.

    Raises:
        ValueError: If effect_size is zero, var is negative, alpha or beta are
            not among predefined values, or if input types are inconsistent.
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
    """Calculates sample size for a ratio metric test using the delta method.

    This determines the sample size needed to detect a specified relative
    effect size in a ratio of two variables.

    Args:
        mean_n: Mean of the numerator.
        var_n: Variance of the numerator. Must be non-negative.
        mean_d: Mean of the denominator. Cannot be zero.
        var_d: Variance of the denominator. Must be non-negative.
        cov: Covariance between the numerator and denominator.
        effect_size: The relative effect size to detect. Cannot be zero.
        alpha: The significance level (Type I error rate).
        beta: The Type II error rate (1 - power).

    Returns:
        The required sample size for each group.

    Raises:
        ValueError: If effect_size or mean_d is zero, any variance is negative,
            or alpha or beta are not among predefined values.
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


# -------------------------- SAMPLE RATIO MISMATCH FUNCTION -------------------------- #


def sample_ratio_mismatch_test(
    observed_counts: list[int] | np.ndarray,
    expected_ratios: list[float] | np.ndarray | None = None,
    alpha: float = 1e-3, 
) -> SRMResult:
    """Sample Ratio Mismatch (SRM) via Pearson's chi-square goodness-of-fit.

    Args:
        observed_counts: 1-D array-like of non-negative integers per arm.
        expected_ratios: 1-D array-like of expected proportions (sum ~ 1).
                         If None, assumes uniform split.
        alpha: Significance level for flagging SRM.

    Returns:
        SRMResult(statistic, p_value, df, expected_counts, observed_counts, allocation, is_srm)
    """
    obs = np.asarray(observed_counts, dtype=float)
    if obs.ndim != 1:
        raise ValueError("observed_counts must be 1-D")
    if obs.size < 2:
        raise ValueError("observed_counts must contain at least 2 groups")
    if np.any(obs < 0):
        raise ValueError("All observed counts must be non-negative")

    N = obs.sum()
    if not np.isfinite(N) or N <= 0:
        raise ValueError("Total count must be positive")

    k = obs.size
    if expected_ratios is None:
        alloc = np.full(k, 1.0 / k, dtype=float)
    else:
        alloc = np.asarray(expected_ratios, dtype=float)
        if alloc.shape != (k,):
            raise ValueError("expected_ratios must have the same length as observed_counts")
        if np.any(alloc < 0):
            raise ValueError("All expected ratios must be non-negative")
        s = alloc.sum()
        if not np.isfinite(s) or s <= 0:
            raise ValueError("expected_ratios must sum to a positive number")
        if not np.isclose(alloc.sum(), 1):
            raise ValueError("expected_ratios must sum to 1")
        alloc = alloc / s

    expected = N * alloc
    if np.any(expected == 0):
        raise ValueError("Expected counts contain zeros; ensure expected_ratios > 0 and N > 0")

    # Pearson's chi-square
    stat = np.sum((obs - expected) ** 2 / expected)
    df = k - 1
    p = chi2.sf(stat, df)

    is_srm = bool(p < alpha)

    return SRMResult(
        statistic=float(stat),
        p_value=float(p),
        df=int(df),
        expected=expected.astype(float),
        observed=obs.astype(int),
        allocation=alloc.astype(float),
        is_srm=is_srm,
    )