import math

import numpy as np
import pandas as pd  # type: ignore
import pytest
from scipy.stats import ttest_ind_from_stats  # type: ignore
from statsmodels.stats.proportion import proportions_ztest  # type: ignore

from src.domain.enums import ExperimentMetricType
from src.services.analytics.calculators import SampleSizeCalculator, SignificanceCalculator
from src.services.analytics.stat_functions import (
    sample_size_proportion_z_test,
    sample_size_ratio_metric,
    sample_size_t_test,
)


# ----------------------------- Test SignificanceCalculator -----------------------------
@pytest.fixture(scope="function")
def valid_precomputes_df():
    """Fixture providing valid DataFrame with precomputed metrics for testing."""
    data = []
    groups = ["control", "treatment1", "treatment2"]
    for metric_type in ExperimentMetricType:
        for i in range(2):
            metric_name = f"{metric_type.value}_metric_{i + 1}"
            metric_display_name = f"{metric_type.value} Metric {i + 1}"

            for group in groups:
                # Generate base values
                if metric_type == ExperimentMetricType.AVG:
                    metric_value = (
                        np.random.normal(100, 10) if group == "control" else np.random.normal(110, 10)
                    )
                    numerator_var = np.random.uniform(50, 100)
                    observation_cnt = np.random.randint(1000, 2000)
                    row = {
                        "group_name": group,
                        "metric_name": metric_name,
                        "metric_display_name": metric_display_name,
                        "metric_type": metric_type,
                        "metric_value": metric_value,
                        "numerator_var": numerator_var,
                        "observation_cnt": observation_cnt,
                    }

                elif metric_type == ExperimentMetricType.RATIO:
                    numerator_avg = (
                        np.random.normal(100, 10) if group == "control" else np.random.normal(110, 10)
                    )
                    denominator_avg = np.random.normal(200, 20)
                    numerator_var = np.random.uniform(50, 100)
                    denominator_var = np.random.uniform(100, 200)
                    covariance = np.random.uniform(-50, 50)
                    metric_value = numerator_avg / denominator_avg
                    observation_cnt = np.random.randint(1000, 2000)
                    row = {
                        "group_name": group,
                        "metric_name": metric_name,
                        "metric_display_name": metric_display_name,
                        "metric_type": metric_type,
                        "metric_value": metric_value,
                        "numerator_avg": numerator_avg,
                        "denominator_avg": denominator_avg,
                        "numerator_var": numerator_var,
                        "denominator_var": denominator_var,
                        "covariance": covariance,
                        "observation_cnt": observation_cnt,
                    }

                elif metric_type == ExperimentMetricType.PROPORTION:
                    metric_value = (
                        np.random.uniform(0.1, 0.3) if group == "control" else np.random.uniform(0.2, 0.4)
                    )
                    observation_cnt = np.random.randint(10000, 20000)
                    row = {
                        "group_name": group,
                        "metric_name": metric_name,
                        "metric_display_name": metric_display_name,
                        "metric_type": metric_type,
                        "metric_value": metric_value,
                        "observation_cnt": observation_cnt,
                    }

                data.append(row)

    return pd.DataFrame(data)


@pytest.fixture
def empty_precomputes_df():
    """Fixture that returns an empty DataFrame with the same structure as mock_precomputes_df"""
    return pd.DataFrame(
        columns=[
            "group_name",
            "metric_name",
            "metric_display_name",
            "metric_type",
            "observation_cnt",
            "metric_value",
            "numerator_avg",
            "denominator_avg",
            "numerator_var",
            "denominator_var",
            "covariance",
        ]
    )


def test_validate_missing_column(valid_precomputes_df):
    """Test SignificanceCalculator validates required columns are present."""
    df = valid_precomputes_df.drop(columns=["numerator_avg"])
    with pytest.raises(ValueError, match="missing required columns"):
        SignificanceCalculator(df)


def test_validate_wrong_dtype(valid_precomputes_df):
    """Test SignificanceCalculator validates column data types."""
    valid_precomputes_df["observation_cnt"] = "a"
    with pytest.raises(TypeError, match="must be numeric"):
        SignificanceCalculator(valid_precomputes_df)


def test_define_control_group_explicit(valid_precomputes_df):
    """Test explicit control group assignment in SignificanceCalculator."""
    calc = SignificanceCalculator(valid_precomputes_df, control_group="treatment1")
    assert calc.control_group == "treatment1"


def test_define_control_group_implicit(valid_precomputes_df):
    """Test implicit control group detection by name containing 'control'."""
    calc = SignificanceCalculator(valid_precomputes_df)
    assert "control" in calc.control_group


def test_calculator_trivial(valid_precomputes_df):
    """Test basic functionality of significance calculation."""
    calculator = SignificanceCalculator(valid_precomputes_df, control_group=None)
    df = calculator.get_metrics_significance_df()
    columns_set = set([col[1] for col in df.columns])
    assert {
        "diff_ratio",
        "metric_type",
        "p_value",
        "diff_abs",
        "metric_value_compared",
        "statistic",
        "metric_display_name",
        "metric_value_control",
        "ci_lower",
        "ci_upper",
        "metric_name",
    } == columns_set


def test_proportion_metric_statistical_correctness(valid_precomputes_df):
    """Test proportion metric calculations match statsmodels results."""

    calculator = SignificanceCalculator(valid_precomputes_df)
    result_df = calculator.calculate_significance_table()

    ratio_filter = result_df.metric_type == ExperimentMetricType.PROPORTION
    for metric in result_df[ratio_filter].to_dict(orient="records"):
        metric_p_value = metric["p_value"]
        metric_statistic = metric["statistic"]

        count_control = float(metric["observation_cnt_control"] * metric["metric_value_control"])
        count_treatment = float(metric["observation_cnt_compared"] * metric["metric_value_compared"])
        n_control = int(metric["observation_cnt_control"])
        n_treatment = int(metric["observation_cnt_compared"])

        sm_stat, sm_p_value = proportions_ztest([count_treatment, count_control], [n_treatment, n_control])
        assert math.isclose(sm_stat, metric_statistic, abs_tol=0.0001, rel_tol=0.0001), (
            f"Statistic mismatch: {sm_stat} vs {metric_statistic}"
        )
        assert math.isclose(sm_p_value, metric_p_value, abs_tol=0.0001, rel_tol=0.0001), (
            f"P-value mismatch: {sm_p_value} vs {metric_p_value}"
        )


def test_average_metric_statistical_correctness(valid_precomputes_df):
    """Test average metric calculations match scipy.stats results."""
    calculator = SignificanceCalculator(valid_precomputes_df)
    result_df = calculator.calculate_significance_table()

    avg_filter = result_df.metric_type == ExperimentMetricType.AVG
    for metric in result_df[avg_filter].to_dict(orient="records"):
        metric_p_value = metric["p_value"]
        metric_statistic = metric["statistic"]

        tt_stat, tt_p_value = ttest_ind_from_stats(
            mean1=metric["metric_value_control"],
            std1=np.sqrt(metric["numerator_var_control"]),
            nobs1=metric["observation_cnt_control"],
            mean2=metric["metric_value_compared"],
            std2=np.sqrt(metric["numerator_var_compared"]),
            nobs2=metric["observation_cnt_compared"],
            equal_var=False,
        )

        # Compare statistics and p-values
        assert math.isclose(abs(tt_stat), abs(metric_statistic), abs_tol=0.0001, rel_tol=0.0001), (
            f"Statistic mismatch: {tt_stat} vs {metric_statistic}"
        )

        assert math.isclose(abs(tt_p_value), abs(metric_p_value), abs_tol=0.0001, rel_tol=0.0001), (
            f"P-value mismatch: {tt_p_value} vs {metric_p_value}"
        )


def test_ratio_metric_statistical_correctness(valid_precomputes_df):
    """Test that ratio metric calculations in calculator match our statistical functions"""
    calculator = SignificanceCalculator(valid_precomputes_df)
    result_df = calculator.calculate_significance_table()

    ratio_filter = result_df.metric_type == ExperimentMetricType.RATIO
    for metric in result_df[ratio_filter].to_dict(orient="records"):
        metric_p_value = metric["p_value"]
        metric_statistic = metric["statistic"]

        # Import the ratio_metric_test function from stat_functions
        from src.services.analytics.stat_functions import (
            ratio_metric_sample_variance,
            ratio_metric_test,
        )

        # Calculate the variances using ratio_metric_sample_variance
        var_control = (
            ratio_metric_sample_variance(
                mean_n=metric["numerator_avg_control"],
                var_n=metric["numerator_var_control"],
                mean_d=metric["denominator_avg_control"],
                var_d=metric["denominator_var_control"],
                cov=metric["covariance_control"],
            )
            / metric["observation_cnt_control"]
        )

        var_compared = (
            ratio_metric_sample_variance(
                mean_n=metric["numerator_avg_compared"],
                var_n=metric["numerator_var_compared"],
                mean_d=metric["denominator_avg_compared"],
                var_d=metric["denominator_var_compared"],
                cov=metric["covariance_compared"],
            )
            / metric["observation_cnt_compared"]
        )

        test_result = ratio_metric_test(
            metric_value_1=metric["metric_value_control"],
            variance_1=var_control,
            metric_value_2=metric["metric_value_compared"],
            variance_2=var_compared,
        )

        assert math.isclose(
            abs(test_result.statistic), abs(metric_statistic), abs_tol=0.0001, rel_tol=0.0001
        ), f"Statistic mismatch: {test_result.statistic} vs {metric_statistic}"
        assert math.isclose(
            abs(test_result.p_value), abs(metric_p_value), abs_tol=0.0001, rel_tol=0.0001
        ), f"P-value mismatch: {test_result.p_value} vs {metric_p_value}"


# ------------------------------ Test SampleSizeCalculator ------------------------------


@pytest.fixture
def valid_metric_results():
    """Fixture providing valid QueryMetricResult objects for sample size calculation tests."""
    from src.domain.results import QueryMetricResult

    results = [
        # PROPORTION metric
        QueryMetricResult(
            group_name="planning",
            metric_name="conversion_rate",
            metric_display_name="Conversion Rate",
            metric_type=ExperimentMetricType.PROPORTION,
            metric_value=0.15,
            observation_cnt=10000,
            numerator_avg=0.15,
            numerator_var=0.15 * 0.85,  # p * (1-p)
            denominator_avg=1,
            denominator_var=1,
            covariance=0,
        ),
        # AVG metric
        QueryMetricResult(
            group_name="planning",
            metric_name="average_order_value",
            metric_display_name="Average Order Value",
            metric_type=ExperimentMetricType.AVG,
            metric_value=50.0,
            observation_cnt=5000,
            numerator_avg=50.0,
            numerator_var=625.0,  # some variance for the average value
            denominator_avg=1,
            denominator_var=100,
            covariance=0,
        ),
        # RATIO metric
        QueryMetricResult(
            group_name="planning",
            metric_name="revenue_per_user",
            metric_display_name="Revenue per User",
            metric_type=ExperimentMetricType.RATIO,
            metric_value=25.0,
            observation_cnt=8000,
            numerator_avg=100.0,  # total revenue
            numerator_var=10000.0,
            denominator_avg=4.0,  # number of orders
            denominator_var=6.0,
            covariance=100.0,  # some covariance between revenue and orders
        ),
    ]
    return results


def test_sample_size_calculator_all_metrics_trivial(valid_metric_results):
    """Test SampleSizeCalculator with all metric types."""
    from src.services.analytics.calculators import SampleSizeCalculator

    effect_sizes = [0.05, 0.1, 0.15]
    reference_period_days = 14

    results = SampleSizeCalculator.calculate(
        metric_results=valid_metric_results,
        effect_sizes=effect_sizes,
        reference_period_days=reference_period_days,
    )

    assert len(results) == len(valid_metric_results), "Should return results for each input metric"

    for i, result in enumerate(results):
        assert result.metric_name == valid_metric_results[i].metric_name, (
            f"Metric name mismatch for item {i}"
        )
        assert len(result.effect_sizes) == len(effect_sizes), (
            f"Effect sizes length mismatch for metric {result.metric_name}"
        )
        assert len(result.sample_sizes) == len(effect_sizes), (
            f"Sample sizes length mismatch for metric {result.metric_name}"
        )
        assert (
            result.observations_per_day == valid_metric_results[i].observation_cnt / reference_period_days
        ), f"Observations per day mismatch for {result.metric_name}"


def test_sample_size_calculator_statistical_correctness(valid_metric_results):
    """Test that sample size calculations match expected statistical functions."""

    # Test parameters
    effect_sizes = [0.05, 0.1]
    reference_period_days = 14
    alpha = 0.05
    beta = 0.2

    # Calculate sample sizes using the calculator
    results = SampleSizeCalculator.calculate(
        metric_results=valid_metric_results,
        effect_sizes=effect_sizes,
        reference_period_days=reference_period_days,
        alpha=alpha,
        beta=beta,
    )

    # Check each metric type individually
    for result in results:
        metric = next(m for m in valid_metric_results if m.metric_name == result.metric_name)

        if metric.metric_type == ExperimentMetricType.PROPORTION:
            # Calculate expected sample sizes directly
            expected_sizes = sample_size_proportion_z_test(
                p1=np.full(len(effect_sizes), metric.metric_value),
                effect_size=np.array(effect_sizes),
                alpha=alpha,
                beta=beta,
            )
            np.testing.assert_allclose(result.sample_sizes, expected_sizes, rtol=1e-4)

        elif metric.metric_type == ExperimentMetricType.AVG:
            # Calculate expected sample sizes directly
            expected_sizes = sample_size_t_test(
                avg1=np.full(len(effect_sizes), metric.metric_value),
                var=np.full(len(effect_sizes), metric.numerator_var),
                effect_size=np.array(effect_sizes),
                alpha=alpha,
                beta=beta,
            )
            np.testing.assert_allclose(result.sample_sizes, expected_sizes, rtol=1e-6)

        elif metric.metric_type == ExperimentMetricType.RATIO:
            # Calculate expected sample sizes directly

            expected_sizes = sample_size_ratio_metric(
                mean_n=np.full(len(effect_sizes), metric.numerator_avg),
                var_n=np.full(len(effect_sizes), metric.numerator_var),
                mean_d=np.full(len(effect_sizes), metric.denominator_avg),
                var_d=np.full(len(effect_sizes), metric.denominator_var),
                cov=np.full(len(effect_sizes), metric.covariance),
                effect_size=np.array(effect_sizes),
                alpha=alpha,
                beta=beta,
            )
            np.testing.assert_allclose(result.sample_sizes, expected_sizes, rtol=1e-6)
