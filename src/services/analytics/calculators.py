"""Statistical calculators for A/B testing experiments.

This module provides tools for calculating statistical significance and required sample sizes
for A/B tests. It supports three metric types:
    - averages (t-test),
    - proportions (z-test),
    - and ratios (delta method).
The module handles both significance testing between groups and
power analysis for experiment design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore

from src.domain.enums import ExperimentMetricType
from src.services.analytics.stat_functions import (
    delta_test_ratio_metric,
    sample_size_proportion_z_test,
    sample_size_ratio_metric,
    sample_size_t_test,
    ttest_welch,
    ztest_proportion,
)

if TYPE_CHECKING:
    from src.domain.results import MetricResult, QueryMetricResult


@dataclass
class SampleSizeCalculation:
    metric_name: str
    effect_sizes: np.ndarray
    sample_sizes: np.ndarray
    observations_per_day: float


class SignificanceCalculator:
    """Calculator for statistical significance between experiment groups.

    This class handles the calculation of statistical significance for different types
    of metrics (average, ratio, proportion) between a control group and other groups
    in an experiment.

    Args:
        precomputed_metrics_df (pd.DataFrame): DataFrame containing precomputed metrics
            for all groups. Must contain specific columns for metric values, variances,
            and observation counts.
        control_group (str | None, optional): Name of the control group. If None,
            will be automatically determined based on group names.

    Raises:
        ValueError: If required columns are missing or if there are insufficient groups.
        TypeError: If numeric columns contain non-numeric data.
    """

    def __init__(self, precomputed_metrics_df: pd.DataFrame, control_group: str | None = None) -> None:
        self.validate(precomputed_metrics_df)

        self.precomputed_metrics_df = precomputed_metrics_df
        self.significance_table = None
        self.control_group = self.define_control_group(control_group)
        self.compared_groups = [
            group for group in precomputed_metrics_df["group_name"].unique() if group != self.control_group
        ]
        self.significance_metrics_df_full = None

    def validate(self, precomputed_metrics_df):
        """Validates the input DataFrame for required columns and data types.

        Args:
            precomputed_metrics_df (pd.DataFrame): DataFrame to validate.

        Raises:
            ValueError: If required columns are missing or if DataFrame is empty.
            TypeError: If numeric columns contain non-numeric data.
        """
        object_columns = [
            "group_name",
            "metric_name",
            "metric_display_name",
            "metric_type",
        ]
        numeric_columns = [
            "observation_cnt",
            "metric_value",
            "numerator_avg",
            "denominator_avg",
            "numerator_var",
            "denominator_var",
            "covariance",
        ]

        required_columns = numeric_columns + object_columns
        missing_columns = [col for col in required_columns if col not in precomputed_metrics_df.columns]
        if len(missing_columns) > 0:
            raise ValueError(f"precomputed_metrics_df is missing required columns: {missing_columns}")
        if precomputed_metrics_df.empty:
            raise ValueError("precomputed_metrics_df is empty. Significance calculation is not possible.")
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(precomputed_metrics_df[col]):
                raise TypeError(f"Column {col} must be numeric")

    def define_control_group(self, control_group: str | None) -> str:
        """Defines the control group for the experiment.

        Args:
            control_group (str | None): Name of the control group. If None, will be
                automatically determined.

        Returns:
            str: Name of the control group.

        Raises:
            ValueError: If there are insufficient groups in the data.
        """
        if self.precomputed_metrics_df.group_name.nunique() <= 1:
            raise ValueError(
                "precomputed_metrics_df has zero or one group. "
                "Significance calculation is not possible. "
                "Check the data in the precomputed_metrics_df."
            )

        if control_group and control_group in self.precomputed_metrics_df["group_name"].unique():
            return control_group
        all_group_names = self.precomputed_metrics_df["group_name"].unique()
        for group_name in all_group_names:
            if isinstance(group_name, str) and "control" in group_name.lower():
                return group_name
        return all_group_names[0]

    @staticmethod
    def _calculate_avg_significance(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates statistical significance for average metrics using Welch's t-test.

        Args:
            df (pd.DataFrame): DataFrame containing average metrics data.

        Returns:
            pd.DataFrame: DataFrame with added significance metrics for average type.
        """
        avg_type_filter = df.metric_type == ExperimentMetricType.AVG
        avg_metrics_df = df[avg_type_filter]
        if avg_metrics_df.shape[0] > 0:
            avg_test_results = ttest_welch(
                mean_1=avg_metrics_df.metric_value_control.to_numpy(dtype=np.float64),
                var_1=avg_metrics_df.numerator_var_control.to_numpy(dtype=np.float64),
                n_1=avg_metrics_df.observation_cnt_control.to_numpy(dtype=np.float64),
                mean_2=avg_metrics_df.metric_value_compared.to_numpy(dtype=np.float64),
                var_2=avg_metrics_df.numerator_var_compared.to_numpy(dtype=np.float64),
                n_2=avg_metrics_df.observation_cnt_compared.to_numpy(dtype=np.float64),
            )

            df.loc[avg_type_filter, "statistic"] = avg_test_results.statistic
            df.loc[avg_type_filter, "p_value"] = avg_test_results.p_value
            df.loc[avg_type_filter, "ci_lower"] = avg_test_results.ci.lower
            df.loc[avg_type_filter, "ci_upper"] = avg_test_results.ci.upper
            df.loc[avg_type_filter, "diff_abs"] = avg_test_results.diff_abs
            df.loc[avg_type_filter, "diff_ratio"] = avg_test_results.diff_ratio
        return df

    @staticmethod
    def _calculate_ratio_significance(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates statistical significance for ratio metrics using delta method.

        Args:
            df (pd.DataFrame): DataFrame containing ratio metrics data.

        Returns:
            pd.DataFrame: DataFrame with added significance metrics for ratio type.
        """
        ratio_type_filter = df.metric_type == ExperimentMetricType.RATIO
        ratio_metrics_df = df[ratio_type_filter]
        if ratio_metrics_df.shape[0] > 0:
            ratio_metrics_results = delta_test_ratio_metric(
                metric_1=ratio_metrics_df.metric_value_control.to_numpy(dtype=np.float64),
                mean_n_1=ratio_metrics_df.numerator_avg_control.to_numpy(dtype=np.float64),
                var_n_1=ratio_metrics_df.numerator_var_control.to_numpy(dtype=np.float64),
                mean_d_1=ratio_metrics_df.denominator_avg_control.to_numpy(dtype=np.float64),
                var_d_1=ratio_metrics_df.denominator_var_control.to_numpy(dtype=np.float64),
                cov_1=ratio_metrics_df.covariance_control.to_numpy(dtype=np.float64),
                n_1=ratio_metrics_df.observation_cnt_control.to_numpy(dtype=np.float64),
                metric_2=ratio_metrics_df.metric_value_compared.to_numpy(dtype=np.float64),
                mean_n_2=ratio_metrics_df.numerator_avg_compared.to_numpy(dtype=np.float64),
                var_n_2=ratio_metrics_df.numerator_var_compared.to_numpy(dtype=np.float64),
                mean_d_2=ratio_metrics_df.denominator_avg_compared.to_numpy(dtype=np.float64),
                var_d_2=ratio_metrics_df.denominator_var_compared.to_numpy(dtype=np.float64),
                cov_2=ratio_metrics_df.covariance_compared.to_numpy(dtype=np.float64),
                n_2=ratio_metrics_df.observation_cnt_compared.to_numpy(dtype=np.float64),
            )

            df.loc[ratio_type_filter, "statistic"] = ratio_metrics_results.statistic
            df.loc[ratio_type_filter, "p_value"] = ratio_metrics_results.p_value
            df.loc[ratio_type_filter, "ci_lower"] = ratio_metrics_results.ci.lower
            df.loc[ratio_type_filter, "ci_upper"] = ratio_metrics_results.ci.upper
            df.loc[ratio_type_filter, "diff_abs"] = ratio_metrics_results.diff_abs
            df.loc[ratio_type_filter, "diff_ratio"] = ratio_metrics_results.diff_ratio
        return df

    @staticmethod
    def _calculate_proportion_significance(df: pd.DataFrame):
        """Calculates statistical significance for proportion metrics using z-test.

        Args:
            df (pd.DataFrame): DataFrame containing proportion metrics data.

        Returns:
            pd.DataFrame: DataFrame with added significance metrics for proportion type.
        """
        cr_type_filter = df.metric_type == ExperimentMetricType.PROPORTION
        cr_metrics_df = df[cr_type_filter]
        if cr_metrics_df.shape[0] > 0:
            cr_metrics_results = ztest_proportion(
                p_1=cr_metrics_df.metric_value_control.to_numpy(dtype=np.float64),
                n_1=cr_metrics_df.observation_cnt_control.to_numpy(dtype=np.float64),
                p_2=cr_metrics_df.metric_value_compared.to_numpy(dtype=np.float64),
                n_2=cr_metrics_df.observation_cnt_compared.to_numpy(dtype=np.float64),
            )
            df.loc[cr_type_filter, "statistic"] = cr_metrics_results.statistic
            df.loc[cr_type_filter, "p_value"] = cr_metrics_results.p_value
            df.loc[cr_type_filter, "ci_lower"] = cr_metrics_results.ci.lower
            df.loc[cr_type_filter, "ci_upper"] = cr_metrics_results.ci.upper
            df.loc[cr_type_filter, "diff_abs"] = cr_metrics_results.diff_abs
            df.loc[cr_type_filter, "diff_ratio"] = cr_metrics_results.diff_ratio
        return df

    def calculate_significance_table(self) -> pd.DataFrame:
        """Calculates significance table comparing control group with other groups.

        Performs statistical tests based on metric type:
        - Average metrics: Welch's t-test
        - Ratio metrics: Delta method test
        - Proportion metrics: Z-test for proportions

        Returns:
            pd.DataFrame: DataFrame containing significance metrics for each group
                and metric type comparison.
        """
        control_group_df = self.precomputed_metrics_df[
            self.precomputed_metrics_df.group_name == self.control_group
        ]
        compared_groups_df = self.precomputed_metrics_df[
            self.precomputed_metrics_df.group_name != self.control_group
        ]
        merge_df = pd.merge(
            control_group_df,
            compared_groups_df,
            how="left",
            on=["metric_name", "metric_type", "metric_display_name"],
            suffixes=("_control", "_compared"),
        )

        merge_df = SignificanceCalculator._calculate_avg_significance(merge_df)
        merge_df = SignificanceCalculator._calculate_ratio_significance(merge_df)
        merge_df = SignificanceCalculator._calculate_proportion_significance(merge_df)

        self.significance_metrics_df_full = merge_df
        return self.significance_metrics_df_full

    def get_metrics_significance_df(
        self, columns_to_show: list | None = None, groups_to_show: list | None = None
    ) -> pd.DataFrame:
        """Gets a formatted DataFrame of metrics significance.

        Args:
            columns_to_show (list, optional): List of columns to include in output.
                Defaults to None, which includes standard columns.
            groups_to_show (list, optional): List of group names to include.
                Defaults to None, which includes all compared groups.

        Returns:
            pd.DataFrame: DataFrame with significance metrics for specified groups
                and columns, formatted with multi-index columns.
        """
        if self.significance_metrics_df_full is None:
            self.calculate_significance_table()

        if groups_to_show is None:
            groups_to_show = self.compared_groups

        if columns_to_show is None:
            columns_to_show = [
                "metric_value_control",
                "metric_value_compared",
                "ci_lower",
                "ci_upper",
                "diff_abs",
                "diff_ratio",
                "p_value",
                "statistic",
            ]

        if self.significance_metrics_df_full is None:
            raise ValueError("Failed to calculate significance table")
        unstack_df = (
            self.significance_metrics_df_full.query("group_name_compared in @groups_to_show")
            .set_index(["group_name_compared", "metric_name", "metric_type", "metric_display_name"])
            .unstack(level=0)
            .reset_index()
            .swaplevel(0, 1, axis=1)
        )

        output_column_index = [
            (group_name, column) for group_name in groups_to_show for column in columns_to_show
        ] + [("", col) for col in ("metric_name", "metric_type", "metric_display_name")]
        result_df = unstack_df[output_column_index].copy()

        return result_df


class SampleSizeCalculator:
    """Calculator for required sample sizes in experiments.

    This class calculates the required sample sizes for different types of metrics
    based on desired effect sizes, statistical power, and significance level.
    """

    @staticmethod
    def calculate(
        metric_results: list[QueryMetricResult] | list[MetricResult],
        effect_sizes: list[float],
        reference_period_days: int,
        alpha: float = 0.05,
        beta: float = 0.2,
    ) -> list[SampleSizeCalculation]:
        """Calculates required sample sizes for experiment metrics.

        Args:
            metric_results (list[QueryMetricResult]): List of metric results to
                calculate sample sizes for.
            effect_sizes (list[float]): List of effect sizes to calculate for.
                If None, will use default range from 0.005 to 0.11.
            reference_period_days (int): Number of days in reference period.
            alpha (float, optional): Significance level. Defaults to 0.05.
            beta (float, optional): Type II error rate (1 - power). Defaults to 0.2.

        Returns:
            list[SampleSizeCalculation]: List of sample size calculations for each metric.

        Raises:
            ValueError: If input parameters are invalid or if no valid effect sizes
                are found for proportion metrics.
        """
        if reference_period_days is None or reference_period_days <= 0:
            raise ValueError("reference_period_days must be a positive integer")

        if alpha is None or alpha <= 0 or alpha >= 1:
            raise ValueError("alpha must be a number between 0 and 1")

        if beta is None or beta <= 0 or beta >= 1:
            raise ValueError("beta must be a number between 0 and 1")

        sample_size_results = []
        if effect_sizes:
            effect_sizes = pd.Series(effect_sizes, name="effect_size")
        else:
            effect_sizes = pd.Series(np.arange(0.005, 0.11, 0.005), name="effect_size")

        df = pd.merge(
            pd.DataFrame.from_records([r.model_dump() for r in metric_results]), effect_sizes, how="cross"
        )
        for metric_result in metric_results:
            metric_slice = df.loc[df.metric_name == metric_result.metric_name]

            if metric_result.metric_type == ExperimentMetricType.PROPORTION:
                p2 = metric_slice.metric_value * (1 + metric_slice.effect_size)
                valid_rows = (p2 < 1) & (p2 > 0)

                metric_slice_proportion = metric_slice[valid_rows]
                metric_effect_size = metric_slice_proportion.effect_size.to_numpy(dtype=np.float64)
                if metric_slice_proportion.empty:
                    raise ValueError(
                        f"No valid effect_size/proportion combinations "
                        f"for metric '{metric_result.metric_name}'"
                    )
                metric_sample_sizes = sample_size_proportion_z_test(
                    p1=metric_slice_proportion.metric_value.to_numpy(dtype=np.float64),
                    effect_size=metric_slice_proportion.effect_size.to_numpy(dtype=np.float64),
                    alpha=alpha,
                    beta=beta,
                )
            elif metric_result.metric_type == ExperimentMetricType.AVG:
                metric_effect_size = metric_slice.effect_size.to_numpy(dtype=np.float64)
                metric_sample_sizes = sample_size_t_test(
                    avg1=metric_slice.metric_value.to_numpy(dtype=np.float64),
                    var=metric_slice.numerator_var.to_numpy(dtype=np.float64),
                    effect_size=metric_slice.effect_size.to_numpy(dtype=np.float64),
                    alpha=alpha,
                    beta=beta,
                )
            elif metric_result.metric_type == ExperimentMetricType.RATIO:
                metric_effect_size = metric_slice.effect_size.to_numpy(dtype=np.float64)
                metric_sample_sizes = sample_size_ratio_metric(
                    mean_n=metric_slice.numerator_avg.to_numpy(dtype=np.float64),
                    var_n=metric_slice.numerator_var.to_numpy(dtype=np.float64),
                    mean_d=metric_slice.denominator_avg.to_numpy(dtype=np.float64),
                    var_d=metric_slice.denominator_var.to_numpy(dtype=np.float64),
                    cov=metric_slice.covariance.to_numpy(dtype=np.float64),
                    effect_size=metric_slice.effect_size.to_numpy(dtype=np.float64),
                    alpha=alpha,
                    beta=beta,
                )
            else:
                raise ValueError(f"Unsupported metric type: {metric_result.metric_type}")

            observations_per_day = metric_result.observation_cnt / reference_period_days

            sample_size_results.append(
                SampleSizeCalculation(
                    metric_name=metric_result.metric_name,
                    effect_sizes=metric_effect_size,
                    sample_sizes=np.array(metric_sample_sizes, dtype=np.float64),
                    observations_per_day=observations_per_day,
                )
            )
        return sample_size_results
