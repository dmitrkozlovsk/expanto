from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field, model_validator

from src.domain.enums import ExperimentMetricType


class QueryMetricResult(BaseModel):
    """Represents the result of a metric query from the database.

    Contains all the statistical information needed for A/B test analysis
    including mean values, variances, and covariances for delta method calculations.

    Attributes:
        group_name: Name of the experimental group (e.g., 'control', 'treatment').
        metric_name: Technical name of the metric.
        metric_display_name: Human-readable name of the metric.
        metric_type: Type of the metric (ratio, proportion, or average).
        observation_cnt: Number of observations used in the calculation.
        metric_value: Computed value of the metric.
        numerator_avg: Average of the numerator for ratio metrics.
        numerator_var: Variance of the numerator for statistical calculations.
        denominator_avg: Average of the denominator for ratio metrics.
        denominator_var: Variance of the denominator for statistical calculations.
        covariance: Covariance between numerator and denominator.
    """

    group_name: str
    metric_name: str
    metric_display_name: str
    metric_type: ExperimentMetricType
    observation_cnt: int = Field(gt=0)
    metric_value: float | int
    numerator_avg: float | int | None = None
    numerator_var: float | int | None = None
    denominator_avg: float | int | None = None
    denominator_var: float | int | None = None
    covariance: float | int | None = None

    @model_validator(mode="after")
    def validate_variances_gt_0(self) -> QueryMetricResult:
        """Validate that variances are non-negative."""

        if self.numerator_var is not None and self.numerator_var < 0:
            raise ValueError(
                f"Numerator variance for metric '{self.metric_name}' must be greater or equal than 0"
            )
        if self.denominator_var is not None and self.denominator_var < 0:
            raise ValueError(
                f"Denominator variance for metric '{self.metric_name}' must be "
                f"must be greater or equal than 0"
            )
        return self

    @model_validator(mode="after")
    def validate_metric_values(self) -> QueryMetricResult:
        """Validate metric values based on their type."""

        if self.metric_type == ExperimentMetricType.PROPORTION:
            if not 0 <= self.metric_value <= 1:
                raise ValueError(f"<'{self.metric_name}'> Proportion metric value must be between 0 and 1")
            # if self.denominator_avg is not None and self.denominator_avg == 0:
            #     raise ValueError(f"<'{self.metric_name}'> Denominator average must be greater than 0")

        elif self.metric_type == ExperimentMetricType.RATIO:
            if any(
                x is None
                for x in [
                    self.numerator_avg,
                    self.numerator_var,
                    self.denominator_avg,
                    self.denominator_var,
                    self.covariance,
                ]
            ):
                raise ValueError(
                    f"<'{self.metric_name}'> Ratio metric requires all statistical values to be present"
                )
            # if self.denominator_avg == 0:
            #     raise ValueError(f"<'{self.metric_name}'> Denominator average must be greater than 0")
        elif self.metric_type == ExperimentMetricType.AVG and self.numerator_var is None:
            raise ValueError(f"<'{self.metric_name}'> Average metric requires numerator_var")
        return self


class MetricResult(QueryMetricResult):
    """Extends QueryMetricResult with job tracking information."""

    job_id: int


@dataclass
class JobResult:
    """Represents the execution result of a calculation job."""

    job_id: int | None
    success: bool
    metric_results: list[MetricResult] | None = None
    error_message: str | None = None

    @classmethod
    def success_result(cls, job_id: int, metric_results: list[MetricResult]) -> JobResult:
        """Create a successful job result with data."""
        return cls(job_id=job_id, success=True, metric_results=metric_results)

    @classmethod
    def error_result(cls, job_id: int | None, error_message: str) -> JobResult:
        """Create an error job result with error message."""
        return cls(job_id=job_id, success=False, error_message=error_message)
