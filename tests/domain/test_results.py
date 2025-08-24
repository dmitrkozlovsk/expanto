import pytest
from pydantic import ValidationError

from src.domain.enums import ExperimentMetricType
from src.domain.results import QueryMetricResult


def test_proportion_metric_with_invalid_bounds():
    """Test proportion metric validation rejects values outside 0-1 range."""
    with pytest.raises(ValidationError, match="must be between 0 and 1"):
        QueryMetricResult(
            group_name="test",
            metric_name="conversion",
            metric_display_name="Conversion",
            metric_type="proportion",
            observation_cnt=100,
            metric_value=1.5,
        )


def test_ratio_metric_missing_required_fields():
    """Test ratio metric validation requires all statistical fields."""
    with pytest.raises(ValidationError, match="Ratio metric requires all statistical values"):
        QueryMetricResult(
            group_name="test",
            metric_name="ctr",
            metric_display_name="Click Through Rate",
            metric_type=ExperimentMetricType.RATIO,
            observation_cnt=100,
            metric_value=0.12,
            numerator_avg=10.0,
            numerator_var=2.0,
        )


def test_avg_metric_missing_variance():
    """Test average metric validation requires numerator variance."""
    with pytest.raises(ValidationError, match="Average metric requires numerator_var"):
        QueryMetricResult(
            group_name="test",
            metric_name="avg_duration",
            metric_display_name="Average Duration",
            metric_type=ExperimentMetricType.AVG,
            observation_cnt=100,
            metric_value=12.0,
            numerator_avg=12.0,
        )


def test_negative_variance_raises_error():
    """Test validation rejects negative variance values."""
    with pytest.raises(ValidationError, match="Numerator variance for metric"):
        QueryMetricResult(
            group_name="test",
            metric_name="avg_time",
            metric_display_name="Average Time",
            metric_type=ExperimentMetricType.AVG,
            observation_cnt=100,
            metric_value=5.0,
            numerator_avg=5.0,
            numerator_var=-0.01,
        )
