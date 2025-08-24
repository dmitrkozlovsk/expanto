from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from src.domain.metrics import ExperimentMetricDefinition
from src.services.metric_register import Metrics


def test_flat_keys(metrics_temp_dir: Path) -> None:
    """Test flat metrics dictionary contains expected metric aliases."""
    metrics = Metrics(metrics_temp_dir)
    flat = metrics.flat
    assert "avg_order_value" in flat
    assert "click_through_rate" in flat
    assert "avg_session_duration" in flat


def test_get_known_aliases(metrics_temp_dir: Path) -> None:
    """Test retrieval of metrics by valid and invalid aliases."""
    metrics = Metrics(metrics_temp_dir)
    result = metrics.get(["avg_order_value", "fa fa fa invalid metric_alias"])
    assert len(result) == 1
    result = metrics.get(["avg_order_value", "click_through_rate"])
    assert len(result) == 2
    assert isinstance(result[0], ExperimentMetricDefinition)


def test_resolve_returns_metric_and_formulas(metrics_temp_dir: Path) -> None:
    """Test resolve method returns experiment metrics and user formulas."""
    metrics = Metrics(metrics_temp_dir)
    exp_metrics, user_formulas = metrics.resolve(["avg_order_value"])
    assert len(exp_metrics) == 1
    assert exp_metrics[0].alias == "avg_order_value"
    assert len(user_formulas) == 2
    assert {f.alias for f in user_formulas} == {"product_revenue_sum", "product_purchase_cnt"}


# --------------------------------- FILTER TESTS ---------------------------------


def test_filter_by_type(metrics_temp_dir: Path) -> None:
    """Test filtering metrics by metric type."""
    metrics = Metrics(metrics_temp_dir)
    result = metrics.filter(types=["avg"])
    assert set(result.keys()) == {"avg_session_duration", "conversion_rate"}


def test_filter_by_tags(metrics_temp_dir: Path) -> None:
    """Test filtering metrics by tags."""
    metrics = Metrics(metrics_temp_dir)
    result = metrics.filter(tags=["sales", "time"])
    assert set(result.keys()) == {"avg_order_value", "avg_session_duration", "conversion_rate"}


def test_filter_by_tags_and_type(metrics_temp_dir: Path) -> None:
    """Test filtering metrics by both tags and type."""
    metrics = Metrics(metrics_temp_dir)
    result = metrics.filter(tags=["time"], types=["avg"])
    assert set(result.keys()) == {"avg_session_duration"}


def test_filter_by_nonexistent_tag(metrics_temp_dir: Path) -> None:
    """Test filtering by nonexistent tag returns empty result."""
    metrics = Metrics(metrics_temp_dir)
    result = metrics.filter(tags=["nonexistent"])
    assert result == {}


def test_filter_by_group_name(metrics_temp_dir: Path) -> None:
    """Test filtering metrics by group name."""
    metrics = Metrics(metrics_temp_dir)
    result = metrics.filter(group_names=["sales_ratios"])
    result = metrics.filter(group_names=["sales_ratios"])
    assert set(result.keys()) == set()


def test_filter_by_multiple_group_names(metrics_temp_dir: Path) -> None:
    """Test filtering metrics by multiple group names."""
    metrics = Metrics(metrics_temp_dir)
    result = metrics.filter(group_names=["Order Metrics", "User Behavior and Engagement Metrics"])
    assert set(result.keys()) == {
        "avg_order_value",
        "click_through_rate",
        "avg_session_duration",
        "users_who_click_product_ratio",
        "conversion_rate",
    }


def test_filter_by_group_name_and_type(metrics_temp_dir: Path) -> None:
    """Test filtering metrics by group name and type combination."""
    metrics = Metrics(metrics_temp_dir)
    result = metrics.filter(group_names=["User Behavior and Engagement Metrics"], types=["proportion"])
    assert set(result.keys()) == {"users_who_click_product_ratio"}
