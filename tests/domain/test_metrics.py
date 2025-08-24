from typing import Any

import pytest
from pydantic import ValidationError

from src.domain.enums import ExperimentMetricType
from src.domain.metrics import (
    ExperimentMetricDefinition,
    MetricFormula,
    UserAggregationFormula,
    YamlGroupOfMetrics,
    YamlMetrics,
)


# ---------------------------------- UserAggregationFormula ----------------------------------
@pytest.mark.parametrize(
    "alias, sql, expected",
    [
        (
            "banner_clicks",
            "COUNT(CASE event_name WHEN 'banner_clicked' THEN event_name ELSE null END)",
            "COUNT(CASE event_name WHEN 'banner_clicked' THEN event_name ELSE null END)",
        ),
        (
            "orders_value",
            "SUM(CASE event_name WHEN 'order_placed' THEN event_value ELSE 0 END)",
            "SUM(CASE event_name WHEN 'order_placed' THEN event_value ELSE 0 END)",
        ),
    ],
)
def test_user_aggregation_formula_valid(alias: str, sql: str, expected: str) -> None:
    """Test valid UserAggregationFormula creation with correct alias and SQL."""
    formula = UserAggregationFormula(alias=alias, sql=sql)
    assert formula.sql == expected
    assert formula.alias == alias


@pytest.mark.parametrize(
    "alias, sql, validation_error_message",
    [
        ("banner_clicks", "SELECT * FROM events", "SQL expression has invalid keywords"),
        ("orders_value", "SUM(field", "SQL expression has unbalanced parentheses"),
        ("clicks", " ", "SQL expression cannot be empty"),
        ("test", "COUNT(field", "SQL expression has unbalanced parentheses"),
        ("SELECT", "SUM(col)", "Alias expression has invalid keywords"),
        ("fa fa fa", "SUM(col)", "alias must be lower-case"),
        ("DELETED", "SUM(col)", "alias must be lower-case"),
        ("54", "SUM(col)", "alias must be lower-case"),
        (".*", "SUM(col)", "alias must be lower-case"),
        ("total--comment", "SUM(col)", "alias must be lower-case"),
    ],
)
def test_user_aggregation_formula_invalid(alias: str, sql: str, validation_error_message: str) -> None:
    """Test UserAggregationFormula validation errors for invalid inputs."""
    with pytest.raises(ValidationError, match=validation_error_message):
        UserAggregationFormula(alias=alias, sql=sql)


# ------------------------------- ExperimentMetricDefinition -------------------------------
@pytest.mark.parametrize(
    "metric_data, validation_error_message",
    [
        (
            {
                "alias": "test_metric",
                "type": "invalid_type",
                "display_name": "Test Metric",
                "description": "",
                "formula": MetricFormula(
                    numerator=UserAggregationFormula(alias="test", sql="COUNT(test)"), denominator=None
                ),
                "owner": None,
                "tags": None,
            },
            "Input should be",
        ),
        (
            {
                "alias": "test_metric",
                "type": "ratio",
                "display_name": "Test Metric",
                "description": "",
                "formula": MetricFormula(
                    numerator=UserAggregationFormula(alias="test", sql="COUNT(test)"), denominator=None
                ),
                "owner": None,
                "tags": None,
            },
            "Denominator is required for ratio type",
        ),
    ],
)
def test_experiment_metric_definition_invalid(
    metric_data: dict[str, Any], validation_error_message: str
) -> None:
    """Test ExperimentMetricDefinition validation errors for invalid configurations."""
    with pytest.raises(ValidationError, match=validation_error_message):
        ExperimentMetricDefinition(**metric_data)


@pytest.mark.parametrize(
    "metric_inputs, expected_formula",
    [
        (("x", "y", "ratio"), "CASE WHEN SUM(y) > 0 THEN SUM(x) / SUM(y) ELSE 0 END"),
        (("n", None, "proportion"), "AVG(n)"),
        (("n", "d", "proportion"), "CASE WHEN SUM(d) > 0 THEN SUM(n) / SUM(d) ELSE 0 END"),
        (("z", "d", "avg"), "AVG(z)"),
        (("n", None, "avg"), "AVG(n)"),
    ],
)
def test_experiment_metric_definition_sql(
    metric_inputs: tuple[str, str | None, str], expected_formula: str
) -> None:
    """Test SQL formula generation for different metric types and configurations."""
    numerator_str = metric_inputs[0]
    denominator_str = metric_inputs[1]
    metric_type = metric_inputs[2]

    metric_def = ExperimentMetricDefinition(
        alias="test_metric",
        type=ExperimentMetricType(metric_type),
        display_name="test",
        description=None,
        formula=MetricFormula(
            numerator=UserAggregationFormula(alias=numerator_str, sql=f"SUM({numerator_str})"),
            denominator=UserAggregationFormula(alias=denominator_str, sql=f"SUM({denominator_str})")
            if denominator_str
            else None,
        ),
        owner=None,
        tags=None,
    )
    assert metric_def.sql == expected_formula


@pytest.fixture
def yaml_metric_groups_fixture() -> list[YamlGroupOfMetrics]:
    group1 = YamlGroupOfMetrics(
        metric_group_name="group_1",
        user_aggregations={"sum_revenue": UserAggregationFormula(alias="sum_revenue", sql="SUM(revenue)")},
        metrics=[
            ExperimentMetricDefinition(
                alias="avg_revenue",
                type=ExperimentMetricType("avg"),
                display_name="Average Revenue",
                description=None,
                formula=MetricFormula(
                    numerator=UserAggregationFormula(alias="sum_revenue", sql="SUM(revenue)"),
                    denominator=None,
                ),
                owner="team_a",
                tags=["financial"],
            )
        ],
    )

    group2 = YamlGroupOfMetrics(
        metric_group_name="group_2",
        user_aggregations={
            "count_users": UserAggregationFormula(alias="count_users", sql="COUNT(user_id)")
        },
        metrics=[
            ExperimentMetricDefinition(
                alias="user_count_avg",
                type=ExperimentMetricType("avg"),
                display_name="User Count Avg",
                description=None,
                formula=MetricFormula(
                    numerator=UserAggregationFormula(alias="count_users", sql="COUNT(user_id)"),
                    denominator=None,
                ),
                owner="team_b",
                tags=["users"],
            )
        ],
    )

    group3 = YamlGroupOfMetrics(
        metric_group_name="group_3",
        user_aggregations={
            "count_users": UserAggregationFormula(alias="count_users", sql="COUNT(user_id)")
        },
        metrics=[
            ExperimentMetricDefinition(
                alias="avg_revenue",
                type=ExperimentMetricType("avg"),
                display_name="User Count Avg",
                description=None,
                formula=MetricFormula(
                    numerator=UserAggregationFormula(alias="count_users", sql="COUNT(user_id)"),
                    denominator=None,
                ),
                owner="team_b",
                tags=["users"],
            )
        ],
    )

    return [group1, group2, group3]


def test_yaml_metrics_invalid(yaml_metric_groups_fixture: list[YamlGroupOfMetrics]) -> None:
    """Test YamlMetrics validation for duplicate aliases across groups."""
    group1, group2, group3 = yaml_metric_groups_fixture
    with pytest.raises(ValidationError, match="Duplicate user aggregation alias found"):
        YamlMetrics(yaml_groups=[group2, group3])
    with pytest.raises(ValidationError, match="Duplicate experiment metric alias found"):
        YamlMetrics(yaml_groups=[group1, group3])
