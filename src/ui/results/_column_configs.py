"""This module defines the column configurations for the results table in the UI.

The `ST_TYPE_OF_COLUMNS_CONFIG` dictionary contains the configuration for each
column, including its type, label, help text, width, and other properties.
This configuration is used to render the results table in a Streamlit app.
"""

from typing import Any

ST_TYPE_OF_COLUMNS_CONFIG: dict[str, dict[str, Any]] = {
    "metric_display_name": {
        "type": "text",
        "label": "Display Metric Name",
        "help": "Metric name",
        "width": 180,
        "pinned": False,
    },
    "metric_name": {
        "type": "text",
        "label": "Yaml Metric Name ",
        "help": "Metric name",
        "width": "larger",
        "pinned": True,
    },
    "metric_type": {"type": "text", "label": "Metric Type", "help": "Metric Type", "pinned": True},
    "metric_value_control": {
        "type": "number",
        "label": "Control Value",
        "help": "Metric value in control group",
        "width": 60,
        "format": "%.3f",
    },
    "metric_value_compared": {
        "type": "number",
        "label": "Metric Value Compared",
        "help": "Metric value in compared group",
        "width": 60,
        "format": "%.3f",
    },
    "ci_lower": {
        "type": "number",
        "label": "CI Lower",
        "help": "Lower bound of the confidence interval",
        "width": "small",
        "format": "compact",
    },
    "ci_upper": {
        "type": "number",
        "label": "CI Upper",
        "help": "Upper bound of the confidence interval",
        "width": "small",
        "format": "compact",
    },
    "diff_abs": {
        "type": "number",
        "label": "Difference",
        "help": "Absolute difference between the metric values in the control and compared groups "
        "(in percentage points)",
        "width": "small",
        "format": "compact",
    },
    "diff_ratio": {
        "type": "number",
        "label": "Difference Ratio",
        "help": "Relative difference between the metric values in the control and compared groups "
        "(as a percentage of the control group)",
        "width": 60,
        "format": "percent",
    },
    "p_value": {
        "type": "number",
        "label": "P-Value",
        "help": "Probability of observing the result (or more extreme) "
        "if there is no real difference between the groups. Lower values indicate "
        "stronger evidence against the null hypothesis.",
        "width": 30,
        "format": "%.3f",
    },
    "statistic": {
        "type": "number",
        "label": "Test Statistic",
        "help": "Test statistic calculated from the chosen statistical test "
        "(e.g., t-test, z-test, or delta method). Used to derive the p-value.",
        "width": "small",
        "format": "compact",
    },
}
