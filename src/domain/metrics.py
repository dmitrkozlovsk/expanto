import re
from typing import Annotated

from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    computed_field,
    constr,
    field_validator,
    model_validator,
)

from src.domain.enums import ExperimentMetricType

INVALID_SQL_KEYWORDS = [
    "SELECT",
    "FROM",
    "INSERT",
    "DELETE",
    "CREATE",
    "MERGE",
    "UPDATE",
    "JOIN",
    "WHERE",
    "GROUP",
    "HAVING",
]

SQL_ALIAS_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
INVALID_SQL_KEYWORDS_PATTERN = r"\b(?:" + "|".join(INVALID_SQL_KEYWORDS) + r")\b"


def are_parentheses_balanced(expression: str) -> bool:
    """
    Checks if the parentheses in the expression are balanced.
    Helper function to check brackets in a SQL-like formula expression.
    """
    stack = []
    for char in expression:
        if char == "(":
            stack.append(char)
        elif char == ")":
            if not stack:
                return False
            stack.pop()
    return len(stack) == 0


def _no_sql_invalid_keywords(v: str) -> str:
    """Validate that SQL alias does not contain invalid SQL keywords.

    Args:
        v (str): The alias string to validate.

    Returns:
        str: The validated alias string.

    Raises:
        ValueError: If the alias contains invalid SQL keywords.
    """
    if re.search(INVALID_SQL_KEYWORDS_PATTERN, v, re.IGNORECASE):
        raise ValueError(f"<{v}> Alias expression has invalid keywords: {', '.join(INVALID_SQL_KEYWORDS)}.")
    return v


def _match_alias_pattern(v: str) -> str:
    """Validate that SQL alias matches the required pattern.

    Args:
        v (str): The alias string to validate.

    Returns:
        str: The validated alias string.

    Raises:
        ValueError: If the alias doesn't match the required pattern.
    """
    if not SQL_ALIAS_RE.fullmatch(v):
        raise ValueError(
            f"<{v}> alias must be lower-case, start with a letter or underscore, "
            "and contain only letters, digits or underscores."
        )
    return v


SQLAlias = Annotated[
    str,
    constr(min_length=1, max_length=50),
    AfterValidator(_no_sql_invalid_keywords),
    AfterValidator(_match_alias_pattern),
]


class UserAggregationFormula(BaseModel):
    """
    Represents a user-level aggregation formula in SQL-like syntax.

    Attributes:
        alias (str): A unique identifier for this aggregation formula.
        sql (str): The SQL-like formula expression.
    """

    alias: SQLAlias
    sql: str

    @field_validator("sql")
    def validate_sql(cls, value: str) -> str:
        """
        Validates SQL-like expressions on base level.

        Performs several validation checks:
        1. Checks for balanced parentheses
        2. Validates against invalid SQL keywords
        3. Ensures non-empty expressions
        4. Verifies proper aggregation function usage
        5. Checks proper expression closure

        """
        # check brackets
        if not are_parentheses_balanced(value):
            raise ValueError(
                f"<{value}> SQL expression has unbalanced parentheses."
                "Please, check the brackets in the formula description"
            )

        # check invalid keywords
        statement_pattern = r"\b(?:" + "|".join(INVALID_SQL_KEYWORDS) + r")\b"
        if re.search(statement_pattern, value, re.IGNORECASE):
            raise ValueError(
                f"<{value}> SQL expression has invalid keywords {', '.join(INVALID_SQL_KEYWORDS)}."
            )

        # check for empty or whitespace-only expressions
        if not value.strip():
            raise ValueError(f"<{value}> SQL expression cannot be empty.")

        if "--" in value or "/*" in value or "*/" in value:
            raise ValueError(f"<{value}> SQL expression contains forbidden comment syntax.")

        return value


class MetricFormula(BaseModel):
    """Represents a metric formula with numerator and optional denominator.

    Attributes:
        numerator (UserAggregationFormula): The numerator formula for the metric.
        denominator (UserAggregationFormula | None): Optional denominator formula for ratio metrics.
    """

    numerator: UserAggregationFormula
    denominator: UserAggregationFormula | None


class ExperimentMetricDefinition(BaseModel):
    """
    This class represents a complete metric definition including its computation
    formula, display properties, and metadata.

    Attributes
        alias (str): Unique identifier for the metric.
        type (str): Metric type (avg, ratio, or proportion).
        display_name (str): Human-readable name for the metric (max 60 chars).
        description (str | None): Optional detailed description of the metric.
        formula (MetricFormula): The computation formula for the metric.
        owner (str | None): Optional owner of the metric.
        tags (list[str] | None): Optional list of tags for categorization.
    """

    alias: SQLAlias
    type: ExperimentMetricType
    display_name: str = Field(max_length=60)
    description: str | None
    formula: MetricFormula
    owner: str | None
    group_name: str | None = None
    tags: list[str] | None

    @model_validator(mode="after")
    @classmethod
    def validate_denominator_for_ratio(cls, data):
        if data.type == ExperimentMetricType.RATIO and not data.formula.denominator:
            raise ValueError(
                f"Denominator is required for ratio type. Check the formula for <{data.alias}>."
            )
        return data

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sql(self) -> str:
        """
        Generates the SQL expression for the metric based on its type.
        """
        if self.type == ExperimentMetricType.AVG.value:
            return f"AVG({self.formula.numerator.alias})"
        elif self.type == ExperimentMetricType.RATIO.value:
            if self.formula.denominator:
                return (
                    f"CASE WHEN SUM({self.formula.denominator.alias}) > 0 THEN "
                    f"SUM({self.formula.numerator.alias}) / "
                    f"SUM({self.formula.denominator.alias}) ELSE 0 END"
                )
            else:
                raise ValueError("Denominator is required for ratio type")
        elif self.type == ExperimentMetricType.PROPORTION.value:
            if self.formula.denominator:
                return (
                    f"CASE WHEN SUM({self.formula.denominator.alias}) > 0 THEN "
                    f"SUM({self.formula.numerator.alias}) / "
                    f"SUM({self.formula.denominator.alias}) ELSE 0 END"
                )
            else:
                return f"AVG({self.formula.numerator.alias})"
        else:
            raise ValueError(f"Unsupported metric type: {self.type}")


class YamlGroupOfMetrics(BaseModel):
    """
    Represents a YAML file containing a group of related metrics.
    """

    metric_group_name: str = Field(max_length=140)
    user_aggregations: dict[str, UserAggregationFormula] | None
    metrics: list[ExperimentMetricDefinition]

    @model_validator(mode="after")
    def attach_group_names(cls, data):
        for metric in data.metrics:
            metric.group_name = data.metric_group_name
        return data


class YamlMetrics(BaseModel):
    yaml_groups: list[YamlGroupOfMetrics]

    @model_validator(mode="after")
    @classmethod
    def validate_unique_aliases_exp_metrics(cls, data):
        """Validate that the alias is unique for all experiment metrics from all groups"""
        seen_aliases = set()
        for group in data.yaml_groups:
            for metric in group.metrics:
                if metric.alias in seen_aliases:
                    raise ValueError(
                        f"Duplicate experiment metric alias found: '{metric.alias}'. "
                        "All experiment metric aliases must be unique across all groups."
                    )
                seen_aliases.add(metric.alias)
        return data

    @model_validator(mode="after")
    def validate_unique_aliases_user_metrics(cls, data):
        """Validate that the alias is unique for all user metrics from all groups"""
        seen_aliases = set()
        all_metrics = (metric for group in data.yaml_groups for metric in group.metrics)
        for metric in all_metrics:
            # Check numerator alias
            if metric.formula.numerator.alias in seen_aliases:
                raise ValueError(
                    f"Duplicate user aggregation alias found: "
                    f"'{metric.formula.numerator.alias}'. "
                    "All user aggregation aliases must be unique across all groups."
                )
            seen_aliases.add(metric.formula.numerator.alias)

            # Check denominator alias if it exists
            if metric.formula.denominator and metric.formula.denominator.alias in seen_aliases:
                raise ValueError(
                    f"Duplicate user aggregation alias found: "
                    f"'{metric.formula.denominator.alias}'. "
                    "All user aggregation aliases must be unique across all groups."
                )
            if metric.formula.denominator:
                seen_aliases.add(metric.formula.denominator.alias)
        return data
