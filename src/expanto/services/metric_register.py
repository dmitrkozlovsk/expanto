"""
Metric Registry Module.

This module provides the Metrics class for managing experiment metric definitions
stored in YAML files. It handles the organization, parsing, filtering, and retrieval
of metric definitions used throughout the experiment analysis system.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import yaml  # type: ignore

if TYPE_CHECKING:
    from src.domain.metrics import (
        ExperimentMetricDefinition,
        UserAggregationFormula,
    )

from src.domain.metrics import YamlGroupOfMetrics


class Metrics:
    """
    Handles the organization, parsing, and filtering of various metric definitions stored as YAML files.

    This class provides a centralized interface for managing experiment metrics by:
    - Loading metric definitions from YAML files in a specified directory
    - Organizing metrics into logical groups
    - Providing filtering and retrieval methods for metric definitions
    - Resolving dependencies between experiment-level and user-level metrics

    Attributes:
        directory (Path): The directory containing YAML metric definition files.
        groups (list[YamlGroupOfMetrics]): List of metric groups loaded from YAML files.
    """

    def __init__(self, directory: str | Path) -> None:
        """
        Initialize the Metrics registry with a directory containing YAML metric files.

        Args:
            directory: Path to the directory containing YAML metric definition files.

        Raises:
            FileNotFoundError: If the specified directory does not exist.
        """
        self.directory = Path(directory)
        if not self.directory.exists():
            raise FileNotFoundError(f"Metrics directory not found: {self.directory}")
        self.groups: list[YamlGroupOfMetrics] = [
            self._parse_yaml(path) for path in self.directory.rglob("*.yaml")
        ]

    def _parse_yaml(self, path: Path) -> YamlGroupOfMetrics:
        """
        Parse a YAML file containing metric definitions.

        Args:
            path: Path to the YAML file to parse.

        Returns:
            YamlGroupOfMetrics: Parsed metric group containing all metrics from the file.

        Raises:
            yaml.YAMLError: If the YAML file is malformed.
            ValidationError: If the YAML structure doesn't match the expected schema.
        """
        with open(path) as file:
            yaml_data = yaml.safe_load(file)
        return YamlGroupOfMetrics(**yaml_data)

    @cached_property
    def flat(self) -> dict[str, ExperimentMetricDefinition]:
        """
        Flatten all metrics into an alias-to-definition mapping.

        Creates a dictionary where keys are metric aliases and values are the corresponding
        ExperimentMetricDefinition objects from all loaded metric groups.

        Returns:
            dict[str, ExperimentMetricDefinition]: Dictionary mapping metric aliases to their definitions.
        """
        return {metric.alias: metric for group in self.groups for metric in group.metrics}

    def get(self, aliases: list[str]) -> list[ExperimentMetricDefinition]:
        """
        Return metric definitions for the given aliases.

        Retrieves metric definitions for the provided list of aliases, silently skipping
        any aliases that don't exist in the registry.

        Args:
            aliases: List of metric aliases to retrieve.

        Returns:
            list[ExperimentMetricDefinition]: List of metric definitions for valid aliases.
        """
        return [self.flat[alias] for alias in aliases if alias in self.flat]

    def filter(
        self,
        types: list[str] | None = None,
        tags: list[str] | None = None,
        group_names: list[str] | None = None,
    ) -> dict[str, ExperimentMetricDefinition]:
        """
        Filter metrics by one or more criteria using AND logic.

        Filters the metrics registry based on the provided criteria. All conditions
        are combined using AND logic (all specified conditions must be met).

        Args:
            types: List of metric types to include (e.g., ['avg', 'ratio']).
            tags: List of tags to match (metrics with any of these tags will be included).
            group_names: List of group names to include.

        Returns:
            dict[str, ExperimentMetricDefinition]: Filtered metrics as alias-to-definition mapping.
        """

        def _match(m: ExperimentMetricDefinition) -> bool:
            return all(
                [
                    not types or m.type in types,
                    not tags or (m.tags and set(tags) & set(m.tags)),
                    not group_names or m.group_name in group_names,
                ]
            )

        return {m.alias: m for m in self.flat.values() if _match(m)}

    def resolve(
        self,
        aliases: list[str] | None = None,
    ) -> tuple[list[ExperimentMetricDefinition], list[UserAggregationFormula]]:
        """
        Return experiment-level metrics plus their required user-level metrics.

        Resolves experiment metrics and automatically includes all required user-level
        aggregation formulas (numerators and denominators) needed for computation.

        Args:
            aliases: List of experiment metric aliases to resolve. If None, resolves all metrics.

        Returns:
            tuple: A pair containing:
                - List of experiment metric definitions
                - List of unique user aggregation formulas required by the experiment metrics
        """
        if not aliases:
            aliases = list(self.flat.keys())
        experiment_metrics_list = self.get(aliases)
        user_agg_formula_dict = {}
        for metric in experiment_metrics_list:
            if metric.formula.numerator.alias not in user_agg_formula_dict:
                user_agg_formula_dict[metric.formula.numerator.alias] = metric.formula.numerator
            if metric.formula.denominator and metric.formula.denominator.alias not in user_agg_formula_dict:
                user_agg_formula_dict[metric.formula.denominator.alias] = metric.formula.denominator
        return experiment_metrics_list, list(user_agg_formula_dict.values())
