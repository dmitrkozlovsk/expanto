from __future__ import annotations

from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader

from src.logger_setup import logger

if TYPE_CHECKING:
    from src.domain.enums import CalculationPurpose
    from src.domain.models import Observation
    from src.services.metric_register import Metrics
    from src.settings import QueryTemplatesConfig


class QueryRenderer:
    """Handles the rendering of SQL queries from templates.

    This class is responsible for loading and rendering SQL query templates
    using Jinja2 templating engine, incorporating metrics and observation data.

    Args:
        query_config (QueryTemplatesConfig): Configuration for query templates.
        metrics (Metrics): Registry of available metrics.
    """

    def __init__(self, query_config: QueryTemplatesConfig, metrics: Metrics):
        self.templates_config = query_config
        self.metrics = metrics
        self.env = Environment(loader=FileSystemLoader(query_config.dir))

    def render(
        self,
        obs: Observation,
        purpose: CalculationPurpose,
        experiment_metric_names: list[str] | None = None,
    ) -> str:
        """Renders the base calculation query using the provided template.

        Args:
            obs (Observation): The observation data to include in the query.
            purpose (CalculationPurpose): The purpose of the calculation.
            experiment_metric_names (list[str] | None, optional): List of metric names
                to include in the calculation. Defaults to None.

        Returns:
            str: The rendered SQL query string.
        """
        calc_scenario_path = self.templates_config.scenarios.get(str(obs.calculation_scenario))
        if not calc_scenario_path:
            error_message = (
                f"Could not find calculation scenario path "
                f"defined for observation <{obs.calculation_scenario}>"
            )
            logger.error(error_message)
            raise Exception(error_message)
        template = self.env.get_template(calc_scenario_path)
        experiment_metrics_list, user_formula_list = self.metrics.resolve(aliases=experiment_metric_names)

        return template.render(
            observation=obs,
            experiment_metrics_list=experiment_metrics_list,
            user_formula_list=user_formula_list,
            purpose=purpose,
        )
