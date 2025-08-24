"""Module for handling query rendering and calculation execution.

This module provides functionality for rendering SQL queries from templates and
executing calculation jobs for metrics computation. It includes classes for query
rendering and job execution management.
"""

from __future__ import annotations

from threading import Thread
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader
from pydantic import ValidationError

from src.domain.enums import CalculationPurpose, JobStatus
from src.domain.models import CalculationJob, Observation
from src.domain.results import JobResult, MetricResult
from src.logger_setup import logger
from src.services.entities.handlers import JobHandler, PrecomputeHandler
from src.services.metric_register import Metrics
from src.services.runners.connectors import PrecomputeConnector
from src.settings import QueryTemplatesConfig

if TYPE_CHECKING:
    from sqlalchemy import Engine

    from src.services.entities.dtos import ObservationDTO


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

    def render_base_calculation_query(
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
            error_message = (f"Could not find calculation scenario path "
                            f"defined for observation <{obs.calculation_scenario}>")
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


class CalculationRunner:
    """Manages the creation, execution, and tracking of metrics calculation jobs.

    This class handles the complete lifecycle of calculation jobs, including query
    rendering, job execution, and result storage. It provides both synchronous and
    asynchronous execution capabilities.

    Args:
        connector (PrecomputeConnector): Database connector for executing queries.
        engine (Engine): SQLAlchemy engine instance.
        renderer (QueryRenderer): Query renderer instance for generating SQL queries.
    """

    def __init__(self, connector: PrecomputeConnector, engine: Engine, renderer: QueryRenderer):
        self._connector = connector
        self._engine = engine
        self._renderer = renderer
        self._job_handler = JobHandler(self._engine)
        self._precompute_handler = PrecomputeHandler(self._engine)

    def _run_job(self, job: CalculationJob) -> JobResult:
        """Executes a calculation job and processes its results.

        Args:
            job (CalculationJob): The job to execute.

        Returns:
            JobResult: The result of the job execution, including any metrics or errors.
        """

        self._job_handler.update(job.id, status=JobStatus.RUNNING)
        logger.info(f"Running job_id #{job.id} \nQuery:\n{job.query}")
        try:
            query_result = self._connector.fetch_results(str(job.query))
            metric_results = [MetricResult(job_id=int(job.id), **qmr.model_dump()) for qmr in query_result]
            logger.info(f"Job id #{job.id} completed with {len(metric_results)} metrics")
            return JobResult.success_result(job_id=int(job.id), metric_results=metric_results)
        except ValidationError as e:
            error_message = f"Job_id: #{job.id} execution failed because of validation error: {str(e)}"
            logger.exception(error_message)
            return JobResult.error_result(job_id=int(job.id), error_message=error_message)
        except Exception as e:
            error_message = f"Job_id: #{job.id} execution failed: {str(e)}"
            logger.exception(error_message)
            return JobResult.error_result(job_id=int(job.id), error_message=error_message)

    def _store_results(self, metric_results: list[MetricResult] | None):
        """Stores the calculation results in the database.

        Args:
            metric_results (list[MetricResult] | None): The results to store.

        Raises:
            Exception: If no results are provided or storage fails.
        """

        if metric_results is None:
            logger.info("No metric results found")
            raise Exception("No metric results found")
        try:
            self._precompute_handler.bulk_insert([mr.model_dump() for mr in metric_results])
            logger.info(f"Successfully stored {len(metric_results)} metrics")
        except Exception:
            logger.exception("Failed to store precompute results")
            raise

    def run_calculation(
        self,
        obs: Observation | ObservationDTO,  # type: ignore[valid-type]
        purpose: CalculationPurpose,
        experiment_metric_names_: list[str] | None = None,
    ) -> JobResult | None:
        """Executes a calculation for a given observation.

        Creates a job, runs it, and stores the results based on the calculation purpose.
        For regular calculations, results are stored in the database.
        For planning purposes, results are returned directly.

        Args:
            obs (Observation | ObservationDTO): The observation to calculate metrics for.
            purpose (CalculationPurpose): The purpose of the calculation.
            experiment_metric_names_ (list[str] | None): List of metric names to include.

        Returns:
            JobResult | None: The job result containing metrics and status information.
        """
        query = self._renderer.render_base_calculation_query(
            obs, purpose=purpose, experiment_metric_names=experiment_metric_names_
        )

        job = self._job_handler.create(
            observation_id=obs.id,  # type: ignore[union-attr]
            query=query,
            status=JobStatus.PENDING,
        )

        if job is None:
            error_message = f"Failed to create job for observation id #{obs.id}"  # type: ignore[union-attr]
            logger.error(error_message)
            raise Exception(error_message)

        job_result = self._run_job(job)

        if job_result.success:
            self._job_handler.update(job.id, status=JobStatus.COMPLETED)
            logger.info(
                f"Calculation for observation id #{obs.id} completed with "  # type: ignore[union-attr]
                f"{len(job_result.metric_results) if job_result.metric_results else 0} "
                f"metrics"
            )
            if purpose == CalculationPurpose.REGULAR and job_result.metric_results:
                self._store_results(job_result.metric_results)
        else:
            self._job_handler.update(
                id_=job.id, status=JobStatus.FAILED, error_message=job_result.error_message
            )
            logger.warning(f"Calculation for observation id #{obs.id} failed: {job_result.error_message}")  # type: ignore[union-attr]
        return job_result

    def run_in_background(self, obs: Observation, purpose: CalculationPurpose):
        """Executes a calculation asynchronously in a background thread.

        Args:
            obs (Observation): The observation to calculate metrics for.
            purpose (CalculationPurpose): The purpose of the calculation.
        """

        thread = Thread(target=self.run_calculation, args=(obs, purpose))
        thread.start()
