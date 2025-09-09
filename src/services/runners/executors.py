"""Manages the lifecycle of metric calculation jobs.

This module provides a set of classes to handle the entire process of running
metric calculations, from creating a job to storing its results. It includes:
- Executing queries against a database.
- Managing job status and storing results in a persistent storage.
- Orchestrating the entire calculation workflow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError
from sqlalchemy import Column

from src.domain.enums import CalculationPurpose, JobStatus
from src.domain.results import JobResult, MetricResult
from src.logger_setup import logger
from src.services.entities.handlers import JobHandler, PrecomputeHandler

if TYPE_CHECKING:
    from sqlalchemy import Engine

    from src.domain.models import Observation
    from src.services.entities.dtos import ObservationDTO
    from src.services.runners.connectors import PrecomputeConnector
    from src.services.runners.renderer import QueryRenderer


def _ok(job_id: int, metrics: list[MetricResult]) -> JobResult:
    return JobResult.success_result(job_id=int(job_id), metric_results=metrics)


def _err(job_id: int | None, msg: str) -> JobResult:
    return JobResult.error_result(job_id=int(job_id) if job_id else None, error_message=msg)


def _trunc(s: str, n: int = 2000) -> str:
    return s if len(s) <= n else s[:n] + " …[truncated]"


class CalculationExecutor:
    """Executes a SQL query and wraps the results in a JobResult.

    This class is responsible for the execution phase of a calculation job. It
    connects to the data source, runs the provided SQL query, and parses the
    results into MetricResult objects. It also handles exceptions during
    execution and validation, packaging the outcome into a standardized
    JobResult object.

    Args:
        connector (PrecomputeConnector): The database connector used to execute
            the SQL query.
    """

    def __init__(self, connector: PrecomputeConnector):
        self._connector = connector

    def run_job(self, job_id: int, sql: str) -> JobResult:
        logger.info("Job #{job_id}: Executing SQL:\n{sql}", job_id=job_id, sql=_trunc(sql))
        try:
            rows = self._connector.fetch_results(sql)
            metrics = [MetricResult(job_id=int(job_id), **r.model_dump()) for r in rows]
            logger.info("Job #{job_id}: received {count} metrics.", job_id=job_id, count=len(metrics))
            return _ok(job_id, metrics)
        except ValidationError as e:
            logger.exception("Job #{job_id}: Failed to validate results.", job_id=job_id)
            return _err(job_id, f"Validation error: {e}")
        except Exception as e:
            logger.exception("Job #{job_id}: SQL execution failed.", job_id=job_id)
            return _err(job_id, f"Execution error: {e}")


class JobGateway:
    """Provides a transactional interface for job and metric persistence.

    This class acts as a facade to the database, handling all state changes for
    calculation jobs and the storage of computed metrics. It ensures that job
    statuses are correctly updated throughout their lifecycle and that results
    are stored atomically.

    Args:
        engine (Engine): The SQLAlchemy engine for database communication.
    """

    def __init__(self, engine: Engine):
        self._jobs = JobHandler(engine)
        self._precompute = PrecomputeHandler(engine)

    def create_pending(self, observation_id: int | None):
        job = self._jobs.create(
            observation_id=observation_id,
            query="",
            status=JobStatus.PENDING,
        )
        if not job or not job.id:
            raise RuntimeError(f"Could not create job for observation #{observation_id}")
        return job

    def update(self, job_id: int, **fields):
        job = self._jobs.update(id_=job_id, **fields)
        if not job or not job.id:
            raise RuntimeError(f"Could not update job #{job_id}")
        return job

    def set_query_and_running(self, job_id: int, sql: str):
        self.update(job_id, query=sql, status=JobStatus.RUNNING)

    def finish(self, job_id: int, result: JobResult):
        """Sets status to COMPLETED on success, or FAILED with a message on error."""
        if result.success:
            self.update(job_id, status=JobStatus.COMPLETED)
        else:
            self.update(job_id, status=JobStatus.FAILED, error_message=result.error_message)

    def store_metrics(self, metrics: list[MetricResult]) -> int:
        payload = [m.model_dump() for m in metrics]
        if not payload:
            return 0
        self._precompute.bulk_insert(payload)
        logger.info("Stored {count} metrics.", count=len(payload))
        return len(payload)


class CalculationRunner:
    """Orchestrates the entire metrics calculation process.

    This class coordinates the different components (rendering, execution, storage)
    to run a metrics calculation job from start to finish. It manages the job's
    lifecycle, from creation to finalization, handling errors at each step.

    The process involves:
    1. Creating a pending job entry in the database.
    2. Rendering the appropriate SQL query for the given observation.
    3. Executing the query and fetching results.
    4. Storing the resulting metrics in the precompute table.
    5. Updating the job's final status (COMPLETED or FAILED).

    Args:
        connector (PrecomputeConnector): Database connector for executing queries.
        engine (Engine): SQLAlchemy engine instance for job management.
        renderer (QueryRenderer): Renderer for generating SQL queries.
    """

    def __init__(self, connector: PrecomputeConnector, engine: Engine, renderer: QueryRenderer):
        self.jobs = JobGateway(engine)
        self.exec = CalculationExecutor(connector)
        self.renderer = renderer

    def run_calculation(
        self,
        obs: Observation | ObservationDTO,  # type: ignore[valid-type]
        purpose: CalculationPurpose,
        experiment_metric_names: list[str] | None = None,
    ) -> JobResult:
        try:
            obs_id = int(obs.id) if isinstance(obs.id, Column | int) else None  # type: ignore[union-attr]
            job = self.jobs.create_pending(obs_id)
        except Exception as e:
            # DB/job creation unavailable - cannot write error to job.
            obs_id = getattr(obs, "id", None)
            logger.exception("CREATE_JOB failed for observation #{obs_id}", obs_id=obs_id)
            # Return JobResult with sentinel job_id=0 (job not created).
            return _err(None, f"Job create failed for observation #{obs_id}: {e}")

        # 2) Render SQL
        try:
            sql = self.renderer.render(
                obs=obs,
                purpose=purpose,
                experiment_metric_names=experiment_metric_names,
            )
        except Exception as e:
            msg = f"Render error: {e}"
            logger.exception("RENDER failed for job #{job_id}", job_id=job.id)
            self.jobs.update(job.id, status=JobStatus.FAILED, error_message=msg)
            return _err(int(job.id), msg)

        # 3) Set SQL and run
        self.jobs.set_query_and_running(job.id, sql)

        # 4) Execute SQL
        result = self.exec.run_job(job.id, sql)

        # 5) Store metrics on success (REGULAR only)
        if result.success and purpose == CalculationPurpose.REGULAR and result.metric_results:
            try:
                self.jobs.store_metrics(result.metric_results)
            except Exception as e:
                logger.exception("STORE failed for job #{job_id}", job_id=job.id)
                result = _err(int(job.id), f"Store error: {e}")

        # 6) Finalize job status based on result
        try:
            self.jobs.finish(job.id, result)
        except Exception:
            # If the final update fails, log it, but return the calculation result.
            logger.exception("FINISH failed for job #{job_id}", job_id=job.id)

        return result
