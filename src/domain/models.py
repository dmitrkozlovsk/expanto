"""Database Models for the Expanto experiment platform.

This module defines the SQLAlchemy ORM models for the experiment platform database.
The models represent the core entities of the A/B testing platform:
- Experiment: The top-level entity representing an A/B test
- Observation: A specific analysis configuration for an experiment
- CalculationJob: A job that calculates metrics for an observation
- Precompute: The computed metrics results from a calculation job

These models form a hierarchical structure where an Experiment contains multiple
Observations, each Observation can have multiple CalculationJobs, and each
CalculationJob produces multiple Precompute records with metric results.
"""

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.types import JSON

from src.domain.base import Base
from src.utils import DatetimeUtils as dt_utils


class Experiment(Base):
    """Represents an A/B experiment with its metadata and configuration.

    An experiment is the top-level entity that contains observations and metrics.
    It defines comprehensive description of an A/B test, including its name, status,
    time period.

    """

    __tablename__ = "experiments"
    __table_args__ = {"comment": "Stores A/B experiments with their metadata and configuration"}

    id = Column(
        Integer, primary_key=True, autoincrement=True, comment="Unique identifier for the experiment"
    )
    name = Column(
        String(255),
        nullable=False,
        unique=True,
        comment="Unique name of the experiment used for identification and querying",
    )
    status = Column(
        String(50),
        nullable=False,
        comment="Current status of the experiment (planned, running, paused, completed, cancelled)",
    )
    description = Column(
        Text,
        nullable=True,
        comment="Detailed description of the experiment purpose and changes being tested",
    )
    hypotheses = Column(Text, nullable=True, comment="Hypotheses being tested in this experiment")
    key_metrics = Column(
        JSON, nullable=True, comment="List of key metrics to be tracked for this experiment"
    )
    design = Column(Text, nullable=True, comment="Description of the experiment design methodology")
    conclusion = Column(Text, nullable=True, comment="Final conclusions after the experiment is completed")
    start_datetime = Column(
        DateTime,
        nullable=True,
        comment="When the experiment started or is scheduled to start (when the treatment period starts)",
    )
    end_datetime = Column(
        DateTime,
        nullable=True,
        comment="When the experiment ended or is scheduled to end (when the treatment period ends)",
    )

    _created_at = Column(
        DateTime,
        nullable=False,
        default=dt_utils.utc_now,
        comment="Timestamp when the experiment was created",
    )
    _updated_at = Column(
        DateTime,
        nullable=False,
        default=dt_utils.utc_now,
        comment="Timestamp when the experiment was updated",
        onupdate=dt_utils.utc_now,
    )

    # Relationships
    observations = relationship("Observation", back_populates="experiment", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return (
            f"<Experiment(id={self.id}, "
            f"name={self.name}, "
            f"status={self.status}, "
            f"description={self.description}, "
            f"hypotheses={self.hypotheses}, "
            f"key_metrics={self.key_metrics}, "
            f"design={self.design}, "
            f"conclusion={self.conclusion}, "
            f"start_datetime={self.start_datetime}, "
            f"end_datetime={self.end_datetime}, "
            f"created_at={self._created_at})>"
        )


class Observation(Base):
    """Represents a specific observation or analysis of an experiment.

    An observation is associated with an experiment and defines how metrics
    should be calculated for a specific segment or time period. It configures
    the parameters for metric calculations, including time periods, audience
    filtering, and which metrics to calculate.

    """

    __tablename__ = "observations"
    __table_args__ = {
        "comment": "Stores observations and analysis configurations for experiments."
        'This entity is responsible for how results will be calculated"'
    }

    id = Column(
        Integer, primary_key=True, autoincrement=True, comment="Unique identifier for the observation"
    )
    experiment_id = Column(
        Integer,
        ForeignKey("experiments.id"),
        nullable=False,
        comment="Foreign key to the parent experiment",
    )

    name = Column(String(100), nullable=False, comment="Name of the observation")

    db_experiment_name = Column(
        String(510), nullable=False, comment="name of the experiment used for identification and querying"
    )

    split_id = Column(
        Text,
        nullable=False,
        comment="""Identifier for the split/variant configuration. 
        It could be a string or a number looks like 'device_id' or 'user_id'""",
    )

    calculation_scenario = Column(
        String(100),
        nullable=False,
        comment="Name of the calculation scenario to use for this observation (linked to SQL templates)",
        default="base",
    )

    # Treatment period
    exposure_start_datetime = Column(DateTime, nullable=True, comment="Start of the treatment period ")
    exposure_end_datetime = Column(DateTime, nullable=True, comment="End of the treatment period ")

    # Calculation period
    calc_start_datetime = Column(
        DateTime,
        nullable=True,
        comment="Start of the additional calculation period for metrics (e.g., retention)",
    )
    calc_end_datetime = Column(
        DateTime, nullable=True, comment="End of the additional calculation period for metrics"
    )

    exposure_event = Column(
        Text, nullable=True, comment="Event when we start to treat users to avoid dilution effect"
    )

    # Filtering and segmentation
    audience_tables = Column(
        JSON, nullable=True, comment="List of custom segment strings/table namesfor inner_join to intersect"
    )
    filters = Column(JSON, nullable=True, comment="List of filter strings (e.g., [\"ios>'1.0.0'\"])")
    custom_test_ids_query = Column(
        Text,
        nullable=True,
        comment="Custom query to calculate the test_ids for the experiment. "
        "The query will be used in jinja template instead of the default query",
    )

    metric_tags = Column(JSON, nullable=True, comment="Filter metrics by tags")
    metric_groups = Column(JSON, nullable=True, comment="Filter metrics by groups")

    _created_at = Column(
        DateTime,
        nullable=False,
        default=dt_utils.utc_now,
        comment="Timestamp when the observation was created",
    )
    _updated_at = Column(
        DateTime,
        nullable=False,
        default=dt_utils.utc_now,
        comment="Timestamp when the observation was updated",
        onupdate=dt_utils.utc_now,
    )

    # Relationships
    experiment = relationship("Experiment", back_populates="observations")
    metrics_calculation_jobs = relationship(
        "CalculationJob", back_populates="observation", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Observation(id={self.id}, "
            f"Observation name='{self.name}', "
            f"split_id={self.split_id}, "
            f"experiment_id={self.experiment_id}, "
            f"exposure_period=({self.exposure_start_datetime}, "
            f"{self.exposure_end_datetime}), "
            f"calc_period=({self.calc_start_datetime}, {self.calc_end_datetime})), "
            f"audience_tables={self.audience_tables}, "
            f"filters={self.filters}, "
            f"metric_tags={self.metric_tags}, "
            f"metric_groups={self.metric_groups}>"
        )


class CalculationJob(Base):
    """Represents a job that calculates metrics for an observation.

    Each job is associated with an observation and contains the query used
    to calculate metrics and the status of the calculation. It tracks the
    execution of metric calculations and stores the results in Precompute records.

    """

    __tablename__ = "calculation_jobs"
    __table_args__ = {
        "comment": "Stores metrics calculation jobs and their execution status. "
        "It is necessary for storing the queries (hystory) and the "
        "status of the calculations"
    }

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Unique identifier for the metrics calculation job",
    )
    observation_id = Column(
        Integer,
        ForeignKey("observations.id"),
        nullable=True,
        comment="Foreign key to the parent observation",
    )
    query = Column(Text, nullable=False, comment="The SQL query used for the metrics calculation")
    status = Column(
        String(50), nullable=False, comment="Current status of the job (e.g., running, completed, failed)"
    )
    error_message = Column(Text, nullable=True, comment="Error message if the job failed")
    extra = Column(
        JSON,
        nullable=True,
        comment="Additional information about the query execution (e.g., bytes read, execution time, etc.)",
    )

    _created_at = Column(
        DateTime,
        nullable=False,
        default=dt_utils.utc_now,
        comment="Timestamp when the calculation job was created",
    )

    # Relationships
    observation = relationship("Observation", back_populates="metrics_calculation_jobs")
    computed_metrics = relationship(
        "Precompute", back_populates="job", foreign_keys="Precompute.job_id", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<CalculationJob(id={self.id}, "
            f"observation_id={self.observation_id}, "
            f"status={self.status}, "
            f"query={self.query}, "
            f"extra={self.extra})>"
        )


class Precompute(Base):
    """Represents the computed metrics results from a metrics calculation job.

    Stores the actual metric values and statistical information calculated
    for an observation. Each record contains the results for a specific metric,
    potentially for a specific group/segment, including statistical information
    needed for significance testing (average, variance, covariance).

    """

    __tablename__ = "precomputes"
    __table_args__ = {"comment": "Stores computed metrics results and statistical information"}

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
        comment="Unique identifier for the computed metrics record",
    )
    job_id = Column(
        Integer,
        ForeignKey("calculation_jobs.id"),
        nullable=False,
        comment="Foreign key to the parent metrics calculation job",
    )

    group_name = Column(Text, nullable=False, comment="Name of the group/segment for this metric")
    metric_name = Column(Text, nullable=False, comment="Name of the metric being computed")
    metric_display_name = Column(Text, nullable=False, comment="Display name of metric")
    metric_type = Column(Text, nullable=False, comment="Type of metric (avg, ratio, proportion)")
    observation_cnt = Column(
        Integer, nullable=False, comment="Count of observations used in the calculation"
    )
    metric_value = Column(Float, nullable=False, comment="Computed value of the metric")
    numerator_avg = Column(Float, nullable=True, comment="Average of the numerator used in ratio metrics")
    denominator_avg = Column(
        Float, nullable=True, comment="Average of the denominator used in ratio metrics"
    )
    numerator_var = Column(Float, nullable=True, comment="Variance of the numerator used in ratio metrics")
    denominator_var = Column(
        Float, nullable=True, comment="Variance of the denominator used in ratio metrics"
    )
    covariance = Column(
        Float,
        nullable=True,
        comment="Covariance between numerator and denominator for statistical calculations",
    )
    _created_at = Column(
        DateTime,
        nullable=False,
        default=dt_utils.utc_now,
        comment="Timestamp when the computed metrics record was created",
    )

    # Relationships
    job = relationship("CalculationJob", back_populates="computed_metrics", foreign_keys=[job_id])

    def __repr__(self) -> str:
        return (
            f"<Precompute(id={self.id}, "
            f"job_id={self.job_id}, "
            f"group_name={self.group_name}, "
            f"metric_name={self.metric_name}, "
            f"observation_cnt={self.observation_cnt}, "
            f"metric_value={self.metric_value}, "
            f"numerator_avg={self.numerator_avg}, "
            f"denominator_avg={self.denominator_avg}, "
            f"numerator_var={self.numerator_var}, "
            f"denominator_var={self.denominator_var}, "
            f"covariance={self.covariance})>"
        )
