from datetime import datetime, timedelta
from tabnanny import verbose

from sqlalchemy import create_engine

from src.domain.models import Base
from src.services.entities.handlers import ExperimentHandler, ObservationHandler, JobHandler, PrecomputeHandler

# Create database engine
engine = create_engine("sqlite:///expanto.db", 
                      echo=False,
                      connect_args={"check_same_thread": False},
                      pool_size=5,
                      max_overflow=10,
                      pool_timeout=10,
                      pool_recycle=900,)

# Create all tables
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

# Create entities handlers with engine
experiments_handler = ExperimentHandler(engine)
observations_handler = ObservationHandler(engine)
jobs_handler = JobHandler(engine)
precomputes_handler = PrecomputeHandler(engine)

# Base start date for all experiments
start_date = datetime.strptime("2025-07-20", "%Y-%m-%d")

try:
    # Create "expanto_implementation" demo experiment - COMPLETED
    exp1 = experiments_handler.create(
        name="🚀 expanto_implementation",
        status="running",
        description="Implementation of Expanto A/B testing platform for data analytics team. Testing productivity impact of replacing manual experimentation workflow with Expanto automated platform.",
        hypotheses="If we implement Expanto platform for our A/B testing workflow, then analyst productivity will increase and bugs/errors will decrease, because automated metric calculations and standardized analysis processes reduce manual work and human errors.",
        key_metrics=["avg_bugs_per_analyst", "avg_debug_time_hours", "avg_metrics_calculation_time", "avg_analysis_time_hours", "avg_experiments_created", "analyst_satisfaction_score", "experiment_success_rate", "high_satisfaction_analyst_proportion"],
        design="Two groups: control (manual Excel/SQL workflow), test (Expanto platform). Testing productivity, quality and satisfaction metrics.",
        conclusion = None,
        start_datetime=start_date - timedelta(days=60),
        end_datetime=start_date - timedelta(days=30)
    )

    # Create observation for demo experiment
    obs1 = observations_handler.create(
        experiment_id=exp1.id,
        name="expanto_implementation_main",
        db_experiment_name="expanto_implementation",
        split_id="analyst_id",
        calculation_scenario="base",
        exposure_start_datetime=start_date - timedelta(days=60),
        exposure_end_datetime=start_date - timedelta(days=30),
        calc_start_datetime=start_date - timedelta(days=60),
        calc_end_datetime=start_date - timedelta(days=25),  # Extra 5 days for delayed effects
        exposure_event="tool_assigned",
    )

    # Create fake calculation job for demo experiment (completed)
    demo_job = jobs_handler.create(
        observation_id=obs1.id,
        query="""
        -- Demo calculation query for Expanto implementation experiment
        SELECT 
            test_group,
            analyst_id,
            AVG(bugs_reported_cnt) as avg_bugs_per_analyst,
            AVG(debug_time_hours) as avg_debug_time_hours,
            AVG(metrics_calculation_time_hours) as avg_metrics_calculation_time,
            AVG(analysis_time_hours) as avg_analysis_time_hours,
            AVG(experiments_created_cnt) as avg_experiments_created,
            AVG(satisfaction_score) as analyst_satisfaction_score
        FROM demo_analytics_productivity_data
        WHERE experiment_date BETWEEN '2025-05-21' AND '2025-06-20'
        GROUP BY test_group, analyst_id
        """,
        status="completed",
        extra={
            "execution_time_seconds": 2.3,
            "bytes_processed": 15648,
            "rows_processed": 2000,
            "warehouse": "demo_warehouse"
        }
    )

    # Create realistic fake precomputed metrics for the demo experiment
    # Control group metrics (manual workflow)
    control_metrics = [
        {
            "job_id": demo_job.id,
            "group_name": "control",
            "metric_name": "avg_bugs_per_analyst",
            "metric_display_name": "Average Bugs Reported per Analyst",
            "metric_type": "avg",
            "observation_cnt": 1000,
            "metric_value": 2.8,  # Higher bugs in manual workflow
            "numerator_avg": 2.8,
            "denominator_avg": 1.0,  # For avg metrics, denominator is always 1
            "numerator_var": 1.44,
            "denominator_var": 0.0,
            "covariance": 0.0
        },
        {
            "job_id": demo_job.id,
            "group_name": "control",
            "metric_name": "avg_debug_time_hours",
            "metric_display_name": "Average Debug Time (Hours)",
            "metric_type": "avg",
            "observation_cnt": 1000,
            "metric_value": 8.5,  # More debug time needed
            "numerator_avg": 8.5,
            "denominator_avg": 1.0,
            "numerator_var": 12.25,
            "denominator_var": 0.0,
            "covariance": 0.0
        },
        {
            "job_id": demo_job.id,
            "group_name": "control",
            "metric_name": "avg_metrics_calculation_time",
            "metric_display_name": "Average Metrics Calculation Time (Hours)",
            "metric_type": "avg",
            "observation_cnt": 1000,
            "metric_value": 16.2,  # Manual Excel/SQL takes much longer
            "numerator_avg": 16.2,
            "denominator_avg": 1.0,
            "numerator_var": 25.6,
            "denominator_var": 0.0,
            "covariance": 0.0
        },
        {
            "job_id": demo_job.id,
            "group_name": "control",
            "metric_name": "avg_analysis_time_hours",
            "metric_display_name": "Average Analysis Time (Hours)",
            "metric_type": "avg",
            "observation_cnt": 1000,
            "metric_value": 12.8,  # Manual report writing
            "numerator_avg": 12.8,
            "denominator_avg": 1.0,
            "numerator_var": 18.5,
            "denominator_var": 0.0,
            "covariance": 0.0
        },
        {
            "job_id": demo_job.id,
            "group_name": "control",
            "metric_name": "avg_experiments_created",
            "metric_display_name": "Average Experiments Created per Analyst",
            "metric_type": "avg",
            "observation_cnt": 1000,
            "metric_value": 1.8,  # Lower throughput with manual process
            "numerator_avg": 1.8,
            "denominator_avg": 1.0,
            "numerator_var": 0.64,
            "denominator_var": 0.0,
            "covariance": 0.0
        },
        {
            "job_id": demo_job.id,
            "group_name": "control",
            "metric_name": "analyst_satisfaction_score",
            "metric_display_name": "Analyst Satisfaction Score",
            "metric_type": "avg",
            "observation_cnt": 1000,
            "metric_value": 5.2,  # Lower satisfaction with manual tools
            "numerator_avg": 5.2,
            "denominator_avg": 1.0,
            "numerator_var": 2.25,
            "denominator_var": 0.0,
            "covariance": 0.0
        },
        {
            "job_id": demo_job.id,
            "group_name": "control",
            "metric_name": "experiment_success_rate",
            "metric_display_name": "Experiment Success Rate",
            "metric_type": "ratio",
            "observation_cnt": 1000,
            "metric_value": 0.65,  # Lower success rate with manual process
            "numerator_avg": 1.17,  # avg successful experiments per analyst
            "denominator_avg": 1.8,   # avg total experiments per analyst
            "numerator_var": 0.64,
            "denominator_var": 0.81,
            "covariance": 0.12
        },
        {
            "job_id": demo_job.id,
            "group_name": "control",
            "metric_name": "high_satisfaction_analyst_proportion",
            "metric_display_name": "High Satisfaction Analysts (%)",
            "metric_type": "proportion",
            "observation_cnt": 1000,
            "metric_value": 0.24,  # Only 24% of analysts are highly satisfied
            "numerator_avg": 0.24,
            "denominator_avg": 1.0,
            "numerator_var": 0.18,  # var = p*(1-p) = 0.24*0.76
            "denominator_var": 0.0,
            "covariance": 0.0
        }
    ]

    # Test group metrics (Expanto platform)
    test_metrics = [
        {
            "job_id": demo_job.id,
            "group_name": "test",
            "metric_name": "avg_bugs_per_analyst",
            "metric_display_name": "Average Bugs Reported per Analyst",
            "metric_type": "avg",
            "observation_cnt": 1000,
            "metric_value": 0.9,  # 68% reduction in bugs!
            "numerator_avg": 0.9,
            "denominator_avg": 1.0,
            "numerator_var": 0.36,
            "denominator_var": 0.0,
            "covariance": 0.0
        },
        {
            "job_id": demo_job.id,
            "group_name": "test",
            "metric_name": "avg_debug_time_hours",
            "metric_display_name": "Average Debug Time (Hours)",
            "metric_type": "avg",
            "observation_cnt": 1000,
            "metric_value": 3.4,  # 60% reduction in debug time
            "numerator_avg": 3.4,
            "denominator_avg": 1.0,
            "numerator_var": 4.8,
            "denominator_var": 0.0,
            "covariance": 0.0
        },
        {
            "job_id": demo_job.id,
            "group_name": "test",
            "metric_name": "avg_metrics_calculation_time",
            "metric_display_name": "Average Metrics Calculation Time (Hours)",
            "metric_type": "avg",
            "observation_cnt": 1000,
            "metric_value": 4.1,  # 75% faster with automated calculations!
            "numerator_avg": 4.1,
            "denominator_avg": 1.0,
            "numerator_var": 6.2,
            "denominator_var": 0.0,
            "covariance": 0.0
        },
        {
            "job_id": demo_job.id,
            "group_name": "test",
            "metric_name": "avg_analysis_time_hours",
            "metric_display_name": "Average Analysis Time (Hours)",
            "metric_type": "avg",
            "observation_cnt": 1000,
            "metric_value": 7.1,  # 45% faster with automated reports
            "numerator_avg": 7.1,
            "denominator_avg": 1.0,
            "numerator_var": 9.8,
            "denominator_var": 0.0,
            "covariance": 0.0
        },
        {
            "job_id": demo_job.id,
            "group_name": "test",
            "metric_name": "avg_experiments_created",
            "metric_display_name": "Average Experiments Created per Analyst",
            "metric_type": "avg",
            "observation_cnt": 1000,
            "metric_value": 3.2,  # 78% more experiments created!
            "numerator_avg": 3.2,
            "denominator_avg": 1.0,
            "numerator_var": 1.1,
            "denominator_var": 0.0,
            "covariance": 0.0
        },
        {
            "job_id": demo_job.id,
            "group_name": "test",
            "metric_name": "analyst_satisfaction_score",
            "metric_display_name": "Analyst Satisfaction Score",
            "metric_type": "avg",
            "observation_cnt": 1000,
            "metric_value": 9.8,  # 89% higher satisfaction! 
            "numerator_avg": 9.8,
            "denominator_avg": 1.0,
            "numerator_var": 0.16,
            "denominator_var": 0.0,
            "covariance": 0.0
        },
        {
            "job_id": demo_job.id,
            "group_name": "test",
            "metric_name": "experiment_success_rate",
            "metric_display_name": "Experiment Success Rate",
            "metric_type": "ratio",
            "observation_cnt": 1000,
            "metric_value": 0.91,  # Much higher success rate with Expanto!
            "numerator_avg": 2.91,  # avg successful experiments per analyst
            "denominator_avg": 3.2,   # avg total experiments per analyst
            "numerator_var": 1.44,
            "denominator_var": 1.62,
            "covariance": 0.28
        },
        {
            "job_id": demo_job.id,
            "group_name": "test",
            "metric_name": "high_satisfaction_analyst_proportion",
            "metric_display_name": "High Satisfaction Analysts (%)",
            "metric_type": "proportion",
            "observation_cnt": 1000,
            "metric_value": 0.94,  # 94% of analysts are highly satisfied with Expanto!
            "numerator_avg": 0.94,
            "denominator_avg": 1.0,
            "numerator_var": 0.056,  # var = p*(1-p) = 0.94*0.06
            "denominator_var": 0.0,
            "covariance": 0.0
        }
    ]

    # Insert all precomputed metrics
    all_metrics = control_metrics + test_metrics
    for metric_data in all_metrics:
        precomputes_handler.create(**metric_data)

    print("Successfully created demo experiment with all metrics!")

except Exception as e:
    print(f"Error occurred: {str(e)}")
    raise

######### VERIFY ###########
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from src.domain.models import Experiment, Observation, CalculationJob, Precompute
engine = create_engine("sqlite:///expanto.db", echo=False)
session = Session(engine)

total_experiments = session.query(Experiment).count()
total_observations = session.query(Observation).count()
total_jobs = session.query(CalculationJob).count()
total_precomputes = session.query(Precompute).count()
print(f"Created: {total_experiments} experiments, {total_observations} observations, {total_jobs} jobs, {total_precomputes} precomputed metrics")
session.close()