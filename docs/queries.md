# Query Templates Documentation

Query templates are the heart of Expanto's metric calculation system. Think of them as functions that take an observation, metrics definitions, and data parameters as input and produce a SQL query that calculates experiment results.

## Concept Overview

The query execution process works like a function:

```
f(observation, metrics, templates) → SQL Query → Results
```

**Input:**
- `observation` — experiment configuration (periods, audience, filters)
- `experiment_metrics_list` — metrics to calculate from YAML definitions
- `user_formula_list` — underlying formulas needed for metrics
- `purpose` — calculation purpose ('regular' or 'planning')

**Output:**
- SQL query that returns metric results with statistical data

**Processing:**
1. Jinja2 renders templates with provided variables
2. Templates include sub-templates for modularity
3. SQL executes against your data warehouse
4. Results are stored as Precompute records

## Template Structure

Templates are organized in scenarios under `queries/` directory. **The structure below is just an example** — you can organize your templates however makes sense for your data and workflow:

```
queries/
├── bigquery_example/         # 📁 EXAMPLE organization
│   ├── __entry__.j2          # Main template (entry point)
│   ├── events.j2             # Events data extraction
│   ├── user_aggregation.j2   # User-level metric calculations
│   ├── events_x_testids.j2   # Events joined with test assignments
│   └── testits.j2            # Test ID assignments
└── snowflake_example/        # 📁 EXAMPLE organization
    ├── base.sql              # Main template (entry point)
    ├── events.sql            # Events data extraction
    ├── user_aggregation.sql  # User-level calculations
    ├── events_x_testids.sql  # Events + test assignments
    ├── testids.sql           # Test ID assignments
    └── dummy_testids.sql     # Mock data for planning
```

**Important:** The modular structure above is just one way to organize templates. You could also:
- Put everything in a single template file
- Use different sub-template organization  
- Create your own naming conventions
- Structure based on your data warehouse setup

**The only requirement is that your main template produces the correct output format.**

### Template Types

#### 1. **Main Template** (Entry Point)
- **BigQuery:** `__entry__.j2`
- **Snowflake:** `base.sql`

The main template orchestrates the entire calculation by including sub-templates and structuring the final output.

#### 2. **Events Template**
Extracts raw event data for the calculation period:

```sql
-- Example: events.j2/events.sql
SELECT
    user_id,
    device_id,
    event_name,
    event_value,
    event_timestamp
FROM events_table
WHERE event_timestamp BETWEEN '{{observation.exposure_start_datetime}}'
    AND '{{observation.calc_end_datetime}}'
```

#### 3. **Test IDs Template**
Defines experiment group assignments:

```sql
-- Example: testids.sql
SELECT
    {{observation.split_id}} as user_id,
    experiment_group as group_name,
    assignment_timestamp as first_split_timestamp
FROM experiment_assignments
WHERE experiment_name = '{{observation.db_experiment_name}}'
```

#### 4. **User Aggregation Template**
Calculates user-level metrics using the formulas from YAML:

```sql
-- Example: user_aggregation.j2
SELECT
    j.{{observation.split_id}},
    j.group_name,
    {% for user_formula in user_formula_list -%}
    {{ user_formula.sql }} AS {{ user_formula.alias }}
    {% endfor -%}
FROM events_x_testids as j
GROUP BY j.{{observation.split_id}}, j.group_name
```

## Available Variables

Templates have access to these Jinja2 variables:

### Observation Object
```python
observation.split_id                    # user identifier field
observation.db_experiment_name          # experiment name in database
observation.exposure_start_datetime     # treatment start
observation.exposure_end_datetime       # treatment end
observation.calc_start_datetime         # calculation period start
observation.calc_end_datetime           # calculation period end
observation.exposure_event              # specific exposure event (optional)
observation.audience_tables             # list of audience segments
observation.filters                     # list of SQL filter conditions
```

### Metrics Lists
```python
experiment_metrics_list                 # List of experiment metrics to calculate
user_formula_list                       # List of user-level formulas needed

# Each metric has:
metric.alias                           # 'conversion_rate'
metric.display_name                    # 'Conversion Rate'
metric.type                           # 'ratio', 'avg', 'proportion'
metric.sql                            # rendered SQL expression
metric.formula.numerator              # numerator formula object
metric.formula.denominator            # denominator formula object
```

### Purpose
```python
purpose                               # 'regular' or 'planning'
```

## BigQuery Example

Here's how the BigQuery main template works:

```sql
-- __entry__.j2
WITH
    events AS (
        {% include "events.j2" %}
    ),
    testids AS (
        {% include "testits.j2" %}
    ),
    events_x_testids AS (
        {% include "events_x_testids.j2" %}
    ),
    user_aggregation AS (
        {% include "user_aggregation.j2" %}
    ),
    aggregated AS (
        SELECT
            group_name,
            ARRAY<STRUCT<...>>[
            {% for experiment_metric in experiment_metrics_list %}
            STRUCT(
                '{{ experiment_metric.alias }}',
                '{{ experiment_metric.type }}',
                '{{ experiment_metric.display_name }}',
                {{ experiment_metric.sql }},
                {% if experiment_metric.type == 'ratio' %}
                    AVG({{ experiment_metric.formula.numerator.alias }}),
                    AVG({{ experiment_metric.formula.denominator.alias }}),
                    VAR_SAMP({{ experiment_metric.formula.numerator.alias }}),
                    VAR_SAMP({{ experiment_metric.formula.denominator.alias }}),
                    COVAR_SAMP(...)
                {% endif %}
            ){% if not loop.last %},{% endif %}
            {% endfor %}
        ] AS metrics
    FROM user_aggregation
    GROUP BY group_name
    )
SELECT
    group_name,
    m.metric_name,
    m.metric_type,
    m.metric_display_name,
    m.metric_value,
    m.numerator_avg,
    m.denominator_avg,
    m.numerator_var,
    m.denominator_var,
    m.covariance
FROM aggregated, UNNEST(metrics) AS m
```

## Snowflake Example

Snowflake uses a simpler UNION ALL approach:

```sql
-- base.sql
WITH user_aggregation AS (
    {% include "user_aggregation.sql" %}
)
{% for experiment_metric in experiment_metrics_list %}
SELECT
    group_name,
    '{{ experiment_metric.alias }}' as metric_name,
    '{{ experiment_metric.type }}' as metric_type,
    '{{ experiment_metric.display_name }}' as metric_display_name,
    COUNT(DISTINCT user_aggregation.{{ observation.split_id }}) as observation_cnt,
    {{ experiment_metric.sql }} as metric_value,
    {% if experiment_metric.type == 'ratio' %}
        AVG({{ experiment_metric.formula.numerator.alias }}) as numerator_avg,
        AVG({{ experiment_metric.formula.denominator.alias }}) as denominator_avg,
        VAR_SAMP({{ experiment_metric.formula.numerator.alias }}) as numerator_var,
        VAR_SAMP({{ experiment_metric.formula.denominator.alias }}) as denominator_var,
        COVAR_SAMP(...) as covariance
    {% else %}
        NULL as numerator_avg,
        -- ... other NULLs for non-ratio metrics
    {% endif %}
FROM user_aggregation
GROUP BY group_name
{% if not loop.last %}UNION ALL{% endif %}
{% endfor %}
```

## Creating Custom Templates

### Step 1: Create Your Scenario

You have complete freedom in how to organize your templates. Choose any approach:

**Option A: Single File**
**Option B: Directory Structure** 

### Step 2: Define Entry Point

Create your main template with any name that makes sense:
- `main.sql`, `base.sql`, `__entry__.j2`, `calculation.sql`, etc.
- The file name doesn't matter — you'll reference it in configuration

### Step 3: Implement Required Output Format

**This is the only strict requirement** — your template must produce results in this exact format:

#### Required Output Columns
```sql
SELECT
    group_name,              -- experiment group (control/treatment)
    metric_name,             -- metric identifier  
    metric_display_name,     -- human-readable name
    metric_type,             -- 'ratio', 'avg', 'proportion'
    observation_cnt,         -- number of observations (must be > 0)
    metric_value,           -- calculated metric value
    numerator_avg,          -- average of numerator (for ratios)
    denominator_avg,        -- average of denominator (for ratios)  
    numerator_var,          -- variance of numerator
    denominator_var,        -- variance of denominator
    covariance              -- covariance (for delta method)
```

These columns must match the `QueryMetricResult` schema exactly. The system validates:

- **observation_cnt** > 0
- **numerator_var, denominator_var** ≥ 0 (if not NULL)
- **proportion metrics**: metric_value between 0 and 1
- **ratio metrics**: all statistical fields must be present (not NULL)
- **average metrics**: numerator_var must be present (not NULL)

#### Handle Different Metric Types
```sql
{% if experiment_metric.type == 'ratio' %}
    -- Calculate all statistical components
    AVG({{ experiment_metric.formula.numerator.alias }}) as numerator_avg,
    AVG({{ experiment_metric.formula.denominator.alias }}) as denominator_avg,
    VAR_SAMP({{ experiment_metric.formula.numerator.alias }}) as numerator_var,
    VAR_SAMP({{ experiment_metric.formula.denominator.alias }}) as denominator_var,
    COVAR_SAMP(
        {{ experiment_metric.formula.numerator.alias }},
        {{ experiment_metric.formula.denominator.alias }}
    ) as covariance
{% elif experiment_metric.type == 'avg' %}
    -- Only numerator variance needed
    NULL as numerator_avg,
    NULL as denominator_avg,
    VAR_SAMP({{ experiment_metric.formula.numerator.alias }}) as numerator_var,
    NULL as denominator_var,
    NULL as covariance
{% else %}
    -- Proportion metrics: all NULLs
    NULL as numerator_avg,
    NULL as denominator_avg,
    NULL as numerator_var,
    NULL as denominator_var,
    NULL as covariance
{% endif %}
```

### Step 4: Update Configuration

Add your scenario to `.streamlit/expanto.toml`:

```toml
[queries.scenarios]
# Point to your main template file
my_custom_scenario = "my_custom_scenario/main.sql"
# OR for single file:
simple_scenario = "my_simple_template.sql"  
# OR whatever path/name you chose:
complex_scenario = "advanced/bigquery_optimized.j2"
```


## Advanced Features

### Conditional Logic
```sql
-- Handle planning vs regular calculations (it's essential for planning page).
{% if purpose == 'planning' %}
    {% include "dummy_testids.sql" %}
{% else %}
    {% include "testids.sql" %}
{% endif %}
```

### Audience Segmentation
```sql
-- Join audience tables if specified
{% if observation.audience_tables %}
    {% for audience_table in observation.audience_tables %}
    INNER JOIN {{ audience_table }} as audience_{{ loop.index }}
        ON events.user_id = audience_{{ loop.index }}.user_id
    {% endfor %}
{% endif %}
```

### Dynamic Filtering
```sql
-- Apply observation filters
WHERE 1=1
{% if observation.filters %}
    {% for filter in observation.filters %}
    AND {{ filter }}
    {% endfor %}
{% endif %}
```

### Exposure Events
```sql
-- Handle specific exposure events
{% if observation.exposure_event %}
INNER JOIN (
    SELECT user_id, MIN(event_timestamp) as exposure_timestamp
    FROM events
    WHERE event_name = '{{ observation.exposure_event }}'
    GROUP BY user_id
) exposure ON events.user_id = exposure.user_id
{% endif %}
```

## Testing Templates

**Important:** Testing your templates is crucial before using them in production. Use the test framework as a starting point.

### 1. Test Framework Template

Expanto includes a test template at `tests/queries/test_sql.py` that you can adapt for your templates. **This file is not run by default** — it's a template for you to customize.

### 2. Setting Up Template Tests

Copy and adapt the test structure:

```python
# tests/queries/test_my_templates.py
import pytest
from datetime import UTC, datetime
from jinja2 import Environment, FileSystemLoader
from src.domain.models import Observation
from src.services.runners.connectors import ConnectorResolver
from src.settings import Config, Secrets

@pytest.fixture(scope="module")
def test_observation():
    """Create test observation with your data."""
    return Observation(
        exposure_start_datetime=datetime(2025, 1, 1, tzinfo=UTC),
        exposure_end_datetime=datetime(2025, 1, 2, tzinfo=UTC),
        calc_start_datetime=datetime(2025, 1, 1, tzinfo=UTC),
        calc_end_datetime=datetime(2025, 1, 2, tzinfo=UTC),
        db_experiment_name="test_experiment",
        split_id="user_id",  # your ID field
        filters=None,
    )

@pytest.fixture(scope="module") 
def jinja_env():
    return Environment(loader=FileSystemLoader('queries/'))

@pytest.mark.integration
def test_my_template_syntax(jinja_env, test_observation):
    """Test that template renders without syntax errors."""
    template = jinja_env.get_template('my_scenario/main.sql')
    
    # Test with minimal data
    query = template.render(
        observation=test_observation,
        experiment_metrics_list=[],
        user_formula_list=[]
    )
    assert query is not None
    assert len(query) > 0
```

### 3. Test Individual Components

Break down your templates into testable parts:

```python
@pytest.mark.integration
def test_events_template(jinja_env, connector, test_observation):
    """Test events extraction works with your data."""
    template = jinja_env.get_template('my_scenario/events.sql')
    query = template.render(observation=test_observation) + " LIMIT 10"
    
    df = connector.run_query_to_df(query)
    
    # Verify expected columns exist
    assert 'user_id' in df.columns
    assert 'event_timestamp' in df.columns
    assert 'event_name' in df.columns
    
    # Verify data is within expected time range
    assert len(df) >= 0

@pytest.mark.integration  
def test_experiment_assignments(jinja_env, connector, test_observation):
    """Test experiment assignment retrieval."""
    template = jinja_env.get_template('my_scenario/testids.sql')
    query = template.render(observation=test_observation) + " LIMIT 10"
    
    df = connector.run_query_to_df(query)
    
    # Verify group assignments
    assert 'group_name' in df.columns
    assert 'user_id' in df.columns
    expected_groups = ['control', 'treatment']  # your groups
    assert df['group_name'].isin(expected_groups).any()
```

### 4. Test Full Template Output

```python
@pytest.mark.integration
def test_complete_template(jinja_env, connector, test_observation):
    """Test complete template produces valid results."""
    from src.domain.metrics import ExperimentMetricDefinition, MetricFormula, UserAggregationFormula
    
    # Create test metrics
    user_formula = UserAggregationFormula(
        alias="sessions", 
        sql="COUNT(DISTINCT session_id)"
    )
    
    experiment_metric = ExperimentMetricDefinition(
        alias="avg_sessions",
        type="avg", 
        display_name="Average Sessions",
        description="Test metric",
        formula=MetricFormula(numerator=user_formula, denominator=None),
        owner="test",
        group_name="test_group",
        tags=None
    )
    
    template = jinja_env.get_template('my_scenario/main.sql')
    query = template.render(
        observation=test_observation,
        experiment_metrics_list=[experiment_metric],
        user_formula_list=[user_formula]
    )
    
    # Execute and validate results
    results = connector.fetch_results(query)
    assert results is not None
    assert len(results) > 0
    
    # Verify required output format
    result = results[0]
    assert hasattr(result, 'group_name')
    assert hasattr(result, 'metric_name')
    assert hasattr(result, 'metric_value')
    assert hasattr(result, 'observation_cnt')
```

### 5. Running Tests

```bash
# Run your custom template tests
pytest tests/queries/test_my_templates.py -v

# Run with integration flag (requires DWH connection)
pytest tests/queries/test_my_templates.py -m integration -v

# Run specific test
pytest tests/queries/test_my_templates.py::test_complete_template -v
```
