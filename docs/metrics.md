# Metrics Documentation

In Expanto, metrics are defined in YAML files with a two-level structure: user-level aggregations and experiment-level metrics.

## Core Concepts

### Two-Level Architecture

Expanto supports a flexible approach to metric definition:

1. **User Aggregations** (optional) — reusable calculations performed at the user level
2. **Experiment Metrics** — business metrics with inline or referenced formulas

You have two options for defining formulas:

**Option A: Reusable User Aggregations** (recommended for complex setups)
```yaml
user_aggregations:
  sessions_count: &sessions_count
    alias: sessions_count
    sql: COUNT(DISTINCT session_id)

metrics:
  - alias: avg_sessions
    formula:
      numerator: *sessions_count  # Reference to user aggregation
```

**Option B: Inline Formulas** (simpler for one-off metrics)
```yaml
metrics:
  - alias: avg_sessions
    formula:
      numerator:
        alias: sessions_count
        sql: COUNT(DISTINCT session_id)  # Defined directly in metric
```

Benefits of user aggregations:
- **Reusability** — same aggregations used in multiple metrics
- **Maintainability** — change SQL logic in one place
- **Performance** — query optimization opportunities

### Metric Calculation Flow

```
Raw Events → User Aggregations → Experiment Metrics → Statistical Analysis
```

**Example:**
```
page_views (events) → sessions_count (user agg) → avg_sessions_per_user (metric) → t-test
```

## YAML File Structure

Metrics are organized in YAML files under the `metrics/` directory. Each file represents a logical group of related metrics.

### Basic Structure

**With User Aggregations (Reusable):**
```yaml
metric_group_name: "Core Business Metrics"
user_aggregations:
  # Reusable user-level calculations
  sessions_count: &sessions_count
    alias: sessions_count
    sql: COUNT(DISTINCT session_id)
  
  revenue_total: &revenue_total
    alias: revenue_total  
    sql: SUM(CASE WHEN event_name = 'purchase' THEN event_value ELSE 0 END)

metrics:
  - alias: avg_sessions_per_user
    type: avg
    display_name: "Average Sessions per User"
    description: "Average number of sessions per user during the experiment period"
    formula:
      numerator: *sessions_count  # Reference to user aggregation
      denominator: null
    owner: product_team
    tags: ["engagement", "core"]
```

**Without User Aggregations (Inline):**
```yaml
metric_group_name: "Core Business Metrics"
# user_aggregations section is optional

metrics:
  - alias: avg_sessions_per_user
    type: avg
    display_name: "Average Sessions per User"
    description: "Average number of sessions per user during the experiment period"
    formula:
      numerator:
        alias: sessions_count
        sql: COUNT(DISTINCT session_id)  # Defined directly in formula
      denominator: null
    owner: product_team
    tags: ["engagement", "core"]
```

### Required Fields

#### File Level
- **metric_group_name** — logical grouping name (max 140 chars)
- **user_aggregations** — dictionary of reusable user-level formulas (optional)
- **metrics** — list of experiment metrics (required)

#### User Aggregation (Optional Section)
- **alias** — unique identifier (lowercase, underscore allowed)
- **sql** — SQL aggregation formula

#### Experiment Metric  
- **alias** — unique identifier across all files
- **type** — metric type: `avg`, `ratio`, or `proportion`
- **display_name** — human-readable name (max 60 chars)
- **description** — detailed explanation (optional)
- **formula** — numerator and optional denominator (can reference user aggregations or be defined inline)
- **owner** — metric owner (optional)
- **tags** — categorization tags (optional)

#### Formula Definition (Two Options)
**Option 1: Reference to User Aggregation**
```yaml
formula:
  numerator: *sessions_count  # YAML reference
  denominator: *users_count
```

**Option 2: Inline Definition**
```yaml
formula:
  numerator:
    alias: sessions_count
    sql: COUNT(DISTINCT session_id)
  denominator:
    alias: users_count
    sql: COUNT(DISTINCT user_id)
```

## Metric Types

### 1. Average Metrics (`avg`)

Calculate the mean of a user-level aggregation across experiment groups.

**With User Aggregations:**
```yaml
- alias: avg_revenue_per_user
  type: avg
  display_name: "Average Revenue per User"
  description: "Mean revenue generated per user during experiment period"
  formula:
    numerator: *revenue_total  # Reference to user aggregation
    denominator: null  # Always null for avg metrics
```

**Inline Definition:**
```yaml
- alias: avg_revenue_per_user
  type: avg
  display_name: "Average Revenue per User"
  description: "Mean revenue generated per user during experiment period"
  formula:
    numerator:
      alias: revenue_total
      sql: SUM(CASE WHEN event_name = 'purchase' THEN event_value ELSE 0 END)
    denominator: null  # Always null for avg metrics
```

**SQL Generated:**
```sql
AVG(revenue_total) as metric_value
```

**Use Cases:** session counts, time spent, revenue amounts, engagement scores

### 2. Ratio Metrics (`ratio`)

Calculate the ratio between two user-level aggregations.

**With User Aggregations:**
```yaml
- alias: conversion_rate
  type: ratio
  display_name: "Conversion Rate"
  description: "Ratio of users who purchased to total active users"
  formula:
    numerator: *purchases_count      # Reference
    denominator: *active_sessions_count  # Reference
```

**Inline Definition:**
```yaml
- alias: conversion_rate
  type: ratio
  display_name: "Conversion Rate"
  description: "Ratio of users who purchased to total active users"
  formula:
    numerator:
      alias: purchases_count
      sql: COUNT(CASE WHEN event_name = 'purchase' THEN 1 END)
    denominator:
      alias: active_sessions_count
      sql: COUNT(DISTINCT session_id)
```

**SQL Generated:**
```sql
CASE WHEN SUM(active_sessions_count) > 0 
     THEN SUM(purchases_count) / SUM(active_sessions_count) 
     ELSE 0 END as metric_value
```

**Use Cases:** conversion rates, click-through rates, success rates

### 3. Proportion Metrics (`proportion`)

Calculate the proportion of users meeting a binary condition.

```yaml
# With denominator (explicit proportion)
- alias: high_engagement_users_pct
  type: proportion
  display_name: "High Engagement Users (%)"
  description: "Percentage of users with 5+ sessions"
  formula:
    numerator: *high_engagement_flag
    denominator: *total_users_count

# Without denominator (average of 0/1 flags)
- alias: conversion_rate_simple
  type: proportion
  display_name: "Conversion Rate (Simple)"
  description: "Proportion of users who converted"
  formula:
    numerator: *converted_flag
    denominator: null
```

**SQL Generated:**
```sql
-- With denominator
CASE WHEN SUM(total_users_count) > 0 
     THEN SUM(high_engagement_flag) / SUM(total_users_count) 
     ELSE 0 END

-- Without denominator  
AVG(converted_flag)
```

**Use Cases:** feature adoption rates, user segment proportions, binary outcome rates

## Validation Rules

Expanto validates metrics strictly to prevent runtime errors:

### Alias Validation
- **Format:** lowercase, start with letter/underscore, only letters/digits/underscores
- **Length:** 1-50 characters
- **Uniqueness:** all aliases must be unique across all YAML files
- **Reserved words:** cannot contain SQL keywords (SELECT, FROM, etc.)

### SQL Formula Validation
- **Balanced parentheses:** all `(` must have matching `)`
- **No SQL statements:** cannot contain SELECT, FROM, JOIN, etc.
- **No comments:** `--` and `/* */` not allowed
- **Non-empty:** must contain actual SQL logic

```yaml
# ✅ Valid SQL
sql: COUNT(DISTINCT user_id)
sql: SUM(CASE WHEN event_name = 'purchase' THEN event_value ELSE 0 END)
sql: AVG(session_duration_seconds)

# ❌ Invalid SQL
sql: "SELECT COUNT(*) FROM events"  # SQL statement
sql: "COUNT("                       # unbalanced parentheses
sql: "COUNT(*) -- comment"          # contains comment
sql: ""                             # empty
```

### Metric Type Validation
- **Average metrics:** denominator must be null
- **Ratio metrics:** denominator is required
- **Proportion metrics:** denominator optional

### Cross-Reference Validation
- All metric aliases must be unique across files
- All user aggregation aliases must be unique across files (when using user_aggregations section)
- Referenced user aggregations must exist (when using YAML references)
- Inline formula aliases must be unique within the metric scope

## Organizing Metrics

### File Organization

Group related metrics logically:

```
metrics/
├── core_business_metrics.yaml      # KPIs, revenue, conversion
├── engagement_metrics.yaml         # sessions, page views, time spent  
├── product_feature_metrics.yaml    # feature adoption, usage
├── user_lifecycle_metrics.yaml     # onboarding, retention, churn
└── quality_metrics.yaml           # errors, performance, satisfaction
```

### Tagging Strategy

Use tags for filtering and organization:

```yaml
tags: ["core_kpi", "revenue"]        # Primary KPIs
tags: ["engagement", "product"]      # Product engagement
tags: ["conversion", "funnel"]       # Conversion funnel
tags: ["retention", "lifecycle"]     # User lifecycle
tags: ["quality", "performance"]     # Quality metrics
tags: ["experiment_specific"]        # One-off metrics
```

### Naming Conventions

**User Aggregation Aliases:** Use descriptive suffixes indicating the aggregation type
```yaml
# ✅ Recommended suffixes
sessions_cnt        # COUNT(...)
revenue_sum         # SUM(...)
converted_flg       # MAX(CASE WHEN ... THEN 1 ELSE 0 END)
duration_avg        # AVG(...)
price_max          # MAX(...)
orders_total       # SUM(order_amount)

# ❌ Generic naming
sessions
revenue
converted
duration
```

**Experiment Metric Aliases:** Use prefixes indicating the metric type
```yaml
# ✅ Recommended prefixes
avg_sessions_per_user      # Average metrics
avg_revenue_per_user
avg_session_duration

cr_checkout               # Conversion rates (proportion)
cr_signup
cr_purchase

ctr_email_campaign        # Click-through rates (ratio)
ctr_banner_ads

share_mobile_users        # Proportions/shares
share_returning_customers
```

**Display Names:** Use clean business language, avoid redundancy
```yaml
# ✅ Clean display names (avoid duplicating the metric type)
alias: avg_sessions_cnt
display_name: "Sessions Count"           # Not "Average Sessions Count"

alias: avg_revenue_per_user  
display_name: "Revenue per User"         # Not "Average Revenue per User"

alias: cr_checkout
display_name: "Checkout Conversion (CR)" # Explicitly mention it's a conversion rate

alias: share_mobile_users
display_name: "Mobile Users (%)"         # Explicitly mention it's a percentage

# ✅ Exception: When conversion context is important
alias: cr_email_signup
display_name: "Email Signup Rate"        # "Rate" is clear in business context

alias: cr_trial_to_paid
display_name: "Trial to Paid Conversion" # "Conversion" adds business context

# ❌ Redundant display names
alias: avg_sessions_cnt
display_name: "Average Sessions Count"   # "Average" is redundant

alias: cr_checkout  
display_name: "Checkout CR"              # Technical abbreviation
```



## Testing and Validation

```bash
# Test metric loading in Expanto
python -c "from src.services.metric_register import Metrics; m = Metrics('metrics'); print('OK')"
```
