# Architecture

The architecture consists of three main layers: user interface, business logic, and data layer.

## System Overview

```
┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │  AI Assistant   │
│                 │    │   (FastAPI)     │
│   • Experiments │◄──►│                 │
│   • Observations│    │  Multi-Agent    │
│   • Results     │    │  System +       │
│   • Planner     │    │  Vector DB      │
│   • Chat        │    │                 │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          └──────────┬───────────┘
                     │
         ┌───────────▼────────────┐
         │    Business Logic      │
         │  (Services Layer)      │
         └───────────┬────────────┘
                     │
    ┌────────────────┼──────────────────┐
    │                │                  │
┌───▼───┐    ┌───────▼─────┐    ┌───────▼──────┐
│SQLite │    │ BigQuery/   │    │ YAML Metrics │
│(Local)│    │ Snowflake   │    │ + Templates  │
└───────┘    └─────────────┘    └──────────────┘
```

## Core Components

### 1. User Interface Layer (`src/ui/`)

**Streamlit Application** — main interface with four pages:
- `experiments/` — experiment management (create, edit, close)
- `observations/` — analysis configuration (periods, audience, metrics)
- `results/` — view results and statistics
- `planner/` — sample size planning

**Chat Interface** (`src/ui/chat/`) — integrated AI assistant for analysis support.

### 2. AI Assistant (`assistant/`)

**Multi-Agent System** based on FastAPI:
- `AgentManager` — manages specialized agents
- `AgentOrchestrator` — routes requests to appropriate agents
- Specialized agents: experiment creator, analyst, SQL expert, router, docs assistant, internet search
- Each agent has its own system prompts and tools
- **Vector Database** — ChromaDB for documentation search and context retrieval
- **Memory System** — TTL cache for conversation history and usage tracking

### 3. Business Logic Layer (`src/services/`)

- **Entities Layer** (`src/services/entities/`) — data handlers for experiments, observations, jobs, precomputes
- **Analytics Layer** (`src/services/analytics/`) — sample size calculator, statistical functions, t-tests
- **Runners Layer** (`src/services/runners/`) — query rendering, calculation execution, DWH connectors
- **Metrics Registry** (`src/services/metric_register.py`) — YAML metrics loading, filtering, dependency resolution

### 4. Domain Layer (`src/domain/`)

**Core Models** — SQLAlchemy models:
- `Experiment` — experiment (name, status, hypotheses, key metrics)
- `Observation` — analysis configuration (periods, audience, filters)
- `CalculationJob` — metrics calculation task (SQL query, status)
- `Precompute` — calculation results (metric values, statistics)

### 5. Metrics (`metrics/`)

YAML-based metric definitions with formulas, tags, and groups for organizing experiment metrics.

→ See [Metrics Documentation](metrics.md) for detailed configuration guide.

### 6. Queries (`queries/`)

Jinja2 SQL templates for different data warehouses (BigQuery, Snowflake) that render experiment calculations.

→ See [Queries Documentation](queries.md) for template structure and examples.

### External Data Warehouse

Connection to BigQuery or Snowflake for:
- Raw event data retrieval
- User-level aggregations calculation
- Complex analytical query execution

## Configuration

- `src/settings.py` — main application settings
- `.streamlit/secrets.toml` — secrets (DWH credentials, API keys)
- `.streamlit/config.toml` — Streamlit UI configuration
- `.streamlit/expanto.toml` — Expanto-specific settings (queries, scenarios)
- `metrics/*.yaml` — metric definitions with formulas, tags, and groups
- `queries/` — SQL templates for different data warehouses
- `assistant/prompts/` — system prompts for AI agents

## Key Technologies

**Core Framework:**
- **Streamlit** — main UI framework for web interface
- **FastAPI** — REST API for AI assistant service
- **SQLAlchemy** — ORM for local SQLite database
- **Pydantic** — data validation and settings management

**AI & Agents:**
- **pydantic-ai** — multi-agent framework for AI assistants
- **ChromaDB** — vector database for documentation search and retrieval
- **Logfire** — observability and logging for AI agents and application

**Data Processing:**
- **Jinja2** — SQL query templating engine
- **PyYAML** — metrics configuration parsing
- **snowflake-connector-python** — Snowflake data warehouse connector
- **google-cloud-bigquery** — BigQuery data warehouse connector

**Analytics:**
- **scipy** — statistical calculations and hypothesis testing
- **statsmodels** — advanced statistical analysis
- **plotly** — interactive data visualization
