# Expanto

**Experimenter Analytics Tools** — A/B testing platform for analysts and data scientists who need full control over their experimentation workflow.

## Design Principles

**One person, full control** — Own the entire workflow from setup to insights using familiar tools (Python, SQL, YAML) without requiring a separate engineering team.

**Flexibility over perfection** — Real experiments rarely fit templates. Expanto prioritizes adaptability, letting you modify metrics, logic, and queries as needed.

## Features

- **Experiment management** — Create, track, and close experiments
- **Flexible calculations** — Define custom metrics and pipelines.
- **Statistical analysis** — Built-in statistical tests and confidence intervals
- **AI assistant** — Multi-agent system for experiment design and analysis (it is an experiment for now).
- **Web interface** — Streamlit-based UI for all operations

## Demo
![demo](https://github.com/user-attachments/assets/8639cc72-62c1-4eb0-87e2-27c9b2eeb189)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/dmitrkozlovsk/expanto.git
cd expanto
make setup

# Configure your data warehouse
# Edit .streamlit/secrets.toml with your credentials

# Start application
make start
```

This will:
- Install dependencies via uv
- Copy configuration templates into .streamlit
- Create example experiment in SQLite database
- Start Streamlit UI and AI assistant

## Documentation

### Core Documentation
- **[Configuration Guide](docs/configuration.md)** — Post-installation setup: data warehouse connections, AI assistant, secrets management
- **[Architecture Overview](docs/architecture.md)** — System design, components, data flow, and technology stack

### Customization Guides  
- **[Query Templates](docs/queries.md)** — Create SQL templates for your data warehouse and experiment calculations
- **[Metrics Configuration](docs/metrics.md)** — Define experiment metrics in YAML: averages, ratios, proportions

### Additional Resources
- **[Metrics Examples](metrics/)** — Example YAML metric definitions
- **[Query Examples](queries/)** — Example SQL templates for BigQuery and Snowflake
