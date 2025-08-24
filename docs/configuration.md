# Configuration Guide

This guide covers the post-installation configuration steps needed to get Expanto fully operational with your data warehouse and AI services.

**Legend:**
- ⭐ **REQUIRED** — must be configured for the system to work
- 🔹 **OPTIONAL** — has sensible defaults, customize as needed

## Overview

After running `make setup`, you'll have configuration templates in `.streamlit/` directory that need to be customized for your environment:

- `.streamlit/secrets.toml` — sensitive credentials (DWH, API keys)
- `.streamlit/expanto.toml` — application settings
- `.streamlit/config.toml` — Streamlit UI configuration

## Required Configuration

### 1. Data Warehouse Connection ⭐ **REQUIRED**

Configure at least one data warehouse in `.streamlit/secrets.toml`:

#### BigQuery Setup

```toml
[bigquery]
file_path = "path/to/your-service-account.json"
project_name = "your-project-id"
connection_type = "service_account"  # or "application_default"
```

**Steps:**
1. Create a service account in Google Cloud Console
2. Download the JSON key file
3. Grant BigQuery permissions: `BigQuery Data Viewer`, `BigQuery Job User`
4. Update `file_path` with absolute path to JSON file
5. Set `connection_type = "service_account"`

#### Snowflake Setup

```toml
[snowflake]
account = "your-account.region.cloud"
user = "your-username"
password = "your-password"
warehouse = "your-warehouse"
database = "your-database"
schema = "your-schema"
```

**Steps:**
1. Ensure your Snowflake user has access to the target database/schema
2. Test connection with the provided credentials
3. Verify the account identifier format (include region and cloud provider)

### 2. AI Assistant Configuration ⭐ **REQUIRED**

#### API Keys ⭐ **REQUIRED**

Add your AI provider API key to `.streamlit/secrets.toml`:

```toml
[api_keys]
PROVIDER_API_KEY = "your-together-ai-api-key"  # ⭐ REQUIRED
TAVILY_API_KEY = "your-tavily-key"  # 🔹 OPTIONAL - for web search
LOGFIRE_TOKEN = "your-logfire-token"  # 🔹 OPTIONAL - for observability
```

**Steps:**
1. ⭐ **REQUIRED:** Sign up at [Together.ai](https://together.ai) and get an API key
2. ⭐ **REQUIRED:** Add the key to `PROVIDER_API_KEY`
3. 🔹 **Optional:** Get Tavily API key for web search capabilities
4. 🔹 **Optional:** Get Logfire token for observability

#### Model Configuration 🔹 **OPTIONAL**

Customize AI models in `.streamlit/expanto.toml` (default models work fine):

```toml
[assistant.models]
fast = "deepseek-ai/DeepSeek-V3"
tool_thinker = "Qwen/Qwen3-235B-A22B-Thinking-2507"
agentic = "moonshotai/Kimi-K2-Instruct"AI Assistant Configuration
```

### 3. Database Selection ⭐ **REQUIRED**

Set which data warehouse to use for calculations in `.streamlit/expanto.toml`:

```toml
[precompute_db]
name = "snowflake"  # or "bigquery"
```

### 4. Internal Database 🔹 **OPTIONAL**

The SQLite configuration is pre-configured and works out of the box. Only customize if needed:

```toml
[internal_db]
engine_str = "sqlite:///expanto.db"
async_engine_str = "sqlite+aiosqlite:///expanto.db"

[internal_db.connect_args]
pool_size = 5
max_overflow = 5
pool_timeout = 10
pool_recycle = 900
connect_args = { check_same_thread = false }
```
