from pathlib import Path

dir_fake_metrics = Path(__file__).parent / "metrics"
dir_fake_query = Path(__file__).parent


fake_configs_dict = {
    "metrics": {"dir": str(dir_fake_metrics)},
    "queries": {
        "dir": str(dir_fake_query),
        "scenarios": {
            "base": "query.j2",
            "base2": "base2",
        },
    },
    "precompute_db": {"name": "snowflake"},
    "assistant": {
        "provider": "super_druper_provider",
        "models": {
            "fast": "Q",
            "agentic": "Q",
            "tool_thinker": "Q",
        },
        "service": {
            "url": "http://127.0.0.1:8000",
            "timeout_seconds": 600,
            "enable_streaming": True,
            "auto_scroll": True,
        },
    },
    "logfire": {"send_to_logfire": False},
}

fake_secrets_dict = {
    "internal_db": {
        "engine_str": "to_replace",
        "async_engine_str": "fake",
        "connect_args": {},
    },
    "bigquery": {
        "file_path": "/fake/path.json",
        "project_name": "proj",
        "connection_type": "service_account",
    },
    "snowflake": {
        "account": "a",
        "user": "u",
        "password": "p",
        "warehouse": "w",
        "database": "d",
        "schema": "s",
    },
    "api_keys": {
        "PROVIDER_API_KEY": "TOGETHER_API_KEY",
        "TAVILY_API_KEY": "TAVILY_API_KEY",
        "LOGFIRE_TOKEN": "LOGFIRE_TOKEN",
    },
}
