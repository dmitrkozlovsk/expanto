import logfire


def get_logfire_config() -> dict:
    """Get Logfire token from secrets, with fallback to default."""

    _logfire_config = {
        "service_name": "Expanto",
        "send_to_logfire": False,
        "distributed_tracing": True,
        "console": logfire.ConsoleOptions(
            span_style="indented",
            include_timestamps=True,
            min_log_level="debug",
            show_project_link=False,
            verbose=False,
            colors="auto",
    ),
    }
    try:
        from src.settings import Config, Secrets
        secrets = Secrets()
        config = Config()
        if config.logfire.send_to_logfire:
            _logfire_config["send_to_logfire"] = True
        if secrets.api_keys.logfire_token:
            _logfire_config["token"] = secrets.api_keys.logfire_token.get_secret_value()
        return _logfire_config
    except Exception:
        pass
    
    return _logfire_config

logfire_config = get_logfire_config()
logfire.configure(**logfire_config)
logfire.instrument_pydantic()
logfire.instrument_httpx()
logger = logfire