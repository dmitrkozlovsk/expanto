from unittest.mock import patch

import pytest


@pytest.fixture
def patch_configs(
    monkeypatch, disable_files_in_settings, fake_load_secrets_cfg, fake_load_expanto_cfg, engine, metrics
):
    """Patch configuration loading for UI testing."""
    monkeypatch.setattr(
        "src.ui.resources.Secrets",
        lambda *a, **kw: fake_load_secrets_cfg,
        raising=True,
    )

    # Same for Config()
    monkeypatch.setattr(
        "src.ui.resources.Config",
        lambda *a, **kw: fake_load_expanto_cfg,
        raising=True,
    )


@pytest.fixture
def patch_client_and_resolver():
    """Mock BigQuery client and connector resolver for testing."""
    with patch("src.services.runners.connectors.bigquery.Client") as MockClient:
        inst = MockClient.from_service_account_json.return_value

        with patch("src.services.runners.connectors.ConnectorResolver.resolve", return_value=inst):
            yield inst
