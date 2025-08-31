import pytest


@pytest.fixture
def patch_configs(
    monkeypatch, disable_files_in_settings, fake_load_secrets_cfg, fake_load_expanto_cfg, engine, metrics
):
    """Patch configuration loading for UI testing."""
    monkeypatch.setattr(
        "assistant.app.Secrets",
        lambda *a, **kw: fake_load_secrets_cfg,
        raising=True,
    )

    # Same for Config()
    monkeypatch.setattr(
        "assistant.app.Config",
        lambda *a, **kw: fake_load_expanto_cfg,
        raising=True,
    )


@pytest.fixture(scope="session")
def docs_temp_dir(tmp_path_factory):
    """Create temporary directory with mock documentation files."""
    temp_dir = tmp_path_factory.mktemp("docs")
    (temp_dir / "integration.md").write_text("Integration documentation. how to start with expanto")
    (temp_dir / "queries.md").write_text("Queries documentation. how to create queries")
    (temp_dir / "precompute.md").write_text("Precompute documentation. how to create precompute")
    (temp_dir / "metrics.md").write_text("Metrics documentation. how to create metrics")
    return temp_dir


@pytest.fixture(scope="session")
def root_dir(tmp_path_factory):
    """Create temporary root directory structure for testing."""
    temp_dir = tmp_path_factory.mktemp("root")
    (temp_dir / "src").mkdir()
    (temp_dir / ".expanto").mkdir()
    (temp_dir / "src" / "agent").mkdir()
    (temp_dir / "src" / "services").mkdir()
    (temp_dir / "src" / ".env").write_text("ENV_VAR=!!!")
    (temp_dir / "src" / "agent" / "vdb.py").write_text("This is a vdb code")
    (temp_dir / "src" / "services" / "metric_register.py").write_text("This is a metric register code")
    (temp_dir / "src" / "services" / "bigquery.py").write_text("This is a bigquery code")
    (temp_dir / ".expanto" / "secrets.toml").write_text("SECRET_KEY=000111")
    return temp_dir
