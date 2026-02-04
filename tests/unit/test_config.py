"""Test configuration settings."""
import pytest
from pathlib import Path
from market_event_ai.config.settings import Settings, Paths


def test_settings_initialization():
    """Test settings can be initialized."""
    settings = Settings()
    assert settings is not None
    assert isinstance(settings.paths, Paths)


def test_paths_creation(tmp_path):
    """Test that paths are created correctly."""
    settings = Settings()
    # Check that default paths exist
    assert settings.paths.data_raw is not None
    assert settings.paths.data_processed is not None


def test_load_assets_config():
    """Test loading assets configuration."""
    settings = Settings()
    config = settings.load_assets_config()
    assert isinstance(config, dict)
    assert 'etfs' in config or 'stocks' in config or 'trading_rules' in config
