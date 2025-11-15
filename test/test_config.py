from __future__ import annotations

from pathlib import Path

import pytest

from bmri.config import (
    CLISettings,
    FittingModel,
    T1Config,
    T1rhoConfig,
    T1rhoSequenceConfig,
    T2Config,
    load_config_from_toml,
)
from bmri.exceptions import ConfigurationError


def test_t2_config_validates_boundaries() -> None:
    config = T2Config()
    assert config.boundary[0][0] < config.boundary[1][0]

    with pytest.raises(ValueError):
        T2Config(boundary=((0.0, 0.0, -1.0), (0.0, 50.0, 1.0)))


def test_t1rho_config_requires_sequence_for_advanced_models() -> None:
    with pytest.raises(ValueError):
        T1rhoConfig(model=FittingModel.ARONEN)

    sequence = T1rhoSequenceConfig(TR=1000, T1=1200, alpha=90, TE=10, T2star=40)
    config = T1rhoConfig(model=FittingModel.ARONEN, sequence=sequence)
    assert config.sequence is not None
    assert config.sequence.TE == 10


def test_cli_settings_have_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("BMRI_VERBOSE=true\n")
    monkeypatch.chdir(tmp_path)
    settings = CLISettings()
    assert settings.verbose is True
    assert settings.log_file is None


def test_load_config_from_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "t2.toml"
    config_path.write_text(
        """
boundary = [[0.9, 5.0, -0.5], [3.0, 40.0, 0.5]]
normalize = true
min_r2 = 0.8
"""
    )

    config = load_config_from_toml(config_path, T2Config)
    assert config.min_r2 == 0.8


def test_load_config_from_toml_errors(tmp_path: Path) -> None:
    missing = tmp_path / "missing.toml"
    with pytest.raises(ConfigurationError):
        load_config_from_toml(missing, T1Config)
