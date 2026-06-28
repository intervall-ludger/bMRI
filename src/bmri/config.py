"""Configuration management for bMRI.

This module provides Pydantic-based configuration classes for all fitting
types and CLI operations. Configurations can be loaded from TOML files or
created programmatically.
"""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from bmri.types import BoundaryTuple


class FittingModel(str, Enum):
    """Available fitting models."""

    MONO_EXP = "mono_exp"  # Simple mono-exponential
    ARONEN = "aronen"  # Aronen model (T1rho/T2prep)
    RAUSCH = "rausch"  # Rausch model (T1rho)


class BaseFittingConfig(BaseModel):
    """Base configuration for all fitting types.

    This class defines common parameters shared across all fitting algorithms.
    """

    boundary: BoundaryTuple = Field(
        default=((-float("inf"),) * 3, (float("inf"),) * 3),
        description="Parameter boundaries as ((lower...), (upper...))",
    )
    normalize: bool = Field(
        default=False,
        description="Normalize signal intensities before fitting",
    )
    min_r2: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum R² threshold for valid fits",
    )
    pools: int = Field(
        default=0,
        ge=0,
        description="Number of CPU cores for parallel processing (0=auto)",
    )

    @field_validator("boundary")
    @classmethod
    def validate_boundary(cls, v: Any) -> BoundaryTuple:
        """Validate boundary format."""
        if not isinstance(v, tuple) or len(v) != 2:
            raise ValueError("Boundary must be ((lower...), (upper...))")

        lower, upper = v
        if not isinstance(lower, tuple) or not isinstance(upper, tuple):
            raise ValueError("Boundary components must be tuples")

        if len(lower) != len(upper):
            raise ValueError(
                f"Boundary length mismatch: lower has {len(lower)}, upper has {len(upper)}"
            )

        # Check that lower < upper
        for i, (lo, hi) in enumerate(zip(lower, upper, strict=False)):
            if lo >= hi:
                raise ValueError(f"Boundary[{i}]: lower ({lo}) must be < upper ({hi})")

        return v

    model_config = {"frozen": False}


class T2Config(BaseFittingConfig):
    """Configuration for T2/T2* fitting.

    Default boundaries optimized for cartilage T2/T2* mapping:
    - T2*: S0=[0.9,2], T2*=[0,50ms], offset=[-1,1]
    - T2:  S0=[0.9,3], T2=[5,40ms], offset=[-0.5,0.5]

    Example:
        >>> config = T2Config(
        ...     boundary=((0.9, 5, -0.5), (3, 40, 0.5)),
        ...     normalize=True,
        ...     min_r2=0.7,
        ... )
    """

    dim: int = Field(default=3, ge=2, le=3, description="Dimensionality (2D or 3D)")
    boundary: BoundaryTuple = Field(
        default=((0.9, 0.0, -1.0), (2.0, 50.0, 1.0)),
        description="Default T2* boundaries: S0=[0.9,2], T2*=[0,50], offset=[-1,1]",
    )
    normalize: bool = Field(
        default=True,
        description="Normalize signal (recommended for T2/T2*)",
    )
    min_r2: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum R² threshold (0.75 recommended for T2*)",
    )


class T1rhoSequenceConfig(BaseModel):
    """MRI sequence parameters for T1rho fitting.

    These parameters define the MRI pulse sequence characteristics required
    for accurate T1rho modeling using the Aronen or Rausch models.
    """

    TR: float = Field(..., gt=0, description="Repetition time (ms)")
    T1: float = Field(..., gt=0, description="Longitudinal relaxation time (ms)")
    alpha: float = Field(..., gt=0, le=180, description="Flip angle (degrees)")
    TE: float | None = Field(default=None, gt=0, description="Echo time (ms)")
    T2star: float | None = Field(default=None, gt=0, description="T2* relaxation time (ms)")

    @model_validator(mode="after")
    def validate_aronen_params(self) -> "T1rhoSequenceConfig":
        """Validate that Aronen model has required parameters."""
        # If TE or T2star is provided, both must be provided for Aronen model
        if (self.TE is not None) != (self.T2star is not None):
            raise ValueError("For Aronen model, both TE and T2star must be provided")
        return self


class T1rhoConfig(BaseFittingConfig):
    """Configuration for T1rho fitting.

    Default boundaries optimized for cartilage T1rho mapping:
    - S0=[1,10000], T1rho=[1,500ms], offset=[-1000,1000]

    Example:
        >>> # Simple mono-exponential
        >>> config = T1rhoConfig(
        ...     model=FittingModel.MONO_EXP,
        ...     boundary=((1, 1, -1000), (10000, 500, 1000)),
        ... )
        >>>
        >>> # Aronen model with sequence parameters
        >>> config = T1rhoConfig(
        ...     model=FittingModel.ARONEN,
        ...     sequence=T1rhoSequenceConfig(
        ...         TR=1000, T1=1500, alpha=90, TE=20, T2star=50
        ...     ),
        ... )
    """

    model: FittingModel = Field(
        default=FittingModel.MONO_EXP,
        description="Fitting model to use",
    )
    sequence: T1rhoSequenceConfig | None = Field(
        default=None,
        description="MRI sequence parameters (required for Aronen/Rausch models)",
    )
    dim: int = Field(default=3, ge=2, le=3, description="Dimensionality (2D or 3D)")
    boundary: BoundaryTuple = Field(
        default=((1.0, 1.0, -1000.0), (10000.0, 500.0, 1000.0)),
        description="Default T1rho boundaries: S0=[1,10000], T1rho=[1,500], offset=[-1000,1000]",
    )
    normalize: bool = Field(
        default=False,
        description="Normalization (typically False for T1rho)",
    )
    min_r2: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum R² threshold (0.3 recommended for T1rho)",
    )

    @model_validator(mode="after")
    def validate_model_params(self) -> "T1rhoConfig":
        """Validate that sequence params are provided for advanced models."""
        if self.model != FittingModel.MONO_EXP and self.sequence is None:
            raise ValueError(
                f"Model {self.model.value} requires sequence parameters (TR, T1, alpha, etc.)"
            )
        return self


class T1Config(BaseFittingConfig):
    """Configuration for T1 fitting.

    Example:
        >>> config = T1Config(
        ...     boundary=((0, 0, -100), (5000, 3000, 100)),
        ...     min_r2=0.8,
        ... )
    """

    dim: int = Field(default=3, ge=2, le=3, description="Dimensionality (2D or 3D)")


class CLISettings(BaseSettings):
    """Global CLI settings loaded from environment or config file.

    These settings can be overridden via:
    1. Environment variables (prefixed with BMRI_)
    2. .env file
    3. bmri.toml file

    Example:
        >>> settings = CLISettings()
        >>> print(settings.verbose)
        False
    """

    verbose: bool = Field(default=False, description="Enable verbose logging")
    quiet: bool = Field(default=False, description="Minimal output")
    log_file: Path | None = Field(default=None, description="Path to log file")

    model_config = SettingsConfigDict(
        env_prefix="BMRI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def load_config_from_toml(config_file: Path, config_class: type[BaseModel]) -> BaseModel:
    """Load configuration from TOML file.

    Args:
        config_file: Path to TOML configuration file
        config_class: Pydantic model class (T2Config, T1rhoConfig, etc.)

    Returns:
        Loaded configuration instance

    Raises:
        ConfigurationError: If file cannot be loaded or is invalid

    Example:
        >>> from bmri.config import load_config_from_toml, T1rhoConfig
        >>> config = load_config_from_toml(Path("t1rho.toml"), T1rhoConfig)
    """
    import tomli

    from bmri.exceptions import ConfigurationError

    if not config_file.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_file}",
            details="Please check the path and try again.",
        )

    try:
        with open(config_file, "rb") as f:
            data = tomli.load(f)
    except tomli.TOMLDecodeError as e:
        raise ConfigurationError(
            f"Invalid TOML syntax in {config_file}",
            details=str(e),
        ) from e

    try:
        return config_class(**data)
    except Exception as e:
        raise ConfigurationError(
            f"Invalid configuration in {config_file}",
            details=str(e),
        ) from e
