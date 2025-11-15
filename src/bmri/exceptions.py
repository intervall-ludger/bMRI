"""Custom exceptions for bMRI.

This module defines a hierarchy of exceptions for structured error handling
throughout the bMRI application.
"""


class BMRIError(Exception):
    """Base exception for all bMRI errors.

    All custom exceptions in bMRI should inherit from this class.
    """

    def __init__(self, message: str, details: str | None = None) -> None:
        """Initialize BMRIError.

        Args:
            message: Main error message
            details: Optional detailed information about the error
        """
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return formatted error message."""
        if self.details:
            return f"{self.message}\n\nDetails: {self.details}"
        return self.message


class ValidationError(BMRIError):
    """Raised when input validation fails.

    This exception is raised when user-provided inputs (files, parameters,
    configuration) do not meet the required constraints.
    """


class DICOMError(BMRIError):
    """Raised when DICOM file operations fail.

    This exception covers errors during DICOM reading, parsing, or validation.
    """


class DICOMReadError(DICOMError):
    """Raised when a DICOM file cannot be read."""


class DICOMStructureError(DICOMError):
    """Raised when DICOM folder structure is invalid.

    This exception is raised when the expected DICOM file organization
    (e.g., echo folders, slice arrangement) is not found.
    """


class MaskError(BMRIError):
    """Raised when mask file operations fail."""


class FittingError(BMRIError):
    """Raised when curve fitting fails.

    This exception is raised when the optimization algorithm cannot
    converge or produces invalid results.
    """


class ConfigurationError(BMRIError):
    """Raised when configuration is invalid.

    This exception is raised when configuration files are malformed or
    contain invalid parameter values.
    """


class VisualizationError(BMRIError):
    """Raised when visualization operations fail."""


class IOError(BMRIError):
    """Raised when file I/O operations fail.

    This exception covers reading/writing NIfTI files, CSV files, etc.
    """
