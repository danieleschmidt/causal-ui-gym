"""
Comprehensive error handling for causal inference systems.
"""

from .causal_exceptions import (
    ErrorSeverity,
    ErrorCategory,
    ErrorContext,
    CausalInferenceError,
    DataValidationError,
    GraphStructureError,
    ComputationError,
    ConvergenceError,
    SecurityViolationError,
    ResourceError,
    UserInputError,
    CausalErrorHandler,
    CausalErrorContext,
    handle_causal_errors,
    global_error_handler
)

__all__ = [
    "ErrorSeverity",
    "ErrorCategory", 
    "ErrorContext",
    "CausalInferenceError",
    "DataValidationError",
    "GraphStructureError",
    "ComputationError",
    "ConvergenceError",
    "SecurityViolationError",
    "ResourceError",
    "UserInputError",
    "CausalErrorHandler",
    "CausalErrorContext",
    "handle_causal_errors",
    "global_error_handler"
]