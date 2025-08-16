"""
Comprehensive error handling for causal inference systems.

This module defines custom exceptions and error handling patterns
specifically for causal inference operations.
"""

import logging
import traceback
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors in causal inference."""
    DATA_VALIDATION = "data_validation"
    GRAPH_STRUCTURE = "graph_structure"
    COMPUTATION = "computation"
    CONVERGENCE = "convergence"
    SECURITY = "security"
    RESOURCE = "resource"
    USER_INPUT = "user_input"
    SYSTEM = "system"


@dataclass
class ErrorContext:
    """Context information for errors."""
    operation: str
    user_id: Optional[str]
    parameters: Dict[str, Any]
    timestamp: datetime
    stack_trace: str
    additional_info: Dict[str, Any]


class CausalInferenceError(Exception):
    """Base exception for causal inference operations."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_code: str = "CAUSAL_ERROR",
        context: Optional[ErrorContext] = None,
        recovery_suggestions: Optional[List[str]] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.error_code = error_code
        self.context = context
        self.recovery_suggestions = recovery_suggestions or []
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "error_code": self.error_code,
            "timestamp": self.timestamp.isoformat(),
            "recovery_suggestions": self.recovery_suggestions,
            "context": {
                "operation": self.context.operation if self.context else None,
                "user_id": self.context.user_id if self.context else None,
                "parameters": self.context.parameters if self.context else {},
                "additional_info": self.context.additional_info if self.context else {}
            }
        }


class DataValidationError(CausalInferenceError):
    """Error in data validation."""
    
    def __init__(
        self,
        message: str,
        invalid_fields: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            message,
            category=ErrorCategory.DATA_VALIDATION,
            error_code="DATA_INVALID",
            **kwargs
        )
        self.invalid_fields = invalid_fields or []


class GraphStructureError(CausalInferenceError):
    """Error in causal graph structure."""
    
    def __init__(
        self,
        message: str,
        graph_issues: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(
            message,
            category=ErrorCategory.GRAPH_STRUCTURE,
            error_code="GRAPH_INVALID",
            **kwargs
        )
        self.graph_issues = graph_issues or []


class ComputationError(CausalInferenceError):
    """Error during causal computation."""
    
    def __init__(
        self,
        message: str,
        computation_step: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            category=ErrorCategory.COMPUTATION,
            error_code="COMPUTATION_FAILED",
            **kwargs
        )
        self.computation_step = computation_step


class ConvergenceError(CausalInferenceError):
    """Error due to convergence failure."""
    
    def __init__(
        self,
        message: str,
        iterations: Optional[int] = None,
        tolerance_achieved: Optional[float] = None,
        target_tolerance: Optional[float] = None,
        **kwargs
    ):
        super().__init__(
            message,
            category=ErrorCategory.CONVERGENCE,
            error_code="CONVERGENCE_FAILED",
            **kwargs
        )
        self.iterations = iterations
        self.tolerance_achieved = tolerance_achieved
        self.target_tolerance = target_tolerance


class SecurityViolationError(CausalInferenceError):
    """Error due to security violation."""
    
    def __init__(
        self,
        message: str,
        violation_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.HIGH,
            error_code="SECURITY_VIOLATION",
            **kwargs
        )
        self.violation_type = violation_type


class ResourceError(CausalInferenceError):
    """Error due to resource constraints."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        requested: Optional[float] = None,
        available: Optional[float] = None,
        **kwargs
    ):
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE,
            error_code="RESOURCE_EXCEEDED",
            **kwargs
        )
        self.resource_type = resource_type
        self.requested = requested
        self.available = available


class UserInputError(CausalInferenceError):
    """Error in user input."""
    
    def __init__(
        self,
        message: str,
        input_field: Optional[str] = None,
        expected_format: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            category=ErrorCategory.USER_INPUT,
            severity=ErrorSeverity.LOW,
            error_code="INVALID_INPUT",
            **kwargs
        )
        self.input_field = input_field
        self.expected_format = expected_format


class CausalErrorHandler:
    """Centralized error handling for causal inference operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_history: List[CausalInferenceError] = []
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[str, callable] = {}
        
        # Register default recovery strategies
        self._register_default_recovery_strategies()
    
    def handle_error(
        self,
        error: Exception,
        operation: str,
        user_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        auto_recover: bool = True
    ) -> Optional[Any]:
        """
        Handle an error with appropriate logging and recovery.
        
        Args:
            error: The error that occurred
            operation: Name of the operation that failed
            user_id: User who triggered the operation
            parameters: Parameters passed to the operation
            auto_recover: Whether to attempt automatic recovery
            
        Returns:
            Recovery result if successful, None otherwise
        """
        # Create error context
        context = ErrorContext(
            operation=operation,
            user_id=user_id,
            parameters=parameters or {},
            timestamp=datetime.now(),
            stack_trace=traceback.format_exc(),
            additional_info={}
        )
        
        # Convert to CausalInferenceError if needed
        if not isinstance(error, CausalInferenceError):
            causal_error = self._convert_to_causal_error(error, context)
        else:
            causal_error = error
            causal_error.context = context
        
        # Log the error
        self._log_error(causal_error)
        
        # Update error statistics
        self._update_error_stats(causal_error)
        
        # Store in history
        self.error_history.append(causal_error)
        
        # Attempt recovery if enabled
        if auto_recover:
            return self._attempt_recovery(causal_error)
        
        return None
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary of errors in the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Error summary statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [
            error for error in self.error_history
            if error.timestamp > cutoff_time
        ]
        
        if not recent_errors:
            return {
                "total_errors": 0,
                "error_rate": 0.0,
                "most_common_errors": {},
                "severity_breakdown": {},
                "category_breakdown": {}
            }
        
        # Calculate statistics
        severity_counts = {}
        category_counts = {}
        error_type_counts = {}
        
        for error in recent_errors:
            # Count by severity
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Count by category
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Count by error type
            error_type = error.__class__.__name__
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        
        return {
            "total_errors": len(recent_errors),
            "error_rate": len(recent_errors) / hours,
            "most_common_errors": dict(sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            "severity_breakdown": severity_counts,
            "category_breakdown": category_counts,
            "critical_errors": sum(1 for e in recent_errors if e.severity == ErrorSeverity.CRITICAL),
            "high_severity_errors": sum(1 for e in recent_errors if e.severity == ErrorSeverity.HIGH)
        }
    
    def register_recovery_strategy(self, error_type: str, strategy: callable) -> None:
        """
        Register a recovery strategy for a specific error type.
        
        Args:
            error_type: Name of the error class
            strategy: Function to call for recovery
        """
        self.recovery_strategies[error_type] = strategy
    
    def _convert_to_causal_error(self, error: Exception, context: ErrorContext) -> CausalInferenceError:
        """Convert generic exception to CausalInferenceError."""
        error_message = str(error)
        
        # Categorize based on error type and message
        if isinstance(error, ValueError):
            if "nan" in error_message.lower() or "inf" in error_message.lower():
                return DataValidationError(
                    f"Invalid numeric values: {error_message}",
                    context=context,
                    recovery_suggestions=["Check for NaN or infinite values in data", "Apply data cleaning"]
                )
            else:
                return UserInputError(
                    f"Invalid input: {error_message}",
                    context=context,
                    recovery_suggestions=["Validate input parameters", "Check data types"]
                )
        
        elif isinstance(error, KeyError):
            return DataValidationError(
                f"Missing required data: {error_message}",
                context=context,
                recovery_suggestions=["Check variable names", "Verify data completeness"]
            )
        
        elif isinstance(error, MemoryError):
            return ResourceError(
                f"Insufficient memory: {error_message}",
                resource_type="memory",
                context=context,
                recovery_suggestions=["Reduce sample size", "Use data streaming", "Increase memory limits"]
            )
        
        elif isinstance(error, TimeoutError):
            return ResourceError(
                f"Operation timed out: {error_message}",
                resource_type="time",
                context=context,
                recovery_suggestions=["Increase timeout limit", "Reduce problem size", "Use faster algorithms"]
            )
        
        else:
            return CausalInferenceError(
                f"Unexpected error: {error_message}",
                context=context,
                recovery_suggestions=["Check system logs", "Contact support if persistent"]
            )
    
    def _log_error(self, error: CausalInferenceError) -> None:
        """Log error with appropriate level."""
        error_dict = error.to_dict()
        
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR: {error.message}", extra=error_dict)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(f"HIGH SEVERITY ERROR: {error.message}", extra=error_dict)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"MEDIUM SEVERITY ERROR: {error.message}", extra=error_dict)
        else:
            self.logger.info(f"LOW SEVERITY ERROR: {error.message}", extra=error_dict)
    
    def _update_error_stats(self, error: CausalInferenceError) -> None:
        """Update error statistics."""
        error_type = error.__class__.__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def _attempt_recovery(self, error: CausalInferenceError) -> Optional[Any]:
        """Attempt to recover from error using registered strategies."""
        error_type = error.__class__.__name__
        
        if error_type in self.recovery_strategies:
            try:
                recovery_result = self.recovery_strategies[error_type](error)
                self.logger.info(f"Successfully recovered from {error_type}")
                return recovery_result
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed for {error_type}: {recovery_error}")
        
        return None
    
    def _register_default_recovery_strategies(self) -> None:
        """Register default recovery strategies."""
        
        def recover_data_validation(error: DataValidationError) -> Optional[Any]:
            """Recovery strategy for data validation errors."""
            if "nan" in error.message.lower():
                # Could implement NaN handling here
                return {"strategy": "remove_nan", "message": "Consider removing NaN values"}
            return None
        
        def recover_convergence_error(error: ConvergenceError) -> Optional[Any]:
            """Recovery strategy for convergence errors."""
            return {
                "strategy": "adjust_parameters",
                "suggestions": [
                    "Increase maximum iterations",
                    "Relax convergence tolerance",
                    "Try different initialization"
                ]
            }
        
        def recover_resource_error(error: ResourceError) -> Optional[Any]:
            """Recovery strategy for resource errors."""
            if error.resource_type == "memory":
                return {
                    "strategy": "reduce_problem_size",
                    "suggestions": [
                        "Use smaller sample size",
                        "Process data in batches",
                        "Use more memory-efficient algorithms"
                    ]
                }
            elif error.resource_type == "time":
                return {
                    "strategy": "optimize_computation",
                    "suggestions": [
                        "Use approximation methods",
                        "Parallelize computation",
                        "Increase timeout limits"
                    ]
                }
            return None
        
        # Register strategies
        self.register_recovery_strategy("DataValidationError", recover_data_validation)
        self.register_recovery_strategy("ConvergenceError", recover_convergence_error)
        self.register_recovery_strategy("ResourceError", recover_resource_error)


# Context manager for error handling
class CausalErrorContext:
    """Context manager for causal inference operations with error handling."""
    
    def __init__(
        self,
        operation: str,
        error_handler: CausalErrorHandler,
        user_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        auto_recover: bool = True
    ):
        self.operation = operation
        self.error_handler = error_handler
        self.user_id = user_id
        self.parameters = parameters
        self.auto_recover = auto_recover
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            recovery_result = self.error_handler.handle_error(
                error=exc_val,
                operation=self.operation,
                user_id=self.user_id,
                parameters=self.parameters,
                auto_recover=self.auto_recover
            )
            
            # If recovery was successful, suppress the exception
            if recovery_result is not None:
                return True
        
        return False


# Decorator for automatic error handling
def handle_causal_errors(
    operation: str,
    error_handler: CausalErrorHandler,
    auto_recover: bool = True
):
    """
    Decorator for automatic error handling in causal inference functions.
    
    Args:
        operation: Name of the operation
        error_handler: Error handler instance
        auto_recover: Whether to attempt automatic recovery
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with CausalErrorContext(
                operation=operation,
                error_handler=error_handler,
                auto_recover=auto_recover
            ):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Global error handler instance
global_error_handler = CausalErrorHandler()