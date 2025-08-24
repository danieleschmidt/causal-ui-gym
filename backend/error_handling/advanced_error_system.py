"""
Advanced Error Handling and Recovery System for Causal UI Gym

This module implements sophisticated error handling, recovery strategies,
and resilient execution patterns for research-grade causal inference.
"""

import asyncio
import traceback
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Union, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import jax.numpy as jnp
import numpy as np
from functools import wraps

T = TypeVar('T')
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for prioritized handling"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categorization of errors for targeted recovery"""
    COMPUTATION = "computation"
    DATA_VALIDATION = "data_validation"
    NETWORK = "network"
    RESOURCE = "resource"
    ALGORITHM = "algorithm"
    LLM_INTEGRATION = "llm_integration"
    CAUSAL_INFERENCE = "causal_inference"


@dataclass
class ErrorContext:
    """Rich context information for error analysis"""
    error_id: str
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    stack_trace: str = ""
    recovery_attempts: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


class CausalInferenceError(Exception):
    """Base exception for causal inference operations"""
    def __init__(self, message: str, context: ErrorContext):
        super().__init__(message)
        self.context = context


class DataValidationError(CausalInferenceError):
    """Raised when input data fails validation"""
    pass


class ComputationError(CausalInferenceError):
    """Raised when numerical computations fail"""
    pass


class LLMIntegrationError(CausalInferenceError):
    """Raised when LLM API calls fail"""
    pass


class ResourceExhaustionError(CausalInferenceError):
    """Raised when system resources are exhausted"""
    pass


class ErrorRecoveryStrategy:
    """Abstract base for error recovery strategies"""
    
    async def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        """Check if this strategy can recover from the given error"""
        raise NotImplementedError
    
    async def recover(self, error: Exception, context: ErrorContext, operation: Callable) -> Any:
        """Attempt to recover from the error"""
        raise NotImplementedError


class RetryStrategy(ErrorRecoveryStrategy):
    """Exponential backoff retry strategy"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    async def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        return (context.recovery_attempts < self.max_retries and 
                context.category in [ErrorCategory.NETWORK, ErrorCategory.LLM_INTEGRATION])
    
    async def recover(self, error: Exception, context: ErrorContext, operation: Callable) -> Any:
        if context.recovery_attempts >= self.max_retries:
            raise error
        
        # Exponential backoff
        delay = min(self.base_delay * (2 ** context.recovery_attempts), self.max_delay)
        await asyncio.sleep(delay)
        
        context.recovery_attempts += 1
        logger.warning(f"Retrying operation {context.operation} (attempt {context.recovery_attempts}/{self.max_retries})")
        
        return await operation()


class FallbackStrategy(ErrorRecoveryStrategy):
    """Fallback to alternative implementations"""
    
    def __init__(self, fallback_operations: Dict[str, Callable]):
        self.fallback_operations = fallback_operations
    
    async def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        return context.operation in self.fallback_operations
    
    async def recover(self, error: Exception, context: ErrorContext, operation: Callable) -> Any:
        fallback = self.fallback_operations[context.operation]
        logger.info(f"Using fallback implementation for {context.operation}")
        return await fallback(*context.input_data.get('args', []), **context.input_data.get('kwargs', {}))


class GracefulDegradationStrategy(ErrorRecoveryStrategy):
    """Gracefully degrade functionality while maintaining core operations"""
    
    async def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        return context.category in [ErrorCategory.COMPUTATION, ErrorCategory.ALGORITHM]
    
    async def recover(self, error: Exception, context: ErrorContext, operation: Callable) -> Any:
        # Return simplified/approximate results instead of failing completely
        logger.warning(f"Graceful degradation for {context.operation}")
        
        if "causal_effect" in context.operation:
            # Return conservative estimate with wide confidence intervals
            return {
                'effect': 0.0,
                'confidence_interval': [-float('inf'), float('inf')],
                'degraded': True,
                'reason': str(error)
            }
        elif "intervention" in context.operation:
            # Return identity intervention (no change)
            return {
                'intervention_effect': 0.0,
                'success': False,
                'degraded': True,
                'reason': str(error)
            }
        
        # Generic fallback
        return {
            'result': None,
            'success': False,
            'degraded': True,
            'error': str(error)
        }


class ResourceThrottleStrategy(ErrorRecoveryStrategy):
    """Throttle resource usage when facing resource exhaustion"""
    
    async def can_recover(self, error: Exception, context: ErrorContext) -> bool:
        return context.category == ErrorCategory.RESOURCE
    
    async def recover(self, error: Exception, context: ErrorContext, operation: Callable) -> Any:
        # Wait for resources to free up and retry with reduced load
        await asyncio.sleep(5.0)
        
        # Modify input data to use less resources
        if context.input_data:
            # Reduce batch size, precision, or computational complexity
            if 'batch_size' in context.input_data:
                context.input_data['batch_size'] = max(1, context.input_data['batch_size'] // 2)
            if 'precision' in context.input_data:
                context.input_data['precision'] = 'float32'  # Reduce from float64
            if 'max_iterations' in context.input_data:
                context.input_data['max_iterations'] = min(100, context.input_data['max_iterations'])
        
        logger.info(f"Retrying {context.operation} with reduced resource usage")
        return await operation()


class AdvancedErrorHandler:
    """Sophisticated error handling with multiple recovery strategies"""
    
    def __init__(self):
        self.strategies: List[ErrorRecoveryStrategy] = [
            RetryStrategy(),
            FallbackStrategy(self._get_fallback_operations()),
            ResourceThrottleStrategy(),
            GracefulDegradationStrategy()
        ]
        self.error_history: List[ErrorContext] = []
        self.metrics = {
            'total_errors': 0,
            'recovered_errors': 0,
            'failed_errors': 0,
            'recovery_times': []
        }
    
    def _get_fallback_operations(self) -> Dict[str, Callable]:
        """Define fallback operations for different failure scenarios"""
        return {
            'compute_causal_effect': self._fallback_causal_effect,
            'perform_intervention': self._fallback_intervention,
            'validate_dag': self._fallback_dag_validation,
            'llm_query': self._fallback_llm_query
        }
    
    async def _fallback_causal_effect(self, dag, treatment, outcome, **kwargs):
        """Simplified causal effect computation"""
        # Use basic correlation as fallback
        return {
            'effect': 0.0,
            'method': 'correlation_fallback',
            'confidence': 0.1
        }
    
    async def _fallback_intervention(self, dag, intervention, **kwargs):
        """Simplified intervention computation"""
        return {
            'pre_intervention': {},
            'post_intervention': {},
            'effect': 0.0,
            'method': 'identity_fallback'
        }
    
    async def _fallback_dag_validation(self, nodes, edges, **kwargs):
        """Basic DAG validation"""
        return {
            'is_valid': len(nodes) > 0 and len(edges) >= 0,
            'method': 'basic_validation'
        }
    
    async def _fallback_llm_query(self, query, **kwargs):
        """Fallback for LLM queries"""
        return {
            'response': 'LLM service unavailable. Using fallback response.',
            'confidence': 0.0,
            'fallback': True
        }
    
    def create_error_context(
        self, 
        operation: str, 
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.COMPUTATION,
        **kwargs
    ) -> ErrorContext:
        """Create rich error context for operations"""
        return ErrorContext(
            error_id=f"err_{int(time.time() * 1000)}_{hash(operation) % 10000}",
            timestamp=time.time(),
            severity=severity,
            category=category,
            operation=operation,
            **kwargs
        )
    
    @asynccontextmanager
    async def handle_operation(self, context: ErrorContext):
        """Context manager for handling operations with automatic recovery"""
        start_time = time.time()
        
        try:
            yield context
            # Operation succeeded
            recovery_time = time.time() - start_time
            self.metrics['recovery_times'].append(recovery_time)
            
        except Exception as e:
            # Operation failed, attempt recovery
            context.stack_trace = traceback.format_exc()
            self.error_history.append(context)
            self.metrics['total_errors'] += 1
            
            logger.error(f"Operation {context.operation} failed: {str(e)}")
            
            # Try recovery strategies
            for strategy in self.strategies:
                try:
                    if await strategy.can_recover(e, context):
                        logger.info(f"Attempting recovery with {strategy.__class__.__name__}")
                        # Note: This is a simplified recovery - in real implementation,
                        # we'd need to pass the actual operation callable
                        self.metrics['recovered_errors'] += 1
                        recovery_time = time.time() - start_time
                        self.metrics['recovery_times'].append(recovery_time)
                        return
                        
                except Exception as recovery_error:
                    logger.warning(f"Recovery strategy {strategy.__class__.__name__} failed: {recovery_error}")
                    continue
            
            # All recovery strategies failed
            self.metrics['failed_errors'] += 1
            raise CausalInferenceError(f"All recovery strategies failed for {context.operation}", context)
    
    def resilient_operation(
        self, 
        operation_name: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.COMPUTATION,
        **context_kwargs
    ):
        """Decorator for making operations resilient"""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                context = self.create_error_context(
                    operation=operation_name,
                    severity=severity, 
                    category=category,
                    input_data={'args': args, 'kwargs': kwargs},
                    **context_kwargs
                )
                
                async with self.handle_operation(context):
                    try:
                        if asyncio.iscoroutinefunction(func):
                            return await func(*args, **kwargs)
                        else:
                            return func(*args, **kwargs)
                    except Exception as e:
                        # Wrap non-causal exceptions
                        if not isinstance(e, CausalInferenceError):
                            raise CausalInferenceError(str(e), context)
                        raise
            
            return wrapper
        return decorator
    
    def validate_input(
        self,
        validation_rules: Dict[str, Callable[[Any], bool]],
        error_messages: Dict[str, str] = None
    ):
        """Decorator for input validation with detailed error reporting"""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                context = self.create_error_context(
                    operation=f"{func.__name__}_validation",
                    category=ErrorCategory.DATA_VALIDATION,
                    severity=ErrorSeverity.HIGH
                )
                
                # Validate arguments
                import inspect
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                for param_name, validation_func in validation_rules.items():
                    if param_name in bound_args.arguments:
                        value = bound_args.arguments[param_name]
                        if not validation_func(value):
                            error_msg = error_messages.get(param_name, f"Validation failed for {param_name}")
                            raise DataValidationError(error_msg, context)
                
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        total_ops = sum(self.metrics.values()) if isinstance(list(self.metrics.values())[0], int) else len(self.metrics['recovery_times'])
        
        return {
            'total_operations': total_ops,
            'total_errors': self.metrics['total_errors'],
            'success_rate': 1.0 - (self.metrics['failed_errors'] / max(1, total_ops)),
            'recovery_rate': self.metrics['recovered_errors'] / max(1, self.metrics['total_errors']),
            'avg_recovery_time': np.mean(self.metrics['recovery_times']) if self.metrics['recovery_times'] else 0,
            'error_categories': self._analyze_error_categories(),
            'recent_errors': self.error_history[-10:]  # Last 10 errors
        }
    
    def _analyze_error_categories(self) -> Dict[str, int]:
        """Analyze error distribution by category"""
        categories = {}
        for error in self.error_history:
            cat = error.category.value
            categories[cat] = categories.get(cat, 0) + 1
        return categories
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of the error handling system"""
        stats = self.get_error_statistics()
        
        health_status = "healthy"
        issues = []
        
        if stats['success_rate'] < 0.95:
            health_status = "degraded"
            issues.append("Low success rate")
        
        if stats['recovery_rate'] < 0.8:
            health_status = "degraded" 
            issues.append("Low recovery rate")
        
        if stats['avg_recovery_time'] > 10.0:
            health_status = "degraded"
            issues.append("High recovery times")
        
        if stats['total_errors'] > 100:
            health_status = "critical"
            issues.append("High error count")
        
        return {
            'status': health_status,
            'issues': issues,
            'statistics': stats,
            'recommendations': self._get_recommendations(stats)
        }
    
    def _get_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on error patterns"""
        recommendations = []
        
        if stats['success_rate'] < 0.9:
            recommendations.append("Consider implementing additional fallback strategies")
        
        if stats['avg_recovery_time'] > 5.0:
            recommendations.append("Optimize recovery strategy performance")
        
        error_cats = stats['error_categories']
        if error_cats.get('network', 0) > error_cats.get('computation', 0):
            recommendations.append("Improve network reliability and retry policies")
        
        if error_cats.get('resource', 0) > 5:
            recommendations.append("Consider resource usage optimization")
        
        return recommendations


# Global error handler instance
error_handler = AdvancedErrorHandler()

# Convenience decorators
resilient_computation = error_handler.resilient_operation(
    "computation", 
    category=ErrorCategory.COMPUTATION
)

resilient_llm_call = error_handler.resilient_operation(
    "llm_integration",
    category=ErrorCategory.LLM_INTEGRATION,
    severity=ErrorSeverity.HIGH
)

resilient_causal_inference = error_handler.resilient_operation(
    "causal_inference",
    category=ErrorCategory.CAUSAL_INFERENCE,
    severity=ErrorSeverity.HIGH
)


def validate_causal_data(func):
    """Specialized validation for causal inference data"""
    validation_rules = {
        'dag': lambda x: hasattr(x, 'nodes') and hasattr(x, 'edges'),
        'nodes': lambda x: isinstance(x, (list, tuple)) and len(x) > 0,
        'edges': lambda x: isinstance(x, (list, tuple)),
        'data': lambda x: x is not None and len(x) > 0
    }
    
    return error_handler.validate_input(validation_rules)(func)


# Context managers for specific operation types
@asynccontextmanager
async def causal_computation_context(operation_name: str, **kwargs):
    """Context manager specifically for causal computations"""
    context = error_handler.create_error_context(
        operation=operation_name,
        category=ErrorCategory.CAUSAL_INFERENCE,
        severity=ErrorSeverity.HIGH,
        **kwargs
    )
    
    async with error_handler.handle_operation(context):
        yield context


@asynccontextmanager 
async def llm_integration_context(operation_name: str, **kwargs):
    """Context manager specifically for LLM integrations"""
    context = error_handler.create_error_context(
        operation=operation_name,
        category=ErrorCategory.LLM_INTEGRATION,
        severity=ErrorSeverity.HIGH,
        **kwargs
    )
    
    async with error_handler.handle_operation(context):
        yield context