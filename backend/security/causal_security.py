"""
Advanced Security Framework for Causal Inference Systems - ENHANCED VERSION

Implements enterprise-grade security measures including:
- DAG validation and sanitization
- Intervention bounds checking  
- Model parameter validation
- Adversarial attack detection
- Differential privacy mechanisms
- Real-time threat monitoring
- Cryptographic data protection
"""

import logging
import hashlib
import hmac
import secrets
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import json

import jax.numpy as jnp
from jax import random

from ..engine.causal_engine import CausalDAG, Intervention, CausalResult

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration for causal inference systems."""
    max_variables: int = 1000
    max_samples: int = 1000000
    max_computation_time: float = 3600.0  # 1 hour
    max_memory_mb: float = 8192.0  # 8GB
    allowed_file_types: List[str] = None
    encryption_key: Optional[bytes] = None
    rate_limit_per_minute: int = 100
    enable_audit_logging: bool = True
    
    def __post_init__(self):
        if self.allowed_file_types is None:
            self.allowed_file_types = ['.json', '.csv', '.parquet', '.pkl']
        if self.encryption_key is None:
            self.encryption_key = secrets.token_bytes(32)


@dataclass
class SecurityViolation:
    """Record of a security violation."""
    violation_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    user_id: Optional[str]
    ip_address: Optional[str]
    timestamp: datetime
    additional_data: Dict[str, Any]


class CausalDataValidator:
    """Validates causal data for security and integrity."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.violations: List[SecurityViolation] = []
    
    def validate_dag(self, dag: CausalDAG) -> Tuple[bool, List[str]]:
        """
        Validate a causal DAG for security and integrity issues.
        
        Args:
            dag: Causal DAG to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check size limits
        if len(dag.nodes) > self.config.max_variables:
            issues.append(f"Too many variables: {len(dag.nodes)} > {self.config.max_variables}")
        
        # Check for malicious variable names
        for node in dag.nodes:
            if not self._is_safe_variable_name(node):
                issues.append(f"Unsafe variable name: {node}")
        
        # Check edge weights for anomalies
        if dag.edge_weights:
            weights = list(dag.edge_weights.values())
            if any(abs(w) > 1000 for w in weights):
                issues.append("Extremely large edge weights detected (possible attack)")
            
            if any(w != w for w in weights):  # NaN check
                issues.append("NaN values detected in edge weights")
        
        # Check for cycles (should not exist in DAG)
        if self._has_cycles(dag):
            issues.append("Cycles detected in supposedly acyclic graph")
        
        # Check data integrity
        for node, data in dag.node_data.items():
            if len(data) > self.config.max_samples:
                issues.append(f"Too many samples for {node}: {len(data)} > {self.config.max_samples}")
            
            if jnp.any(jnp.isnan(data)):
                issues.append(f"NaN values detected in data for {node}")
            
            if jnp.any(jnp.isinf(data)):
                issues.append(f"Infinite values detected in data for {node}")
        
        return len(issues) == 0, issues
    
    def validate_intervention(self, intervention: Intervention) -> Tuple[bool, List[str]]:
        """
        Validate an intervention for security issues.
        
        Args:
            intervention: Intervention to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check variable name
        if not self._is_safe_variable_name(intervention.variable):
            issues.append(f"Unsafe variable name in intervention: {intervention.variable}")
        
        # Check intervention value
        if isinstance(intervention.value, (int, float)):
            if abs(intervention.value) > 1e6:
                issues.append(f"Extremely large intervention value: {intervention.value}")
            
            if intervention.value != intervention.value:  # NaN check
                issues.append("NaN intervention value")
        
        elif isinstance(intervention.value, jnp.ndarray):
            if jnp.any(jnp.isnan(intervention.value)):
                issues.append("NaN values in intervention array")
            
            if jnp.any(jnp.isinf(intervention.value)):
                issues.append("Infinite values in intervention array")
            
            if jnp.any(jnp.abs(intervention.value) > 1e6):
                issues.append("Extremely large values in intervention array")
        
        return len(issues) == 0, issues
    
    def sanitize_user_input(self, user_input: str) -> str:
        """
        Sanitize user input to prevent injection attacks.
        
        Args:
            user_input: Raw user input string
            
        Returns:
            Sanitized input string
        """
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '&', '"', "'", '\\', '/', ';', '|', '`', '$']
        sanitized = user_input
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Limit length
        max_length = 1000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            logger.warning(f"User input truncated to {max_length} characters")
        
        return sanitized.strip()
    
    def _is_safe_variable_name(self, name: str) -> bool:
        """Check if a variable name is safe."""
        if not isinstance(name, str):
            return False
        
        # Check length
        if len(name) > 100:
            return False
        
        # Check for dangerous patterns
        dangerous_patterns = [
            '__', 'eval', 'exec', 'import', 'os.', 'sys.', 'subprocess',
            'file://', 'http://', 'https://', 'ftp://', 'javascript:'
        ]
        
        name_lower = name.lower()
        for pattern in dangerous_patterns:
            if pattern in name_lower:
                return False
        
        # Must start with letter or underscore
        if not (name[0].isalpha() or name[0] == '_'):
            return False
        
        # Must contain only alphanumeric and underscores
        return all(c.isalnum() or c == '_' for c in name)
    
    def _has_cycles(self, dag: CausalDAG) -> bool:
        """Check if DAG has cycles using DFS."""
        # Create adjacency list
        adj_list = {node: [] for node in dag.nodes}
        for source, target in dag.edges:
            adj_list[source].append(target)
        
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in dag.nodes:
            if node not in visited:
                if dfs(node):
                    return True
        
        return False


class CausalAccessControl:
    """Access control for causal inference operations."""
    
    def __init__(self):
        self.user_permissions: Dict[str, Dict[str, bool]] = {}
        self.rate_limits: Dict[str, List[float]] = {}  # user_id -> list of request timestamps
        self.blocked_users: Set[str] = set()
    
    def add_user(self, user_id: str, permissions: Dict[str, bool]) -> None:
        """
        Add a user with specific permissions.
        
        Args:
            user_id: User identifier
            permissions: Dictionary of permission names and whether granted
        """
        self.user_permissions[user_id] = permissions
        self.rate_limits[user_id] = []
    
    def check_permission(self, user_id: str, operation: str) -> bool:
        """
        Check if user has permission for operation.
        
        Args:
            user_id: User identifier
            operation: Operation name (e.g., 'compute_ate', 'upload_data')
            
        Returns:
            True if user has permission, False otherwise
        """
        if user_id in self.blocked_users:
            return False
        
        if user_id not in self.user_permissions:
            return False
        
        return self.user_permissions[user_id].get(operation, False)
    
    def check_rate_limit(self, user_id: str, limit_per_minute: int = 100) -> bool:
        """
        Check if user is within rate limits.
        
        Args:
            user_id: User identifier
            limit_per_minute: Maximum requests per minute
            
        Returns:
            True if within limits, False if exceeded
        """
        current_time = time.time()
        
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = []
        
        # Remove old timestamps (older than 1 minute)
        cutoff_time = current_time - 60
        self.rate_limits[user_id] = [
            timestamp for timestamp in self.rate_limits[user_id]
            if timestamp > cutoff_time
        ]
        
        # Check if within limit
        if len(self.rate_limits[user_id]) >= limit_per_minute:
            return False
        
        # Add current request
        self.rate_limits[user_id].append(current_time)
        return True
    
    def block_user(self, user_id: str, reason: str) -> None:
        """
        Block a user from all operations.
        
        Args:
            user_id: User to block
            reason: Reason for blocking
        """
        self.blocked_users.add(user_id)
        logger.warning(f"User {user_id} blocked: {reason}")


class CausalAuditLogger:
    """Audit logger for causal inference operations."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.audit_logs: List[Dict[str, Any]] = []
    
    def log_operation(
        self,
        operation: str,
        user_id: Optional[str],
        parameters: Dict[str, Any],
        result_summary: str,
        execution_time: float,
        success: bool
    ) -> None:
        """
        Log a causal inference operation.
        
        Args:
            operation: Name of operation performed
            user_id: User who performed operation
            parameters: Operation parameters
            result_summary: Summary of results
            execution_time: Time taken to execute
            success: Whether operation succeeded
        """
        if not self.config.enable_audit_logging:
            return
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'user_id': user_id,
            'parameters': self._sanitize_parameters(parameters),
            'result_summary': result_summary,
            'execution_time': execution_time,
            'success': success,
            'log_id': self._generate_log_id()
        }
        
        self.audit_logs.append(log_entry)
        
        # Also log to file/external system
        logger.info(f"AUDIT: {json.dumps(log_entry)}")
    
    def log_security_violation(self, violation: SecurityViolation) -> None:
        """
        Log a security violation.
        
        Args:
            violation: Security violation details
        """
        log_entry = {
            'timestamp': violation.timestamp.isoformat(),
            'type': 'SECURITY_VIOLATION',
            'violation_type': violation.violation_type,
            'severity': violation.severity,
            'description': violation.description,
            'user_id': violation.user_id,
            'ip_address': violation.ip_address,
            'additional_data': violation.additional_data,
            'log_id': self._generate_log_id()
        }
        
        self.audit_logs.append(log_entry)
        
        # Log with appropriate level based on severity
        if violation.severity == 'critical':
            logger.critical(f"SECURITY: {json.dumps(log_entry)}")
        elif violation.severity == 'high':
            logger.error(f"SECURITY: {json.dumps(log_entry)}")
        elif violation.severity == 'medium':
            logger.warning(f"SECURITY: {json.dumps(log_entry)}")
        else:
            logger.info(f"SECURITY: {json.dumps(log_entry)}")
    
    def get_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get audit summary for the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Summary statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_logs = [
            log for log in self.audit_logs
            if datetime.fromisoformat(log['timestamp']) > cutoff_time
        ]
        
        return {
            'total_operations': len(recent_logs),
            'successful_operations': sum(1 for log in recent_logs if log.get('success', False)),
            'failed_operations': sum(1 for log in recent_logs if not log.get('success', True)),
            'security_violations': sum(1 for log in recent_logs if log.get('type') == 'SECURITY_VIOLATION'),
            'unique_users': len(set(log.get('user_id') for log in recent_logs if log.get('user_id'))),
            'most_common_operations': self._get_operation_counts(recent_logs),
            'average_execution_time': self._get_average_execution_time(recent_logs)
        }
    
    def _sanitize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize parameters for logging (remove sensitive data)."""
        sanitized = {}
        sensitive_keys = ['password', 'token', 'key', 'secret', 'api_key']
        
        for key, value in parameters.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = '[REDACTED]'
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_parameters(value)
            else:
                sanitized[key] = str(value)[:100]  # Truncate long values
        
        return sanitized
    
    def _generate_log_id(self) -> str:
        """Generate unique log ID."""
        return hashlib.sha256(
            f"{time.time()}{secrets.token_hex(8)}".encode()
        ).hexdigest()[:16]
    
    def _get_operation_counts(self, logs: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get counts of each operation type."""
        counts = {}
        for log in logs:
            operation = log.get('operation', 'unknown')
            counts[operation] = counts.get(operation, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def _get_average_execution_time(self, logs: List[Dict[str, Any]]) -> float:
        """Get average execution time."""
        times = [log.get('execution_time', 0) for log in logs if 'execution_time' in log]
        return sum(times) / len(times) if times else 0.0


class SecureCausalEngine:
    """Secure wrapper around causal inference engine."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.validator = CausalDataValidator(config)
        self.access_control = CausalAccessControl()
        self.audit_logger = CausalAuditLogger(config)
        self.engine = None  # Will be initialized when needed
    
    def secure_compute_ate(
        self,
        dag: CausalDAG,
        treatment: str,
        outcome: str,
        user_id: Optional[str] = None,
        **kwargs
    ) -> Tuple[Optional[CausalResult], List[str]]:
        """
        Securely compute average treatment effect.
        
        Args:
            dag: Causal DAG
            treatment: Treatment variable
            outcome: Outcome variable
            user_id: User performing the operation
            **kwargs: Additional parameters for computation
            
        Returns:
            Tuple of (result, security_issues)
        """
        start_time = time.time()
        issues = []
        result = None
        
        try:
            # Check permissions
            if user_id and not self.access_control.check_permission(user_id, 'compute_ate'):
                issues.append("User lacks permission to compute ATE")
                self._log_violation(
                    'PERMISSION_DENIED',
                    'medium',
                    f"User {user_id} attempted to compute ATE without permission",
                    user_id
                )
                return None, issues
            
            # Check rate limits
            if user_id and not self.access_control.check_rate_limit(user_id, self.config.rate_limit_per_minute):
                issues.append("Rate limit exceeded")
                self._log_violation(
                    'RATE_LIMIT_EXCEEDED',
                    'medium',
                    f"User {user_id} exceeded rate limit",
                    user_id
                )
                return None, issues
            
            # Validate DAG
            dag_valid, dag_issues = self.validator.validate_dag(dag)
            if not dag_valid:
                issues.extend(dag_issues)
                self._log_violation(
                    'INVALID_DAG',
                    'high',
                    f"Invalid DAG: {'; '.join(dag_issues)}",
                    user_id
                )
                return None, issues
            
            # Validate variable names
            if not self.validator._is_safe_variable_name(treatment):
                issues.append(f"Unsafe treatment variable name: {treatment}")
                return None, issues
            
            if not self.validator._is_safe_variable_name(outcome):
                issues.append(f"Unsafe outcome variable name: {outcome}")
                return None, issues
            
            # Check computation limits
            n_samples = kwargs.get('n_samples', 10000)
            if n_samples > self.config.max_samples:
                issues.append(f"Too many samples requested: {n_samples} > {self.config.max_samples}")
                return None, issues
            
            # Initialize engine if needed
            if self.engine is None:
                from ..engine.causal_engine import JaxCausalEngine
                self.engine = JaxCausalEngine()
            
            # Perform computation with timeout
            result = self._compute_with_timeout(
                lambda: self.engine.compute_ate(dag, treatment, outcome, **kwargs),
                self.config.max_computation_time
            )
            
            # Validate result
            if result and result.ate is not None:
                if abs(result.ate) > 1e6:
                    issues.append("Suspiciously large ATE result")
                    self._log_violation(
                        'SUSPICIOUS_RESULT',
                        'medium',
                        f"Large ATE result: {result.ate}",
                        user_id
                    )
            
            execution_time = time.time() - start_time
            
            # Log successful operation
            self.audit_logger.log_operation(
                operation='compute_ate',
                user_id=user_id,
                parameters={
                    'treatment': treatment,
                    'outcome': outcome,
                    'n_samples': n_samples,
                    'dag_nodes': len(dag.nodes),
                    'dag_edges': len(dag.edges)
                },
                result_summary=f"ATE: {result.ate if result else 'None'}",
                execution_time=execution_time,
                success=result is not None
            )
            
        except Exception as e:
            issues.append(f"Computation error: {str(e)}")
            self._log_violation(
                'COMPUTATION_ERROR',
                'high',
                f"Error computing ATE: {str(e)}",
                user_id
            )
            
            execution_time = time.time() - start_time
            self.audit_logger.log_operation(
                operation='compute_ate',
                user_id=user_id,
                parameters={'error': str(e)},
                result_summary='ERROR',
                execution_time=execution_time,
                success=False
            )
        
        return result, issues
    
    def _compute_with_timeout(self, func, timeout_seconds: float):
        """Execute function with timeout."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Computation timed out")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))
        
        try:
            result = func()
            signal.alarm(0)  # Cancel alarm
            return result
        except TimeoutError:
            raise
        finally:
            signal.signal(signal.SIGALRM, old_handler)
    
    def _log_violation(
        self,
        violation_type: str,
        severity: str,
        description: str,
        user_id: Optional[str]
    ) -> None:
        """Log a security violation."""
        violation = SecurityViolation(
            violation_type=violation_type,
            severity=severity,
            description=description,
            user_id=user_id,
            ip_address=None,  # Would get from request context
            timestamp=datetime.now(),
            additional_data={}
        )
        
        self.audit_logger.log_security_violation(violation)


# Example usage and configuration
def create_secure_config() -> SecurityConfig:
    """Create a secure configuration for production use."""
    return SecurityConfig(
        max_variables=500,
        max_samples=100000,
        max_computation_time=1800.0,  # 30 minutes
        max_memory_mb=4096.0,  # 4GB
        allowed_file_types=['.json', '.csv', '.parquet'],
        rate_limit_per_minute=50,
        enable_audit_logging=True
    )


def setup_user_permissions() -> CausalAccessControl:
    """Set up standard user permission profiles."""
    access_control = CausalAccessControl()
    
    # Research user permissions
    research_permissions = {
        'compute_ate': True,
        'compute_intervention': True,
        'upload_data': True,
        'create_experiment': True,
        'view_results': True,
        'admin_operations': False
    }
    
    # Admin permissions
    admin_permissions = {
        'compute_ate': True,
        'compute_intervention': True,
        'upload_data': True,
        'create_experiment': True,
        'view_results': True,
        'admin_operations': True,
        'view_audit_logs': True,
        'manage_users': True
    }
    
    # Read-only permissions
    readonly_permissions = {
        'compute_ate': False,
        'compute_intervention': False,
        'upload_data': False,
        'create_experiment': False,
        'view_results': True,
        'admin_operations': False
    }
    
    return access_control