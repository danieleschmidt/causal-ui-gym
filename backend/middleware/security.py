"""
Security middleware for Causal UI Gym API.

This module provides comprehensive security measures including
rate limiting, input validation, authentication, and threat detection.
"""

import time
import hashlib
import hmac
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict, deque

from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import re

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Exception raised when rate limit is exceeded."""
    def __init__(self, message: str, retry_after: int):
        self.message = message
        self.retry_after = retry_after
        super().__init__(message)


class SecurityThreat(Exception):
    """Exception raised when security threat is detected."""
    def __init__(self, message: str, threat_type: str, details: Dict[str, Any]):
        self.message = message
        self.threat_type = threat_type
        self.details = details
        super().__init__(message)


class RateLimiter:
    """Token bucket rate limiter with sliding window."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, deque] = defaultdict(deque)
    
    def is_allowed(self, identifier: str) -> tuple[bool, int]:
        """
        Check if request is allowed for given identifier.
        
        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        request_times = self.requests[identifier]
        while request_times and request_times[0] < window_start:
            request_times.popleft()
        
        # Check if within limits
        if len(request_times) >= self.max_requests:
            # Calculate retry after time
            oldest_request = request_times[0]
            retry_after = int(oldest_request + self.window_seconds - now) + 1
            return False, retry_after
        
        # Add current request
        request_times.append(now)
        return True, 0


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    # Dangerous patterns that might indicate attacks
    DANGEROUS_PATTERNS = [
        # SQL Injection
        r"(?i)(union\s+select|drop\s+table|insert\s+into|delete\s+from)",
        r"(?i)(\'\s*or\s*\'\d*\'\s*=\s*\'\d*|\'\s*or\s*\d*\s*=\s*\d*)",
        
        # XSS
        r"(?i)(<script|javascript:|vbscript:|onload=|onerror=)",
        r"(?i)(alert\s*\(|confirm\s*\(|prompt\s*\()",
        
        # Command Injection
        r"(?i)(;\s*rm\s|;\s*cat\s|;\s*ls\s|;\s*pwd|&&\s*rm\s)",
        r"(?i)(\$\(.*\)|`.*`|\|\s*sh|\|\s*bash)",
        
        # Path Traversal
        r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e%5c)",
        
        # LDAP Injection
        r"(\*\)|\(\||\)\(|\*\|\*)",
        
        # XXE
        r"(?i)(<!entity|<!doctype.*entity|SYSTEM\s+[\"'])",
    ]
    
    # Compilation for performance
    COMPILED_PATTERNS = [re.compile(pattern) for pattern in DANGEROUS_PATTERNS]
    
    # Maximum allowed lengths for different fields
    MAX_LENGTHS = {
        'experiment_name': 200,
        'experiment_description': 2000,
        'node_id': 50,
        'node_label': 100,
        'agent_id': 50,
        'query_text': 5000,
        'general_string': 1000
    }
    
    @staticmethod
    def validate_string(value: str, field_type: str = 'general_string') -> bool:
        """Validate string input for security threats."""
        if not isinstance(value, str):
            return False
            
        # Check length
        max_length = InputValidator.MAX_LENGTHS.get(field_type, 1000)
        if len(value) > max_length:
            raise SecurityThreat(
                f"Input too long for field type {field_type}",
                "input_length_exceeded",
                {"field_type": field_type, "max_length": max_length, "actual_length": len(value)}
            )
        
        # Check for dangerous patterns
        for pattern in InputValidator.COMPILED_PATTERNS:
            if pattern.search(value):
                raise SecurityThreat(
                    "Potentially malicious input detected",
                    "malicious_pattern",
                    {"pattern": pattern.pattern, "input": value[:100]}
                )
        
        return True
    
    @staticmethod
    def validate_numeric(value: Any, min_val: float = None, max_val: float = None) -> bool:
        """Validate numeric input."""
        if not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                return False
        
        if not (-1e308 <= value <= 1e308):  # Check for reasonable bounds
            raise SecurityThreat(
                "Numeric value out of reasonable bounds",
                "numeric_bounds_exceeded",
                {"value": value}
            )
        
        if min_val is not None and value < min_val:
            raise SecurityThreat(
                f"Value {value} below minimum {min_val}",
                "value_below_minimum",
                {"value": value, "minimum": min_val}
            )
        
        if max_val is not None and value > max_val:
            raise SecurityThreat(
                f"Value {value} above maximum {max_val}",
                "value_above_maximum", 
                {"value": value, "maximum": max_val}
            )
        
        return True
    
    @staticmethod
    def sanitize_dag_input(dag_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize and validate DAG input data."""
        if not isinstance(dag_data, dict):
            raise SecurityThreat("DAG data must be a dictionary", "invalid_type", {"type": type(dag_data)})
        
        # Validate nodes
        if 'nodes' in dag_data:
            if not isinstance(dag_data['nodes'], list):
                raise SecurityThreat("Nodes must be a list", "invalid_type", {})
            
            if len(dag_data['nodes']) > 100:  # Reasonable limit
                raise SecurityThreat("Too many nodes", "resource_limit", {"count": len(dag_data['nodes'])})
            
            for i, node in enumerate(dag_data['nodes']):
                if not isinstance(node, dict):
                    raise SecurityThreat(f"Node {i} must be a dictionary", "invalid_type", {})
                
                if 'id' in node:
                    InputValidator.validate_string(node['id'], 'node_id')
                if 'label' in node:
                    InputValidator.validate_string(node['label'], 'node_label')
        
        # Validate edges
        if 'edges' in dag_data:
            if not isinstance(dag_data['edges'], list):
                raise SecurityThreat("Edges must be a list", "invalid_type", {})
                
            if len(dag_data['edges']) > 500:  # Reasonable limit
                raise SecurityThreat("Too many edges", "resource_limit", {"count": len(dag_data['edges'])})
        
        return dag_data


class ThreatDetector:
    """Advanced threat detection system."""
    
    def __init__(self):
        self.suspicious_ips: Dict[str, Dict[str, Any]] = {}
        self.blocked_ips: set = set()
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
    
    def analyze_request(self, request: Request) -> Dict[str, Any]:
        """Analyze request for security threats."""
        client_ip = self._get_client_ip(request)
        threat_score = 0
        threats = []
        
        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            raise SecurityThreat(
                "IP address is blocked",
                "blocked_ip",
                {"ip": client_ip}
            )
        
        # Analyze request headers
        threat_score += self._analyze_headers(request.headers, threats)
        
        # Analyze user agent
        threat_score += self._analyze_user_agent(request.headers.get('user-agent', ''), threats)
        
        # Check for suspicious patterns in URL
        threat_score += self._analyze_url(str(request.url), threats)
        
        # Check request frequency
        threat_score += self._analyze_request_frequency(client_ip, threats)
        
        # Update suspicious IP tracking
        if threat_score > 0:
            self._update_suspicious_ip(client_ip, threat_score, threats)
        
        # Block IP if threat score is too high
        if threat_score >= 100:
            self.blocked_ips.add(client_ip)
            raise SecurityThreat(
                "High threat score detected",
                "high_threat_score",
                {"ip": client_ip, "score": threat_score, "threats": threats}
            )
        
        return {
            "ip": client_ip,
            "threat_score": threat_score,
            "threats": threats
        }
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers (be careful with these in production)
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip.strip()
        
        return request.client.host if request.client else 'unknown'
    
    def _analyze_headers(self, headers: Dict, threats: List[str]) -> int:
        """Analyze request headers for threats."""
        score = 0
        
        # Missing important headers
        if not headers.get('user-agent'):
            threats.append("missing_user_agent")
            score += 20
        
        # Suspicious header values
        suspicious_headers = ['x-forwarded-for', 'x-real-ip', 'x-originating-ip']
        for header in suspicious_headers:
            if header in headers:
                value = headers[header]
                if len(value.split(',')) > 5:  # Too many forwarded IPs
                    threats.append(f"too_many_forwarded_ips_{header}")
                    score += 15
        
        return score
    
    def _analyze_user_agent(self, user_agent: str, threats: List[str]) -> int:
        """Analyze user agent for threats."""
        score = 0
        
        if not user_agent:
            return 0
        
        # Known malicious user agents
        malicious_patterns = [
            r'sqlmap',
            r'nikto',
            r'nessus',
            r'w3af',
            r'arachni',
            r'skipfish',
            r'burp'
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, user_agent, re.IGNORECASE):
                threats.append(f"malicious_user_agent_{pattern}")
                score += 50
        
        # Suspicious patterns
        if len(user_agent) > 500:
            threats.append("user_agent_too_long")
            score += 10
        
        if re.search(r'<script|javascript:|eval\(', user_agent, re.IGNORECASE):
            threats.append("user_agent_xss_pattern")
            score += 30
        
        return score
    
    def _analyze_url(self, url: str, threats: List[str]) -> int:
        """Analyze URL for threats."""
        score = 0
        
        # Check for path traversal
        if '../' in url or '%2e%2e%2f' in url.lower():
            threats.append("path_traversal")
            score += 40
        
        # Check for SQL injection patterns in URL
        sql_patterns = ['union', 'select', 'drop', 'insert', 'delete']
        for pattern in sql_patterns:
            if pattern in url.lower():
                threats.append(f"sql_injection_url_{pattern}")
                score += 30
        
        # Check URL length
        if len(url) > 2000:
            threats.append("url_too_long")
            score += 15
        
        return score
    
    def _analyze_request_frequency(self, ip: str, threats: List[str]) -> int:
        """Analyze request frequency for this IP."""
        score = 0
        now = datetime.now()
        
        # Clean old failed attempts
        cutoff = now - timedelta(minutes=15)
        self.failed_attempts[ip] = [
            attempt for attempt in self.failed_attempts[ip] 
            if attempt > cutoff
        ]
        
        # Check recent failed attempts
        recent_failures = len(self.failed_attempts[ip])
        if recent_failures > 10:
            threats.append("too_many_failed_attempts")
            score += 25
        elif recent_failures > 5:
            threats.append("multiple_failed_attempts")
            score += 10
        
        return score
    
    def _update_suspicious_ip(self, ip: str, score: int, threats: List[str]) -> None:
        """Update suspicious IP tracking."""
        if ip not in self.suspicious_ips:
            self.suspicious_ips[ip] = {
                'first_seen': datetime.now(),
                'total_score': 0,
                'threat_count': 0,
                'threats': set()
            }
        
        self.suspicious_ips[ip]['total_score'] += score
        self.suspicious_ips[ip]['threat_count'] += 1
        self.suspicious_ips[ip]['threats'].update(threats)
        self.suspicious_ips[ip]['last_seen'] = datetime.now()
    
    def record_failed_attempt(self, ip: str) -> None:
        """Record a failed authentication attempt."""
        self.failed_attempts[ip].append(datetime.now())


class SecurityMiddleware(BaseHTTPMiddleware):
    """Main security middleware."""
    
    def __init__(self, app, rate_limit_requests: int = 100, rate_limit_window: int = 60):
        super().__init__(app)
        self.rate_limiter = RateLimiter(rate_limit_requests, rate_limit_window)
        self.threat_detector = ThreatDetector()
        self.validator = InputValidator()
    
    async def dispatch(self, request: Request, call_next):
        """Process request through security layers."""
        start_time = time.time()
        
        try:
            # Get client identifier for rate limiting
            client_id = self._get_client_identifier(request)
            
            # Check rate limits
            allowed, retry_after = self.rate_limiter.is_allowed(client_id)
            if not allowed:
                logger.warning(f"Rate limit exceeded for {client_id}")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "retry_after": retry_after
                    },
                    headers={"Retry-After": str(retry_after)}
                )
            
            # Threat detection
            threat_analysis = self.threat_detector.analyze_request(request)
            
            # Log security metrics
            self._log_security_metrics(request, threat_analysis, time.time() - start_time)
            
            # Process request
            response = await call_next(request)
            
            # Add security headers to response
            response = self._add_security_headers(response)
            
            return response
            
        except SecurityThreat as e:
            logger.warning(f"Security threat detected: {e.message}", extra={
                "threat_type": e.threat_type,
                "details": e.details,
                "ip": self._get_client_ip(request)
            })
            
            # Record failed attempt
            self.threat_detector.record_failed_attempt(self._get_client_ip(request))
            
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": "Security threat detected",
                    "message": "Request blocked for security reasons",
                    "threat_type": e.threat_type
                }
            )
            
        except RateLimitError as e:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": e.message,
                    "retry_after": e.retry_after
                },
                headers={"Retry-After": str(e.retry_after)}
            )
            
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Internal security error"}
            )
    
    def _get_client_identifier(self, request: Request) -> str:
        """Get unique client identifier for rate limiting."""
        # In production, you might want to use API keys or user IDs
        ip = self._get_client_ip(request)
        user_agent = request.headers.get('user-agent', '')
        
        # Create a hash of IP + User Agent for rate limiting
        identifier = f"{ip}:{hashlib.sha256(user_agent.encode()).hexdigest()[:16]}"
        return identifier
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        return self.threat_detector._get_client_ip(request)
    
    def _add_security_headers(self, response):
        """Add security headers to response."""
        security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'X-Permitted-Cross-Domain-Policies': 'none'
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response
    
    def _log_security_metrics(self, request: Request, threat_analysis: Dict, processing_time: float):
        """Log security metrics for monitoring."""
        logger.info("Security analysis completed", extra={
            "ip": threat_analysis.get("ip"),
            "threat_score": threat_analysis.get("threat_score"),
            "threats": threat_analysis.get("threats"),
            "processing_time": processing_time,
            "path": request.url.path,
            "method": request.method,
            "user_agent": request.headers.get('user-agent', '')[:100]
        })


# Authentication utilities
class APIKeyValidator:
    """API key validation and management."""
    
    def __init__(self, valid_keys: Optional[List[str]] = None):
        self.valid_keys = set(valid_keys) if valid_keys else set()
        self.key_usage: Dict[str, Dict[str, Any]] = {}
    
    def validate_key(self, api_key: str) -> bool:
        """Validate API key."""
        if not api_key or api_key not in self.valid_keys:
            return False
        
        # Track usage
        if api_key not in self.key_usage:
            self.key_usage[api_key] = {
                'first_used': datetime.now(),
                'usage_count': 0,
                'last_used': None
            }
        
        self.key_usage[api_key]['usage_count'] += 1
        self.key_usage[api_key]['last_used'] = datetime.now()
        
        return True
    
    def add_key(self, api_key: str) -> None:
        """Add a new API key."""
        self.valid_keys.add(api_key)
    
    def revoke_key(self, api_key: str) -> None:
        """Revoke an API key."""
        self.valid_keys.discard(api_key)
        if api_key in self.key_usage:
            del self.key_usage[api_key]


# Security decorators
def require_api_key(api_key_validator: APIKeyValidator):
    """Decorator to require valid API key."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract API key from request (this is a simplified example)
            request = kwargs.get('request') 
            if not request:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key required"
                )
            
            api_key = request.headers.get('X-API-Key')
            if not api_key_validator.validate_key(api_key):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def validate_input(validator_func):
    """Decorator to validate request input."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # This would need to be adapted based on your specific needs
            try:
                # Apply validation to request data
                if 'request_data' in kwargs:
                    validator_func(kwargs['request_data'])
                return await func(*args, **kwargs)
            except SecurityThreat as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Input validation failed: {e.message}"
                )
        return wrapper
    return decorator