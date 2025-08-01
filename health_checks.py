"""
Comprehensive Health Check System for Causal UI Gym

Provides detailed health checks for all system components including
database, external services, disk space, memory, and application-specific checks.
"""

import asyncio
import time
import psutil
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import os
import subprocess

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: str
    details: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SystemHealthReport:
    """Complete system health report."""
    overall_status: HealthStatus
    timestamp: str
    duration_ms: float
    version: str
    environment: str
    checks: List[HealthCheckResult]
    summary: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            **asdict(self),
            'checks': [asdict(check) for check in self.checks],
            'overall_status': self.overall_status.value,
            'summary': {
                status.value: count for status, count in self.summary.items()
            }
        }


class HealthChecker:
    """Main health check coordinator."""
    
    def __init__(self, 
                 app_version: str = "1.0.0",
                 environment: str = "development",
                 timeout_seconds: float = 30.0):
        self.app_version = app_version
        self.environment = environment
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(__name__)
        self._checks: Dict[str, Callable] = {}
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("application_startup", self._check_application_startup)
        
        if ASYNCPG_AVAILABLE:
            self.register_check("database", self._check_database)
        
        if REDIS_AVAILABLE:
            self.register_check("redis", self._check_redis)
        
        if JAX_AVAILABLE:
            self.register_check("jax_environment", self._check_jax_environment)
        
        self.register_check("external_apis", self._check_external_apis)
        self.register_check("configuration", self._check_configuration)
    
    def register_check(self, name: str, check_func: Callable):
        """Register a custom health check."""
        self._checks[name] = check_func
    
    async def run_health_checks(self, 
                              include_checks: Optional[List[str]] = None,
                              exclude_checks: Optional[List[str]] = None) -> SystemHealthReport:
        """Run all health checks and return comprehensive report."""
        start_time = time.time()
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Determine which checks to run
        checks_to_run = self._get_checks_to_run(include_checks, exclude_checks)
        
        # Run all checks concurrently with timeout
        check_results = []
        
        try:
            tasks = [
                self._run_single_check(name, check_func)
                for name, check_func in checks_to_run.items()
            ]
            
            check_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            self.logger.error(f"Health checks timed out after {self.timeout_seconds}s")
            check_results = [
                HealthCheckResult(
                    name="timeout",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health checks timed out after {self.timeout_seconds}s",
                    duration_ms=(time.time() - start_time) * 1000,
                    timestamp=timestamp
                )
            ]
        
        # Handle exceptions in results
        final_results = []
        for i, result in enumerate(check_results):
            if isinstance(result, Exception):
                check_name = list(checks_to_run.keys())[i] if i < len(checks_to_run) else "unknown"
                final_results.append(
                    HealthCheckResult(
                        name=check_name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check failed with exception: {str(result)}",
                        duration_ms=0,
                        timestamp=timestamp,
                        details={"exception_type": type(result).__name__}
                    )
                )
            else:
                final_results.append(result)
        
        # Calculate overall status and summary
        overall_status = self._calculate_overall_status(final_results)
        summary = self._calculate_summary(final_results)
        total_duration = (time.time() - start_time) * 1000
        
        return SystemHealthReport(
            overall_status=overall_status,
            timestamp=timestamp,
            duration_ms=total_duration,
            version=self.app_version,
            environment=self.environment,
            checks=final_results,
            summary=summary
        )
    
    def _get_checks_to_run(self, 
                          include_checks: Optional[List[str]], 
                          exclude_checks: Optional[List[str]]) -> Dict[str, Callable]:
        """Determine which checks to run based on include/exclude lists."""
        checks_to_run = self._checks.copy()
        
        if include_checks:
            checks_to_run = {
                name: func for name, func in checks_to_run.items()
                if name in include_checks
            }
        
        if exclude_checks:
            checks_to_run = {
                name: func for name, func in checks_to_run.items()
                if name not in exclude_checks
            }
        
        return checks_to_run
    
    async def _run_single_check(self, name: str, check_func: Callable) -> HealthCheckResult:
        """Run a single health check with timing and error handling."""
        start_time = time.time()
        timestamp = datetime.now(timezone.utc).isoformat()
        
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(result, HealthCheckResult):
                result.duration_ms = duration_ms
                result.timestamp = timestamp
                return result
            else:
                # Handle simple return values
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    message="Check completed" if result else "Check failed",
                    duration_ms=duration_ms,
                    timestamp=timestamp
                )
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.exception(f"Health check '{name}' failed")
            
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                duration_ms=duration_ms,
                timestamp=timestamp,
                details={
                    "exception_type": type(e).__name__,
                    "exception_message": str(e)
                }
            )
    
    def _calculate_overall_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """Calculate overall system health from individual check results."""
        if not results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in results]
        
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def _calculate_summary(self, results: List[HealthCheckResult]) -> Dict[HealthStatus, int]:
        """Calculate summary statistics of health check results."""
        summary = {status: 0 for status in HealthStatus}
        
        for result in results:
            summary[result.status] += 1
        
        return summary
    
    # Default Health Checks
    
    def _check_system_resources(self) -> HealthCheckResult:
        """Check system CPU and memory usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_total": memory.total,
                "memory_available": memory.available,
                "memory_used": memory.used
            }
            
            # Determine status based on thresholds
            if cpu_percent > 90 or memory_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"High resource usage: CPU {cpu_percent}%, Memory {memory_percent}%"
            elif cpu_percent > 70 or memory_percent > 70:
                status = HealthStatus.DEGRADED
                message = f"Moderate resource usage: CPU {cpu_percent}%, Memory {memory_percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Resource usage normal: CPU {cpu_percent}%, Memory {memory_percent}%"
            
            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=message,
                duration_ms=0,  # Will be set by caller
                timestamp="",  # Will be set by caller
                details=details
            )
        
        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check system resources: {str(e)}",
                duration_ms=0,
                timestamp=""
            )
    
    def _check_disk_space(self) -> HealthCheckResult:
        """Check available disk space."""
        try:
            disk_usage = psutil.disk_usage('/')
            free_percent = (disk_usage.free / disk_usage.total) * 100
            used_percent = (disk_usage.used / disk_usage.total) * 100
            
            details = {
                "total_bytes": disk_usage.total,
                "used_bytes": disk_usage.used,
                "free_bytes": disk_usage.free,
                "used_percent": used_percent,
                "free_percent": free_percent
            }
            
            if free_percent < 5:
                status = HealthStatus.UNHEALTHY
                message = f"Critical disk space: {free_percent:.1f}% free"
            elif free_percent < 15:
                status = HealthStatus.DEGRADED
                message = f"Low disk space: {free_percent:.1f}% free"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space OK: {free_percent:.1f}% free"
            
            return HealthCheckResult(
                name="disk_space",
                status=status,
                message=message,
                duration_ms=0,
                timestamp="",
                details=details
            )
        
        except Exception as e:
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check disk space: {str(e)}",
                duration_ms=0,
                timestamp=""
            )
    
    async def _check_database(self) -> HealthCheckResult:
        """Check database connectivity and basic operations."""
        if not ASYNCPG_AVAILABLE:
            return HealthCheckResult(
                name="database",
                status=HealthStatus.UNKNOWN,
                message="asyncpg not available",
                duration_ms=0,
                timestamp=""
            )
        
        database_url = os.getenv('DATABASE_URL', 'postgresql://localhost/causal_ui_gym')
        
        try:
            conn = await asyncpg.connect(database_url)
            
            # Test basic query
            result = await conn.fetchval('SELECT 1')
            
            # Check connection pool status if available
            pool_size = getattr(conn, '_pool_size', 'unknown')
            
            await conn.close()
            
            details = {
                "query_result": result,
                "pool_size": pool_size,
                "database_url": database_url.split('@')[-1] if '@' in database_url else "hidden"
            }
            
            return HealthCheckResult(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                duration_ms=0,
                timestamp="",
                details=details
            )
        
        except Exception as e:
            return HealthCheckResult(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                duration_ms=0,
                timestamp="",
                details={"error": str(e)}
            )
    
    async def _check_redis(self) -> HealthCheckResult:
        """Check Redis connectivity."""
        if not REDIS_AVAILABLE:
            return HealthCheckResult(
                name="redis",
                status=HealthStatus.UNKNOWN,
                message="redis package not available",
                duration_ms=0,
                timestamp=""
            )
        
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        
        try:
            r = redis.Redis.from_url(redis_url)
            
            # Test basic operations
            test_key = "health_check_test"
            await r.set(test_key, "test_value", ex=60)
            result = await r.get(test_key)
            await r.delete(test_key)
            
            # Get Redis info
            info = await r.info()
            
            details = {
                "test_result": result.decode() if result else None,
                "redis_version": info.get('redis_version'),
                "connected_clients": info.get('connected_clients'),
                "used_memory": info.get('used_memory'),
                "used_memory_human": info.get('used_memory_human')
            }
            
            return HealthCheckResult(
                name="redis",
                status=HealthStatus.HEALTHY,
                message="Redis connection successful",
                duration_ms=0,
                timestamp="",
                details=details
            )
        
        except Exception as e:
            return HealthCheckResult(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis connection failed: {str(e)}",
                duration_ms=0,
                timestamp="",
                details={"error": str(e)}
            )
    
    def _check_jax_environment(self) -> HealthCheckResult:
        """Check JAX availability and basic functionality."""
        if not JAX_AVAILABLE:
            return HealthCheckResult(
                name="jax_environment",
                status=HealthStatus.UNKNOWN,
                message="JAX not available",
                duration_ms=0,
                timestamp=""
            )
        
        try:
            # Test basic JAX operations
            x = jnp.array([1.0, 2.0, 3.0])
            y = jnp.sum(x)
            
            # Get device information
            devices = jax.devices()
            default_backend = jax.default_backend()
            
            details = {
                "jax_version": jax.__version__,
                "default_backend": default_backend,
                "available_devices": [str(device) for device in devices],
                "device_count": len(devices),
                "test_computation_result": float(y)
            }
            
            return HealthCheckResult(
                name="jax_environment",
                status=HealthStatus.HEALTHY,
                message=f"JAX environment OK with {len(devices)} device(s)",
                duration_ms=0,
                timestamp="",
                details=details
            )
        
        except Exception as e:
            return HealthCheckResult(
                name="jax_environment",
                status=HealthStatus.UNHEALTHY,
                message=f"JAX environment check failed: {str(e)}",
                duration_ms=0,
                timestamp="",
                details={"error": str(e)}
            )
    
    async def _check_external_apis(self) -> HealthCheckResult:
        """Check external API connectivity."""
        if not HTTPX_AVAILABLE:
            return HealthCheckResult(
                name="external_apis",
                status=HealthStatus.UNKNOWN,
                message="httpx not available for external API checks",
                duration_ms=0,
                timestamp=""
            )
        
        api_checks = {
            "openai": os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1/models'),
            "anthropic": os.getenv('ANTHROPIC_API_BASE', 'https://api.anthropic.com')
        }
        
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for api_name, url in api_checks.items():
                try:
                    # Only do a HEAD request or basic connectivity check
                    response = await client.get(url, 
                                              headers={"User-Agent": "HealthCheck/1.0"})
                    
                    if response.status_code < 500:
                        results[api_name] = {
                            "status": "healthy",
                            "status_code": response.status_code,
                            "response_time_ms": response.elapsed.total_seconds() * 1000
                        }
                    else:
                        results[api_name] = {
                            "status": "degraded",
                            "status_code": response.status_code,
                            "error": f"Server error: {response.status_code}"
                        }
                        overall_status = HealthStatus.DEGRADED
                
                except Exception as e:
                    results[api_name] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                    overall_status = HealthStatus.DEGRADED  # Not fully unhealthy for external APIs
        
        healthy_apis = sum(1 for r in results.values() if r.get("status") == "healthy")
        total_apis = len(results)
        
        message = f"External APIs: {healthy_apis}/{total_apis} healthy"
        
        return HealthCheckResult(
            name="external_apis",
            status=overall_status,
            message=message,
            duration_ms=0,
            timestamp="",
            details=results
        )
    
    def _check_configuration(self) -> HealthCheckResult:
        """Check application configuration."""
        try:
            required_env_vars = [
                'NODE_ENV',
                'DATABASE_URL',
                'REDIS_URL'
            ]
            
            optional_env_vars = [
                'OPENAI_API_KEY',
                'ANTHROPIC_API_KEY',
                'MONITORING_ENABLED'
            ]
            
            missing_required = []
            present_optional = []
            
            for var in required_env_vars:
                if not os.getenv(var):
                    missing_required.append(var)
            
            for var in optional_env_vars:
                if os.getenv(var):
                    present_optional.append(var)
            
            details = {
                "required_env_vars": {
                    "total": len(required_env_vars),
                    "present": len(required_env_vars) - len(missing_required),
                    "missing": missing_required
                },
                "optional_env_vars": {
                    "total": len(optional_env_vars),
                    "present": len(present_optional),
                    "configured": present_optional
                },
                "environment": os.getenv('NODE_ENV', 'unknown'),
                "python_version": os.sys.version.split()[0],
                "working_directory": os.getcwd()
            }
            
            if missing_required:
                status = HealthStatus.UNHEALTHY
                message = f"Missing required configuration: {', '.join(missing_required)}"
            else:
                status = HealthStatus.HEALTHY
                message = "Configuration OK"
            
            return HealthCheckResult(
                name="configuration",
                status=status,
                message=message,
                duration_ms=0,
                timestamp="",
                details=details
            )
        
        except Exception as e:
            return HealthCheckResult(
                name="configuration",
                status=HealthStatus.UNHEALTHY,
                message=f"Configuration check failed: {str(e)}",
                duration_ms=0,
                timestamp=""
            )
    
    def _check_application_startup(self) -> HealthCheckResult:
        """Check if the application started successfully."""
        try:
            # Check if critical files exist
            critical_files = [
                'package.json',
                'pyproject.toml',
                'requirements.txt'
            ]
            
            missing_files = []
            for file_path in critical_files:
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
            
            # Check if we can import critical modules
            import_results = {}
            critical_imports = [
                ('jax', 'jax'),
                ('fastapi', 'fastapi'),
                ('react', 'package.json')  # We'll check if React is in package.json
            ]
            
            for module_name, import_path in critical_imports:
                try:
                    if import_path == 'package.json' and os.path.exists('package.json'):
                        with open('package.json', 'r') as f:
                            package_data = json.load(f)
                            has_react = 'react' in package_data.get('dependencies', {})
                            import_results[module_name] = has_react
                    else:
                        __import__(import_path)
                        import_results[module_name] = True
                except Exception:
                    import_results[module_name] = False
            
            details = {
                "critical_files": {
                    "total": len(critical_files),
                    "present": len(critical_files) - len(missing_files),
                    "missing": missing_files
                },
                "critical_imports": import_results,
                "startup_time": time.time(),
                "process_id": os.getpid()
            }
            
            if missing_files or not all(import_results.values()):
                status = HealthStatus.DEGRADED
                message = "Application startup has issues"
            else:
                status = HealthStatus.HEALTHY
                message = "Application startup successful"
            
            return HealthCheckResult(
                name="application_startup",
                status=status,
                message=message,
                duration_ms=0,
                timestamp="",
                details=details
            )
        
        except Exception as e:
            return HealthCheckResult(
                name="application_startup",
                status=HealthStatus.UNHEALTHY,
                message=f"Application startup check failed: {str(e)}",
                duration_ms=0,
                timestamp=""
            )


# Global health checker instance
health_checker = HealthChecker(
    app_version=os.getenv('APP_VERSION', '1.0.0'),
    environment=os.getenv('NODE_ENV', 'development')
)


# FastAPI integration
async def health_check_endpoint(include: Optional[str] = None, 
                              exclude: Optional[str] = None) -> Dict[str, Any]:
    """FastAPI endpoint for health checks."""
    include_list = include.split(',') if include else None
    exclude_list = exclude.split(',') if exclude else None
    
    report = await health_checker.run_health_checks(
        include_checks=include_list,
        exclude_checks=exclude_list
    )
    
    return report.to_dict()


async def readiness_check() -> Dict[str, Any]:
    """Kubernetes readiness probe endpoint."""
    # Only check critical components for readiness
    report = await health_checker.run_health_checks(
        include_checks=['database', 'redis', 'configuration']
    )
    
    if report.overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
        return {"status": "ready", "timestamp": report.timestamp}
    else:
        raise HTTPException(status_code=503, detail="Service not ready")


async def liveness_check() -> Dict[str, Any]:
    """Kubernetes liveness probe endpoint."""
    # Basic liveness check
    return {
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime": time.time()
    }


if __name__ == "__main__":
    # CLI usage example
    import asyncio
    
    async def main():
        print("Running health checks...")
        report = await health_checker.run_health_checks()
        
        print(f"\nOverall Status: {report.overall_status.value.upper()}")
        print(f"Duration: {report.duration_ms:.2f}ms")
        print(f"Environment: {report.environment}")
        print(f"Version: {report.version}")
        
        print(f"\nSummary:")
        for status, count in report.summary.items():
            print(f"  {status.value}: {count}")
        
        print(f"\nDetailed Results:")
        for check in report.checks:
            status_icon = {
                HealthStatus.HEALTHY: "✅",
                HealthStatus.DEGRADED: "⚠️",
                HealthStatus.UNHEALTHY: "❌",
                HealthStatus.UNKNOWN: "❓"
            }.get(check.status, "❓")
            
            print(f"  {status_icon} {check.name}: {check.message} ({check.duration_ms:.2f}ms)")
        
        return report.overall_status != HealthStatus.UNHEALTHY
    
    success = asyncio.run(main())
    exit(0 if success else 1)