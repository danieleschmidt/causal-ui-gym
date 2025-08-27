"""
Advanced Monitoring and Observability System for Causal UI Gym

Provides comprehensive monitoring, metrics collection, alerting, and observability
for research-grade causal inference operations with real-time dashboards.
"""

import asyncio
import time
import logging
import json
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import threading
from datetime import datetime, timedelta
import numpy as np
import jax.numpy as jnp
from contextlib import asynccontextmanager
import hashlib
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected by the monitoring system"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"
    DISTRIBUTION = "distribution"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceProfile:
    """Performance profiling information"""
    operation_name: str
    duration_ms: float
    cpu_usage_percent: float
    memory_usage_mb: float
    gpu_usage_percent: Optional[float]
    io_operations: int
    cache_hit_rate: float
    timestamp: float
    call_stack: Optional[List[str]] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None


@dataclass
class Alert:
    """System alert information"""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    threshold_value: float
    actual_value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class SystemHealth:
    """Overall system health metrics"""
    status: str  # healthy, degraded, critical, down
    overall_score: float  # 0-100
    component_health: Dict[str, float]
    active_alerts: List[Alert]
    performance_summary: Dict[str, float]
    resource_utilization: Dict[str, float]
    recent_errors: List[Dict[str, Any]]
    uptime_seconds: float
    last_updated: float


class MetricsCollector:
    """High-performance metrics collection with buffering"""
    
    def __init__(self, buffer_size: int = 10000, flush_interval: float = 30.0):
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        # Thread-safe metric storage
        self._metrics_buffer = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        
        # Aggregated metrics for fast queries
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        
        # Background flush task
        self._flush_task = None
        self._running = False
        
    async def start(self):
        """Start the metrics collector background tasks"""
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        
    async def stop(self):
        """Stop the metrics collector"""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
                
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        tags: Optional[Dict[str, str]] = None,
        **metadata
    ):
        """Record a metric (thread-safe)"""
        
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=time.time(),
            tags=tags or {},
            metadata=metadata
        )
        
        with self._lock:
            self._metrics_buffer.append(metric)
            
            # Update aggregated metrics for fast access
            if metric_type == MetricType.COUNTER:
                self._counters[name] += value
            elif metric_type == MetricType.GAUGE:
                self._gauges[name] = value
            elif metric_type == MetricType.HISTOGRAM:
                self._histograms[name].append(value)
                # Keep only recent values
                if len(self._histograms[name]) > 1000:
                    self._histograms[name] = self._histograms[name][-1000:]
            elif metric_type == MetricType.TIMER:
                self._timers[name].append(value)
                if len(self._timers[name]) > 1000:
                    self._timers[name] = self._timers[name][-1000:]
                    
    def increment_counter(self, name: str, value: float = 1.0, **kwargs):
        """Convenience method for counters"""
        self.record_metric(name, value, MetricType.COUNTER, **kwargs)
        
    def set_gauge(self, name: str, value: float, **kwargs):
        """Convenience method for gauges"""
        self.record_metric(name, value, MetricType.GAUGE, **kwargs)
        
    def record_timer(self, name: str, duration_ms: float, **kwargs):
        """Convenience method for timers"""
        self.record_metric(name, duration_ms, MetricType.TIMER, **kwargs)
        
    def record_histogram(self, name: str, value: float, **kwargs):
        """Convenience method for histograms"""
        self.record_metric(name, value, MetricType.HISTOGRAM, **kwargs)
        
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """Get summary statistics for a metric"""
        with self._lock:
            if name in self._counters:
                return {
                    "type": "counter",
                    "value": self._counters[name]
                }
            elif name in self._gauges:
                return {
                    "type": "gauge",
                    "value": self._gauges[name]
                }
            elif name in self._histograms:
                values = self._histograms[name]
                if values:
                    return {
                        "type": "histogram",
                        "count": len(values),
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "p50": np.percentile(values, 50),
                        "p95": np.percentile(values, 95),
                        "p99": np.percentile(values, 99)
                    }
            elif name in self._timers:
                values = self._timers[name]
                if values:
                    return {
                        "type": "timer",
                        "count": len(values),
                        "mean_ms": np.mean(values),
                        "std_ms": np.std(values),
                        "min_ms": np.min(values),
                        "max_ms": np.max(values),
                        "p50_ms": np.percentile(values, 50),
                        "p95_ms": np.percentile(values, 95),
                        "p99_ms": np.percentile(values, 99)
                    }
                    
        return {"type": "unknown", "error": "Metric not found"}
        
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics"""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: self.get_metric_summary(k) for k in self._histograms.keys()},
                "timers": {k: self.get_metric_summary(k) for k in self._timers.keys()}
            }
            
    async def _flush_loop(self):
        """Background task to flush metrics periodically"""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics flush loop: {e}")
                
    async def _flush_metrics(self):
        """Flush buffered metrics to storage/logging"""
        with self._lock:
            if self._metrics_buffer:
                # In production, this would send to a metrics backend
                # For now, we log summary statistics
                buffer_copy = list(self._metrics_buffer)
                
        if buffer_copy:
            logger.info(f"Metrics flush: {len(buffer_copy)} metrics processed")
            
            # Example: Group by metric name and log summaries
            metric_groups = defaultdict(list)
            for metric in buffer_copy:
                metric_groups[metric.name].append(metric.value)
                
            for name, values in metric_groups.items():
                logger.debug(f"Metric {name}: count={len(values)}, avg={np.mean(values):.3f}")


class PerformanceProfiler:
    """Advanced performance profiling for causal inference operations"""
    
    def __init__(self):
        self.profiles: List[PerformanceProfile] = []
        self._active_profiles: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        
    @asynccontextmanager
    async def profile_operation(self, operation_name: str, **metadata):
        """Context manager for profiling operations"""
        
        profile_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        # Start profiling
        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=None)
        start_memory = psutil.virtual_memory().used / (1024 * 1024)  # MB
        
        gc.collect()  # Clean up before measurement
        
        with self._lock:
            self._active_profiles[profile_id] = {
                'start_time': start_time,
                'start_cpu': start_cpu,
                'start_memory': start_memory,
                'operation_name': operation_name,
                'metadata': metadata
            }
            
        try:
            yield profile_id
        finally:
            # End profiling
            end_time = time.time()
            end_cpu = psutil.cpu_percent(interval=None)
            end_memory = psutil.virtual_memory().used / (1024 * 1024)
            
            with self._lock:
                if profile_id in self._active_profiles:
                    profile_data = self._active_profiles.pop(profile_id)
                    
                    duration_ms = (end_time - profile_data['start_time']) * 1000
                    cpu_usage = max(0, end_cpu - profile_data['start_cpu'])
                    memory_usage = end_memory - profile_data['start_memory']
                    
                    profile = PerformanceProfile(
                        operation_name=operation_name,
                        duration_ms=duration_ms,
                        cpu_usage_percent=cpu_usage,
                        memory_usage_mb=memory_usage,
                        gpu_usage_percent=self._get_gpu_usage(),
                        io_operations=0,  # Would need more sophisticated tracking
                        cache_hit_rate=0.0,  # Would need cache instrumentation
                        timestamp=end_time,
                        input_size=metadata.get('input_size'),
                        output_size=metadata.get('output_size')
                    )
                    
                    self.profiles.append(profile)
                    
                    # Keep only recent profiles
                    if len(self.profiles) > 1000:
                        self.profiles = self.profiles[-1000:]
                        
    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage if available"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except ImportError:
            pass
        return None
        
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get performance statistics for a specific operation"""
        
        relevant_profiles = [p for p in self.profiles if p.operation_name == operation_name]
        
        if not relevant_profiles:
            return {"error": "No profiles found for operation"}
            
        durations = [p.duration_ms for p in relevant_profiles]
        cpu_usages = [p.cpu_usage_percent for p in relevant_profiles]
        memory_usages = [p.memory_usage_mb for p in relevant_profiles]
        
        return {
            "operation_name": operation_name,
            "total_calls": len(relevant_profiles),
            "duration_stats": {
                "mean_ms": np.mean(durations),
                "std_ms": np.std(durations),
                "min_ms": np.min(durations),
                "max_ms": np.max(durations),
                "p95_ms": np.percentile(durations, 95)
            },
            "cpu_usage_stats": {
                "mean_percent": np.mean(cpu_usages),
                "max_percent": np.max(cpu_usages)
            },
            "memory_usage_stats": {
                "mean_mb": np.mean(memory_usages),
                "max_mb": np.max(memory_usages)
            },
            "recent_profiles": relevant_profiles[-10:]  # Last 10 profiles
        }
        
    def get_all_operation_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all operations"""
        
        operation_names = set(p.operation_name for p in self.profiles)
        return {name: self.get_operation_stats(name) for name in operation_names}


class AlertManager:
    """Advanced alerting system with thresholds and escalation"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.notification_handlers: List[Callable[[Alert], None]] = []
        self._alert_lock = threading.Lock()
        
    def add_alert_rule(
        self,
        metric_name: str,
        threshold: float,
        comparison: str,  # 'gt', 'lt', 'eq', 'gte', 'lte'
        severity: AlertSeverity,
        title: str,
        description: str = "",
        cooldown_seconds: float = 300.0,
        **kwargs
    ):
        """Add an alert rule for metric monitoring"""
        
        self.alert_rules[metric_name] = {
            'threshold': threshold,
            'comparison': comparison,
            'severity': severity,
            'title': title,
            'description': description,
            'cooldown_seconds': cooldown_seconds,
            'last_triggered': 0.0,
            'metadata': kwargs
        }
        
    def check_metric_alerts(self, metric_name: str, value: float):
        """Check if a metric value triggers any alerts"""
        
        if metric_name not in self.alert_rules:
            return
            
        rule = self.alert_rules[metric_name]
        current_time = time.time()
        
        # Check cooldown
        if current_time - rule['last_triggered'] < rule['cooldown_seconds']:
            return
            
        # Check threshold
        triggered = False
        comparison = rule['comparison']
        threshold = rule['threshold']
        
        if comparison == 'gt' and value > threshold:
            triggered = True
        elif comparison == 'lt' and value < threshold:
            triggered = True
        elif comparison == 'gte' and value >= threshold:
            triggered = True
        elif comparison == 'lte' and value <= threshold:
            triggered = True
        elif comparison == 'eq' and abs(value - threshold) < 1e-10:
            triggered = True
            
        if triggered:
            alert = Alert(
                alert_id=self._generate_alert_id(),
                severity=rule['severity'],
                title=rule['title'],
                description=rule['description'],
                metric_name=metric_name,
                threshold_value=threshold,
                actual_value=value,
                timestamp=current_time,
                tags=rule['metadata']
            )
            
            with self._alert_lock:
                self.alerts.append(alert)
                rule['last_triggered'] = current_time
                
            # Notify handlers
            self._notify_alert(alert)
            
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        return f"alert_{int(time.time() * 1000)}_{hash(str(time.time())) % 10000}"
        
    def _notify_alert(self, alert: Alert):
        """Send alert to all notification handlers"""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert notification handler: {e}")
                
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler for alerts"""
        self.notification_handlers.append(handler)
        
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get currently active (unresolved) alerts"""
        with self._alert_lock:
            alerts = [a for a in self.alerts if not a.resolved]
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            return alerts
            
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        with self._alert_lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.resolution_time = time.time()
                    break
                    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        with self._alert_lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    break


class AdvancedMonitoringSystem:
    """Comprehensive monitoring system for Causal UI Gym"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Components
        self.metrics = MetricsCollector(
            buffer_size=self.config.get('metrics_buffer_size', 10000),
            flush_interval=self.config.get('metrics_flush_interval', 30.0)
        )
        self.profiler = PerformanceProfiler()
        self.alerts = AlertManager()
        
        # System state
        self._start_time = time.time()
        self._running = False
        
        # Background tasks
        self._monitoring_task = None
        self._health_check_task = None
        
        # Setup default alert rules
        self._setup_default_alerts()
        
        # Setup default notification handlers
        self._setup_default_notifications()
        
    def _setup_default_alerts(self):
        """Setup default alert rules for common issues"""
        
        # High error rate
        self.alerts.add_alert_rule(
            metric_name="error_rate",
            threshold=0.05,  # 5% error rate
            comparison="gt",
            severity=AlertSeverity.WARNING,
            title="High Error Rate Detected",
            description="System error rate exceeds acceptable threshold"
        )
        
        # High response time
        self.alerts.add_alert_rule(
            metric_name="response_time_p95",
            threshold=5000,  # 5 seconds
            comparison="gt",
            severity=AlertSeverity.WARNING,
            title="High Response Time",
            description="95th percentile response time is too high"
        )
        
        # High memory usage
        self.alerts.add_alert_rule(
            metric_name="memory_usage_percent",
            threshold=90.0,
            comparison="gt",
            severity=AlertSeverity.CRITICAL,
            title="High Memory Usage",
            description="System memory usage is critically high"
        )
        
        # High CPU usage
        self.alerts.add_alert_rule(
            metric_name="cpu_usage_percent",
            threshold=95.0,
            comparison="gt",
            severity=AlertSeverity.ERROR,
            title="High CPU Usage",
            description="System CPU usage is very high"
        )
        
    def _setup_default_notifications(self):
        """Setup default notification handlers"""
        
        def log_alert_handler(alert: Alert):
            level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.ERROR: logging.ERROR,
                AlertSeverity.CRITICAL: logging.CRITICAL
            }[alert.severity]
            
            logger.log(level, f"ALERT: {alert.title} - {alert.description} (value: {alert.actual_value}, threshold: {alert.threshold_value})")
            
        self.alerts.add_notification_handler(log_alert_handler)
        
    async def start(self):
        """Start the monitoring system"""
        self._running = True
        
        # Start metrics collection
        await self.metrics.start()
        
        # Start background monitoring tasks
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("Advanced monitoring system started")
        
    async def stop(self):
        """Stop the monitoring system"""
        self._running = False
        
        # Stop metrics collection
        await self.metrics.stop()
        
        # Cancel background tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._health_check_task:
            self._health_check_task.cancel()
            
        # Wait for tasks to complete
        tasks = [t for t in [self._monitoring_task, self._health_check_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
        logger.info("Advanced monitoring system stopped")
        
    @asynccontextmanager
    async def monitor_operation(self, operation_name: str, **metadata):
        """Context manager for monitoring operations with metrics and profiling"""
        
        # Start profiling
        async with self.profiler.profile_operation(operation_name, **metadata) as profile_id:
            start_time = time.time()
            
            # Record operation start
            self.metrics.increment_counter(f"{operation_name}_started")
            
            try:
                yield profile_id
                
                # Record success
                duration_ms = (time.time() - start_time) * 1000
                self.metrics.record_timer(f"{operation_name}_duration", duration_ms)
                self.metrics.increment_counter(f"{operation_name}_success")
                
            except Exception as e:
                # Record failure
                self.metrics.increment_counter(f"{operation_name}_error")
                self.metrics.increment_counter("total_errors")
                
                # Check error rate alerts
                self._check_error_rate_alerts()
                
                raise
                
    def record_causal_inference_metrics(
        self,
        method_name: str,
        dataset_size: int,
        computation_time: float,
        accuracy: float,
        confidence: float,
        **additional_metrics
    ):
        """Record metrics specific to causal inference operations"""
        
        base_tags = {"method": method_name}
        
        self.metrics.record_timer("causal_inference_time", computation_time * 1000, tags=base_tags)
        self.metrics.record_histogram("causal_accuracy", accuracy, tags=base_tags)
        self.metrics.record_histogram("causal_confidence", confidence, tags=base_tags)
        self.metrics.record_histogram("dataset_size", dataset_size, tags=base_tags)
        
        # Record method-specific metrics
        for metric_name, value in additional_metrics.items():
            self.metrics.record_histogram(f"causal_{metric_name}", value, tags=base_tags)
            
    def record_llm_interaction_metrics(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        response_time: float,
        success: bool,
        **additional_metrics
    ):
        """Record metrics for LLM interactions"""
        
        base_tags = {"model": model_name}
        
        self.metrics.record_timer("llm_response_time", response_time * 1000, tags=base_tags)
        self.metrics.record_histogram("llm_prompt_tokens", prompt_tokens, tags=base_tags)
        self.metrics.record_histogram("llm_completion_tokens", completion_tokens, tags=base_tags)
        
        if success:
            self.metrics.increment_counter("llm_success", tags=base_tags)
        else:
            self.metrics.increment_counter("llm_error", tags=base_tags)
            
        # Additional metrics
        for metric_name, value in additional_metrics.items():
            self.metrics.record_histogram(f"llm_{metric_name}", value, tags=base_tags)
            
    def _check_error_rate_alerts(self):
        """Check and trigger error rate alerts"""
        
        total_ops = self.metrics._counters.get("total_operations", 0)
        total_errors = self.metrics._counters.get("total_errors", 0)
        
        if total_ops > 0:
            error_rate = total_errors / total_ops
            self.alerts.check_metric_alerts("error_rate", error_rate)
            
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self._running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(10.0)  # Collect every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5.0)
                
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics.set_gauge("cpu_usage_percent", cpu_percent)
        self.alerts.check_metric_alerts("cpu_usage_percent", cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        self.metrics.set_gauge("memory_usage_percent", memory_percent)
        self.metrics.set_gauge("memory_usage_mb", memory.used / (1024 * 1024))
        self.alerts.check_metric_alerts("memory_usage_percent", memory_percent)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        self.metrics.set_gauge("disk_usage_percent", disk_percent)
        
        # Network I/O
        network = psutil.net_io_counters()
        self.metrics.set_gauge("network_bytes_sent", network.bytes_sent)
        self.metrics.set_gauge("network_bytes_recv", network.bytes_recv)
        
        # Process count
        process_count = len(psutil.pids())
        self.metrics.set_gauge("process_count", process_count)
        
    async def _health_check_loop(self):
        """Background health check loop"""
        while self._running:
            try:
                health = await self.get_system_health()
                self.metrics.set_gauge("system_health_score", health.overall_score)
                
                # Alert on low health score
                if health.overall_score < 70:
                    self.alerts.check_metric_alerts("system_health_score", health.overall_score)
                    
                await asyncio.sleep(60.0)  # Health check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(30.0)
                
    async def get_system_health(self) -> SystemHealth:
        """Get comprehensive system health status"""
        
        current_time = time.time()
        uptime = current_time - self._start_time
        
        # Component health scores
        component_health = {
            "metrics_collector": 100.0 if self.metrics._running else 0.0,
            "profiler": 100.0,  # Always healthy if system is running
            "alerts": 100.0,  # Always healthy if system is running
        }
        
        # Performance health based on recent metrics
        performance_health = self._calculate_performance_health()
        component_health.update(performance_health)
        
        # Overall health score (weighted average)
        weights = {
            "metrics_collector": 0.2,
            "profiler": 0.1,
            "alerts": 0.1,
            "cpu_health": 0.2,
            "memory_health": 0.2,
            "error_health": 0.2
        }
        
        overall_score = sum(component_health.get(comp, 100.0) * weight 
                          for comp, weight in weights.items())
        
        # Determine status
        if overall_score >= 90:
            status = "healthy"
        elif overall_score >= 70:
            status = "degraded"
        elif overall_score >= 40:
            status = "critical"
        else:
            status = "down"
            
        # Active alerts
        active_alerts = self.alerts.get_active_alerts()
        
        # Recent errors (last 10)
        recent_errors = []
        # Would pull from error logging system in production
        
        # Performance summary
        performance_summary = {
            "avg_response_time_ms": self.metrics.get_metric_summary("response_time").get("mean", 0),
            "error_rate": self._calculate_error_rate(),
            "throughput_ops_per_sec": self._calculate_throughput()
        }
        
        # Resource utilization
        resource_utilization = {
            "cpu_percent": self.metrics._gauges.get("cpu_usage_percent", 0),
            "memory_percent": self.metrics._gauges.get("memory_usage_percent", 0),
            "disk_percent": self.metrics._gauges.get("disk_usage_percent", 0)
        }
        
        return SystemHealth(
            status=status,
            overall_score=overall_score,
            component_health=component_health,
            active_alerts=active_alerts,
            performance_summary=performance_summary,
            resource_utilization=resource_utilization,
            recent_errors=recent_errors,
            uptime_seconds=uptime,
            last_updated=current_time
        )
        
    def _calculate_performance_health(self) -> Dict[str, float]:
        """Calculate health scores for performance metrics"""
        
        health = {}
        
        # CPU health
        cpu_usage = self.metrics._gauges.get("cpu_usage_percent", 0)
        health["cpu_health"] = max(0, 100 - cpu_usage)  # Lower CPU usage is better
        
        # Memory health
        memory_usage = self.metrics._gauges.get("memory_usage_percent", 0)
        health["memory_health"] = max(0, 100 - memory_usage)
        
        # Error health
        error_rate = self._calculate_error_rate()
        health["error_health"] = max(0, 100 - (error_rate * 1000))  # Scale error rate
        
        return health
        
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        total_ops = sum(v for k, v in self.metrics._counters.items() if k.endswith('_started'))
        total_errors = self.metrics._counters.get("total_errors", 0)
        
        if total_ops > 0:
            return total_errors / total_ops
        return 0.0
        
    def _calculate_throughput(self) -> float:
        """Calculate operations per second"""
        uptime = time.time() - self._start_time
        total_ops = sum(v for k, v in self.metrics._counters.items() if k.endswith('_started'))
        
        if uptime > 0:
            return total_ops / uptime
        return 0.0
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        
        return {
            "system_health": asyncio.create_task(self.get_system_health()),
            "metrics_summary": self.metrics.get_all_metrics(),
            "performance_stats": self.profiler.get_all_operation_stats(),
            "active_alerts": self.alerts.get_active_alerts(),
            "alert_history": self.alerts.alerts[-50:],  # Last 50 alerts
            "uptime_seconds": time.time() - self._start_time
        }
        
    def export_metrics(
        self, 
        format_type: str = "json", 
        time_range: Optional[Tuple[float, float]] = None
    ) -> str:
        """Export metrics in various formats"""
        
        data = {
            "timestamp": time.time(),
            "metrics": self.metrics.get_all_metrics(),
            "performance": self.profiler.get_all_operation_stats(),
            "alerts": [asdict(alert) for alert in self.alerts.alerts],
            "system_info": {
                "uptime_seconds": time.time() - self._start_time,
                "platform": psutil.WINDOWS or psutil.LINUX or psutil.OSX,
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3)
            }
        }
        
        if format_type == "json":
            return json.dumps(data, indent=2, default=str)
        elif format_type == "prometheus":
            return self._export_prometheus_format(data)
        else:
            return str(data)
            
    def _export_prometheus_format(self, data: Dict[str, Any]) -> str:
        """Export metrics in Prometheus format"""
        
        prometheus_lines = []
        
        # Export counters
        for name, value in data["metrics"].get("counters", {}).items():
            prometheus_lines.append(f'# TYPE {name} counter')
            prometheus_lines.append(f'{name} {value}')
            
        # Export gauges
        for name, value in data["metrics"].get("gauges", {}).items():
            prometheus_lines.append(f'# TYPE {name} gauge')
            prometheus_lines.append(f'{name} {value}')
            
        # Export histograms (simplified)
        for name, stats in data["metrics"].get("histograms", {}).items():
            if isinstance(stats, dict) and "mean" in stats:
                prometheus_lines.append(f'# TYPE {name} histogram')
                prometheus_lines.append(f'{name}_sum {stats["mean"] * stats.get("count", 1)}')
                prometheus_lines.append(f'{name}_count {stats.get("count", 1)}')
                
        return "\n".join(prometheus_lines)


# Global monitoring instance
monitoring = AdvancedMonitoringSystem()

# Convenience decorators
def monitor_causal_operation(operation_name: str):
    """Decorator for monitoring causal inference operations"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with monitoring.monitor_operation(operation_name):
                return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
        return wrapper
    return decorator


def monitor_llm_operation(model_name: str):
    """Decorator for monitoring LLM operations"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                # Extract metrics from result if available
                prompt_tokens = getattr(result, 'prompt_tokens', 0)
                completion_tokens = getattr(result, 'completion_tokens', 0)
                
                monitoring.record_llm_interaction_metrics(
                    model_name=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    response_time=time.time() - start_time,
                    success=True
                )
                
                return result
            except Exception as e:
                monitoring.record_llm_interaction_metrics(
                    model_name=model_name,
                    prompt_tokens=0,
                    completion_tokens=0,
                    response_time=time.time() - start_time,
                    success=False
                )
                raise
        return wrapper
    return decorator


# Export main classes
__all__ = [
    "AdvancedMonitoringSystem",
    "MetricsCollector",
    "PerformanceProfiler", 
    "AlertManager",
    "SystemHealth",
    "monitoring",
    "monitor_causal_operation",
    "monitor_llm_operation"
]
