"""
Advanced Performance Optimizer for Causal UI Gym

Comprehensive performance optimization including intelligent caching,
memory management, computation optimization, and adaptive resource allocation.
"""

import asyncio
import time
import logging
import gc
import psutil
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from enum import Enum
import threading
from contextlib import asynccontextmanager
import hashlib
import pickle
import weakref
import numpy as np
import jax
import jax.numpy as jnp
from functools import lru_cache, wraps
import concurrent.futures

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Different caching strategies for optimization"""
    LRU = "lru"
    LFU = "lfu"
    ADAPTIVE = "adaptive"
    TIME_AWARE = "time_aware"
    SEMANTIC = "semantic"


class OptimizationLevel(Enum):
    """Performance optimization levels"""
    BASIC = "basic"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    RESEARCH_GRADE = "research_grade"


@dataclass
class CacheEntry:
    """Cache entry with metadata for intelligent eviction"""
    key: str
    value: Any
    size_bytes: int
    access_count: int
    last_access: float
    creation_time: float
    computation_time: float
    access_frequency: float
    semantic_hash: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    
    def update_access(self):
        """Update access statistics"""
        current_time = time.time()
        self.access_count += 1
        time_since_last = current_time - self.last_access
        
        # Exponential moving average for frequency
        if time_since_last > 0:
            self.access_frequency = 0.9 * self.access_frequency + 0.1 * (1.0 / time_since_last)
        
        self.last_access = current_time
        
    def get_value_score(self) -> float:
        """Calculate cache value score for eviction decisions"""
        current_time = time.time()
        age = current_time - self.creation_time
        recency = current_time - self.last_access
        
        # Factors: frequency, recency, size efficiency, computation cost
        frequency_score = self.access_frequency
        recency_score = 1.0 / (1.0 + recency)
        size_efficiency = self.computation_time / max(1, self.size_bytes / 1024)  # time per KB
        
        return frequency_score * recency_score * size_efficiency


@dataclass 
class PerformanceProfile:
    """Performance profiling data for operations"""
    operation_name: str
    avg_duration_ms: float
    min_duration_ms: float
    max_duration_ms: float
    total_calls: int
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_rate: float
    optimization_opportunities: List[str]
    bottlenecks: List[str]
    recommended_optimizations: List[str]


class IntelligentCache:
    """Advanced caching system with multiple strategies and automatic optimization"""
    
    def __init__(
        self,
        max_size_mb: int = 1024,
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        ttl_seconds: Optional[float] = None
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.strategy = strategy
        self.ttl_seconds = ttl_seconds
        
        # Cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'current_size_bytes': 0,
            'total_requests': 0
        }
        
        # Adaptive strategy parameters
        self._strategy_performance = {
            CacheStrategy.LRU: {'hit_rate': 0.0, 'samples': 0},
            CacheStrategy.LFU: {'hit_rate': 0.0, 'samples': 0},
            CacheStrategy.TIME_AWARE: {'hit_rate': 0.0, 'samples': 0}
        }
        
        # Background optimization task
        self._optimization_task = None
        self._running = False
        
    async def start(self):
        """Start background optimization tasks"""
        self._running = True
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        logger.info(f"Started intelligent cache with {self.strategy.value} strategy")
        
    async def stop(self):
        """Stop background tasks and cleanup"""
        self._running = False
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
                
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with intelligent access tracking"""
        with self._lock:
            self.stats['total_requests'] += 1
            
            if key in self._cache:
                entry = self._cache[key]
                
                # Check TTL expiration
                if self.ttl_seconds and time.time() - entry.creation_time > self.ttl_seconds:
                    self._remove_entry(key)
                    self.stats['misses'] += 1
                    return None
                    
                # Update access statistics
                entry.update_access()
                
                # Move to end for LRU strategy
                self._cache.move_to_end(key)
                
                self.stats['hits'] += 1
                return entry.value
            else:
                self.stats['misses'] += 1
                return None
                
    def put(
        self, 
        key: str, 
        value: Any, 
        computation_time: float = 0.0,
        semantic_hash: Optional[str] = None,
        dependencies: Optional[List[str]] = None
    ):
        """Put value in cache with metadata"""
        
        with self._lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 1024  # Default estimate
                
            current_time = time.time()
            
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                access_count=1,
                last_access=current_time,
                creation_time=current_time,
                computation_time=computation_time,
                access_frequency=1.0,
                semantic_hash=semantic_hash,
                dependencies=dependencies or []
            )
            
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self.stats['current_size_bytes'] -= old_entry.size_bytes
                
            # Add new entry
            self._cache[key] = entry
            self.stats['current_size_bytes'] += size_bytes
            
            # Evict if necessary
            self._evict_if_necessary()
            
    def _evict_if_necessary(self):
        """Evict entries based on current strategy"""
        
        while self.stats['current_size_bytes'] > self.max_size_bytes and self._cache:
            if self.strategy == CacheStrategy.LRU:
                # Remove least recently used
                key_to_remove = next(iter(self._cache))
            elif self.strategy == CacheStrategy.LFU:
                # Remove least frequently used
                key_to_remove = min(self._cache.keys(), 
                                   key=lambda k: self._cache[k].access_count)
            elif self.strategy == CacheStrategy.ADAPTIVE:
                # Use best performing strategy
                key_to_remove = self._adaptive_eviction()
            elif self.strategy == CacheStrategy.TIME_AWARE:
                # Remove based on age and access pattern
                key_to_remove = self._time_aware_eviction()
            else:
                key_to_remove = next(iter(self._cache))
                
            self._remove_entry(key_to_remove)
            
    def _remove_entry(self, key: str):
        """Remove cache entry and update statistics"""
        if key in self._cache:
            entry = self._cache.pop(key)
            self.stats['current_size_bytes'] -= entry.size_bytes
            self.stats['evictions'] += 1
            
    def _adaptive_eviction(self) -> str:
        """Adaptive eviction based on value scores"""
        return min(self._cache.keys(), key=lambda k: self._cache[k].get_value_score())
        
    def _time_aware_eviction(self) -> str:
        """Time-aware eviction considering access patterns"""
        current_time = time.time()
        
        def time_score(key):
            entry = self._cache[key]
            age = current_time - entry.creation_time
            recency = current_time - entry.last_access
            return age + recency - entry.access_frequency * 100
            
        return max(self._cache.keys(), key=time_score)
        
    async def _optimization_loop(self):
        """Background optimization loop"""
        while self._running:
            try:
                await self._optimize_cache()
                await asyncio.sleep(60.0)  # Optimize every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache optimization: {e}")
                await asyncio.sleep(30.0)
                
    async def _optimize_cache(self):
        """Perform cache optimization"""
        with self._lock:
            # Clean up expired entries
            if self.ttl_seconds:
                current_time = time.time()
                expired_keys = [
                    key for key, entry in self._cache.items()
                    if current_time - entry.creation_time > self.ttl_seconds
                ]
                for key in expired_keys:
                    self._remove_entry(key)
                    
            # Update adaptive strategy if needed
            if self.strategy == CacheStrategy.ADAPTIVE:
                self._update_adaptive_strategy()
                
            # Garbage collect if memory usage is high
            if self.stats['current_size_bytes'] > self.max_size_bytes * 0.8:
                gc.collect()
                
    def _update_adaptive_strategy(self):
        """Update adaptive caching strategy based on performance"""
        if self.stats['total_requests'] < 100:
            return  # Need more data
            
        hit_rate = self.stats['hits'] / self.stats['total_requests']
        
        # This is a simplified adaptive mechanism
        # In practice, you'd run A/B tests with different strategies
        if hit_rate < 0.7:  # Poor performance
            logger.info("Adapting cache strategy due to low hit rate")
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self._lock:
            total_requests = self.stats['total_requests']
            hit_rate = self.stats['hits'] / max(1, total_requests)
            
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'miss_rate': 1.0 - hit_rate,
                'utilization': self.stats['current_size_bytes'] / self.max_size_bytes,
                'avg_entry_size': self.stats['current_size_bytes'] / max(1, len(self._cache)),
                'total_entries': len(self._cache),
                'strategy': self.strategy.value
            }
            
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self.stats['current_size_bytes'] = 0
            self.stats['evictions'] += len(self._cache)


class MemoryManager:
    """Advanced memory management for optimal performance"""
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.memory_pools: Dict[str, List[Any]] = defaultdict(list)
        self._monitoring_task = None
        self._running = False
        
    async def start(self):
        """Start memory monitoring"""
        self._running = True
        self._monitoring_task = asyncio.create_task(self._memory_monitor())
        
    async def stop(self):
        """Stop memory monitoring"""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            
    async def _memory_monitor(self):
        """Monitor memory usage and take action"""
        while self._running:
            try:
                memory_info = psutil.virtual_memory()
                usage_percent = memory_info.percent / 100.0
                
                if usage_percent > self.critical_threshold:
                    await self._emergency_memory_cleanup()
                elif usage_percent > self.warning_threshold:
                    await self._gentle_memory_cleanup()
                    
                await asyncio.sleep(10.0)  # Check every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                await asyncio.sleep(5.0)
                
    async def _gentle_memory_cleanup(self):
        """Gentle memory cleanup"""
        logger.info("Performing gentle memory cleanup")
        gc.collect()
        
        # Clear unused memory pools
        for pool_name, pool in self.memory_pools.items():
            if len(pool) > 10:  # Keep some objects
                self.memory_pools[pool_name] = pool[:5]
                
    async def _emergency_memory_cleanup(self):
        """Emergency memory cleanup"""
        logger.warning("Performing emergency memory cleanup")
        
        # Aggressive garbage collection
        for _ in range(3):
            gc.collect()
            
        # Clear all memory pools
        self.memory_pools.clear()
        
        # Force JAX to clear its cache
        try:
            jax.clear_caches()
        except:
            pass
            
    def get_memory_pool(self, pool_name: str, factory: Callable[[], Any]) -> Any:
        """Get object from memory pool or create new one"""
        pool = self.memory_pools[pool_name]
        
        if pool:
            return pool.pop()
        else:
            return factory()
            
    def return_to_pool(self, pool_name: str, obj: Any):
        """Return object to memory pool for reuse"""
        pool = self.memory_pools[pool_name]
        
        if len(pool) < 50:  # Limit pool size
            pool.append(obj)
            
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        memory_info = psutil.virtual_memory()
        
        return {
            'total_gb': memory_info.total / (1024**3),
            'available_gb': memory_info.available / (1024**3),
            'used_gb': memory_info.used / (1024**3),
            'percent_used': memory_info.percent,
            'memory_pools': {name: len(pool) for name, pool in self.memory_pools.items()},
            'gc_counts': gc.get_count()
        }


class ComputationOptimizer:
    """Advanced computation optimization with JAX compilation and batching"""
    
    def __init__(self):
        self.compiled_functions: Dict[str, Callable] = {}
        self.batch_processors: Dict[str, Any] = {}
        self.optimization_history: Dict[str, List[float]] = defaultdict(list)
        
    def optimize_function(
        self,
        func: Callable,
        optimization_level: OptimizationLevel = OptimizationLevel.MODERATE
    ) -> Callable:
        """Optimize function based on optimization level"""
        
        func_name = func.__name__
        
        if optimization_level == OptimizationLevel.BASIC:
            return self._basic_optimization(func)
        elif optimization_level == OptimizationLevel.MODERATE:
            return self._moderate_optimization(func)
        elif optimization_level == OptimizationLevel.AGGRESSIVE:
            return self._aggressive_optimization(func)
        else:  # RESEARCH_GRADE
            return self._research_grade_optimization(func)
            
    def _basic_optimization(self, func: Callable) -> Callable:
        """Basic optimization with simple caching"""
        
        @lru_cache(maxsize=128)
        @wraps(func)
        def optimized_func(*args, **kwargs):
            return func(*args, **kwargs)
            
        return optimized_func
        
    def _moderate_optimization(self, func: Callable) -> Callable:
        """Moderate optimization with JAX compilation"""
        
        try:
            # Try to JIT compile the function
            jitted_func = jax.jit(func)
            
            @wraps(func)
            def optimized_func(*args, **kwargs):
                start_time = time.time()
                try:
                    result = jitted_func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    self.optimization_history[func.__name__].append(execution_time)
                    return result
                except:
                    # Fallback to original function
                    return func(*args, **kwargs)
                    
            return optimized_func
            
        except:
            # JAX compilation failed, use basic optimization
            return self._basic_optimization(func)
            
    def _aggressive_optimization(self, func: Callable) -> Callable:
        """Aggressive optimization with vectorization and batching"""
        
        try:
            # Try vectorization
            vectorized_func = jax.vmap(func)
            jitted_func = jax.jit(vectorized_func)
            
            @wraps(func)
            def optimized_func(*args, **kwargs):
                start_time = time.time()
                
                # Check if inputs can be batched
                batch_size = kwargs.get('batch_size', 1)
                
                if batch_size > 1:
                    # Use vectorized version
                    try:
                        result = jitted_func(*args, **kwargs)
                        execution_time = time.time() - start_time
                        self.optimization_history[func.__name__].append(execution_time)
                        return result
                    except:
                        pass
                        
                # Fallback to single execution
                return func(*args, **kwargs)
                
            return optimized_func
            
        except:
            return self._moderate_optimization(func)
            
    def _research_grade_optimization(self, func: Callable) -> Callable:
        """Research-grade optimization with all available techniques"""
        
        try:
            # Multi-device compilation if available
            if len(jax.devices()) > 1:
                pmapped_func = jax.pmap(func)
                
                @wraps(func)
                def optimized_func(*args, **kwargs):
                    start_time = time.time()
                    
                    # Try parallel execution across devices
                    try:
                        # Replicate inputs across devices
                        replicated_args = jax.device_put_replicated(args[0], jax.devices())
                        result = pmapped_func(replicated_args, **kwargs)
                        
                        # Average results from all devices
                        if isinstance(result, jnp.ndarray):
                            result = jnp.mean(result, axis=0)
                            
                        execution_time = time.time() - start_time
                        self.optimization_history[func.__name__].append(execution_time)
                        return result
                    except:
                        # Fallback to aggressive optimization
                        return self._aggressive_optimization(func)(*args, **kwargs)
                        
                return optimized_func
            else:
                return self._aggressive_optimization(func)
                
        except:
            return self._aggressive_optimization(func)
            
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        
        stats = {}
        
        for func_name, times in self.optimization_history.items():
            if times:
                stats[func_name] = {
                    'total_calls': len(times),
                    'avg_time_ms': np.mean(times) * 1000,
                    'min_time_ms': np.min(times) * 1000,
                    'max_time_ms': np.max(times) * 1000,
                    'total_time_s': np.sum(times)
                }
                
        return {
            'function_stats': stats,
            'compiled_functions': len(self.compiled_functions),
            'total_optimized_calls': sum(len(times) for times in self.optimization_history.values())
        }


class AdvancedPerformanceOptimizer:
    """Main performance optimizer coordinating all optimization subsystems"""
    
    def __init__(
        self,
        cache_size_mb: int = 1024,
        optimization_level: OptimizationLevel = OptimizationLevel.MODERATE,
        enable_monitoring: bool = True
    ):
        self.optimization_level = optimization_level
        
        # Initialize subsystems
        self.cache = IntelligentCache(
            max_size_mb=cache_size_mb,
            strategy=CacheStrategy.ADAPTIVE
        )
        self.memory_manager = MemoryManager()
        self.computation_optimizer = ComputationOptimizer()
        
        # Performance monitoring
        self.enable_monitoring = enable_monitoring
        self.performance_profiles: Dict[str, PerformanceProfile] = {}
        
        # Background tasks
        self._monitoring_task = None
        self._running = False
        
    async def start(self):
        """Start all optimization subsystems"""
        self._running = True
        
        await self.cache.start()
        await self.memory_manager.start()
        
        if self.enable_monitoring:
            self._monitoring_task = asyncio.create_task(self._performance_monitor())
            
        logger.info(f"Advanced performance optimizer started with {self.optimization_level.value} level")
        
    async def stop(self):
        """Stop all optimization subsystems"""
        self._running = False
        
        await self.cache.stop()
        await self.memory_manager.stop()
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            
    @asynccontextmanager
    async def optimize_operation(self, operation_name: str, **metadata):
        """Context manager for optimizing operations with comprehensive monitoring"""
        
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        
        try:
            yield
            
        finally:
            # Record performance metrics
            end_time = time.time()
            end_memory = psutil.virtual_memory().used
            
            duration_ms = (end_time - start_time) * 1000
            memory_delta_mb = (end_memory - start_memory) / (1024 * 1024)
            
            # Update performance profile
            if operation_name in self.performance_profiles:
                profile = self.performance_profiles[operation_name]
                
                # Update rolling averages
                total_calls = profile.total_calls + 1
                profile.avg_duration_ms = (
                    (profile.avg_duration_ms * profile.total_calls + duration_ms) / total_calls
                )
                profile.min_duration_ms = min(profile.min_duration_ms, duration_ms)
                profile.max_duration_ms = max(profile.max_duration_ms, duration_ms)
                profile.total_calls = total_calls
                profile.memory_usage_mb = memory_delta_mb
                
            else:
                # Create new profile
                cache_stats = self.cache.get_statistics()
                
                self.performance_profiles[operation_name] = PerformanceProfile(
                    operation_name=operation_name,
                    avg_duration_ms=duration_ms,
                    min_duration_ms=duration_ms,
                    max_duration_ms=duration_ms,
                    total_calls=1,
                    memory_usage_mb=memory_delta_mb,
                    cpu_usage_percent=psutil.cpu_percent(),
                    cache_hit_rate=cache_stats['hit_rate'],
                    optimization_opportunities=[],
                    bottlenecks=[],
                    recommended_optimizations=[]
                )
                
    def cached_computation(self, cache_key: str, ttl_seconds: Optional[float] = None):
        """Decorator for cached computations"""
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate unique cache key
                full_key = f"{cache_key}_{hash(str(args) + str(sorted(kwargs.items())))}"
                
                # Try cache first
                cached_result = self.cache.get(full_key)
                if cached_result is not None:
                    return cached_result
                    
                # Compute result
                start_time = time.time()
                result = func(*args, **kwargs)
                computation_time = time.time() - start_time
                
                # Cache result
                self.cache.put(
                    key=full_key,
                    value=result,
                    computation_time=computation_time
                )
                
                return result
                
            return wrapper
        return decorator
        
    def optimize_function(
        self, 
        func: Callable,
        cache_key: Optional[str] = None,
        optimization_level: Optional[OptimizationLevel] = None
    ) -> Callable:
        """Comprehensively optimize a function"""
        
        opt_level = optimization_level or self.optimization_level
        
        # Apply computation optimization
        optimized_func = self.computation_optimizer.optimize_function(func, opt_level)
        
        # Add caching if requested
        if cache_key:
            optimized_func = self.cached_computation(cache_key)(optimized_func)
            
        return optimized_func
        
    async def _performance_monitor(self):
        """Background performance monitoring and optimization"""
        
        while self._running:
            try:
                await self._analyze_performance()
                await self._suggest_optimizations()
                await asyncio.sleep(30.0)  # Monitor every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(10.0)
                
    async def _analyze_performance(self):
        """Analyze current performance and identify issues"""
        
        for operation_name, profile in self.performance_profiles.items():
            # Identify bottlenecks
            profile.bottlenecks.clear()
            
            if profile.avg_duration_ms > 1000:  # Slow operations
                profile.bottlenecks.append("High average execution time")
                
            if profile.memory_usage_mb > 100:  # High memory usage
                profile.bottlenecks.append("High memory usage")
                
            if profile.cache_hit_rate < 0.5:  # Poor cache performance
                profile.bottlenecks.append("Low cache hit rate")
                
            # Identify optimization opportunities
            profile.optimization_opportunities.clear()
            
            if profile.total_calls > 10 and profile.avg_duration_ms > 100:
                profile.optimization_opportunities.append("Candidate for JIT compilation")
                
            if profile.total_calls > 100 and profile.cache_hit_rate < 0.7:
                profile.optimization_opportunities.append("Improve caching strategy")
                
    async def _suggest_optimizations(self):
        """Generate optimization recommendations"""
        
        for operation_name, profile in self.performance_profiles.items():
            profile.recommended_optimizations.clear()
            
            if "High average execution time" in profile.bottlenecks:
                if self.optimization_level.value != OptimizationLevel.RESEARCH_GRADE.value:
                    profile.recommended_optimizations.append(
                        "Increase optimization level to research_grade"
                    )
                profile.recommended_optimizations.append(
                    "Consider parallel/distributed execution"
                )
                
            if "Low cache hit rate" in profile.bottlenecks:
                profile.recommended_optimizations.append(
                    "Implement semantic caching for better hit rates"
                )
                profile.recommended_optimizations.append(
                    "Increase cache size or adjust eviction strategy"
                )
                
            if "High memory usage" in profile.bottlenecks:
                profile.recommended_optimizations.append(
                    "Implement memory pooling for object reuse"
                )
                profile.recommended_optimizations.append(
                    "Consider streaming/chunked processing"
                )
                
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        
        return {
            'cache_stats': self.cache.get_statistics(),
            'memory_stats': self.memory_manager.get_memory_stats(),
            'computation_stats': self.computation_optimizer.get_optimization_stats(),
            'performance_profiles': {
                name: {
                    'avg_duration_ms': profile.avg_duration_ms,
                    'total_calls': profile.total_calls,
                    'memory_usage_mb': profile.memory_usage_mb,
                    'cache_hit_rate': profile.cache_hit_rate,
                    'bottlenecks': profile.bottlenecks,
                    'optimizations': profile.recommended_optimizations
                }
                for name, profile in self.performance_profiles.items()
            },
            'optimization_level': self.optimization_level.value,
            'system_metrics': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'available_devices': len(jax.devices()),
                'device_types': [d.device_kind for d in jax.devices()]
            }
        }
        
    def generate_optimization_report(self) -> str:
        """Generate comprehensive optimization report"""
        
        stats = self.get_comprehensive_stats()
        
        report_lines = [
            "# Advanced Performance Optimization Report",
            f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Optimization Level: {self.optimization_level.value}",
            "",
            "## Cache Performance",
            f"Hit Rate: {stats['cache_stats']['hit_rate']:.2%}",
            f"Utilization: {stats['cache_stats']['utilization']:.2%}",
            f"Total Entries: {stats['cache_stats']['total_entries']}",
            "",
            "## Memory Usage",
            f"Used: {stats['memory_stats']['used_gb']:.1f}GB / {stats['memory_stats']['total_gb']:.1f}GB",
            f"Usage: {stats['memory_stats']['percent_used']:.1f}%",
            "",
            "## Operation Performance"
        ]
        
        for op_name, profile in stats['performance_profiles'].items():
            report_lines.extend([
                f"### {op_name}",
                f"- Average Duration: {profile['avg_duration_ms']:.1f}ms",
                f"- Total Calls: {profile['total_calls']}",
                f"- Cache Hit Rate: {profile['cache_hit_rate']:.2%}",
                f"- Bottlenecks: {', '.join(profile['bottlenecks']) if profile['bottlenecks'] else 'None'}",
                f"- Recommended Optimizations: {', '.join(profile['optimizations']) if profile['optimizations'] else 'None'}",
                ""
            ])
            
        return "\n".join(report_lines)


# Global optimizer instance
performance_optimizer = AdvancedPerformanceOptimizer()

# Convenience decorators
def optimize_causal_computation(
    cache_key: Optional[str] = None,
    optimization_level: OptimizationLevel = OptimizationLevel.MODERATE
):
    """Decorator for optimizing causal computations"""
    def decorator(func):
        return performance_optimizer.optimize_function(
            func, cache_key=cache_key, optimization_level=optimization_level
        )
    return decorator


def cached_result(cache_key: str, ttl_seconds: Optional[float] = None):
    """Decorator for caching function results"""
    return performance_optimizer.cached_computation(cache_key, ttl_seconds)


@asynccontextmanager
async def monitor_performance(operation_name: str, **metadata):
    """Context manager for performance monitoring"""
    async with performance_optimizer.optimize_operation(operation_name, **metadata):
        yield


# Export main classes
__all__ = [
    "AdvancedPerformanceOptimizer",
    "IntelligentCache",
    "MemoryManager",
    "ComputationOptimizer",
    "OptimizationLevel",
    "CacheStrategy",
    "performance_optimizer",
    "optimize_causal_computation",
    "cached_result",
    "monitor_performance"
]
