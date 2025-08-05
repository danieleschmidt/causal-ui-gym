"""
Advanced caching system for Causal UI Gym backend.

This module provides multi-level caching with Redis, in-memory caching,
and intelligent cache invalidation strategies.
"""

import asyncio
import json
import hashlib
import logging
import pickle
import time
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from functools import wraps
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import weakref

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached item with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: List[str] = None
    size_bytes: int = 0

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.last_accessed is None:
            self.last_accessed = self.created_at
        if self.size_bytes == 0:
            self.size_bytes = len(pickle.dumps(self.value))

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def access(self) -> None:
        """Mark entry as accessed."""
        self.access_count += 1
        self.last_accessed = datetime.now()


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all values from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass


class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, max_size_bytes: int = 100 * 1024 * 1024):
        self.max_size = max_size
        self.max_size_bytes = max_size_bytes
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.total_size_bytes = 0
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        async with self._lock:
            entry = self.cache.get(key)
            if entry is None:
                return None
            
            if entry.is_expired():
                await self._remove_entry(key)
                return None
            
            entry.access()
            self._update_access_order(key)
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        async with self._lock:
            now = datetime.now()
            expires_at = now + timedelta(seconds=ttl) if ttl else None
            
            # Remove existing entry if present
            if key in self.cache:
                await self._remove_entry(key)
            
            # Create new entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                expires_at=expires_at
            )
            
            # Check if we need to evict entries
            await self._ensure_capacity(entry.size_bytes)
            
            # Add new entry
            self.cache[key] = entry
            self.access_order.append(key)
            self.total_size_bytes += entry.size_bytes
            
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        async with self._lock:
            if key in self.cache:
                await self._remove_entry(key)
                return True
            return False
    
    async def clear(self) -> bool:
        """Clear all values from memory cache."""
        async with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.total_size_bytes = 0
            return True
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        async with self._lock:
            entry = self.cache.get(key)
            if entry is None:
                return False
            
            if entry.is_expired():
                await self._remove_entry(key)
                return False
            
            return True
    
    async def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self.cache:
            entry = self.cache[key]
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
            self.total_size_bytes -= entry.size_bytes
    
    def _update_access_order(self, key: str) -> None:
        """Update access order for LRU."""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    async def _ensure_capacity(self, new_entry_size: int) -> None:
        """Ensure cache has capacity for new entry."""
        # Evict by count
        while len(self.cache) >= self.max_size and self.access_order:
            oldest_key = self.access_order[0]
            await self._remove_entry(oldest_key)
        
        # Evict by size
        while (self.total_size_bytes + new_entry_size > self.max_size_bytes and 
               self.access_order):
            oldest_key = self.access_order[0]
            await self._remove_entry(oldest_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'entries': len(self.cache),
            'max_size': self.max_size,
            'size_bytes': self.total_size_bytes,
            'max_size_bytes': self.max_size_bytes,
            'utilization': len(self.cache) / self.max_size if self.max_size > 0 else 0,
            'memory_utilization': self.total_size_bytes / self.max_size_bytes if self.max_size_bytes > 0 else 0
        }


class RedisCacheBackend(CacheBackend):
    """Redis cache backend for distributed caching."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", prefix: str = "causal_ui:"):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available. Install redis package.")
        
        self.redis_url = redis_url
        self.prefix = prefix
        self.redis_client: Optional[redis.Redis] = None
    
    async def _get_client(self) -> redis.Redis:
        """Get Redis client, creating if necessary."""
        if self.redis_client is None:
            self.redis_client = redis.from_url(self.redis_url)
        return self.redis_client
    
    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            client = await self._get_client()
            data = await client.get(self._make_key(key))
            if data is None:
                return None
            
            # Deserialize the data
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Redis GET error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        try:
            client = await self._get_client()
            data = pickle.dumps(value)
            
            if ttl:
                result = await client.setex(self._make_key(key), ttl, data)
            else:
                result = await client.set(self._make_key(key), data)
            
            return bool(result)
        except Exception as e:
            logger.error(f"Redis SET error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        try:
            client = await self._get_client()
            result = await client.delete(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Redis DELETE error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all values with prefix from Redis cache."""
        try:
            client = await self._get_client()
            keys = await client.keys(f"{self.prefix}*")
            if keys:
                await client.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Redis CLEAR error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        try:
            client = await self._get_client()
            result = await client.exists(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Redis EXISTS error: {e}")
            return False
    
    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()


class MultiLevelCache:
    """Multi-level cache with L1 (memory) and L2 (Redis) backends."""
    
    def __init__(
        self,
        l1_backend: Optional[CacheBackend] = None,
        l2_backend: Optional[CacheBackend] = None,
        enable_l2: bool = True
    ):
        self.l1 = l1_backend or MemoryCacheBackend()
        self.l2 = l2_backend if enable_l2 else None
        self.hit_stats = {
            'l1_hits': 0,
            'l2_hits': 0,
            'misses': 0,
            'total_requests': 0
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache."""
        self.hit_stats['total_requests'] += 1
        
        # Try L1 cache first
        value = await self.l1.get(key)
        if value is not None:
            self.hit_stats['l1_hits'] += 1
            return value
        
        # Try L2 cache if available
        if self.l2:
            value = await self.l2.get(key)
            if value is not None:
                self.hit_stats['l2_hits'] += 1
                # Populate L1 cache
                await self.l1.set(key, value)
                return value
        
        self.hit_stats['misses'] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in multi-level cache."""
        # Set in L1
        l1_success = await self.l1.set(key, value, ttl)
        
        # Set in L2 if available
        l2_success = True
        if self.l2:
            l2_success = await self.l2.set(key, value, ttl)
        
        return l1_success and l2_success
    
    async def delete(self, key: str) -> bool:
        """Delete value from multi-level cache."""
        l1_success = await self.l1.delete(key)
        
        l2_success = True
        if self.l2:
            l2_success = await self.l2.delete(key)
        
        return l1_success or l2_success
    
    async def clear(self) -> bool:
        """Clear all values from multi-level cache."""
        l1_success = await self.l1.clear()
        
        l2_success = True
        if self.l2:
            l2_success = await self.l2.clear()
        
        return l1_success and l2_success
    
    def get_hit_rate(self) -> Dict[str, float]:
        """Get cache hit rate statistics."""
        total = self.hit_stats['total_requests']
        if total == 0:
            return {'l1_hit_rate': 0.0, 'l2_hit_rate': 0.0, 'overall_hit_rate': 0.0}
        
        return {
            'l1_hit_rate': self.hit_stats['l1_hits'] / total,
            'l2_hit_rate': self.hit_stats['l2_hits'] / total,
            'overall_hit_rate': (self.hit_stats['l1_hits'] + self.hit_stats['l2_hits']) / total,
            'miss_rate': self.hit_stats['misses'] / total
        }


class CacheManager:
    """Advanced cache manager with intelligent caching strategies."""
    
    def __init__(self, cache: Optional[MultiLevelCache] = None):
        self.cache = cache or MultiLevelCache()
        self.tag_registry: Dict[str, set[str]] = {}
        self._weak_refs = weakref.WeakSet()
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from function arguments."""
        key_parts = [prefix]
        
        # Add positional arguments
        for arg in args:
            if hasattr(arg, '__dict__'):
                key_parts.append(str(sorted(arg.__dict__.items())))
            else:
                key_parts.append(str(arg))
        
        # Add keyword arguments
        if kwargs:
            key_parts.append(str(sorted(kwargs.items())))
        
        # Create hash for long keys
        key_string = ':'.join(key_parts)
        if len(key_string) > 200:
            key_string = hashlib.sha256(key_string.encode()).hexdigest()
        
        return key_string
    
    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> Any:
        """Get value from cache or set it using factory function."""
        value = await self.cache.get(key)
        if value is not None:
            return value
        
        # Generate value using factory
        value = await factory() if asyncio.iscoroutinefunction(factory) else factory()
        
        # Cache the value
        await self.cache.set(key, value, ttl)
        
        # Register tags
        if tags:
            self._register_tags(key, tags)
        
        return value
    
    async def invalidate_by_tags(self, tags: List[str]) -> int:
        """Invalidate all cache entries with given tags."""
        keys_to_invalidate = set()
        
        for tag in tags:
            if tag in self.tag_registry:
                keys_to_invalidate.update(self.tag_registry[tag])
        
        # Remove keys from cache
        invalidated_count = 0
        for key in keys_to_invalidate:
            if await self.cache.delete(key):
                invalidated_count += 1
                self._unregister_key_from_tags(key)
        
        return invalidated_count
    
    def _register_tags(self, key: str, tags: List[str]) -> None:
        """Register tags for a cache key."""
        for tag in tags:
            if tag not in self.tag_registry:
                self.tag_registry[tag] = set()
            self.tag_registry[tag].add(key)
    
    def _unregister_key_from_tags(self, key: str) -> None:
        """Unregister key from all tags."""
        for tag_keys in self.tag_registry.values():
            tag_keys.discard(key)
    
    def cached(
        self,
        ttl: Optional[int] = None,
        key_prefix: Optional[str] = None,
        tags: Optional[List[str]] = None,
        vary_on: Optional[List[str]] = None
    ):
        """Decorator for caching function results."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                prefix = key_prefix or f"{func.__module__}.{func.__name__}"
                
                # Add vary_on parameters to key generation
                if vary_on:
                    vary_kwargs = {k: kwargs.get(k) for k in vary_on if k in kwargs}
                    cache_key = self._generate_cache_key(prefix, *args, **vary_kwargs)
                else:
                    cache_key = self._generate_cache_key(prefix, *args, **kwargs)
                
                # Try to get from cache
                cached_result = await self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Cache result
                await self.cache.set(cache_key, result, ttl)
                
                # Register tags
                if tags:
                    self._register_tags(cache_key, tags)
                
                return result
            
            return wrapper
        return decorator
    
    async def warm_cache(self, warmup_functions: List[Tuple[Callable, List[Any], Dict[str, Any]]]) -> None:
        """Warm up cache with predefined function calls."""
        tasks = []
        
        for func, args, kwargs in warmup_functions:
            if asyncio.iscoroutinefunction(func):
                tasks.append(func(*args, **kwargs))
            else:
                # Wrap sync function in async
                async def async_wrapper():
                    return func(*args, **kwargs)
                tasks.append(async_wrapper())
        
        # Execute all warmup functions concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        hit_rates = self.cache.get_hit_rate()
        
        stats = {
            'hit_rates': hit_rates,
            'tag_count': len(self.tag_registry),
            'keys_per_tag': {
                tag: len(keys) for tag, keys in self.tag_registry.items()
            }
        }
        
        # Add L1 stats if available
        if hasattr(self.cache.l1, 'get_stats'):
            stats['l1_cache'] = self.cache.l1.get_stats()
        
        return stats


# Global cache manager instance
cache_manager = CacheManager()

# Convenience decorators
def cached(ttl: Optional[int] = None, tags: Optional[List[str]] = None):
    """Convenience decorator using global cache manager."""
    return cache_manager.cached(ttl=ttl, tags=tags)


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments."""
    return cache_manager._generate_cache_key("manual", *args, **kwargs)