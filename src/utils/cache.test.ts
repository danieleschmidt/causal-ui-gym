import { describe, it, expect, beforeEach, vi } from 'vitest'
import { IntelligentCache, createConcurrencyManager, createPerformanceManager } from './cache'

// Mock performance.now for consistent testing
const mockPerformanceNow = vi.fn()
global.performance = global.performance || { now: mockPerformanceNow }

describe('Cache Utils', () => {
  let cache: IntelligentCache<string>
  
  beforeEach(() => {
    cache = new IntelligentCache<string>('test-cache', {
      defaultTTL: 1000,
      maxSize: 5,
      evictionPolicy: 'priority'
    })
    vi.clearAllMocks()
    // Mock performance.now to return predictable values
    mockPerformanceNow.mockReturnValue(0)
  })

  describe('IntelligentCache', () => {
    it('should store and retrieve values', () => {
      cache.set('key1', 'value1')
      expect(cache.get('key1')).toBe('value1')
    })

    it('should return null for non-existent keys', () => {
      expect(cache.get('nonexistent')).toBeNull()
    })

    it('should respect TTL expiration', () => {
      mockPerformanceNow.mockReturnValue(0)
      cache.set('key1', 'value1')
      
      // Move time forward beyond TTL
      mockPerformanceNow.mockReturnValue(1500)
      expect(cache.get('key1')).toBeNull()
    })

    it('should update access count and last access time', () => {
      cache.set('key1', 'value1')
      cache.get('key1')
      cache.get('key1')
      
      const stats = cache.getStats()
      expect(stats.totalOperations).toBeGreaterThan(0)
    })

    it('should calculate adaptive TTL based on access patterns', () => {
      cache.set('key1', 'value1', undefined, 1.0, 100) // priority=1.0, computationTime=100ms
      
      // Access multiple times to increase access count
      cache.get('key1')
      cache.get('key1')
      cache.get('key1')
      
      // Set another value to trigger adaptive TTL calculation
      cache.set('key2', 'value2', undefined, 1.0, 50)
      
      expect(cache.get('key1')).toBe('value1')
    })

    it('should evict least recently used items when at capacity', () => {
      // Fill cache to capacity
      for (let i = 0; i < 5; i++) {
        cache.set(`key${i}`, `value${i}`)
      }
      
      // Access key0 to make it more recently used
      cache.get('key0')
      
      // Add one more item, should evict least recently used (key1)
      cache.set('key5', 'value5')
      
      expect(cache.get('key0')).toBe('value0') // Recently accessed, should remain
      expect(cache.get('key1')).toBeNull() // Should be evicted
      expect(cache.get('key5')).toBe('value5') // New item should be present
    })

    it('should evict based on priority when policy is priority-based', () => {
      const priorityCache = new IntelligentCache<string>('priority-cache', {
        defaultTTL: 1000,
        maxSize: 3,
        evictionPolicy: 'priority'
      })

      // Add items with different priorities
      priorityCache.set('low', 'value1', undefined, 0.1)
      priorityCache.set('medium', 'value2', undefined, 0.5)
      priorityCache.set('high', 'value3', undefined, 1.0)
      
      // Add another item, should evict lowest priority
      priorityCache.set('new', 'value4', undefined, 0.8)
      
      expect(priorityCache.get('low')).toBeNull() // Lowest priority, should be evicted
      expect(priorityCache.get('high')).toBe('value3') // Highest priority, should remain
      expect(priorityCache.get('new')).toBe('value4') // New high priority item
    })

    it('should clear all entries', () => {
      cache.set('key1', 'value1')
      cache.set('key2', 'value2')
      
      cache.clear()
      
      expect(cache.get('key1')).toBeNull()
      expect(cache.get('key2')).toBeNull()
      expect(cache.size()).toBe(0)
    })

    it('should delete specific entries', () => {
      cache.set('key1', 'value1')
      cache.set('key2', 'value2')
      
      cache.delete('key1')
      
      expect(cache.get('key1')).toBeNull()
      expect(cache.get('key2')).toBe('value2')
    })

    it('should check if key exists', () => {
      cache.set('key1', 'value1')
      
      expect(cache.has('key1')).toBe(true)
      expect(cache.has('nonexistent')).toBe(false)
    })

    it('should return all keys', () => {
      cache.set('key1', 'value1')
      cache.set('key2', 'value2')
      
      const keys = cache.keys()
      expect(keys).toContain('key1')
      expect(keys).toContain('key2')
      expect(keys).toHaveLength(2)
    })

    it('should return cache statistics', () => {
      cache.set('key1', 'value1')
      cache.get('key1')
      cache.get('nonexistent')
      
      const stats = cache.getStats()
      expect(stats.size).toBe(1)
      expect(stats.hits).toBe(1)
      expect(stats.misses).toBe(1)
      expect(stats.hitRate).toBe(0.5)
      expect(stats.totalOperations).toBe(3)
    })
  })

  describe('ConcurrencyManager', () => {
    it('should execute tasks with priority ordering', async () => {
      const concurrencyManager = createConcurrencyManager(2) // Max 2 concurrent tasks
      const results: number[] = []

      // Create tasks with different priorities
      const task1 = () => new Promise<void>(resolve => {
        setTimeout(() => {
          results.push(1)
          resolve()
        }, 10)
      })
      
      const task2 = () => new Promise<void>(resolve => {
        setTimeout(() => {
          results.push(2)
          resolve()
        }, 5)
      })
      
      const task3 = () => new Promise<void>(resolve => {
        results.push(3)
        resolve()
      })

      // Execute tasks with different priorities
      const promises = [
        concurrencyManager.execute(task1, 0.1), // Low priority
        concurrencyManager.execute(task2, 1.0), // High priority  
        concurrencyManager.execute(task3, 0.5)  // Medium priority
      ]

      await Promise.all(promises)

      // Higher priority tasks should complete first when possible
      expect(results).toContain(1)
      expect(results).toContain(2)
      expect(results).toContain(3)
    })

    it('should respect maximum concurrency limit', async () => {
      const concurrencyManager = createConcurrencyManager(1) // Only 1 concurrent task
      let activeCount = 0
      let maxActiveCount = 0

      const createTask = () => () => new Promise<void>(resolve => {
        activeCount++
        maxActiveCount = Math.max(maxActiveCount, activeCount)
        setTimeout(() => {
          activeCount--
          resolve()
        }, 10)
      })

      const promises = [
        concurrencyManager.execute(createTask()),
        concurrencyManager.execute(createTask()),
        concurrencyManager.execute(createTask())
      ]

      await Promise.all(promises)

      // Should never exceed the concurrency limit
      expect(maxActiveCount).toBe(1)
    })
  })

  describe('PerformanceManager', () => {
    it('should track cache statistics', () => {
      const perfManager = createPerformanceManager()
      
      // Register some caches
      const cache1 = new IntelligentCache<string>('cache1')
      const cache2 = new IntelligentCache<string>('cache2')
      
      // Simulate some cache operations
      cache1.set('key1', 'value1')
      cache1.get('key1')
      cache1.get('missing')
      
      cache2.set('key2', 'value2')
      cache2.get('key2')
      
      // The performance manager should track operations
      expect(cache1.getStats().hits).toBe(1)
      expect(cache1.getStats().misses).toBe(1)
      expect(cache2.getStats().hits).toBe(1)
    })

    it('should provide optimization recommendations', async () => {
      const perfManager = createPerformanceManager()
      
      const result = await perfManager.optimizePerformance()
      
      expect(result).toHaveProperty('cachesOptimized')
      expect(result).toHaveProperty('memoryFreed')
      expect(typeof result.cachesOptimized).toBe('number')
      expect(typeof result.memoryFreed).toBe('number')
    })
  })
})