import { useState, useEffect, useCallback, useRef, useMemo } from 'react'
import { debounce } from '../utils'
import { metrics } from '../utils/metrics'

// Performance monitoring hook
export function usePerformanceMonitor(componentName: string) {
  const startTime = useRef<number>(Date.now())
  const renderCount = useRef<number>(0)
  const [performanceData, setPerformanceData] = useState({
    averageRenderTime: 0,
    renderCount: 0,
    memoryUsage: 0
  })

  useEffect(() => {
    renderCount.current += 1
    const renderTime = Date.now() - startTime.current
    
    // Track component render performance
    metrics.trackRenderTime(componentName, renderTime)
    
    // Update performance data
    setPerformanceData(prev => ({
      averageRenderTime: (prev.averageRenderTime * (prev.renderCount - 1) + renderTime) / prev.renderCount,
      renderCount: renderCount.current,
      memoryUsage: (performance as any).memory?.usedJSHeapSize || 0
    }))
    
    startTime.current = Date.now()
  })

  return performanceData
}

// Optimized data fetching hook with caching
export function useOptimizedFetch<T>(
  url: string,
  options?: RequestInit,
  cacheTime: number = 5 * 60 * 1000 // 5 minutes default
) {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<Error | null>(null)
  const cache = useRef<Map<string, { data: T; timestamp: number }>>(new Map())
  const abortController = useRef<AbortController | null>(null)

  const fetchData = useCallback(async () => {
    const cacheKey = `${url}:${JSON.stringify(options)}`
    const cached = cache.current.get(cacheKey)
    
    // Return cached data if still valid
    if (cached && Date.now() - cached.timestamp < cacheTime) {
      setData(cached.data)
      return cached.data
    }

    // Abort previous request if still pending
    if (abortController.current) {
      abortController.current.abort()
    }

    abortController.current = new AbortController()
    setLoading(true)
    setError(null)

    try {
      const response = await fetch(url, {
        ...options,
        signal: abortController.current.signal
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const result = await response.json()
      
      // Cache the result
      cache.current.set(cacheKey, { data: result, timestamp: Date.now() })
      
      setData(result)
      return result
    } catch (err) {
      if (err instanceof Error && err.name !== 'AbortError') {
        setError(err)
        metrics.trackError('fetch_error', err.message, 'useOptimizedFetch')
      }
    } finally {
      setLoading(false)
    }
  }, [url, options, cacheTime])

  useEffect(() => {
    fetchData()
    
    return () => {
      if (abortController.current) {
        abortController.current.abort()
      }
    }
  }, [fetchData])

  const invalidateCache = useCallback(() => {
    cache.current.clear()
    fetchData()
  }, [fetchData])

  return { data, loading, error, refetch: fetchData, invalidateCache }
}

// Virtual scrolling hook for large lists
export function useVirtualScrolling<T>(
  items: T[],
  itemHeight: number,
  containerHeight: number,
  overscan: number = 5
) {
  const [scrollTop, setScrollTop] = useState(0)
  
  const visibleRange = useMemo(() => {
    const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan)
    const endIndex = Math.min(
      items.length - 1,
      Math.ceil((scrollTop + containerHeight) / itemHeight) + overscan
    )
    
    return { startIndex, endIndex }
  }, [scrollTop, itemHeight, containerHeight, items.length, overscan])

  const visibleItems = useMemo(() => {
    return items.slice(visibleRange.startIndex, visibleRange.endIndex + 1).map((item, index) => ({
      item,
      index: visibleRange.startIndex + index
    }))
  }, [items, visibleRange])

  const totalHeight = items.length * itemHeight
  const offsetY = visibleRange.startIndex * itemHeight

  const handleScroll = useCallback((event: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(event.currentTarget.scrollTop)
  }, [])

  return {
    visibleItems,
    totalHeight,
    offsetY,
    handleScroll
  }
}

// Debounced state hook for performance
export function useDebouncedState<T>(
  initialValue: T,
  delay: number = 300
): [T, T, (value: T) => void] {
  const [immediateValue, setImmediateValue] = useState<T>(initialValue)
  const [debouncedValue, setDebouncedValue] = useState<T>(initialValue)

  const debouncedSetValue = useMemo(
    () => debounce((value: T) => setDebouncedValue(value), delay),
    [delay]
  )

  const setValue = useCallback((value: T) => {
    setImmediateValue(value)
    debouncedSetValue(value)
  }, [debouncedSetValue])

  return [immediateValue, debouncedValue, setValue]
}

// Intersection Observer hook for lazy loading
export function useIntersectionObserver(
  elementRef: React.RefObject<Element>,
  options?: IntersectionObserverInit
) {
  const [isIntersecting, setIsIntersecting] = useState(false)
  const [hasIntersected, setHasIntersected] = useState(false)

  useEffect(() => {
    const element = elementRef.current
    if (!element) return

    const observer = new IntersectionObserver(([entry]) => {
      setIsIntersecting(entry.isIntersecting)
      if (entry.isIntersecting) {
        setHasIntersected(true)
      }
    }, options)

    observer.observe(element)

    return () => {
      observer.unobserve(element)
    }
  }, [elementRef, options])

  return { isIntersecting, hasIntersected }
}

// Optimized computation hook with memoization
export function useOptimizedComputation<T, R>(
  computation: (input: T) => R,
  input: T,
  dependencies: any[] = []
): R {
  const cache = useRef<Map<string, R>>(new Map())
  
  return useMemo(() => {
    const key = JSON.stringify(input)
    
    if (cache.current.has(key)) {
      return cache.current.get(key)!
    }

    const startTime = performance.now()
    const result = computation(input)
    const computationTime = performance.now() - startTime
    
    // Track computation performance
    metrics.histogram('computation_time', 'Time spent on computation', {
      computation_type: computation.name || 'anonymous'
    }, computationTime)
    
    // Cache result with size limit
    if (cache.current.size > 100) {
      const firstKey = cache.current.keys().next().value
      cache.current.delete(firstKey)
    }
    
    cache.current.set(key, result)
    return result
  }, [computation, input, ...dependencies])
}

// WebWorker hook for heavy computations
export function useWebWorker<T, R>(
  workerScript: string,
  data: T,
  enabled: boolean = true
) {
  const [result, setResult] = useState<R | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const workerRef = useRef<Worker | null>(null)

  useEffect(() => {
    if (!enabled || !data) return

    setLoading(true)
    setError(null)

    try {
      // Create worker if not exists
      if (!workerRef.current) {
        workerRef.current = new Worker(workerScript)
      }

      const worker = workerRef.current

      const handleMessage = (event: MessageEvent<R>) => {
        setResult(event.data)
        setLoading(false)
      }

      const handleError = (event: ErrorEvent) => {
        setError(new Error(event.message))
        setLoading(false)
      }

      worker.addEventListener('message', handleMessage)
      worker.addEventListener('error', handleError)

      // Send data to worker
      worker.postMessage(data)

      return () => {
        worker.removeEventListener('message', handleMessage)
        worker.removeEventListener('error', handleError)
      }
    } catch (err) {
      setError(err as Error)
      setLoading(false)
    }
  }, [workerScript, data, enabled])

  useEffect(() => {
    return () => {
      if (workerRef.current) {
        workerRef.current.terminate()
        workerRef.current = null
      }
    }
  }, [])

  return { result, loading, error }
}

// Resource preloading hook
export function useResourcePreloader(resources: string[]) {
  const [loadedResources, setLoadedResources] = useState<Set<string>>(new Set())
  const [failedResources, setFailedResources] = useState<Set<string>>(new Set())

  useEffect(() => {
    const preloadResource = (url: string) => {
      return new Promise<void>((resolve, reject) => {
        if (url.endsWith('.js')) {
          const script = document.createElement('script')
          script.src = url
          script.onload = () => resolve()
          script.onerror = () => reject(new Error(`Failed to load script: ${url}`))
          document.head.appendChild(script)
        } else if (url.endsWith('.css')) {
          const link = document.createElement('link')
          link.rel = 'stylesheet'
          link.href = url
          link.onload = () => resolve()
          link.onerror = () => reject(new Error(`Failed to load stylesheet: ${url}`))
          document.head.appendChild(link)
        } else {
          // Preload other resources (images, etc.)
          const link = document.createElement('link')
          link.rel = 'preload'
          link.href = url
          link.as = 'fetch'
          link.onload = () => resolve()
          link.onerror = () => reject(new Error(`Failed to preload: ${url}`))
          document.head.appendChild(link)
        }
      })
    }

    Promise.allSettled(resources.map(preloadResource)).then(results => {
      results.forEach((result, index) => {
        const resource = resources[index]
        if (result.status === 'fulfilled') {
          setLoadedResources(prev => new Set(prev).add(resource))
        } else {
          setFailedResources(prev => new Set(prev).add(resource))
        }
      })
    })
  }, [resources])

  return { loadedResources, failedResources }
}

// Optimized event handler hook
export function useOptimizedEventHandler<T extends (...args: any[]) => any>(
  handler: T,
  delay: number = 0,
  options: { leading?: boolean; trailing?: boolean } = { trailing: true }
): T {
  const debouncedHandler = useMemo(() => {
    if (delay === 0) return handler
    
    return debounce(handler, delay) as T
  }, [handler, delay])

  return debouncedHandler
}

// Memory usage monitoring hook
export function useMemoryMonitor() {
  const [memoryInfo, setMemoryInfo] = useState({
    usedJSHeapSize: 0,
    totalJSHeapSize: 0,
    jsHeapSizeLimit: 0
  })

  useEffect(() => {
    const updateMemoryInfo = () => {
      if ((performance as any).memory) {
        const memory = (performance as any).memory
        setMemoryInfo({
          usedJSHeapSize: memory.usedJSHeapSize,
          totalJSHeapSize: memory.totalJSHeapSize,
          jsHeapSizeLimit: memory.jsHeapSizeLimit
        })
      }
    }

    updateMemoryInfo()
    const interval = setInterval(updateMemoryInfo, 5000) // Update every 5 seconds

    return () => clearInterval(interval)
  }, [])

  return memoryInfo
}

// Batch operation hook for API calls
export function useBatchOperations<T, R>(
  operation: (items: T[]) => Promise<R[]>,
  batchSize: number = 10,
  delay: number = 100
) {
  const [queue, setQueue] = useState<T[]>([])
  const [results, setResults] = useState<Map<string, R>>(new Map())
  const [loading, setLoading] = useState(false)
  const timeoutRef = useRef<NodeJS.Timeout>()

  const addToQueue = useCallback((item: T) => {
    setQueue(prev => [...prev, item])
  }, [])

  const processBatch = useCallback(async () => {
    if (queue.length === 0) return

    setLoading(true)
    try {
      const batch = queue.slice(0, batchSize)
      const batchResults = await operation(batch)
      
      const newResults = new Map(results)
      batch.forEach((item, index) => {
        const key = JSON.stringify(item)
        newResults.set(key, batchResults[index])
      })
      
      setResults(newResults)
      setQueue(prev => prev.slice(batchSize))
    } catch (error) {
      metrics.trackError('batch_operation_error', (error as Error).message, 'useBatchOperations')
    } finally {
      setLoading(false)
    }
  }, [queue, operation, batchSize, results])

  useEffect(() => {
    if (queue.length > 0) {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
      
      timeoutRef.current = setTimeout(processBatch, delay)
    }

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [queue, processBatch, delay])

  const getResult = useCallback((item: T): R | undefined => {
    const key = JSON.stringify(item)
    return results.get(key)
  }, [results])

  return {
    addToQueue,
    getResult,
    processBatch,
    loading,
    queueLength: queue.length
  }
}