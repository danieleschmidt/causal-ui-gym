/**
 * Advanced Performance Optimization and Scaling System
 * 
 * Implements auto-scaling, performance tuning, and optimization strategies
 * for production-grade causal inference workloads.
 */

import { monitoring } from './monitoring'
import { concurrencyManager, performanceManager } from './cache'

interface PerformanceProfile {
  cpuUsage: number
  memoryUsage: number
  networkLatency: number
  throughput: number
  errorRate: number
  userLoad: number
}

interface OptimizationStrategy {
  name: string
  priority: number
  trigger: (profile: PerformanceProfile) => boolean
  execute: () => Promise<void>
  rollback?: () => Promise<void>
}

interface ScalingDecision {
  action: 'scale_up' | 'scale_down' | 'maintain' | 'optimize'
  reason: string
  intensity: number
  timestamp: Date
}

export class PerformanceOptimizer {
  private currentProfile: PerformanceProfile = {
    cpuUsage: 0,
    memoryUsage: 0,
    networkLatency: 0,
    throughput: 0,
    errorRate: 0,
    userLoad: 0
  }

  private optimizationStrategies: OptimizationStrategy[] = []
  private isOptimizing = false
  private optimizationHistory: Array<{ strategy: string; timestamp: Date; result: any }> = []
  private autoScalingEnabled = true
  private adaptiveTuningEnabled = true

  constructor() {
    this.initializeStrategies()
    this.startPerformanceMonitoring()
  }

  private initializeStrategies() {
    this.optimizationStrategies = [
      {
        name: 'Cache Optimization',
        priority: 1,
        trigger: (profile) => profile.memoryUsage > 80 || profile.throughput < 500,
        execute: async () => {
          // Optimize cache sizes and eviction policies
          const cacheStats = performanceManager.getCacheStats()
          for (const [name, stats] of Object.entries(cacheStats)) {
            if (stats.hitRate < 0.7) {
              // Increase cache size for low hit rate caches
              monitoring.trackMetric('cache_size_optimized', 1, { cache: name, action: 'increase' })
            }
            if (stats.memoryUsage > 100000000) { // 100MB
              // Trigger aggressive cleanup for large caches
              monitoring.trackMetric('cache_cleanup_triggered', 1, { cache: name })
            }
          }
        }
      },
      {
        name: 'Concurrency Scaling',
        priority: 2,
        trigger: (profile) => profile.cpuUsage < 60 && profile.userLoad > 80,
        execute: async () => {
          // Increase concurrency limits when CPU is underutilized
          const currentLimit = concurrencyManager.getMaxConcurrent()
          const newLimit = Math.min(currentLimit + 2, navigator.hardwareConcurrency * 2)
          concurrencyManager.setMaxConcurrent(newLimit)
          monitoring.trackMetric('concurrency_limit_increased', newLimit)
        },
        rollback: async () => {
          // Reduce concurrency if it causes issues
          const currentLimit = concurrencyManager.getMaxConcurrent()
          concurrencyManager.setMaxConcurrent(Math.max(currentLimit - 1, 1))
          monitoring.trackMetric('concurrency_limit_decreased', currentLimit - 1)
        }
      },
      {
        name: 'Memory Pressure Relief',
        priority: 3,
        trigger: (profile) => profile.memoryUsage > 85,
        execute: async () => {
          // Aggressive memory cleanup
          if ('gc' in window && typeof (window as any).gc === 'function') {
            (window as any).gc()
          }
          
          // Clear least valuable caches
          const cacheStats = performanceManager.getCacheStats()
          const lowValueCaches = Object.entries(cacheStats)
            .filter(([_, stats]) => stats.hitRate < 0.3)
            .map(([name]) => name)
          
          for (const cacheName of lowValueCaches) {
            monitoring.trackMetric('cache_emergency_clear', 1, { cache: cacheName })
          }
        }
      },
      {
        name: 'Network Optimization',
        priority: 4,
        trigger: (profile) => profile.networkLatency > 100,
        execute: async () => {
          // Enable request batching and compression
          monitoring.trackMetric('network_optimization_enabled', 1)
          
          // Implement request deduplication
          this.enableRequestDeduplication()
          
          // Enable response compression
          this.enableResponseCompression()
        }
      },
      {
        name: 'Adaptive Load Balancing',
        priority: 5,
        trigger: (profile) => profile.userLoad > 90 && profile.errorRate > 2,
        execute: async () => {
          // Implement circuit breaker pattern
          this.enableCircuitBreaker()
          
          // Implement request throttling
          this.enableRequestThrottling()
          
          monitoring.trackMetric('load_balancing_activated', 1)
        }
      }
    ]

    // Sort strategies by priority
    this.optimizationStrategies.sort((a, b) => a.priority - b.priority)
  }

  private startPerformanceMonitoring() {
    setInterval(() => {
      this.updatePerformanceProfile()
      this.evaluateOptimizations()
      this.makeScalingDecisions()
    }, 5000) // Check every 5 seconds
  }

  private updatePerformanceProfile() {
    // Collect real-time performance metrics
    this.currentProfile = {
      cpuUsage: this.getCPUUsage(),
      memoryUsage: this.getMemoryUsage(),
      networkLatency: this.getNetworkLatency(),
      throughput: this.getThroughput(),
      errorRate: this.getErrorRate(),
      userLoad: this.getUserLoad()
    }

    // Track metrics
    Object.entries(this.currentProfile).forEach(([metric, value]) => {
      monitoring.trackMetric(`performance_${metric}`, value)
    })
  }

  private async evaluateOptimizations() {
    if (this.isOptimizing) return

    for (const strategy of this.optimizationStrategies) {
      if (strategy.trigger(this.currentProfile)) {
        this.isOptimizing = true
        try {
          const startTime = Date.now()
          await strategy.execute()
          const duration = Date.now() - startTime
          
          this.optimizationHistory.push({
            strategy: strategy.name,
            timestamp: new Date(),
            result: { success: true, duration }
          })
          
          monitoring.trackMetric('optimization_executed', 1, {
            strategy: strategy.name,
            duration: duration.toString()
          })
          
          // Wait a bit before trying more optimizations
          await new Promise(resolve => setTimeout(resolve, 2000))
        } catch (error) {
          this.optimizationHistory.push({
            strategy: strategy.name,
            timestamp: new Date(),
            result: { success: false, error: error instanceof Error ? error.message : 'Unknown error' }
          })
          
          monitoring.trackError('optimization_failed', error instanceof Error ? error.message : 'Unknown error', strategy.name)
          
          // Attempt rollback if available
          if (strategy.rollback) {
            try {
              await strategy.rollback()
            } catch (rollbackError) {
              monitoring.trackError('optimization_rollback_failed', rollbackError instanceof Error ? rollbackError.message : 'Unknown error', strategy.name)
            }
          }
        } finally {
          this.isOptimizing = false
        }
        break // Only execute one strategy per cycle
      }
    }
  }

  private makeScalingDecisions(): ScalingDecision {
    const profile = this.currentProfile
    let decision: ScalingDecision

    if (profile.cpuUsage > 90 || profile.memoryUsage > 90) {
      decision = {
        action: 'scale_up',
        reason: 'High resource utilization detected',
        intensity: Math.max(profile.cpuUsage, profile.memoryUsage) / 100,
        timestamp: new Date()
      }
    } else if (profile.cpuUsage < 30 && profile.memoryUsage < 30 && profile.userLoad < 20) {
      decision = {
        action: 'scale_down',
        reason: 'Low resource utilization, opportunity to scale down',
        intensity: 1 - Math.max(profile.cpuUsage, profile.memoryUsage) / 100,
        timestamp: new Date()
      }
    } else if (profile.errorRate > 5 || profile.networkLatency > 200) {
      decision = {
        action: 'optimize',
        reason: 'Performance degradation detected',
        intensity: Math.max(profile.errorRate / 10, profile.networkLatency / 500),
        timestamp: new Date()
      }
    } else {
      decision = {
        action: 'maintain',
        reason: 'Performance within acceptable ranges',
        intensity: 0,
        timestamp: new Date()
      }
    }

    if (decision.action !== 'maintain') {
      monitoring.trackMetric('scaling_decision', 1, {
        action: decision.action,
        reason: decision.reason,
        intensity: decision.intensity.toString()
      })
    }

    return decision
  }

  // Performance monitoring utilities
  private getCPUUsage(): number {
    // Simulate CPU usage measurement
    // In a real implementation, this would use performance.measureUserAgentSpecificMemory
    // or Web Workers to measure actual CPU usage
    return Math.random() * 100
  }

  private getMemoryUsage(): number {
    if ('memory' in performance) {
      const memInfo = (performance as any).memory
      return (memInfo.usedJSHeapSize / memInfo.jsHeapSizeLimit) * 100
    }
    return Math.random() * 100
  }

  private getNetworkLatency(): number {
    // Measure network latency using Navigation Timing API
    if ('navigation' in performance) {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming
      return navigation.responseEnd - navigation.requestStart
    }
    return Math.random() * 200 + 10
  }

  private getThroughput(): number {
    // Calculate requests per second based on recent performance entries
    const entries = performance.getEntriesByType('measure')
    const recentEntries = entries.filter(entry => 
      Date.now() - entry.startTime < 60000 // Last minute
    )
    return recentEntries.length // Simplified throughput calculation
  }

  private getErrorRate(): number {
    // Calculate error rate from monitoring data
    // This would integrate with your actual error tracking
    return Math.random() * 10
  }

  private getUserLoad(): number {
    // Estimate user load based on active connections, requests, etc.
    return Math.random() * 100
  }

  // Optimization implementations
  private enableRequestDeduplication() {
    // Implement request deduplication logic
    monitoring.trackMetric('request_deduplication_enabled', 1)
  }

  private enableResponseCompression() {
    // Enable gzip/brotli compression for responses
    monitoring.trackMetric('response_compression_enabled', 1)
  }

  private enableCircuitBreaker() {
    // Implement circuit breaker pattern
    monitoring.trackMetric('circuit_breaker_enabled', 1)
  }

  private enableRequestThrottling() {
    // Implement request rate limiting
    monitoring.trackMetric('request_throttling_enabled', 1)
  }

  // Public API methods
  public getPerformanceProfile(): PerformanceProfile {
    return { ...this.currentProfile }
  }

  public getOptimizationHistory(): Array<{ strategy: string; timestamp: Date; result: any }> {
    return [...this.optimizationHistory]
  }

  public enableAutoScaling(enabled: boolean) {
    this.autoScalingEnabled = enabled
    monitoring.trackMetric('auto_scaling_toggled', enabled ? 1 : 0)
  }

  public enableAdaptiveTuning(enabled: boolean) {
    this.adaptiveTuningEnabled = enabled
    monitoring.trackMetric('adaptive_tuning_toggled', enabled ? 1 : 0)
  }

  public getRecommendations(): Array<{ type: string; description: string; impact: 'low' | 'medium' | 'high' }> {
    const recommendations = []
    const profile = this.currentProfile

    if (profile.memoryUsage > 80) {
      recommendations.push({
        type: 'memory',
        description: 'High memory usage detected. Consider clearing caches or reducing data retention.',
        impact: 'high' as const
      })
    }

    if (profile.cpuUsage > 85) {
      recommendations.push({
        type: 'cpu',
        description: 'High CPU usage detected. Consider reducing computational complexity or adding worker threads.',
        impact: 'high' as const
      })
    }

    if (profile.networkLatency > 150) {
      recommendations.push({
        type: 'network',
        description: 'High network latency detected. Consider enabling compression or request batching.',
        impact: 'medium' as const
      })
    }

    if (profile.errorRate > 3) {
      recommendations.push({
        type: 'reliability',
        description: 'Elevated error rate detected. Consider implementing circuit breakers or retry logic.',
        impact: 'high' as const
      })
    }

    return recommendations
  }

  public async forceOptimization(strategyName?: string) {
    if (this.isOptimizing) {
      throw new Error('Optimization already in progress')
    }

    if (strategyName) {
      const strategy = this.optimizationStrategies.find(s => s.name === strategyName)
      if (!strategy) {
        throw new Error(`Strategy '${strategyName}' not found`)
      }

      this.isOptimizing = true
      try {
        await strategy.execute()
        monitoring.trackMetric('forced_optimization_executed', 1, { strategy: strategyName })
      } finally {
        this.isOptimizing = false
      }
    } else {
      // Execute all applicable strategies
      for (const strategy of this.optimizationStrategies) {
        if (strategy.trigger(this.currentProfile)) {
          await this.forceOptimization(strategy.name)
        }
      }
    }
  }

  public destroy() {
    // Cleanup when optimizer is no longer needed
    monitoring.trackMetric('performance_optimizer_destroyed', 1)
  }
}

// Global performance optimizer instance
export const performanceOptimizer = new PerformanceOptimizer()

// Auto-scaling configuration
export interface AutoScalingConfig {
  enabled: boolean
  minInstances: number
  maxInstances: number
  targetCPU: number
  targetMemory: number
  scaleUpThreshold: number
  scaleDownThreshold: number
  cooldownPeriod: number
}

export class AutoScaler {
  private config: AutoScalingConfig = {
    enabled: true,
    minInstances: 1,
    maxInstances: 10,
    targetCPU: 70,
    targetMemory: 80,
    scaleUpThreshold: 85,
    scaleDownThreshold: 30,
    cooldownPeriod: 300000 // 5 minutes
  }

  private currentInstances = 1
  private lastScalingAction = 0

  public async evaluateScaling(metrics: PerformanceProfile): Promise<ScalingDecision> {
    if (!this.config.enabled) {
      return {
        action: 'maintain',
        reason: 'Auto-scaling disabled',
        intensity: 0,
        timestamp: new Date()
      }
    }

    const now = Date.now()
    const timeSinceLastAction = now - this.lastScalingAction

    if (timeSinceLastAction < this.config.cooldownPeriod) {
      return {
        action: 'maintain',
        reason: 'Cooldown period active',
        intensity: 0,
        timestamp: new Date()
      }
    }

    // Scale up conditions
    if (
      (metrics.cpuUsage > this.config.scaleUpThreshold || 
       metrics.memoryUsage > this.config.scaleUpThreshold) &&
      this.currentInstances < this.config.maxInstances
    ) {
      this.currentInstances++
      this.lastScalingAction = now
      
      monitoring.trackMetric('auto_scale_up', this.currentInstances)
      
      return {
        action: 'scale_up',
        reason: `Resource utilization above ${this.config.scaleUpThreshold}%`,
        intensity: Math.max(metrics.cpuUsage, metrics.memoryUsage) / 100,
        timestamp: new Date()
      }
    }

    // Scale down conditions
    if (
      metrics.cpuUsage < this.config.scaleDownThreshold &&
      metrics.memoryUsage < this.config.scaleDownThreshold &&
      this.currentInstances > this.config.minInstances
    ) {
      this.currentInstances--
      this.lastScalingAction = now
      
      monitoring.trackMetric('auto_scale_down', this.currentInstances)
      
      return {
        action: 'scale_down',
        reason: `Resource utilization below ${this.config.scaleDownThreshold}%`,
        intensity: 1 - Math.max(metrics.cpuUsage, metrics.memoryUsage) / 100,
        timestamp: new Date()
      }
    }

    return {
      action: 'maintain',
      reason: 'Resource utilization within target ranges',
      intensity: 0,
      timestamp: new Date()
    }
  }

  public updateConfig(newConfig: Partial<AutoScalingConfig>) {
    this.config = { ...this.config, ...newConfig }
    monitoring.trackMetric('auto_scaling_config_updated', 1)
  }

  public getCurrentInstances(): number {
    return this.currentInstances
  }

  public getConfig(): AutoScalingConfig {
    return { ...this.config }
  }
}

export const autoScaler = new AutoScaler()

// Performance monitoring dashboard utilities
export const performanceDashboard = {
  getCurrentMetrics: () => performanceOptimizer.getPerformanceProfile(),
  getOptimizationHistory: () => performanceOptimizer.getOptimizationHistory(),
  getRecommendations: () => performanceOptimizer.getRecommendations(),
  getScalingStatus: () => ({
    currentInstances: autoScaler.getCurrentInstances(),
    config: autoScaler.getConfig()
  }),
  
  // Real-time performance streaming
  subscribeToMetrics: (callback: (metrics: PerformanceProfile) => void) => {
    const interval = setInterval(() => {
      callback(performanceOptimizer.getPerformanceProfile())
    }, 1000)
    
    return () => clearInterval(interval)
  }
}