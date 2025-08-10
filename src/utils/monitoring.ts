/**
 * Monitoring and telemetry utilities for Causal UI Gym
 * 
 * This module provides comprehensive monitoring, error tracking,
 * performance measurement, and user analytics capabilities.
 */

import { CausalResult, ExperimentConfig, Intervention } from '../types'

interface MetricEvent {
  name: string
  value: number
  timestamp: Date
  tags?: Record<string, string>
  metadata?: Record<string, any>
}

interface ErrorEvent {
  type: string
  message: string
  component: string
  stack?: string
  timestamp: Date
  userId?: string
  sessionId?: string
  metadata?: Record<string, any>
}

interface PerformanceEvent {
  name: string
  duration: number
  timestamp: Date
  tags?: Record<string, string>
  metadata?: Record<string, any>
}

interface UserEvent {
  action: string
  component: string
  timestamp: Date
  userId?: string
  sessionId?: string
  metadata?: Record<string, any>
}

class MonitoringService {
  private sessionId: string
  private userId?: string
  private isProduction: boolean
  private events: Array<MetricEvent | ErrorEvent | PerformanceEvent | UserEvent> = []
  private performanceTimers = new Map<string, number>()

  constructor() {
    this.sessionId = this.generateSessionId()
    this.isProduction = process.env.NODE_ENV === 'production'
    this.initializeSession()
  }

  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  private initializeSession() {
    this.trackUser('session_start', 'monitoring', {
      userAgent: navigator.userAgent,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight
      },
      url: window.location.href,
      referrer: document.referrer
    })
  }

  setUserId(userId: string) {
    this.userId = userId
    this.trackUser('user_identified', 'monitoring', { userId })
  }

  // Error tracking
  trackError(type: string, message: string, component: string, stack?: string, metadata?: Record<string, any>) {
    const errorEvent: ErrorEvent = {
      type,
      message,
      component,
      stack,
      timestamp: new Date(),
      userId: this.userId,
      sessionId: this.sessionId,
      metadata
    }

    this.events.push(errorEvent)
    
    if (this.isProduction) {
      this.sendToErrorService(errorEvent)
    } else {
      console.error('🚨 Error tracked:', errorEvent)
    }

    // Also track as metric for dashboards
    this.trackMetric('error_count', 1, { error_type: type, component })
  }

  // Performance tracking
  startTimer(name: string) {
    this.performanceTimers.set(name, performance.now())
  }

  endTimer(name: string, tags?: Record<string, string>, metadata?: Record<string, any>) {
    const startTime = this.performanceTimers.get(name)
    if (startTime === undefined) {
      console.warn(`No timer found for: ${name}`)
      return
    }

    const duration = performance.now() - startTime
    this.performanceTimers.delete(name)

    const perfEvent: PerformanceEvent = {
      name,
      duration,
      timestamp: new Date(),
      tags,
      metadata
    }

    this.events.push(perfEvent)
    
    if (this.isProduction) {
      this.sendToPerformanceService(perfEvent)
    } else {
      console.log(`⚡ Performance: ${name} took ${duration.toFixed(2)}ms`)
    }

    // Track as metric
    this.trackMetric('performance_duration', duration, { operation: name, ...tags })
  }

  // Metric tracking
  trackMetric(name: string, value: number, tags?: Record<string, string>, metadata?: Record<string, any>) {
    const metricEvent: MetricEvent = {
      name,
      value,
      timestamp: new Date(),
      tags,
      metadata
    }

    this.events.push(metricEvent)
    
    if (this.isProduction) {
      this.sendToMetricService(metricEvent)
    } else {
      console.log(`📊 Metric: ${name} = ${value}`, tags)
    }
  }

  // User action tracking
  trackUser(action: string, component: string, metadata?: Record<string, any>) {
    const userEvent: UserEvent = {
      action,
      component,
      timestamp: new Date(),
      userId: this.userId,
      sessionId: this.sessionId,
      metadata
    }

    this.events.push(userEvent)
    
    if (this.isProduction) {
      this.sendToAnalyticsService(userEvent)
    } else {
      console.log(`👤 User action: ${action} in ${component}`, metadata)
    }
  }

  // Causal-specific tracking
  trackExperiment(config: ExperimentConfig) {
    this.trackUser('experiment_created', 'ExperimentBuilder', {
      experiment_id: config.id,
      experiment_name: config.name,
      node_count: config.dag.nodes.length,
      edge_count: config.dag.edges.length,
      intervention_count: config.interventions.length,
      sample_size: config.sample_size
    })

    this.trackMetric('experiment_complexity', config.dag.nodes.length * config.dag.edges.length, {
      experiment_id: config.id
    })
  }

  trackIntervention(intervention: Intervention, component: string, metadata?: Record<string, any>) {
    this.trackUser('intervention_applied', component, {
      variable: intervention.variable,
      value: intervention.value.toString(),
      intervention_type: intervention.intervention_type,
      ...metadata
    })

    this.trackMetric('intervention_value', typeof intervention.value === 'number' ? intervention.value : 0, {
      variable: intervention.variable,
      intervention_type: intervention.intervention_type
    })
  }

  trackCausalResult(result: CausalResult) {
    this.trackUser('causal_result_computed', 'CausalEngine', {
      result_id: result.id,
      intervention_variable: result.intervention.variable,
      outcome_variable: result.outcome_variable,
      metric_count: result.metrics.length
    })

    // Track individual metrics
    result.metrics.forEach(metric => {
      this.trackMetric(`causal_${metric.metric_type}`, metric.value, {
        intervention_variable: result.intervention.variable,
        outcome_variable: result.outcome_variable
      })

      if (metric.computation_time) {
        this.trackMetric('causal_computation_time', metric.computation_time, {
          metric_type: metric.metric_type,
          sample_size: metric.sample_size.toString()
        })
      }
    })
  }

  // Component lifecycle tracking
  trackComponentMount(componentName: string, props?: Record<string, any>) {
    this.trackUser('component_mount', componentName, {
      props_keys: props ? Object.keys(props).join(',') : 'none'
    })
  }

  trackComponentError(componentName: string, error: Error, errorInfo?: any) {
    this.trackError(
      'react_error',
      error.message,
      componentName,
      error.stack,
      {
        error_boundary: true,
        component_stack: errorInfo?.componentStack
      }
    )
  }

  // Graph interaction tracking
  trackGraphInteraction(action: string, nodeId?: string, metadata?: Record<string, any>) {
    this.trackUser('graph_interaction', 'CausalGraph', {
      action,
      node_id: nodeId,
      ...metadata
    })
  }

  // API call tracking
  trackApiCall(endpoint: string, method: string, status: number, duration: number) {
    this.trackMetric('api_call_duration', duration, {
      endpoint,
      method,
      status: status.toString()
    })

    if (status >= 400) {
      this.trackError(
        'api_error',
        `API call failed: ${method} ${endpoint}`,
        'api',
        undefined,
        { status, endpoint, method }
      )
    }
  }

  // Validation tracking
  trackValidation(type: string, isValid: boolean, errors: string[] = [], component: string = 'unknown') {
    this.trackMetric('validation_result', isValid ? 1 : 0, {
      validation_type: type,
      component
    })

    if (!isValid && errors.length > 0) {
      errors.forEach(error => {
        this.trackError(
          'validation_error',
          error,
          component,
          undefined,
          { validation_type: type }
        )
      })
    }
  }

  // Performance monitoring utilities
  measureAsync<T>(name: string, asyncFn: () => Promise<T>, tags?: Record<string, string>): Promise<T> {
    this.startTimer(name)
    return asyncFn().finally(() => {
      this.endTimer(name, tags)
    })
  }

  measureSync<T>(name: string, syncFn: () => T, tags?: Record<string, string>): T {
    this.startTimer(name)
    try {
      return syncFn()
    } finally {
      this.endTimer(name, tags)
    }
  }

  // Memory usage tracking
  trackMemoryUsage(component: string) {
    if ('memory' in performance) {
      const memInfo = (performance as any).memory
      this.trackMetric('memory_used', memInfo.usedJSHeapSize, { component })
      this.trackMetric('memory_total', memInfo.totalJSHeapSize, { component })
      this.trackMetric('memory_limit', memInfo.jsHeapSizeLimit, { component })
    }
  }

  // Browser performance tracking
  trackPageLoad() {
    window.addEventListener('load', () => {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming
      
      this.trackMetric('page_load_time', navigation.loadEventEnd - navigation.fetchStart, {
        page: window.location.pathname
      })
      
      this.trackMetric('dom_content_loaded', navigation.domContentLoadedEventEnd - navigation.fetchStart, {
        page: window.location.pathname
      })
    })
  }

  // Event aggregation and batching
  getEventSummary(): {
    errors: number
    metrics: number
    performance: number
    user_actions: number
  } {
    const summary = { errors: 0, metrics: 0, performance: 0, user_actions: 0 }
    
    this.events.forEach(event => {
      if ('type' in event && event.type) summary.errors++
      else if ('value' in event) summary.metrics++
      else if ('duration' in event) summary.performance++
      else if ('action' in event) summary.user_actions++
    })
    
    return summary
  }

  exportEvents(format: 'json' | 'csv' = 'json'): string {
    if (format === 'csv') {
      const csvHeader = 'timestamp,type,category,name,value,metadata\n'
      const csvRows = this.events.map(event => {
        const timestamp = event.timestamp.toISOString()
        let type = 'unknown'
        let category = 'unknown'
        let name = 'unknown'
        let value = ''
        
        if ('type' in event && event.type) {
          type = 'error'
          category = event.component
          name = event.message
        } else if ('value' in event) {
          type = 'metric'
          category = 'metric'
          name = event.name
          value = event.value.toString()
        } else if ('duration' in event) {
          type = 'performance'
          category = 'timing'
          name = event.name
          value = event.duration.toString()
        } else if ('action' in event) {
          type = 'user'
          category = event.component
          name = event.action
        }
        
        const metadata = JSON.stringify(('metadata' in event && event.metadata) || {})
        return `${timestamp},${type},${category},${name},${value},"${metadata}"`
      })
      
      return csvHeader + csvRows.join('\n')
    }
    
    return JSON.stringify(this.events, null, 2)
  }

  // Service integration methods (implement based on your monitoring stack)
  private async sendToErrorService(error: ErrorEvent) {
    // Example: Sentry, Rollbar, etc.
    // await fetch('/api/errors', { method: 'POST', body: JSON.stringify(error) })
  }

  private async sendToMetricService(metric: MetricEvent) {
    // Example: DataDog, New Relic, etc.
    // await fetch('/api/metrics', { method: 'POST', body: JSON.stringify(metric) })
  }

  private async sendToPerformanceService(performance: PerformanceEvent) {
    // Example: SpeedCurve, WebPageTest, etc.
    // await fetch('/api/performance', { method: 'POST', body: JSON.stringify(performance) })
  }

  private async sendToAnalyticsService(user: UserEvent) {
    // Example: Google Analytics, Mixpanel, etc.
    // await fetch('/api/analytics', { method: 'POST', body: JSON.stringify(user) })
  }

  // Health check
  healthCheck(): {
    status: 'healthy' | 'degraded' | 'unhealthy'
    events_count: number
    memory_usage?: number
    last_error?: Date
  } {
    const recentErrors = this.events.filter(e => 
      'type' in e && e.type && 
      (Date.now() - e.timestamp.getTime()) < 60000 // Last minute
    )
    
    let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy'
    if (recentErrors.length > 10) status = 'unhealthy'
    else if (recentErrors.length > 5) status = 'degraded'
    
    const lastError = recentErrors.length > 0 ? recentErrors[recentErrors.length - 1].timestamp : undefined
    
    return {
      status,
      events_count: this.events.length,
      memory_usage: 'memory' in performance ? (performance as any).memory?.usedJSHeapSize : undefined,
      last_error: lastError
    }
  }
}

// Export singleton instance
export const monitoring = new MonitoringService()

// React hook for easy component integration
export function useMonitoring() {
  return {
    trackError: monitoring.trackError.bind(monitoring),
    trackUser: monitoring.trackUser.bind(monitoring),
    trackMetric: monitoring.trackMetric.bind(monitoring),
    startTimer: monitoring.startTimer.bind(monitoring),
    endTimer: monitoring.endTimer.bind(monitoring),
    measureAsync: monitoring.measureAsync.bind(monitoring),
    measureSync: monitoring.measureSync.bind(monitoring)
  }
}

// Automatic performance monitoring decorator
export function withPerformanceTracking<T extends (...args: any[]) => any>(
  fn: T, 
  name: string,
  tags?: Record<string, string>
): T {
  return ((...args: any[]) => {
    return monitoring.measureSync(name, () => fn(...args), tags)
  }) as T
}

// Error boundary integration
export function withErrorTracking<T extends (...args: any[]) => any>(
  fn: T,
  component: string,
  context?: Record<string, any>
): T {
  return ((...args: any[]) => {
    try {
      return fn(...args)
    } catch (error) {
      monitoring.trackError(
        'function_error',
        (error as Error).message,
        component,
        (error as Error).stack,
        context
      )
      throw error
    }
  }) as T
}

// Initialize page load tracking
if (typeof window !== 'undefined') {
  monitoring.trackPageLoad()
}

export type { MetricEvent, ErrorEvent, PerformanceEvent, UserEvent }