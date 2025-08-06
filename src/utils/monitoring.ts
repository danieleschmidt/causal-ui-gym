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
    result.metrics.forEach(metric => {\n      this.trackMetric(`causal_${metric.metric_type}`, metric.value, {\n        intervention_variable: result.intervention.variable,\n        outcome_variable: result.outcome_variable\n      })\n\n      if (metric.computation_time) {\n        this.trackMetric('causal_computation_time', metric.computation_time, {\n          metric_type: metric.metric_type,\n          sample_size: metric.sample_size.toString()\n        })\n      }\n    })\n  }

  // Component lifecycle tracking
  trackComponentMount(componentName: string, props?: Record<string, any>) {\n    this.trackUser('component_mount', componentName, {\n      props_keys: props ? Object.keys(props).join(',') : 'none'\n    })\n  }\n\n  trackComponentError(componentName: string, error: Error, errorInfo?: any) {\n    this.trackError(\n      'react_error',\n      error.message,\n      componentName,\n      error.stack,\n      {\n        error_boundary: true,\n        component_stack: errorInfo?.componentStack\n      }\n    )\n  }\n\n  // Graph interaction tracking\n  trackGraphInteraction(action: string, nodeId?: string, metadata?: Record<string, any>) {\n    this.trackUser('graph_interaction', 'CausalGraph', {\n      action,\n      node_id: nodeId,\n      ...metadata\n    })\n  }\n\n  // API call tracking\n  trackApiCall(endpoint: string, method: string, status: number, duration: number) {\n    this.trackMetric('api_call_duration', duration, {\n      endpoint,\n      method,\n      status: status.toString()\n    })\n\n    if (status >= 400) {\n      this.trackError(\n        'api_error',\n        `API call failed: ${method} ${endpoint}`,\n        'api',\n        undefined,\n        { status, endpoint, method }\n      )\n    }\n  }\n\n  // Validation tracking\n  trackValidation(type: string, isValid: boolean, errors: string[] = [], component: string = 'unknown') {\n    this.trackMetric('validation_result', isValid ? 1 : 0, {\n      validation_type: type,\n      component\n    })\n\n    if (!isValid && errors.length > 0) {\n      errors.forEach(error => {\n        this.trackError(\n          'validation_error',\n          error,\n          component,\n          undefined,\n          { validation_type: type }\n        )\n      })\n    }\n  }\n\n  // Performance monitoring utilities\n  measureAsync<T>(name: string, asyncFn: () => Promise<T>, tags?: Record<string, string>): Promise<T> {\n    this.startTimer(name)\n    return asyncFn().finally(() => {\n      this.endTimer(name, tags)\n    })\n  }\n\n  measureSync<T>(name: string, syncFn: () => T, tags?: Record<string, string>): T {\n    this.startTimer(name)\n    try {\n      return syncFn()\n    } finally {\n      this.endTimer(name, tags)\n    }\n  }\n\n  // Memory usage tracking\n  trackMemoryUsage(component: string) {\n    if ('memory' in performance) {\n      const memInfo = (performance as any).memory\n      this.trackMetric('memory_used', memInfo.usedJSHeapSize, { component })\n      this.trackMetric('memory_total', memInfo.totalJSHeapSize, { component })\n      this.trackMetric('memory_limit', memInfo.jsHeapSizeLimit, { component })\n    }\n  }\n\n  // Browser performance tracking\n  trackPageLoad() {\n    window.addEventListener('load', () => {\n      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming\n      \n      this.trackMetric('page_load_time', navigation.loadEventEnd - navigation.fetchStart, {\n        page: window.location.pathname\n      })\n      \n      this.trackMetric('dom_content_loaded', navigation.domContentLoadedEventEnd - navigation.fetchStart, {\n        page: window.location.pathname\n      })\n    })\n  }\n\n  // Event aggregation and batching\n  getEventSummary(): {\n    errors: number\n    metrics: number\n    performance: number\n    user_actions: number\n  } {\n    const summary = { errors: 0, metrics: 0, performance: 0, user_actions: 0 }\n    \n    this.events.forEach(event => {\n      if ('type' in event && event.type) summary.errors++\n      else if ('value' in event) summary.metrics++\n      else if ('duration' in event) summary.performance++\n      else if ('action' in event) summary.user_actions++\n    })\n    \n    return summary\n  }\n\n  exportEvents(format: 'json' | 'csv' = 'json'): string {\n    if (format === 'csv') {\n      const csvHeader = 'timestamp,type,category,name,value,metadata\\n'\n      const csvRows = this.events.map(event => {\n        const timestamp = event.timestamp.toISOString()\n        let type = 'unknown'\n        let category = 'unknown'\n        let name = 'unknown'\n        let value = ''\n        \n        if ('type' in event && event.type) {\n          type = 'error'\n          category = event.component\n          name = event.message\n        } else if ('value' in event) {\n          type = 'metric'\n          category = 'metric'\n          name = event.name\n          value = event.value.toString()\n        } else if ('duration' in event) {\n          type = 'performance'\n          category = 'timing'\n          name = event.name\n          value = event.duration.toString()\n        } else if ('action' in event) {\n          type = 'user'\n          category = event.component\n          name = event.action\n        }\n        \n        const metadata = JSON.stringify(('metadata' in event && event.metadata) || {})\n        return `${timestamp},${type},${category},${name},${value},\"${metadata}\"`\n      })\n      \n      return csvHeader + csvRows.join('\\n')\n    }\n    \n    return JSON.stringify(this.events, null, 2)\n  }\n\n  // Service integration methods (implement based on your monitoring stack)\n  private async sendToErrorService(error: ErrorEvent) {\n    // Example: Sentry, Rollbar, etc.\n    // await fetch('/api/errors', { method: 'POST', body: JSON.stringify(error) })\n  }\n\n  private async sendToMetricService(metric: MetricEvent) {\n    // Example: DataDog, New Relic, etc.\n    // await fetch('/api/metrics', { method: 'POST', body: JSON.stringify(metric) })\n  }\n\n  private async sendToPerformanceService(performance: PerformanceEvent) {\n    // Example: SpeedCurve, WebPageTest, etc.\n    // await fetch('/api/performance', { method: 'POST', body: JSON.stringify(performance) })\n  }\n\n  private async sendToAnalyticsService(user: UserEvent) {\n    // Example: Google Analytics, Mixpanel, etc.\n    // await fetch('/api/analytics', { method: 'POST', body: JSON.stringify(user) })\n  }\n\n  // Health check\n  healthCheck(): {\n    status: 'healthy' | 'degraded' | 'unhealthy'\n    events_count: number\n    memory_usage?: number\n    last_error?: Date\n  } {\n    const recentErrors = this.events.filter(e => \n      'type' in e && e.type && \n      (Date.now() - e.timestamp.getTime()) < 60000 // Last minute\n    )\n    \n    let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy'\n    if (recentErrors.length > 10) status = 'unhealthy'\n    else if (recentErrors.length > 5) status = 'degraded'\n    \n    const lastError = recentErrors.length > 0 ? recentErrors[recentErrors.length - 1].timestamp : undefined\n    \n    return {\n      status,\n      events_count: this.events.length,\n      memory_usage: 'memory' in performance ? (performance as any).memory?.usedJSHeapSize : undefined,\n      last_error: lastError\n    }\n  }\n}\n\n// Export singleton instance\nexport const monitoring = new MonitoringService()\n\n// React hook for easy component integration\nexport function useMonitoring() {\n  return {\n    trackError: monitoring.trackError.bind(monitoring),\n    trackUser: monitoring.trackUser.bind(monitoring),\n    trackMetric: monitoring.trackMetric.bind(monitoring),\n    startTimer: monitoring.startTimer.bind(monitoring),\n    endTimer: monitoring.endTimer.bind(monitoring),\n    measureAsync: monitoring.measureAsync.bind(monitoring),\n    measureSync: monitoring.measureSync.bind(monitoring)\n  }\n}\n\n// Automatic performance monitoring decorator\nexport function withPerformanceTracking<T extends (...args: any[]) => any>(\n  fn: T, \n  name: string,\n  tags?: Record<string, string>\n): T {\n  return ((...args: any[]) => {\n    return monitoring.measureSync(name, () => fn(...args), tags)\n  }) as T\n}\n\n// Error boundary integration\nexport function withErrorTracking<T extends (...args: any[]) => any>(\n  fn: T,\n  component: string,\n  context?: Record<string, any>\n): T {\n  return ((...args: any[]) => {\n    try {\n      return fn(...args)\n    } catch (error) {\n      monitoring.trackError(\n        'function_error',\n        (error as Error).message,\n        component,\n        (error as Error).stack,\n        context\n      )\n      throw error\n    }\n  }) as T\n}\n\n// Initialize page load tracking\nif (typeof window !== 'undefined') {\n  monitoring.trackPageLoad()\n}\n\nexport type { MetricEvent, ErrorEvent, PerformanceEvent, UserEvent }\n"