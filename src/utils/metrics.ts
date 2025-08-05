/**
 * Application Metrics Collection Framework
 * 
 * Provides comprehensive metrics collection for the Causal UI Gym application.
 * Integrates with Prometheus for monitoring and alerting.
 */

export interface MetricLabels {
  [key: string]: string | number;
}

export interface CustomMetric {
  name: string;
  help: string;
  labels?: MetricLabels;
  value: number;
  timestamp?: number;
}

export class ApplicationMetrics {
  private static instance: ApplicationMetrics;
  private metricsBuffer: CustomMetric[] = [];
  private isProduction = process.env.NODE_ENV === 'production';
  private metricsEndpoint = process.env.REACT_APP_METRICS_ENDPOINT || '/api/metrics';

  private constructor() {
    this.setupPerformanceObserver();
    this.setupErrorTracking();
    this.startMetricsFlush();
  }

  public static getInstance(): ApplicationMetrics {
    if (!ApplicationMetrics.instance) {
      ApplicationMetrics.instance = new ApplicationMetrics();
    }
    return ApplicationMetrics.instance;
  }

  /**
   * Record a counter metric (incremental values)
   */
  public counter(name: string, help: string, labels?: MetricLabels, value: number = 1): void {
    this.recordMetric({
      name: `${name}_total`,
      help,
      labels,
      value,
      timestamp: Date.now()
    });
  }

  /**
   * Record a gauge metric (point-in-time values)
   */
  public gauge(name: string, help: string, labels?: MetricLabels, value: number): void {
    this.recordMetric({
      name,
      help,
      labels,
      value,
      timestamp: Date.now()
    });
  }

  /**
   * Record a histogram metric for duration measurements
   */
  public histogram(name: string, help: string, labels?: MetricLabels, value: number): void {
    this.recordMetric({
      name: `${name}_duration_ms`,
      help,
      labels,
      value,
      timestamp: Date.now()
    });
  }

  /**
   * Track causal graph operations
   */
  public trackCausalGraphOperation(
    operation: 'create' | 'update' | 'delete' | 'analyze',
    duration: number,
    nodeCount: number,
    edgeCount: number,
    success: boolean
  ): void {
    const labels = {
      operation,
      node_count_bucket: this.getBucket(nodeCount, [10, 50, 100, 500]),
      edge_count_bucket: this.getBucket(edgeCount, [20, 100, 500, 1000]),
      status: success ? 'success' : 'error'
    };

    this.counter('causal_graph_operations', 'Total causal graph operations', labels);
    this.histogram('causal_graph_operation', 'Causal graph operation duration', labels, duration);
    this.gauge('causal_graph_nodes', 'Number of nodes in causal graph', labels, nodeCount);
    this.gauge('causal_graph_edges', 'Number of edges in causal graph', labels, edgeCount);
  }

  /**
   * Track experiment execution metrics
   */
  public trackExperiment(
    experimentType: string,
    duration: number,
    sampleSize: number,
    success: boolean,
    effect_size?: number
  ): void {
    const labels = {
      experiment_type: experimentType,
      sample_size_bucket: this.getBucket(sampleSize, [100, 1000, 10000, 100000]),
      status: success ? 'success' : 'error'
    };

    this.counter('experiments', 'Total experiments executed', labels);
    this.histogram('experiment_execution', 'Experiment execution duration', labels, duration);
    this.gauge('experiment_sample_size', 'Experiment sample size', labels, sampleSize);

    if (effect_size !== undefined) {
      this.gauge('experiment_effect_size', 'Measured effect size', labels, effect_size);
    }
  }

  /**
   * Track UI interaction metrics
   */
  public trackUIInteraction(
    component: string,
    action: string,
    duration?: number,
    metadata?: Record<string, any>
  ): void {
    const labels = {
      component,
      action,
      ...(metadata || {})
    };

    this.counter('ui_interactions', 'Total UI interactions', labels);
    
    if (duration !== undefined) {
      this.histogram('ui_interaction', 'UI interaction duration', labels, duration);
    }
  }

  /**
   * Track API call metrics
   */
  public trackAPICall(
    endpoint: string,
    method: string,
    statusCode: number,
    duration: number,
    payloadSize?: number
  ): void {
    const labels = {
      endpoint: this.sanitizeEndpoint(endpoint),
      method: method.toUpperCase(),
      status_code: statusCode.toString(),
      status_class: `${Math.floor(statusCode / 100)}xx`
    };

    this.counter('api_requests', 'Total API requests', labels);
    this.histogram('api_request', 'API request duration', labels, duration);

    if (payloadSize !== undefined) {
      this.gauge('api_payload_size', 'API payload size in bytes', labels, payloadSize);
    }
  }

  /**
   * Track performance metrics
   */
  public trackPerformance(
    metric: 'FCP' | 'LCP' | 'FID' | 'CLS' | 'TTFB',
    value: number,
    rating: 'good' | 'needs-improvement' | 'poor'
  ): void {
    const labels = {
      metric: metric.toLowerCase(),
      rating
    };

    this.gauge('web_vitals', `Web Vitals ${metric} metric`, labels, value);
  }

  /**
   * Track error occurrences
   */
  public trackError(
    errorType: string,
    errorMessage: string,
    component?: string,
    stackTrace?: string
  ): void {
    const labels = {
      error_type: errorType,
      component: component || 'unknown',
      // Hash the error message to avoid high cardinality
      error_hash: this.hashString(errorMessage).toString()
    };

    this.counter('application_errors', 'Total application errors', labels);

    // Log detailed error information (not as metrics due to cardinality)
    if (this.isProduction) {
      console.error('Application Error:', {
        type: errorType,
        message: errorMessage,
        component,
        stackTrace: stackTrace?.substring(0, 1000), // Truncate for logging
        timestamp: new Date().toISOString()
      });
    }
  }

  /**
   * Track business metrics
   */
  public trackBusinessMetric(
    metric: 'user_engagement' | 'feature_adoption' | 'completion_rate',
    value: number,
    labels?: MetricLabels
  ): void {
    this.gauge(`business_${metric}`, `Business metric: ${metric}`, labels, value);
  }

  private recordMetric(metric: CustomMetric): void {
    if (!this.isProduction && process.env.NODE_ENV !== 'test') {
      console.debug('Metric recorded:', metric);
    }

    this.metricsBuffer.push(metric);

    // Prevent memory leaks by limiting buffer size
    if (this.metricsBuffer.length > 1000) {
      this.metricsBuffer = this.metricsBuffer.slice(-500);
    }
  }

  private setupPerformanceObserver(): void {
    if (typeof window === 'undefined' || !window.PerformanceObserver) return;

    // Web Vitals tracking
    try {
      const observer = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
          if (entry.entryType === 'measure') {
            this.histogram('custom_performance', 'Custom performance measurements', 
              { measure_name: entry.name }, entry.duration);
          }
        });
      });

      observer.observe({ entryTypes: ['measure', 'navigation', 'resource'] });
    } catch (error) {
      console.warn('Performance Observer setup failed:', error);
    }
  }

  private setupErrorTracking(): void {
    if (typeof window === 'undefined') return;

    window.addEventListener('error', (event) => {
      this.trackError(
        'javascript_error',
        event.message,
        event.filename?.split('/').pop(),
        event.error?.stack
      );
    });

    window.addEventListener('unhandledrejection', (event) => {
      this.trackError(
        'unhandled_promise_rejection',
        event.reason?.message || String(event.reason),
        'async_operation',
        event.reason?.stack
      );
    });
  }

  private startMetricsFlush(): void {
    if (!this.isProduction) return;

    // Flush metrics every 30 seconds
    setInterval(() => {
      this.flushMetrics();
    }, 30000);

    // Flush metrics before page unload
    if (typeof window !== 'undefined') {
      window.addEventListener('beforeunload', () => {
        this.flushMetrics();
      });
    }
  }

  private async flushMetrics(): Promise<void> {
    if (this.metricsBuffer.length === 0) return;

    const metricsToFlush = [...this.metricsBuffer];
    this.metricsBuffer = [];

    try {
      await fetch(this.metricsEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          metrics: metricsToFlush,
          timestamp: Date.now(),
          user_agent: navigator?.userAgent,
          url: window?.location?.href
        }),
        // Don't wait for response to avoid blocking
        keepalive: true
      });
    } catch (error) {
      console.warn('Failed to flush metrics:', error);
      // Re-add metrics to buffer for retry
      this.metricsBuffer.unshift(...metricsToFlush.slice(-100)); // Keep last 100
    }
  }

  private getBucket(value: number, buckets: number[]): string {
    for (const bucket of buckets) {
      if (value <= bucket) {
        return `le_${bucket}`;
      }
    }
    return `gt_${buckets[buckets.length - 1]}`;
  }

  private sanitizeEndpoint(endpoint: string): string {
    // Replace dynamic path parameters with placeholders
    return endpoint
      .replace(/\/\d+/g, '/:id')
      .replace(/\/[a-f0-9-]{36}/g, '/:uuid')
      .replace(/\/[a-zA-Z0-9]{20,}/g, '/:hash');
  }

  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  }

  /**
   * Get current metrics snapshot for debugging
   */
  public getMetricsSnapshot(): CustomMetric[] {
    return [...this.metricsBuffer];
  }

  /**
   * Clear metrics buffer (useful for testing)
   */
  public clearMetrics(): void {
    this.metricsBuffer = [];
  }
}

// Export singleton instance
export const metrics = ApplicationMetrics.getInstance();

// React Hook for easy metrics integration
export function useMetrics() {
  const trackComponentMount = (componentName: string) => {
    metrics.counter('component_mounts', 'React component mount count', { component: componentName });
  };

  const trackComponentUnmount = (componentName: string) => {
    metrics.counter('component_unmounts', 'React component unmount count', { component: componentName });
  };

  const trackUserAction = (action: string, component: string, metadata?: Record<string, any>) => {
    metrics.trackUIInteraction(component, action, undefined, metadata);
  };

  const trackRenderTime = (componentName: string, duration: number) => {
    metrics.histogram('component_render', 'React component render duration', 
      { component: componentName }, duration);
  };

  return {
    trackComponentMount,
    trackComponentUnmount,
    trackUserAction,
    trackRenderTime,
    metrics
  };
}

// Performance measurement helper
export function measurePerformance<T>(
  name: string,
  fn: () => T | Promise<T>,
  labels?: MetricLabels
): Promise<T> {
  const start = performance.now();
  
  const finish = (result: T) => {
    const duration = performance.now() - start;
    metrics.histogram('operation_performance', `Performance for ${name}`, 
      { operation: name, ...labels }, duration);
    return result;
  };

  try {
    const result = fn();
    if (result instanceof Promise) {
      return result.then(finish).catch((error) => {
        metrics.trackError('performance_measurement_error', error.message, name);
        throw error;
      });
    }
    return Promise.resolve(finish(result));
  } catch (error) {
    metrics.trackError('performance_measurement_error', (error as Error).message, name);
    throw error;
  }
}

// Causal-specific metrics and utilities
import { CausalDAG, CausalMetric, Intervention } from '../types'

export const calculateBasicStats = (data: number[]) => {
  if (data.length === 0) return { mean: 0, variance: 0, stdDev: 0 }
  
  const sum = data.reduce((acc, val) => acc + val, 0)
  const mean = sum / data.length
  const variance = data.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / data.length
  const stdDev = Math.sqrt(variance)
  
  return { mean, variance, stdDev }
}

export const calculateATE = (
  treatmentOutcomes: number[],
  controlOutcomes: number[]
): CausalMetric => {
  const treatmentStats = calculateBasicStats(treatmentOutcomes)
  const controlStats = calculateBasicStats(controlOutcomes)
  
  const ate = treatmentStats.mean - controlStats.mean
  const pooledVariance = (treatmentStats.variance + controlStats.variance) / 2
  const standardError = Math.sqrt(pooledVariance * (1/treatmentOutcomes.length + 1/controlOutcomes.length))
  
  const tStatistic = ate / standardError
  const degreesOfFreedom = treatmentOutcomes.length + controlOutcomes.length - 2
  const pValue = 2 * (1 - studentTCDF(Math.abs(tStatistic), degreesOfFreedom))
  
  const confidenceLevel = 0.95
  const tCritical = studentTInverse((1 - confidenceLevel) / 2, degreesOfFreedom)
  const marginOfError = tCritical * standardError
  
  return {
    metric_type: 'ate',
    value: ate,
    confidence_interval: [ate - marginOfError, ate + marginOfError],
    standard_error: standardError,
    p_value: pValue,
    sample_size: treatmentOutcomes.length + controlOutcomes.length,
    computation_time: 0,
    metadata: {
      treatment_mean: treatmentStats.mean,
      control_mean: controlStats.mean,
      t_statistic: tStatistic,
      degrees_of_freedom: degreesOfFreedom
    }
  }
}

export const calculateCausalAccuracy = (
  predictedEffects: number[],
  trueEffects: number[]
): number => {
  if (predictedEffects.length !== trueEffects.length) {
    throw new Error('Predicted and true effects arrays must have the same length')
  }
  
  const squaredErrors = predictedEffects.map((pred, i) => 
    Math.pow(pred - trueEffects[i], 2)
  )
  
  const mse = squaredErrors.reduce((sum, error) => sum + error, 0) / squaredErrors.length
  const trueMean = trueEffects.reduce((sum, val) => sum + val, 0) / trueEffects.length
  const totalVariance = trueEffects.reduce((sum, val) => sum + Math.pow(val - trueMean, 2), 0)
  
  const r2 = 1 - (squaredErrors.reduce((sum, error) => sum + error, 0) / totalVariance)
  
  return Math.max(0, r2)
}

function studentTCDF(t: number, df: number): number {
  if (df <= 0) return 0.5
  if (t === 0) return 0.5
  
  const x = df / (t * t + df)
  return 0.5 + 0.5 * Math.sign(t) * (1 - incompleteBeta(0.5, df / 2, x))
}

function studentTInverse(p: number, df: number): number {
  if (p <= 0 || p >= 1) throw new Error('p must be between 0 and 1')
  
  const t_table: { [key: number]: number[] } = {
    1: [0, 1.000, 3.078, 6.314, 12.71, 31.82],
    2: [0, 0.816, 1.886, 2.920, 4.303, 6.965],
    5: [0, 0.727, 1.476, 2.015, 2.571, 3.365],
    10: [0, 0.700, 1.372, 1.812, 2.228, 2.764],
    20: [0, 0.687, 1.325, 1.725, 2.086, 2.528],
    30: [0, 0.683, 1.310, 1.697, 2.042, 2.457]
  }
  
  const alpha = 1 - p
  const closestDf = Object.keys(t_table).map(Number).reduce((prev, curr) =>
    Math.abs(curr - df) < Math.abs(prev - df) ? curr : prev
  )
  
  if (alpha <= 0.025) return t_table[closestDf][3]
  if (alpha <= 0.05) return t_table[closestDf][2]
  if (alpha <= 0.1) return t_table[closestDf][1]
  
  return t_table[closestDf][4]
}

function incompleteBeta(a: number, b: number, x: number): number {
  if (x <= 0) return 0
  if (x >= 1) return 1
  
  let result = 0
  let term = 1
  
  for (let n = 0; n < 100; n++) {
    if (n > 0) {
      term *= (a + n - 1) * x / n
    }
    
    const factor = Math.pow(1 - x, b) / (b + n)
    result += term * factor
    
    if (Math.abs(term * factor) < 1e-10) break
  }
  
  return Math.pow(x, a) * result
}