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