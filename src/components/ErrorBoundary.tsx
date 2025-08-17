import React, { Component, ErrorInfo, ReactNode, useCallback } from 'react'
import { 
  Alert, 
  AlertTitle, 
  Box, 
  Button, 
  Card, 
  CardContent, 
  Typography, 
  Collapse,
  CircularProgress
} from '@mui/material'
import { 
  Error as ErrorIcon, 
  ExpandMore as ExpandMoreIcon,
  Refresh as RefreshIcon,
  BugReport as BugReportIcon,
  Info as InfoIcon,
  Warning as EmergencyIcon
} from '@mui/icons-material'
import { metrics } from '../utils/metrics'

interface Props {
  children: ReactNode
  fallback?: ReactNode
  onError?: (error: Error, errorInfo: ErrorInfo) => void
  enableAutoRetry?: boolean
  maxRetries?: number
  retryDelay?: number
  enableErrorReporting?: boolean
  enableTelemetry?: boolean
  component?: string
}

interface State {
  hasError: boolean
  error: Error | null
  errorInfo: ErrorInfo | null
  showDetails: boolean
  errorId: string
  retryCount: number
  isRetrying: boolean
  errorSeverity: 'low' | 'medium' | 'high' | 'critical'
  errorCategory: 'ui' | 'data' | 'network' | 'computation' | 'unknown'
  userContext: any
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false,
      errorId: '',
      retryCount: 0,
      isRetrying: false,
      errorSeverity: 'medium',
      errorCategory: 'unknown',
      userContext: this.captureUserContext()
    }
  }
  
  private captureUserContext() {
    return {
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight
      },
      localStorage: this.getRelevantLocalStorage(),
      sessionId: this.getSessionId()
    }
  }
  
  private getRelevantLocalStorage() {
    try {
      return {
        theme: localStorage.getItem('theme'),
        lastExperiment: localStorage.getItem('lastExperiment'),
        userPreferences: localStorage.getItem('userPreferences')
      }
    } catch {
      return {}
    }
  }
  
  private getSessionId() {
    let sessionId = sessionStorage.getItem('sessionId')
    if (!sessionId) {
      sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      sessionStorage.setItem('sessionId', sessionId)
    }
    return sessionId
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    const errorId = `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    return {
      hasError: true,
      error,
      errorId
    }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    const errorCategory = this.categorizeError(error)
    const errorSeverity = this.assessErrorSeverity(error, errorInfo)
    
    // Enhanced error tracking with telemetry
    if (this.props.enableTelemetry !== false) {
      metrics.trackError(
        'react_error_boundary',
        error.message,
        this.props.component || 'ErrorBoundary',
        error.stack,
        {
          category: errorCategory,
          severity: errorSeverity,
          retryCount: this.state.retryCount,
          userContext: this.state.userContext,
          componentStack: errorInfo?.componentStack || 'No component stack',
          props: this.sanitizeProps()
        }
      )
    }

    // Store comprehensive error info in state
    this.setState({
      error,
      errorInfo,
      errorCategory,
      errorSeverity
    })

    // Call optional error handler with enhanced context
    if (this.props.onError) {
      this.props.onError(error, errorInfo)
    }

    // Enhanced logging with security considerations
    const sanitizedLog = {
      errorId: this.state.errorId,
      message: error.message,
      category: errorCategory,
      severity: errorSeverity,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: this.sanitizeUrl(window.location.href),
      componentStack: errorInfo.componentStack?.substring(0, 1000), // Limit size
      stack: error.stack?.substring(0, 2000) // Limit size
    }
    
    console.error('Enhanced Error Boundary caught an error:', sanitizedLog)
    
    // Auto-retry for transient errors
    if (this.props.enableAutoRetry && this.shouldAutoRetry(error, errorSeverity)) {
      this.scheduleAutoRetry()
    }
    
    // Send to error reporting service
    if (this.props.enableErrorReporting !== false && errorSeverity !== 'low') {
      this.sendToErrorReporting(sanitizedLog)
    }
  }
  
  private categorizeError(error: Error): 'ui' | 'data' | 'network' | 'computation' | 'unknown' {
    const message = error.message.toLowerCase()
    const stack = error.stack?.toLowerCase() || ''
    
    if (message.includes('network') || message.includes('fetch') || message.includes('xhr')) {
      return 'network'
    }
    if (message.includes('computation') || message.includes('jax') || message.includes('calculation')) {
      return 'computation'
    }
    if (message.includes('render') || message.includes('component') || stack.includes('react')) {
      return 'ui'
    }
    if (message.includes('data') || message.includes('parse') || message.includes('json')) {
      return 'data'
    }
    return 'unknown'
  }
  
  private assessErrorSeverity(error: Error, errorInfo: ErrorInfo): 'low' | 'medium' | 'high' | 'critical' {
    // Critical errors that completely break the application
    if (error.message.includes('ChunkLoadError') || 
        error.message.includes('Script error') ||
        errorInfo?.componentStack?.includes('App')) {
      return 'critical'
    }
    
    // High severity for core functionality
    if (error.message.includes('causal') || 
        error.message.includes('experiment') ||
        errorInfo?.componentStack?.includes('CausalGraph') ||
        errorInfo?.componentStack?.includes('ExperimentBuilder')) {
      return 'high'
    }
    
    // Medium for UI components
    if (errorInfo?.componentStack?.includes('Dashboard') || 
        errorInfo?.componentStack?.includes('Metrics')) {
      return 'medium'
    }
    
    return 'low'
  }
  
  private sanitizeProps() {
    // Remove sensitive data from props before logging
    const { children, onError, ...safeProps } = this.props
    return safeProps
  }
  
  private sanitizeUrl(url: string) {
    try {
      const urlObj = new URL(url)
      // Remove sensitive query parameters
      urlObj.searchParams.delete('token')
      urlObj.searchParams.delete('key')
      urlObj.searchParams.delete('secret')
      return urlObj.toString()
    } catch {
      return url.split('?')[0] // Fallback to path only
    }
  }
  
  private shouldAutoRetry(error: Error, severity: string): boolean {
    return (
      this.state.retryCount < (this.props.maxRetries || 3) &&
      severity !== 'critical' &&
      !error.message.includes('ChunkLoadError')
    )
  }
  
  private scheduleAutoRetry() {
    this.setState({ isRetrying: true })
    
    const delay = (this.props.retryDelay || 2000) * Math.pow(2, this.state.retryCount) // Exponential backoff
    
    setTimeout(() => {
      this.setState(prevState => ({
        hasError: false,
        error: null,
        errorInfo: null,
        showDetails: false,
        retryCount: prevState.retryCount + 1,
        isRetrying: false
      }))
    }, delay)
  }
  
  private async sendToErrorReporting(errorData: any) {
    try {
      // In production, this would send to error reporting service like Sentry
      await fetch('/api/errors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(errorData)
      })
    } catch {
      // Silently fail - error reporting shouldn't break the app
    }
  }

  handleRetry = () => {
    // Track manual retry
    metrics.trackEvent('error_boundary_manual_retry', {
      errorId: this.state.errorId,
      retryCount: this.state.retryCount,
      errorCategory: this.state.errorCategory
    })
    
    this.setState(prevState => ({
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false,
      retryCount: prevState.retryCount + 1,
      isRetrying: false,
      errorId: ''
    }))
  }

  handleToggleDetails = () => {
    this.setState(prevState => ({
      showDetails: !prevState.showDetails
    }))
  }

  handleReportBug = () => {
    const { error, errorInfo, errorId, errorCategory, errorSeverity, userContext } = this.state
    
    const enhancedBugReport = {
      errorId,
      category: errorCategory,
      severity: errorSeverity,
      message: error?.message,
      stack: error?.stack?.substring(0, 2000), // Limit stack trace size
      componentStack: errorInfo?.componentStack?.substring(0, 1000),
      userContext: {
        ...userContext,
        url: this.sanitizeUrl(userContext.url) // Remove sensitive URL params
      },
      reproductionSteps: this.gatherReproductionSteps(),
      systemInfo: this.gatherSystemInfo(),
      experimentalContext: this.gatherExperimentContext()
    }

    // Track bug report submission
    metrics.trackEvent('bug_report_submitted', {
      errorId,
      category: errorCategory,
      severity: errorSeverity
    })
    
    console.log('Enhanced bug report generated:', enhancedBugReport)
    
    // Copy enhanced report to clipboard
    const reportText = this.formatBugReport(enhancedBugReport)
    navigator.clipboard.writeText(reportText)
      .then(() => {
        alert('📋 Detailed bug report copied to clipboard!\nPlease paste it when reporting the issue.')
      })
      .catch(() => {
        alert('Bug report logged to console - please check developer tools.')
      })
  }
  
  private gatherReproductionSteps() {
    // Try to reconstruct user actions from local storage and session storage
    try {
      const recentActions = JSON.parse(sessionStorage.getItem('recentActions') || '[]')
      return recentActions.slice(-5) // Last 5 actions
    } catch {
      return ['Unable to gather reproduction steps']
    }
  }
  
  private gatherSystemInfo() {
    return {
      platform: navigator.platform,
      language: navigator.language,
      cookiesEnabled: navigator.cookieEnabled,
      onLine: navigator.onLine,
      hardwareConcurrency: navigator.hardwareConcurrency,
      memory: (performance as any).memory ? {
        usedJSHeapSize: (performance as any).memory.usedJSHeapSize,
        totalJSHeapSize: (performance as any).memory.totalJSHeapSize,
        limit: (performance as any).memory.jsHeapSizeLimit
      } : undefined
    }
  }
  
  private gatherExperimentContext() {
    try {
      return {
        currentExperiment: localStorage.getItem('currentExperiment'),
        lastAction: sessionStorage.getItem('lastAction'),
        activeComponents: this.getActiveComponents()
      }
    } catch {
      return {}
    }
  }
  
  private getActiveComponents() {
    // Simple heuristic to identify active components
    const elements = document.querySelectorAll('[data-component]')
    return Array.from(elements).map(el => el.getAttribute('data-component')).filter(Boolean)
  }
  
  private formatBugReport(report: any) {
    return `# Causal UI Gym Bug Report

## Error Information
- **Error ID**: ${report.errorId}
- **Category**: ${report.category}
- **Severity**: ${report.severity}
- **Message**: ${report.message}

## System Information
- **Browser**: ${report.userContext.userAgent}
- **Platform**: ${report.systemInfo.platform}
- **Viewport**: ${report.userContext.viewport.width}x${report.userContext.viewport.height}
- **Online**: ${report.systemInfo.onLine}

## Context
- **URL**: ${report.userContext.url}
- **Timestamp**: ${report.userContext.timestamp}
- **Session ID**: ${report.userContext.sessionId}

## Technical Details
\`\`\`
${report.stack || 'No stack trace available'}
\`\`\`

## Component Stack
\`\`\`
${report.componentStack || 'No component stack available'}
\`\`\`

---
*Generated automatically by Causal UI Gym Error Boundary*`
  }

  render() {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback
      }

      const { error, errorInfo, showDetails, errorId } = this.state

      return (
        <Box p={3} maxWidth="800px" mx="auto">
          <Card sx={{ border: '2px solid', borderColor: 'error.main' }}>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <ErrorIcon color="error" sx={{ mr: 1, fontSize: 32 }} />
                <Box>
                  <Typography variant="h5" color="error">
                    {this.getErrorTitle()}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {this.getErrorDescription()}
                  </Typography>
                </Box>
              </Box>

              <Alert 
                severity={this.getAlertSeverity()} 
                sx={{ mb: 2 }}
                action={
                  this.state.isRetrying ? (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <CircularProgress size={16} />
                      <Typography variant="caption">Retrying...</Typography>
                    </Box>
                  ) : null
                }
              >
                <AlertTitle>{this.getAlertTitle()}</AlertTitle>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  {this.getRecoveryMessage()}
                  {this.props.enableErrorReporting !== false && ' The development team has been notified automatically.'}
                </Typography>
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="caption" color="text.secondary">
                    Error ID: {errorId} | Category: {this.state.errorCategory} | Severity: {this.state.errorSeverity}
                  </Typography>
                  {this.state.retryCount > 0 && (
                    <Typography variant="caption" color="warning.main">
                      Retry #{this.state.retryCount}
                    </Typography>
                  )}
                </Box>
              </Alert>

              <Box display="flex" gap={2} mb={2}>
                <Button
                  variant="contained"
                  startIcon={<RefreshIcon />}
                  onClick={this.handleRetry}
                  color="primary"
                >
                  Try Again
                </Button>
                
                <Button
                  variant="outlined"
                  startIcon={<BugReportIcon />}
                  onClick={this.handleReportBug}
                  color="secondary"
                >
                  Report Bug
                </Button>
                
                <Button
                  variant="text"
                  endIcon={<ExpandMoreIcon 
                    sx={{ 
                      transform: showDetails ? 'rotate(180deg)' : 'rotate(0deg)',
                      transition: 'transform 0.2s'
                    }} 
                  />}
                  onClick={this.handleToggleDetails}
                >
                  {showDetails ? 'Hide' : 'Show'} Details
                </Button>
              </Box>

              <Collapse in={showDetails}>
                <Box sx={{ bgcolor: 'grey.100', p: 2, borderRadius: 1 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Error Details:
                  </Typography>
                  
                  {error && (
                    <Box mb={2}>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                        <strong>Message:</strong> {error.message}
                      </Typography>
                    </Box>
                  )}

                  {error?.stack && (
                    <Box mb={2}>
                      <Typography variant="body2" gutterBottom>
                        <strong>Stack Trace:</strong>
                      </Typography>
                      <Box 
                        sx={{ 
                          bgcolor: 'white', 
                          p: 1, 
                          borderRadius: 1, 
                          maxHeight: '200px', 
                          overflow: 'auto',
                          fontFamily: 'monospace',
                          fontSize: '0.75rem',
                          whiteSpace: 'pre-wrap'
                        }}
                      >
                        {error.stack}
                      </Box>
                    </Box>
                  )}

                  {errorInfo?.componentStack && (
                    <Box>
                      <Typography variant="body2" gutterBottom>
                        <strong>Component Stack:</strong>
                      </Typography>
                      <Box 
                        sx={{ 
                          bgcolor: 'white', 
                          p: 1, 
                          borderRadius: 1, 
                          maxHeight: '150px', 
                          overflow: 'auto',
                          fontFamily: 'monospace',
                          fontSize: '0.75rem',
                          whiteSpace: 'pre-wrap'
                        }}
                      >
                        {errorInfo.componentStack}
                      </Box>
                    </Box>
                  )}
                </Box>
              </Collapse>

              <Box mt={2}>
                <Alert severity="info" variant="outlined">
                  <Typography variant="caption">
                    💡 <strong>What can you do?</strong><br/>
                    • Try refreshing the page or clicking "Try Again"<br/>
                    • Check your internet connection<br/>
                    • Clear your browser cache if the problem persists<br/>
                    • Contact support with the Error ID if you need immediate assistance
                  </Typography>
                </Alert>
              </Box>
            </CardContent>
          </Card>
        </Box>
      )
    }

    return this.props.children
  }
  
  // Helper methods for enhanced error UI
  private getErrorTitle(): string {
    switch (this.state.errorSeverity) {
      case 'critical': return 'Critical System Error'
      case 'high': return 'Application Error'
      case 'medium': return 'Component Error'
      case 'low': return 'Minor Issue Detected'
      default: return 'Something went wrong'
    }
  }
  
  private getErrorDescription(): string {
    switch (this.state.errorCategory) {
      case 'network': return 'Network connectivity issue detected'
      case 'computation': return 'Computational processing error'
      case 'data': return 'Data processing or validation error'
      case 'ui': return 'User interface rendering error'
      default: return `${this.state.errorCategory} error occurred`
    }
  }
  
  private getAlertSeverity(): 'error' | 'warning' | 'info' {
    switch (this.state.errorSeverity) {
      case 'critical':
      case 'high':
        return 'error'
      case 'medium':
        return 'warning'
      case 'low':
        return 'info'
      default:
        return 'error'
    }
  }
  
  private getAlertTitle(): string {
    switch (this.state.errorSeverity) {
      case 'critical': return 'Critical Error - Immediate Action Required'
      case 'high': return 'Application Error'
      case 'medium': return 'Component Error'
      case 'low': return 'Minor Issue'
      default: return 'Application Error'
    }
  }
  
  private getRecoveryMessage(): string {
    switch (this.state.errorCategory) {
      case 'network': 
        return 'Please check your internet connection and try again.'
      case 'computation':
        return 'A computational error occurred. The system will attempt to recover automatically.'
      case 'data':
        return 'There was an issue processing your data. Please verify your inputs and retry.'
      case 'ui':
        return 'A display issue occurred. Refreshing the page should resolve this.'
      default:
        return 'An unexpected error occurred. We\'re working to resolve this automatically.'
    }
  }
}

// Higher-order component for wrapping components with error boundary
export function withErrorBoundary<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  errorFallback?: ReactNode,
  onError?: (error: Error, errorInfo: ErrorInfo) => void
) {
  const WithErrorBoundaryComponent = (props: P) => (
    <ErrorBoundary fallback={errorFallback} onError={onError}>
      <WrappedComponent {...props} />
    </ErrorBoundary>
  )

  WithErrorBoundaryComponent.displayName = `withErrorBoundary(${WrappedComponent.displayName || WrappedComponent.name})`

  return WithErrorBoundaryComponent
}

// Enhanced hook for handling async errors with recovery strategies
export function useErrorHandler() {
  const handleError = useCallback((error: Error, context?: string, options?: {
    showToast?: boolean
    severity?: 'low' | 'medium' | 'high' | 'critical'
    recoverable?: boolean
    retryCallback?: () => Promise<void>
  }) => {
    const errorId = `async_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    const severity = options?.severity || 'medium'
    
    // Enhanced error tracking
    metrics.trackError(
      'async_error',
      error.message,
      context || 'unknown_component',
      error.stack,
      {
        errorId,
        severity,
        recoverable: options?.recoverable,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        url: window.location.href
      }
    )

    // Comprehensive logging with safety measures
    const sanitizedLog = {
      errorId,
      message: error.message,
      context,
      severity,
      recoverable: options?.recoverable,
      timestamp: new Date().toISOString(),
      stack: error.stack?.substring(0, 1000) // Limit stack trace size
    }
    
    console.error('Enhanced async error caught:', sanitizedLog)
    
    // Show user-friendly notification if requested
    if (options?.showToast) {
      // This would integrate with a toast notification system
      showErrorToast({
        message: getErrorMessage(error, severity),
        severity,
        errorId,
        retry: options.retryCallback
      })
    }
    
    // Automatic error reporting for high/critical errors
    if (severity === 'high' || severity === 'critical') {
      sendErrorReport(sanitizedLog)
    }
    
    return errorId
  }, [])

  const handleAsyncError = useCallback(async (
    asyncOperation: () => Promise<any>,
    context?: string,
    options?: {
      fallback?: any
      retries?: number
      retryDelay?: number
      onError?: (error: Error) => void
    }
  ): Promise<any> => {
    const maxRetries = options?.retries || 0
    const retryDelay = options?.retryDelay || 1000
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await asyncOperation()
      } catch (error) {
        const errorId = handleError(error as Error, context, {
          severity: attempt === maxRetries ? 'high' : 'medium',
          recoverable: attempt < maxRetries
        })
        
        if (options?.onError) {
          options.onError(error as Error)
        }
        
        // If this was the last attempt, return fallback or null
        if (attempt === maxRetries) {
          return options?.fallback ?? null
        }
        
        // Wait before retry with exponential backoff
        await new Promise(resolve => 
          setTimeout(resolve, retryDelay * Math.pow(2, attempt))
        )
      }
    }
    
    return null
  }, [handleError])

  return { handleError, handleAsyncError }
}

// Helper functions for error handling
function getErrorMessage(error: Error, severity: string): string {
  if (severity === 'critical') {
    return 'A critical error occurred. Please refresh the page.'
  }
  if (severity === 'high') {
    return 'An error occurred while processing your request. Please try again.'
  }
  if (error.message.includes('network')) {
    return 'Network error. Please check your connection and try again.'
  }
  return 'An unexpected error occurred. The issue has been reported.'
}

function showErrorToast(options: {
  message: string
  severity: string
  errorId: string
  retry?: () => Promise<void>
}) {
  // This would integrate with your toast notification system
  console.log('Would show toast:', options)
}

async function sendErrorReport(errorData: any) {
  try {
    await fetch('/api/errors/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(errorData)
    })
  } catch {
    // Silently fail - error reporting shouldn't break the app
  }
}