import React, { Component, ErrorInfo, ReactNode } from 'react'
import { 
  Alert, 
  AlertTitle, 
  Box, 
  Button, 
  Card, 
  CardContent, 
  Typography, 
  Collapse,
  IconButton
} from '@mui/material'
import { 
  Error as ErrorIcon, 
  ExpandMore as ExpandMoreIcon,
  Refresh as RefreshIcon,
  BugReport as BugReportIcon 
} from '@mui/icons-material'
import { metrics } from '../utils/metrics'

interface Props {
  children: ReactNode
  fallback?: ReactNode
  onError?: (error: Error, errorInfo: ErrorInfo) => void
}

interface State {
  hasError: boolean
  error: Error | null
  errorInfo: ErrorInfo | null
  showDetails: boolean
  errorId: string
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false,
      errorId: ''
    }
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
    // Log error to metrics system
    metrics.trackError(
      'react_error_boundary',
      error.message,
      'ErrorBoundary',
      error.stack
    )

    // Store error info in state
    this.setState({
      error,
      errorInfo
    })

    // Call optional error handler
    if (this.props.onError) {
      this.props.onError(error, errorInfo)
    }

    // Log detailed error information
    console.error('Error Boundary caught an error:', {
      error,
      errorInfo,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href
    })
  }

  handleRetry = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      showDetails: false,
      errorId: ''
    })
  }

  handleToggleDetails = () => {
    this.setState(prevState => ({
      showDetails: !prevState.showDetails
    }))
  }

  handleReportBug = () => {
    const { error, errorInfo, errorId } = this.state
    
    const bugReport = {
      errorId,
      message: error?.message,
      stack: error?.stack,
      componentStack: errorInfo?.componentStack,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href
    }

    // In a real app, this would send to error reporting service
    console.log('Bug report generated:', bugReport)
    
    // Copy to clipboard for easy sharing
    navigator.clipboard.writeText(JSON.stringify(bugReport, null, 2))
      .then(() => {
        alert('Bug report copied to clipboard!')
      })
      .catch(() => {
        alert('Bug report logged to console')
      })
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
                <Typography variant="h5" color="error">
                  Something went wrong
                </Typography>
              </Box>

              <Alert severity="error" sx={{ mb: 2 }}>
                <AlertTitle>Application Error</AlertTitle>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  An unexpected error occurred while rendering this component. 
                  The development team has been notified.
                </Typography>
                
                <Typography variant="caption" color="text.secondary">
                  Error ID: {errorId}
                </Typography>
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
                <Typography variant="caption" color="text.secondary">
                  If this problem persists, please contact support with the Error ID above.
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Box>
      )
    }

    return this.props.children
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

// Hook for handling async errors in components
export function useErrorHandler() {
  const handleError = (error: Error, context?: string) => {
    // Log error to metrics
    metrics.trackError(
      'async_error',
      error.message,
      context || 'unknown_component',
      error.stack
    )

    // Log to console
    console.error('Async error caught:', {
      error,
      context,
      timestamp: new Date().toISOString()
    })

    // In a real app, you might want to show a toast notification
    // or update some global error state
  }

  return { handleError }
}