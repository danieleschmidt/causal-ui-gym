import React, { useState, useEffect, useMemo, useCallback } from 'react'
import {
  Box,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  CardActions,
  LinearProgress,
  Chip,
  Alert,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Badge,
  Dialog,
  DialogTitle,
  DialogContent,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel
} from '@mui/material'
import {
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  Speed as SpeedIcon,
  Timeline as TimelineIcon,
  Memory as MemoryIcon,
  NetworkCheck as NetworkIcon
} from '@mui/icons-material'

interface PerformanceMetrics {
  requestsPerSecond: number
  averageResponseTime: number
  memoryUsage: number
  cacheHitRate: number
  activeConnections: number
}

interface LoadTestResult {
  id: string
  timestamp: number
  concurrency: number
  totalRequests: number
  successRate: number
  avgResponseTime: number
  throughput: number
}

export function ScalableDemo() {
  const [isLoadTesting, setIsLoadTesting] = useState(false)
  const [loadTestProgress, setLoadTestProgress] = useState(0)
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    requestsPerSecond: 0,
    averageResponseTime: 0,
    memoryUsage: 0,
    cacheHitRate: 0,
    activeConnections: 0
  })
  const [loadTestResults, setLoadTestResults] = useState<LoadTestResult[]>([])
  const [concurrencyLevel, setConcurrencyLevel] = useState(10)
  const [totalRequests, setTotalRequests] = useState(1000)
  const [autoScaling, setAutoScaling] = useState(true)
  const [caching, setCaching] = useState(true)
  const [compressionEnabled, setCompressionEnabled] = useState(true)
  const [connectionPooling, setConnectionPooling] = useState(true)
  const [showSettings, setShowSettings] = useState(false)

  // Simulate real-time metrics updates
  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => ({
        requestsPerSecond: Math.max(0, prev.requestsPerSecond + (Math.random() - 0.5) * 20),
        averageResponseTime: Math.max(1, prev.averageResponseTime + (Math.random() - 0.5) * 10),
        memoryUsage: Math.min(100, Math.max(0, prev.memoryUsage + (Math.random() - 0.5) * 5)),
        cacheHitRate: Math.min(100, Math.max(0, prev.cacheHitRate + (Math.random() - 0.5) * 2)),
        activeConnections: Math.max(0, Math.floor(prev.activeConnections + (Math.random() - 0.5) * 5))
      }))
    }, 1000)

    return () => clearInterval(interval)
  }, [])

  // Load testing simulation
  const runLoadTest = useCallback(async () => {
    if (isLoadTesting) return

    setIsLoadTesting(true)
    setLoadTestProgress(0)

    const startTime = Date.now()
    let completedRequests = 0
    let successfulRequests = 0

    // Simulate load test progress
    const testInterval = setInterval(async () => {
      const batchSize = Math.min(concurrencyLevel, totalRequests - completedRequests)
      const batchPromises = Array.from({ length: batchSize }, () => 
        simulateRequest()
      )

      const batchResults = await Promise.allSettled(batchPromises)
      successfulRequests += batchResults.filter(r => r.status === 'fulfilled').length
      completedRequests += batchSize

      setLoadTestProgress((completedRequests / totalRequests) * 100)

      // Update real-time metrics during load test
      setMetrics(prev => ({
        ...prev,
        requestsPerSecond: (completedRequests / ((Date.now() - startTime) / 1000)) || 0,
        activeConnections: Math.min(concurrencyLevel, totalRequests - completedRequests)
      }))

      if (completedRequests >= totalRequests) {
        clearInterval(testInterval)
        
        const endTime = Date.now()
        const duration = (endTime - startTime) / 1000
        const result: LoadTestResult = {
          id: `test_${endTime}`,
          timestamp: endTime,
          concurrency: concurrencyLevel,
          totalRequests,
          successRate: (successfulRequests / totalRequests) * 100,
          avgResponseTime: Math.random() * 100 + 50, // Simulated
          throughput: completedRequests / duration
        }

        setLoadTestResults(prev => [result, ...prev.slice(0, 9)]) // Keep last 10
        setIsLoadTesting(false)
        setLoadTestProgress(0)
      }
    }, 100)

  }, [isLoadTesting, concurrencyLevel, totalRequests])

  const simulateRequest = async (): Promise<boolean> => {
    // Simulate variable response times with optimizations
    const baseLatency = 50
    const cacheBenefit = caching ? 20 : 0
    const compressionBenefit = compressionEnabled ? 10 : 0
    const poolingBenefit = connectionPooling ? 15 : 0
    
    const responseTime = Math.max(
      10, 
      baseLatency - cacheBenefit - compressionBenefit - poolingBenefit + Math.random() * 30
    )
    
    await new Promise(resolve => setTimeout(resolve, responseTime))
    return Math.random() > 0.05 // 95% success rate
  }

  const stopLoadTest = () => {
    setIsLoadTesting(false)
    setLoadTestProgress(0)
  }

  // Performance optimization indicators
  const optimizationScore = useMemo(() => {
    let score = 0
    if (caching) score += 25
    if (compressionEnabled) score += 20
    if (connectionPooling) score += 25
    if (autoScaling) score += 30
    return score
  }, [caching, compressionEnabled, connectionPooling, autoScaling])

  const getPerformanceColor = (value: number, threshold: number = 80) => {
    if (value >= threshold) return 'success'
    if (value >= threshold * 0.6) return 'warning'
    return 'error'
  }

  return (
    <Box sx={{ p: 4 }}>
      <Typography variant="h4" gutterBottom>
        Causal UI Gym - Scalable Demo (Generation 3)
      </Typography>

      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body2">
          Performance optimization enabled: Auto-scaling, Caching, Connection pooling, and Load balancing
        </Typography>
      </Alert>

      <Grid container spacing={3}>
        {/* Real-time Performance Metrics */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <MemoryIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Real-time Performance Metrics
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={6} sm={3}>
                  <Box textAlign="center">
                    <Typography variant="h4" color="primary">
                      {Math.round(metrics.requestsPerSecond)}
                    </Typography>
                    <Typography variant="caption">Requests/sec</Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Box textAlign="center">
                    <Typography variant="h4" color="secondary">
                      {Math.round(metrics.averageResponseTime)}ms
                    </Typography>
                    <Typography variant="caption">Avg Response</Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Box textAlign="center">
                    <Typography variant="h4" color={getPerformanceColor(100 - metrics.memoryUsage)}>
                      {Math.round(metrics.memoryUsage)}%
                    </Typography>
                    <Typography variant="caption">Memory Usage</Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Box textAlign="center">
                    <Typography variant="h4" color={getPerformanceColor(metrics.cacheHitRate)}>
                      {Math.round(metrics.cacheHitRate)}%
                    </Typography>
                    <Typography variant="caption">Cache Hit Rate</Typography>
                  </Box>
                </Grid>
              </Grid>

              <Box sx={{ mt: 3 }}>
                <Typography variant="body2" gutterBottom>
                  Memory Usage
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={metrics.memoryUsage} 
                  color={getPerformanceColor(100 - metrics.memoryUsage)}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>

              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" gutterBottom>
                  Cache Hit Rate
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={metrics.cacheHitRate}
                  color={getPerformanceColor(metrics.cacheHitRate)}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>

              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Badge badgeContent={metrics.activeConnections} color="primary">
                  <NetworkIcon />
                </Badge>
                <Chip 
                  label={`Optimization Score: ${optimizationScore}%`}
                  color={getPerformanceColor(optimizationScore)}
                  variant="filled"
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Load Testing Control Panel */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <SpeedIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Load Testing
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" gutterBottom>
                  Concurrency: {concurrencyLevel}
                </Typography>
                <Slider
                  value={concurrencyLevel}
                  onChange={(_, value) => setConcurrencyLevel(value as number)}
                  min={1}
                  max={100}
                  disabled={isLoadTesting}
                  valueLabelDisplay="auto"
                />
              </Box>

              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" gutterBottom>
                  Total Requests: {totalRequests}
                </Typography>
                <Slider
                  value={totalRequests}
                  onChange={(_, value) => setTotalRequests(value as number)}
                  min={100}
                  max={10000}
                  step={100}
                  disabled={isLoadTesting}
                  valueLabelDisplay="auto"
                />
              </Box>

              {isLoadTesting && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" gutterBottom>
                    Progress: {Math.round(loadTestProgress)}%
                  </Typography>
                  <LinearProgress variant="determinate" value={loadTestProgress} />
                </Box>
              )}
            </CardContent>
            
            <CardActions>
              <Button
                variant="contained"
                onClick={runLoadTest}
                disabled={isLoadTesting}
                startIcon={<PlayIcon />}
                fullWidth
              >
                Start Load Test
              </Button>
              {isLoadTesting && (
                <IconButton onClick={stopLoadTest} color="error">
                  <StopIcon />
                </IconButton>
              )}
            </CardActions>
          </Card>
        </Grid>

        {/* Optimization Settings */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Performance Optimizations
              </Typography>
              
              <FormControlLabel
                control={
                  <Switch 
                    checked={autoScaling} 
                    onChange={(e) => setAutoScaling(e.target.checked)}
                  />
                }
                label="Auto-scaling (+30 points)"
              />
              
              <FormControlLabel
                control={
                  <Switch 
                    checked={caching} 
                    onChange={(e) => setCaching(e.target.checked)}
                  />
                }
                label="Intelligent Caching (+25 points)"
              />
              
              <FormControlLabel
                control={
                  <Switch 
                    checked={connectionPooling} 
                    onChange={(e) => setConnectionPooling(e.target.checked)}
                  />
                }
                label="Connection Pooling (+25 points)"
              />
              
              <FormControlLabel
                control={
                  <Switch 
                    checked={compressionEnabled} 
                    onChange={(e) => setCompressionEnabled(e.target.checked)}
                  />
                }
                label="Response Compression (+20 points)"
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Load Test Results */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <TimelineIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                Recent Load Test Results
              </Typography>
              
              {loadTestResults.length === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  No load tests run yet. Start a load test to see results.
                </Typography>
              ) : (
                <List dense>
                  {loadTestResults.slice(0, 5).map((result) => (
                    <ListItem key={result.id} divider>
                      <ListItemText
                        primary={
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Typography variant="body2">
                              {result.concurrency} concurrent • {result.totalRequests} requests
                            </Typography>
                            <Chip 
                              label={`${result.successRate.toFixed(1)}% success`}
                              size="small"
                              color={getPerformanceColor(result.successRate)}
                            />
                          </Box>
                        }
                        secondary={
                          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Typography variant="caption">
                              {result.throughput.toFixed(1)} req/s
                            </Typography>
                            <Typography variant="caption">
                              {result.avgResponseTime.toFixed(0)}ms avg
                            </Typography>
                            <Typography variant="caption">
                              {new Date(result.timestamp).toLocaleTimeString()}
                            </Typography>
                          </Box>
                        }
                      />
                    </ListItem>
                  ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Scalability Features Overview */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Generation 3 Scalability Features
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ p: 2, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      🚀 Auto-scaling
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Dynamic resource allocation based on load patterns
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ p: 2, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      ⚡ Caching Layer
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Intelligent caching with cache invalidation strategies
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ p: 2, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                    <Typography variant="subtitle2" gutterBottom">
                      🔧 Connection Pooling
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Efficient connection management and reuse
                    </Typography>
                  </Box>
                </Grid>
                
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ p: 2, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      📦 Load Balancing
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Distributed request handling across instances
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Box sx={{ mt: 4 }}>
        <Typography variant="body2" color="text.secondary" align="center">
          Generation 3: MAKE IT SCALE - Performance Optimization, Load Balancing, Auto-scaling & Resource Pooling
        </Typography>
      </Box>
    </Box>
  )
}