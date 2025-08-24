/**
 * Advanced Performance Dashboard for Real-time System Monitoring
 * 
 * Provides comprehensive performance metrics, optimization recommendations,
 * and automated system tuning for production-grade causal inference.
 */

import React, { useState, useEffect, useRef } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  LinearProgress,
  Chip,
  Alert,
  Button,
  IconButton,
  Tooltip,
  Switch,
  FormControlLabel,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper
} from '@mui/material'
import {
  TrendingUp,
  Speed,
  Memory,
  Cpu,
  Storage,
  NetworkCheck,
  Warning,
  CheckCircle,
  Settings,
  Refresh,
  Download,
  Timeline,
  Assessment
} from '@mui/icons-material'
import { Line, Bar, Doughnut, Radar } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  ArcElement,
  RadialLinearScale
} from 'chart.js'

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  ChartTooltip,
  Legend,
  ArcElement,
  RadialLinearScale
)

interface PerformanceMetrics {
  cpu_usage: number
  memory_usage: number
  gpu_usage: number
  disk_usage: number
  network_throughput: number
  cache_hit_rate: number
  computation_throughput: number
  error_rate: number
  response_time: number
  queue_size: number
  active_workers: number
  completed_tasks: number
  failed_tasks: number
  system_health_score: number
}

interface OptimizationRecommendation {
  type: 'performance' | 'resource' | 'stability' | 'cost'
  priority: 'low' | 'medium' | 'high' | 'critical'
  title: string
  description: string
  estimated_improvement: string
  implementation_effort: 'low' | 'medium' | 'high'
  auto_applicable: boolean
}

interface SystemAlert {
  id: string
  severity: 'info' | 'warning' | 'error' | 'critical'
  title: string
  message: string
  timestamp: Date
  acknowledged: boolean
  auto_resolvable: boolean
}

const AdvancedPerformanceDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    cpu_usage: 0,
    memory_usage: 0,
    gpu_usage: 0,
    disk_usage: 0,
    network_throughput: 0,
    cache_hit_rate: 0,
    computation_throughput: 0,
    error_rate: 0,
    response_time: 0,
    queue_size: 0,
    active_workers: 0,
    completed_tasks: 0,
    failed_tasks: 0,
    system_health_score: 0
  })

  const [recommendations, setRecommendations] = useState<OptimizationRecommendation[]>([])
  const [alerts, setAlerts] = useState<SystemAlert[]>([])
  const [metricsHistory, setMetricsHistory] = useState<PerformanceMetrics[]>([])
  const [autoOptimization, setAutoOptimization] = useState(false)
  const [refreshInterval, setRefreshInterval] = useState(5000)
  const [showDetailModal, setShowDetailModal] = useState(false)
  const [selectedMetric, setSelectedMetric] = useState<string>('')

  const intervalRef = useRef<NodeJS.Timeout>()

  // Simulate real-time metrics updates
  useEffect(() => {
    const updateMetrics = () => {
      const newMetrics: PerformanceMetrics = {
        cpu_usage: Math.random() * 100,
        memory_usage: Math.random() * 100,
        gpu_usage: Math.random() * 100,
        disk_usage: Math.random() * 100,
        network_throughput: Math.random() * 1000,
        cache_hit_rate: 80 + Math.random() * 20,
        computation_throughput: 100 + Math.random() * 500,
        error_rate: Math.random() * 5,
        response_time: 50 + Math.random() * 200,
        queue_size: Math.floor(Math.random() * 50),
        active_workers: Math.floor(Math.random() * 10),
        completed_tasks: metrics.completed_tasks + Math.floor(Math.random() * 5),
        failed_tasks: metrics.failed_tasks + Math.floor(Math.random() * 2),
        system_health_score: 70 + Math.random() * 30
      }

      setMetrics(newMetrics)
      setMetricsHistory(prev => [...prev.slice(-99), newMetrics]) // Keep last 100 points

      // Generate recommendations based on metrics
      updateRecommendations(newMetrics)
      
      // Generate alerts for critical conditions
      updateAlerts(newMetrics)
    }

    intervalRef.current = setInterval(updateMetrics, refreshInterval)
    updateMetrics() // Initial load

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [refreshInterval, metrics.completed_tasks, metrics.failed_tasks])

  const updateRecommendations = (currentMetrics: PerformanceMetrics) => {
    const newRecommendations: OptimizationRecommendation[] = []

    if (currentMetrics.cpu_usage > 80) {
      newRecommendations.push({
        type: 'performance',
        priority: 'high',
        title: 'High CPU Usage',
        description: 'CPU utilization is consistently high. Consider scaling horizontally or optimizing algorithms.',
        estimated_improvement: '20-30% response time reduction',
        implementation_effort: 'medium',
        auto_applicable: false
      })
    }

    if (currentMetrics.cache_hit_rate < 70) {
      newRecommendations.push({
        type: 'performance',
        priority: 'medium',
        title: 'Low Cache Hit Rate',
        description: 'Cache efficiency is below optimal. Adjust cache size or TTL policies.',
        estimated_improvement: '15-25% faster response times',
        implementation_effort: 'low',
        auto_applicable: true
      })
    }

    if (currentMetrics.error_rate > 3) {
      newRecommendations.push({
        type: 'stability',
        priority: 'critical',
        title: 'High Error Rate',
        description: 'System error rate is above acceptable threshold. Review recent changes.',
        estimated_improvement: 'Improved system stability',
        implementation_effort: 'high',
        auto_applicable: false
      })
    }

    if (currentMetrics.memory_usage > 90) {
      newRecommendations.push({
        type: 'resource',
        priority: 'high',
        title: 'Memory Pressure',
        description: 'Memory usage is critically high. Consider garbage collection tuning.',
        estimated_improvement: 'Prevent memory-related crashes',
        implementation_effort: 'low',
        auto_applicable: true
      })
    }

    setRecommendations(newRecommendations)
  }

  const updateAlerts = (currentMetrics: PerformanceMetrics) => {
    const newAlerts: SystemAlert[] = []

    if (currentMetrics.system_health_score < 60) {
      newAlerts.push({
        id: `health_${Date.now()}`,
        severity: 'critical',
        title: 'System Health Critical',
        message: `Overall system health has dropped to ${currentMetrics.system_health_score.toFixed(1)}%`,
        timestamp: new Date(),
        acknowledged: false,
        auto_resolvable: false
      })
    }

    if (currentMetrics.queue_size > 30) {
      newAlerts.push({
        id: `queue_${Date.now()}`,
        severity: 'warning',
        title: 'High Queue Size',
        message: `Task queue has grown to ${currentMetrics.queue_size} items`,
        timestamp: new Date(),
        acknowledged: false,
        auto_resolvable: true
      })
    }

    // Only add new unique alerts
    setAlerts(prev => {
      const existing = prev.filter(alert => 
        !newAlerts.some(newAlert => 
          newAlert.title === alert.title && !alert.acknowledged
        )
      )
      return [...existing, ...newAlerts].slice(-10) // Keep last 10 alerts
    })
  }

  const getHealthColor = (score: number) => {
    if (score >= 80) return 'success'
    if (score >= 60) return 'warning'
    return 'error'
  }

  const getUsageColor = (usage: number) => {
    if (usage < 60) return 'success'
    if (usage < 80) return 'warning'
    return 'error'
  }

  const applyOptimization = async (recommendation: OptimizationRecommendation) => {
    if (!recommendation.auto_applicable) return

    // Simulate applying optimization
    console.log(`Applying optimization: ${recommendation.title}`)
    
    // In a real implementation, this would call the backend
    try {
      const response = await fetch('/api/optimization/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          optimization_type: recommendation.type,
          title: recommendation.title 
        })
      })
      
      if (response.ok) {
        // Remove applied recommendation
        setRecommendations(prev => 
          prev.filter(rec => rec.title !== recommendation.title)
        )
      }
    } catch (error) {
      console.error('Failed to apply optimization:', error)
    }
  }

  const acknowledgeAlert = (alertId: string) => {
    setAlerts(prev => 
      prev.map(alert => 
        alert.id === alertId ? { ...alert, acknowledged: true } : alert
      )
    )
  }

  const exportMetrics = () => {
    const dataStr = JSON.stringify({
      current_metrics: metrics,
      history: metricsHistory,
      recommendations: recommendations,
      alerts: alerts,
      timestamp: new Date().toISOString()
    }, null, 2)
    
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = `performance_metrics_${Date.now()}.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  // Chart data for metrics history
  const chartData = {
    labels: metricsHistory.map((_, index) => index.toString()),
    datasets: [
      {
        label: 'CPU Usage %',
        data: metricsHistory.map(m => m.cpu_usage),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        tension: 0.1
      },
      {
        label: 'Memory Usage %',
        data: metricsHistory.map(m => m.memory_usage),
        borderColor: 'rgb(54, 162, 235)',
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        tension: 0.1
      },
      {
        label: 'GPU Usage %',
        data: metricsHistory.map(m => m.gpu_usage),
        borderColor: 'rgb(255, 205, 86)',
        backgroundColor: 'rgba(255, 205, 86, 0.2)',
        tension: 0.1
      }
    ]
  }

  const throughputChartData = {
    labels: ['Computation', 'Network', 'Cache Hits', 'Queue Processing'],
    datasets: [{
      label: 'Throughput Metrics',
      data: [
        metrics.computation_throughput,
        metrics.network_throughput,
        metrics.cache_hit_rate * 10, // Scale for visibility
        Math.max(0, 100 - metrics.queue_size * 2) // Inverse queue size
      ],
      backgroundColor: [
        'rgba(255, 99, 132, 0.8)',
        'rgba(54, 162, 235, 0.8)',
        'rgba(255, 205, 86, 0.8)',
        'rgba(75, 192, 192, 0.8)'
      ]
    }]
  }

  const radarChartData = {
    labels: ['Performance', 'Stability', 'Efficiency', 'Scalability', 'Reliability'],
    datasets: [{
      label: 'System Health Radar',
      data: [
        100 - metrics.response_time / 5, // Performance (inverse of response time)
        100 - metrics.error_rate * 20,   // Stability (inverse of error rate)
        metrics.cache_hit_rate,          // Efficiency
        Math.min(100, metrics.active_workers * 10), // Scalability
        metrics.system_health_score      // Reliability
      ],
      backgroundColor: 'rgba(54, 162, 235, 0.2)',
      borderColor: 'rgba(54, 162, 235, 1)',
      borderWidth: 2
    }]
  }

  return (
    <Box sx={{ p: 3, backgroundColor: '#f5f5f5', minHeight: '100vh' }}>
      <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <Assessment color="primary" />
        Advanced Performance Dashboard
        <Chip 
          label={`Health: ${metrics.system_health_score.toFixed(1)}%`}
          color={getHealthColor(metrics.system_health_score)}
          variant="outlined"
        />
      </Typography>

      {/* Control Panel */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item>
              <FormControlLabel
                control={
                  <Switch 
                    checked={autoOptimization}
                    onChange={(e) => setAutoOptimization(e.target.checked)}
                  />
                }
                label="Auto Optimization"
              />
            </Grid>
            <Grid item>
              <FormControl size="small" sx={{ minWidth: 120 }}>
                <InputLabel>Refresh Rate</InputLabel>
                <Select
                  value={refreshInterval}
                  label="Refresh Rate"
                  onChange={(e) => setRefreshInterval(Number(e.target.value))}
                >
                  <MenuItem value={1000}>1s</MenuItem>
                  <MenuItem value={5000}>5s</MenuItem>
                  <MenuItem value={10000}>10s</MenuItem>
                  <MenuItem value={30000}>30s</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item>
              <Button 
                variant="outlined" 
                startIcon={<Refresh />}
                onClick={() => window.location.reload()}
              >
                Refresh
              </Button>
            </Grid>
            <Grid item>
              <Button 
                variant="outlined" 
                startIcon={<Download />}
                onClick={exportMetrics}
              >
                Export
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Active Alerts */}
      {alerts.filter(a => !a.acknowledged).length > 0 && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              <Warning color="warning" sx={{ mr: 1 }} />
              Active Alerts ({alerts.filter(a => !a.acknowledged).length})
            </Typography>
            {alerts.filter(a => !a.acknowledged).slice(0, 3).map((alert) => (
              <Alert 
                key={alert.id}
                severity={alert.severity}
                sx={{ mb: 1 }}
                action={
                  <Button 
                    color="inherit" 
                    size="small"
                    onClick={() => acknowledgeAlert(alert.id)}
                  >
                    Acknowledge
                  </Button>
                }
              >
                <strong>{alert.title}</strong>: {alert.message}
              </Alert>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Resource Usage Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Cpu color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">CPU Usage</Typography>
              </Box>
              <Typography variant="h4" color={getUsageColor(metrics.cpu_usage)}>
                {metrics.cpu_usage.toFixed(1)}%
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={metrics.cpu_usage} 
                color={getUsageColor(metrics.cpu_usage)}
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Memory color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Memory</Typography>
              </Box>
              <Typography variant="h4" color={getUsageColor(metrics.memory_usage)}>
                {metrics.memory_usage.toFixed(1)}%
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={metrics.memory_usage} 
                color={getUsageColor(metrics.memory_usage)}
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Speed color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">GPU Usage</Typography>
              </Box>
              <Typography variant="h4" color={getUsageColor(metrics.gpu_usage)}>
                {metrics.gpu_usage.toFixed(1)}%
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={metrics.gpu_usage} 
                color={getUsageColor(metrics.gpu_usage)}
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Storage color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Disk Usage</Typography>
              </Box>
              <Typography variant="h4" color={getUsageColor(metrics.disk_usage)}>
                {metrics.disk_usage.toFixed(1)}%
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={metrics.disk_usage} 
                color={getUsageColor(metrics.disk_usage)}
                sx={{ mt: 1 }}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Performance Metrics */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Cache Hit Rate</Typography>
              <Typography variant="h4" color="primary">
                {metrics.cache_hit_rate.toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Target: >95%
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Response Time</Typography>
              <Typography variant="h4" color={metrics.response_time < 100 ? 'success' : 'warning'}>
                {metrics.response_time.toFixed(0)}ms
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Target: &lt;100ms
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Throughput</Typography>
              <Typography variant="h4" color="primary">
                {metrics.computation_throughput.toFixed(0)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                tasks/min
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <Timeline sx={{ mr: 1 }} />
                Resource Usage Over Time
              </Typography>
              <Box sx={{ height: 300 }}>
                <Line data={chartData} options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    y: {
                      beginAtZero: true,
                      max: 100
                    }
                  }
                }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>System Health Radar</Typography>
              <Box sx={{ height: 300 }}>
                <Radar data={radarChartData} options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    r: {
                      beginAtZero: true,
                      max: 100
                    }
                  }
                }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Optimization Recommendations */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            <TrendingUp sx={{ mr: 1 }} />
            Optimization Recommendations ({recommendations.length})
          </Typography>
          
          {recommendations.length === 0 ? (
            <Alert severity="success" icon={<CheckCircle />}>
              System is performing optimally. No recommendations at this time.
            </Alert>
          ) : (
            <Grid container spacing={2}>
              {recommendations.map((rec, index) => (
                <Grid item xs={12} md={6} key={index}>
                  <Card variant="outlined">
                    <CardContent>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                        <Typography variant="h6" color={rec.priority === 'critical' ? 'error' : 'primary'}>
                          {rec.title}
                        </Typography>
                        <Chip 
                          size="small"
                          label={rec.priority}
                          color={rec.priority === 'critical' ? 'error' : rec.priority === 'high' ? 'warning' : 'default'}
                        />
                      </Box>
                      
                      <Typography variant="body2" sx={{ mb: 2 }}>
                        {rec.description}
                      </Typography>
                      
                      <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                        <Chip size="small" label={`${rec.type} optimization`} />
                        <Chip size="small" label={`${rec.implementation_effort} effort`} />
                      </Box>
                      
                      <Typography variant="body2" color="success.main" sx={{ mb: 2 }}>
                        Expected: {rec.estimated_improvement}
                      </Typography>
                      
                      {rec.auto_applicable && autoOptimization && (
                        <Button 
                          size="small" 
                          variant="contained" 
                          onClick={() => applyOptimization(rec)}
                        >
                          Apply Automatically
                        </Button>
                      )}
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          )}
        </CardContent>
      </Card>

      {/* Task Statistics */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Task Statistics</Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Metric</TableCell>
                      <TableCell align="right">Current</TableCell>
                      <TableCell align="right">Status</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow>
                      <TableCell>Active Workers</TableCell>
                      <TableCell align="right">{metrics.active_workers}</TableCell>
                      <TableCell align="right">
                        <Chip size="small" label="Normal" color="success" />
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Queue Size</TableCell>
                      <TableCell align="right">{metrics.queue_size}</TableCell>
                      <TableCell align="right">
                        <Chip 
                          size="small" 
                          label={metrics.queue_size > 20 ? "High" : "Normal"} 
                          color={metrics.queue_size > 20 ? "warning" : "success"} 
                        />
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Completed Tasks</TableCell>
                      <TableCell align="right">{metrics.completed_tasks}</TableCell>
                      <TableCell align="right">
                        <Chip size="small" label="↗" color="success" />
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Failed Tasks</TableCell>
                      <TableCell align="right">{metrics.failed_tasks}</TableCell>
                      <TableCell align="right">
                        <Chip 
                          size="small" 
                          label={metrics.failed_tasks > 10 ? "High" : "Low"} 
                          color={metrics.failed_tasks > 10 ? "error" : "success"} 
                        />
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Error Rate</TableCell>
                      <TableCell align="right">{metrics.error_rate.toFixed(2)}%</TableCell>
                      <TableCell align="right">
                        <Chip 
                          size="small" 
                          label={metrics.error_rate > 3 ? "High" : "Normal"} 
                          color={metrics.error_rate > 3 ? "error" : "success"} 
                        />
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>Throughput Analysis</Typography>
              <Box sx={{ height: 300 }}>
                <Bar data={throughputChartData} options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  indexAxis: 'y' as const,
                  plugins: {
                    legend: {
                      display: false
                    }
                  }
                }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}

export default AdvancedPerformanceDashboard