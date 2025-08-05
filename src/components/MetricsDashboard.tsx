import React, { useState, useEffect } from 'react'
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Alert,
  IconButton,
  Tooltip
} from '@mui/material'
import {
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Info as InfoIcon
} from '@mui/icons-material'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts'
import { CausalMetric, CausalResult } from '../types'
import { formatMetricValue, downloadResults } from '../utils'

interface MetricsDashboardProps {
  experimentId: string
  results: CausalResult[]
  isLoading?: boolean
  onRefresh?: () => void
  className?: string
}

interface MetricsData {
  ate_values: number[]
  confidence_intervals: [number, number][]
  p_values: number[]
  computation_times: number[]
  sample_sizes: number[]
  timestamps: string[]
}

export function MetricsDashboard({
  experimentId,
  results,
  isLoading = false,
  onRefresh,
  className
}: MetricsDashboardProps) {
  const [metricsData, setMetricsData] = useState<MetricsData>({
    ate_values: [],
    confidence_intervals: [],
    p_values: [],
    computation_times: [],
    sample_sizes: [],
    timestamps: []
  })

  useEffect(() => {
    if (results.length > 0) {
      const processedData: MetricsData = {
        ate_values: [],
        confidence_intervals: [],
        p_values: [],
        computation_times: [],
        sample_sizes: [],
        timestamps: []
      }

      results.forEach((result) => {
        const ateMetric = result.metrics.find(m => m.metric_type === 'ate')
        if (ateMetric) {
          processedData.ate_values.push(ateMetric.value)
          processedData.confidence_intervals.push(ateMetric.confidence_interval || [0, 0])
          processedData.p_values.push(ateMetric.p_value || 1)
          processedData.computation_times.push(ateMetric.computation_time)
          processedData.sample_sizes.push(ateMetric.sample_size)
          processedData.timestamps.push(result.created_at.toISOString())
        }
      })

      setMetricsData(processedData)
    }
  }, [results])

  const handleDownload = () => {
    const exportData = results.map((result, index) => ({
      experiment_id: experimentId,
      intervention_variable: result.intervention.variable,
      intervention_value: result.intervention.value,
      outcome_variable: result.outcome_variable,
      ate: metricsData.ate_values[index],
      ci_lower: metricsData.confidence_intervals[index]?.[0],
      ci_upper: metricsData.confidence_intervals[index]?.[1],
      p_value: metricsData.p_values[index],
      computation_time: metricsData.computation_times[index],
      sample_size: metricsData.sample_sizes[index],
      timestamp: metricsData.timestamps[index]
    }))

    downloadResults(exportData, `experiment_${experimentId}_results`, 'csv')
  }

  const averageATE = metricsData.ate_values.length > 0 
    ? metricsData.ate_values.reduce((sum, val) => sum + val, 0) / metricsData.ate_values.length
    : 0

  const significantResults = metricsData.p_values.filter(p => p < 0.05).length
  const significanceRate = metricsData.p_values.length > 0 
    ? (significantResults / metricsData.p_values.length) * 100
    : 0

  const averageComputationTime = metricsData.computation_times.length > 0
    ? metricsData.computation_times.reduce((sum, val) => sum + val, 0) / metricsData.computation_times.length
    : 0

  const chartData = metricsData.ate_values.map((ate, index) => ({
    index: index + 1,
    ate: ate,
    ci_lower: metricsData.confidence_intervals[index]?.[0] || 0,
    ci_upper: metricsData.confidence_intervals[index]?.[1] || 0,
    timestamp: new Date(metricsData.timestamps[index]).toLocaleTimeString()
  }))

  return (
    <Box className={`metrics-dashboard ${className || ''}`}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h5" component="h2">
          Experiment Metrics
        </Typography>
        <Box>
          <Tooltip title="Download Results">
            <IconButton onClick={handleDownload} disabled={results.length === 0}>
              <DownloadIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Refresh Data">
            <IconButton onClick={onRefresh} disabled={isLoading}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {isLoading && <LinearProgress sx={{ mb: 2 }} />}

      {results.length === 0 && !isLoading && (
        <Alert severity="info" sx={{ mb: 2 }}>
          No experimental results available yet. Run some interventions to see metrics.
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Summary Statistics */}
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="primary" gutterBottom>
                Average Treatment Effect
              </Typography>
              <Typography variant="h4">
                {formatMetricValue(averageATE, 'decimal', 3)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Across {results.length} interventions
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="primary" gutterBottom>
                Significance Rate
              </Typography>
              <Typography variant="h4">
                {formatMetricValue(significanceRate, 'percentage', 1)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Results with p &lt; 0.05
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="primary" gutterBottom>
                Avg Computation Time
              </Typography>
              <Typography variant="h4">
                {formatMetricValue(averageComputationTime * 1000, 'integer')}ms
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Per intervention
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" color="primary" gutterBottom>
                Total Interventions
              </Typography>
              <Typography variant="h4">
                {results.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Completed successfully
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* ATE Trend Chart */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Typography variant="h6" sx={{ flexGrow: 1 }}>
                  Treatment Effect Over Time
                </Typography>
                <Tooltip title="Shows average treatment effect and confidence intervals for each intervention">
                  <InfoIcon color="action" fontSize="small" />
                </Tooltip>
              </Box>
              
              {chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="index" 
                      label={{ value: 'Intervention #', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis 
                      label={{ value: 'ATE', angle: -90, position: 'insideLeft' }}
                    />
                    <RechartsTooltip 
                      formatter={(value: number, name: string) => [
                        formatMetricValue(value, 'decimal', 3),
                        name === 'ate' ? 'ATE' : name === 'ci_lower' ? 'CI Lower' : 'CI Upper'
                      ]}
                      labelFormatter={(label) => `Intervention ${label}`}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="ate" 
                      stroke="#1976d2" 
                      strokeWidth={2}
                      dot={{ fill: '#1976d2' }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="ci_lower" 
                      stroke="#1976d2" 
                      strokeWidth={1}
                      strokeDasharray="5 5"
                      dot={false}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="ci_upper" 
                      stroke="#1976d2" 
                      strokeWidth={1}
                      strokeDasharray="5 5"
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <Box 
                  display="flex" 
                  justifyContent="center" 
                  alignItems="center" 
                  height={300}
                  bgcolor="grey.50"
                >
                  <Typography color="text.secondary">
                    No data available for chart
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Results Table Preview */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Results
              </Typography>
              {results.slice(-5).map((result, index) => (
                <Box 
                  key={result.id}
                  display="flex" 
                  justifyContent="space-between" 
                  alignItems="center"
                  py={1}
                  borderBottom={index < 4 ? 1 : 0}
                  borderColor="divider"
                >
                  <Box>
                    <Typography variant="body2">
                      {result.intervention.variable} = {result.intervention.value}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {new Date(result.created_at).toLocaleString()}
                    </Typography>
                  </Box>
                  <Box textAlign="right">
                    <Typography variant="body2" fontWeight="medium">
                      ATE: {formatMetricValue(
                        result.metrics.find(m => m.metric_type === 'ate')?.value || 0,
                        'decimal',
                        3
                      )}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      p = {formatMetricValue(
                        result.metrics.find(m => m.metric_type === 'ate')?.p_value || 1,
                        'decimal',
                        3
                      )}
                    </Typography>
                  </Box>
                </Box>
              ))}
              {results.length === 0 && (
                <Typography color="text.secondary" textAlign="center" py={2}>
                  No results yet
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}