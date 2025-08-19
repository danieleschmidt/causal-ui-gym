import React, { useState, useEffect, useRef, useCallback } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Paper,
  Chip,
  LinearProgress,
  Alert,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Divider,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  CircularProgress,
  Badge
} from '@mui/material'
import {
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  Stop as StopIcon,
  Save as SaveIcon,
  CloudUpload as CloudUploadIcon,
  Assessment as AssessmentIcon,
  Security as SecurityIcon,
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  NetworkCheck as NetworkIcon,
  BugReport as BugReportIcon,
  Verified as VerifiedIcon,
  Science as ScienceIcon
} from '@mui/icons-material'
import { CausalDAG, ExperimentConfig } from '../types'
import { monitoring } from '../utils/monitoring'
import { performanceUtils } from '../utils/cache'

interface ProductionExperimentRunnerProps {
  experiment: ExperimentConfig
  dag: CausalDAG
  onComplete?: (results: any) => void
  enableProductionFeatures?: boolean
}

interface ExperimentStep {
  id: string
  name: string
  description: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  startTime?: Date
  endTime?: Date
  results?: any
  errors?: string[]
}

interface SystemMetrics {
  memoryUsage: number
  cpuUsage: number
  networkLatency: number
  cacheHitRate: number
  errorRate: number
  throughput: number
}

interface QualityGateResult {
  gate: string
  passed: boolean
  score: number
  threshold: number
  details: string
}

export function ProductionExperimentRunner({
  experiment,
  dag,
  onComplete,
  enableProductionFeatures = true
}: ProductionExperimentRunnerProps) {
  const [isRunning, setIsRunning] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [steps, setSteps] = useState<ExperimentStep[]>([])
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({
    memoryUsage: 0,
    cpuUsage: 0,
    networkLatency: 0,
    cacheHitRate: 0,
    errorRate: 0,
    throughput: 0
  })
  const [qualityGates, setQualityGates] = useState<QualityGateResult[]>([])
  const [showSaveDialog, setShowSaveDialog] = useState(false)
  const [exportFormat, setExportFormat] = useState('json')
  const [autoRetry, setAutoRetry] = useState(true)
  const [maxRetries, setMaxRetries] = useState(3)
  const [circuitBreakerEnabled, setCircuitBreakerEnabled] = useState(true)
  const [realTimeMonitoring, setRealTimeMonitoring] = useState(true)
  
  const metricsInterval = useRef<NodeJS.Timeout>()
  const abortController = useRef<AbortController>()

  const initializeSteps = useCallback(() => {
    const experimentSteps: ExperimentStep[] = [
      {
        id: 'validation',
        name: 'Input Validation',
        description: 'Validate DAG structure and experiment configuration',
        status: 'pending',
        progress: 0
      },
      {
        id: 'preprocessing',
        name: 'Data Preprocessing',
        description: 'Clean and prepare data for causal analysis',
        status: 'pending',
        progress: 0
      },
      {
        id: 'causal_discovery',
        name: 'Causal Discovery',
        description: 'Discover causal relationships in the data',
        status: 'pending',
        progress: 0
      },
      {
        id: 'intervention_analysis',
        name: 'Intervention Analysis',
        description: 'Analyze causal effects of interventions',
        status: 'pending',
        progress: 0
      },
      {
        id: 'quality_gates',
        name: 'Quality Gates',
        description: 'Run quality checks and validation tests',
        status: 'pending',
        progress: 0
      },
      {
        id: 'results_generation',
        name: 'Results Generation',
        description: 'Generate comprehensive analysis results',
        status: 'pending',
        progress: 0
      }
    ]

    if (enableProductionFeatures) {
      experimentSteps.push(
        {
          id: 'security_scan',
          name: 'Security Scan',
          description: 'Scan for security vulnerabilities and data leaks',
          status: 'pending',
          progress: 0
        },
        {
          id: 'performance_optimization',
          name: 'Performance Optimization',
          description: 'Optimize and tune performance parameters',
          status: 'pending',
          progress: 0
        },
        {
          id: 'deployment_preparation',
          name: 'Deployment Preparation',
          description: 'Prepare results for production deployment',
          status: 'pending',
          progress: 0
        }
      )
    }

    setSteps(experimentSteps)
  }, [enableProductionFeatures])

  useEffect(() => {
    initializeSteps()
  }, [initializeSteps])

  const startMetricsMonitoring = useCallback(() => {
    if (!realTimeMonitoring) return

    metricsInterval.current = setInterval(() => {
      // Simulate real-time metrics collection
      setSystemMetrics(prev => ({
        memoryUsage: Math.random() * 100,
        cpuUsage: Math.random() * 100,
        networkLatency: Math.random() * 100 + 10,
        cacheHitRate: Math.random() * 100,
        errorRate: Math.random() * 5,
        throughput: Math.random() * 1000 + 500
      }))

      // Track metrics with monitoring system
      monitoring.trackMetric('experiment_memory_usage', systemMetrics.memoryUsage)
      monitoring.trackMetric('experiment_cpu_usage', systemMetrics.cpuUsage)
      monitoring.trackMetric('experiment_throughput', systemMetrics.throughput)
    }, 1000)
  }, [realTimeMonitoring, systemMetrics])

  const stopMetricsMonitoring = useCallback(() => {
    if (metricsInterval.current) {
      clearInterval(metricsInterval.current)
    }
  }, [])

  const runQualityGates = useCallback(async (): Promise<boolean> => {
    const gates = [
      {
        gate: 'Performance Threshold',
        threshold: 80,
        score: systemMetrics.throughput / 10,
        details: 'Throughput must exceed minimum requirements'
      },
      {
        gate: 'Memory Efficiency',
        threshold: 70,
        score: 100 - systemMetrics.memoryUsage,
        details: 'Memory usage must remain within acceptable limits'
      },
      {
        gate: 'Error Rate',
        threshold: 95,
        score: 100 - systemMetrics.errorRate,
        details: 'Error rate must be below threshold'
      },
      {
        gate: 'Cache Performance',
        threshold: 85,
        score: systemMetrics.cacheHitRate,
        details: 'Cache hit rate must meet performance standards'
      },
      {
        gate: 'Statistical Significance',
        threshold: 95,
        score: Math.random() * 100,
        details: 'Results must achieve statistical significance'
      }
    ]

    const results = gates.map(gate => ({
      ...gate,
      passed: gate.score >= gate.threshold
    }))

    setQualityGates(results)
    return results.every(r => r.passed)
  }, [systemMetrics])

  const executeStep = useCallback(async (stepIndex: number, retryCount = 0): Promise<boolean> => {
    const step = steps[stepIndex]
    if (!step) return false

    try {
      // Update step status
      setSteps(prev => prev.map((s, i) => 
        i === stepIndex ? { ...s, status: 'running', startTime: new Date() } : s
      ))

      // Simulate step execution with progress updates
      for (let progress = 0; progress <= 100; progress += 10) {
        if (abortController.current?.signal.aborted) {
          throw new Error('Execution aborted')
        }

        if (isPaused) {
          await new Promise(resolve => {
            const checkPause = () => {
              if (!isPaused) resolve(void 0)
              else setTimeout(checkPause, 100)
            }
            checkPause()
          })
        }

        setSteps(prev => prev.map((s, i) => 
          i === stepIndex ? { ...s, progress } : s
        ))

        await new Promise(resolve => setTimeout(resolve, 200 + Math.random() * 300))
      }

      // Special handling for quality gates step
      if (step.id === 'quality_gates') {
        const gatesPassed = await runQualityGates()
        if (!gatesPassed && circuitBreakerEnabled) {
          throw new Error('Quality gates failed - circuit breaker activated')
        }
      }

      // Simulate step completion
      const results = {
        timestamp: new Date(),
        stepId: step.id,
        success: true,
        metrics: { ...systemMetrics },
        data: `Results for ${step.name}`
      }

      setSteps(prev => prev.map((s, i) => 
        i === stepIndex ? { 
          ...s, 
          status: 'completed', 
          progress: 100, 
          endTime: new Date(),
          results 
        } : s
      ))

      monitoring.trackUser('experiment_step_completed', 'production_runner', {
        step_id: step.id,
        step_name: step.name,
        duration: Date.now() - (step.startTime?.getTime() || Date.now())
      })

      return true
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error'
      
      setSteps(prev => prev.map((s, i) => 
        i === stepIndex ? { 
          ...s, 
          status: 'failed', 
          endTime: new Date(),
          errors: [...(s.errors || []), errorMessage]
        } : s
      ))

      monitoring.trackError('experiment_step_failed', errorMessage, step.id)

      // Auto-retry logic
      if (autoRetry && retryCount < maxRetries) {
        monitoring.trackUser('experiment_step_retry', 'production_runner', {
          step_id: step.id,
          retry_count: retryCount + 1
        })
        
        await new Promise(resolve => setTimeout(resolve, 1000 * (retryCount + 1)))
        return executeStep(stepIndex, retryCount + 1)
      }

      return false
    }
  }, [steps, isPaused, runQualityGates, circuitBreakerEnabled, autoRetry, maxRetries, systemMetrics])

  const startExperiment = useCallback(async () => {
    setIsRunning(true)
    setCurrentStep(0)
    abortController.current = new AbortController()
    startMetricsMonitoring()

    monitoring.trackUser('experiment_started', 'production_runner', {
      experiment_id: experiment.id || 'unknown',
      production_features: enableProductionFeatures.toString()
    })

    try {
      for (let i = 0; i < steps.length; i++) {
        if (abortController.current.signal.aborted) break

        setCurrentStep(i)
        const success = await executeStep(i)
        
        if (!success) {
          monitoring.trackUser('experiment_failed', 'production_runner', {
            failed_step: steps[i].id,
            step_index: i
          })
          break
        }
      }

      const allCompleted = steps.every(s => s.status === 'completed')
      if (allCompleted) {
        monitoring.trackUser('experiment_completed', 'production_runner', {
          experiment_id: experiment.id || 'unknown',
          total_duration: Date.now() - (steps[0].startTime?.getTime() || Date.now())
        })

        if (onComplete) {
          const results = {
            steps: steps.map(s => s.results),
            metrics: systemMetrics,
            qualityGates,
            timestamp: new Date()
          }
          onComplete(results)
        }
      }
    } catch (error) {
      monitoring.trackError('experiment_error', error instanceof Error ? error.message : 'Unknown error', 'production_runner')
    } finally {
      setIsRunning(false)
      stopMetricsMonitoring()
    }
  }, [steps, experiment, enableProductionFeatures, executeStep, startMetricsMonitoring, stopMetricsMonitoring, systemMetrics, qualityGates, onComplete])

  const stopExperiment = useCallback(() => {
    if (abortController.current) {
      abortController.current.abort()
    }
    setIsRunning(false)
    setIsPaused(false)
    stopMetricsMonitoring()
    
    monitoring.trackUser('experiment_stopped', 'production_runner', {
      experiment_id: experiment.id || 'unknown',
      current_step: currentStep
    })
  }, [currentStep, experiment, stopMetricsMonitoring])

  const pauseExperiment = useCallback(() => {
    setIsPaused(!isPaused)
    monitoring.trackUser(isPaused ? 'experiment_resumed' : 'experiment_paused', 'production_runner')
  }, [isPaused])

  const getMetricColor = (value: number, threshold: number, inverse = false) => {
    const isGood = inverse ? value < threshold : value > threshold
    return isGood ? 'success' : value > threshold * 0.7 ? 'warning' : 'error'
  }

  const exportResults = useCallback(() => {
    const results = {
      experiment: experiment,
      steps: steps,
      metrics: systemMetrics,
      qualityGates: qualityGates,
      timestamp: new Date().toISOString()
    }

    const dataStr = exportFormat === 'json' 
      ? JSON.stringify(results, null, 2)
      : `# Experiment Results\n\nExperiment: ${experiment.name || 'Unknown'}\nTimestamp: ${results.timestamp}\n\n## Steps\n${steps.map(s => `- ${s.name}: ${s.status}`).join('\n')}`

    const blob = new Blob([dataStr], { type: exportFormat === 'json' ? 'application/json' : 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `experiment-results.${exportFormat}`
    a.click()
    URL.revokeObjectURL(url)

    setShowSaveDialog(false)
    monitoring.trackUser('experiment_results_exported', 'production_runner', {
      format: exportFormat
    })
  }, [experiment, steps, systemMetrics, qualityGates, exportFormat])

  return (
    <Box sx={{ width: '100%' }}>
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Typography variant="h5">
              Production Experiment Runner
            </Typography>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Tooltip title="Save Results">
                <IconButton onClick={() => setShowSaveDialog(true)} disabled={steps.every(s => s.status === 'pending')}>
                  <SaveIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="Export to Cloud">
                <IconButton disabled={!enableProductionFeatures}>
                  <CloudUploadIcon />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>

          {enableProductionFeatures && (
            <Grid container spacing={2} sx={{ mb: 3 }}>
              <Grid item xs={12} md={6}>
                <FormControlLabel
                  control={<Switch checked={autoRetry} onChange={(e) => setAutoRetry(e.target.checked)} />}
                  label="Auto Retry on Failure"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControlLabel
                  control={<Switch checked={circuitBreakerEnabled} onChange={(e) => setCircuitBreakerEnabled(e.target.checked)} />}
                  label="Circuit Breaker"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <FormControlLabel
                  control={<Switch checked={realTimeMonitoring} onChange={(e) => setRealTimeMonitoring(e.target.checked)} />}
                  label="Real-time Monitoring"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  label="Max Retries"
                  type="number"
                  value={maxRetries}
                  onChange={(e) => setMaxRetries(parseInt(e.target.value))}
                  size="small"
                  inputProps={{ min: 0, max: 10 }}
                />
              </Grid>
            </Grid>
          )}

          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Paper sx={{ p: 2 }}>
                <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                  <Button
                    variant="contained"
                    startIcon={<PlayIcon />}
                    onClick={startExperiment}
                    disabled={isRunning}
                  >
                    Start
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={isPaused ? <PlayIcon /> : <PauseIcon />}
                    onClick={pauseExperiment}
                    disabled={!isRunning}
                  >
                    {isPaused ? 'Resume' : 'Pause'}
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<StopIcon />}
                    onClick={stopExperiment}
                    disabled={!isRunning}
                  >
                    Stop
                  </Button>
                </Box>

                <Stepper activeStep={currentStep} orientation="vertical">
                  {steps.map((step, index) => (
                    <Step key={step.id}>
                      <StepLabel
                        error={step.status === 'failed'}
                        icon={
                          step.status === 'running' ? (
                            <CircularProgress size={24} />
                          ) : step.status === 'completed' ? (
                            <VerifiedIcon color="success" />
                          ) : step.status === 'failed' ? (
                            <BugReportIcon color="error" />
                          ) : undefined
                        }
                      >
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography>{step.name}</Typography>
                          <Chip
                            label={step.status.toUpperCase()}
                            size="small"
                            color={
                              step.status === 'completed' ? 'success' :
                              step.status === 'running' ? 'primary' :
                              step.status === 'failed' ? 'error' : 'default'
                            }
                          />
                        </Box>
                      </StepLabel>
                      <StepContent>
                        <Typography variant="body2" color="text.secondary">
                          {step.description}
                        </Typography>
                        {step.status === 'running' && (
                          <LinearProgress 
                            variant="determinate" 
                            value={step.progress} 
                            sx={{ mt: 1 }}
                          />
                        )}
                        {step.errors && step.errors.length > 0 && (
                          <Alert severity="error" sx={{ mt: 1 }}>
                            {step.errors.join(', ')}
                          </Alert>
                        )}
                      </StepContent>
                    </Step>
                  ))}
                </Stepper>
              </Paper>
            </Grid>

            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 2, mb: 2 }}>
                <Typography variant="h6" gutterBottom>
                  System Metrics
                </Typography>
                <Grid container spacing={1}>
                  <Grid item xs={6}>
                    <Box sx={{ textAlign: 'center' }}>
                      <MemoryIcon color={getMetricColor(systemMetrics.memoryUsage, 80, true) as any} />
                      <Typography variant="caption" display="block">Memory</Typography>
                      <Typography variant="h6">{systemMetrics.memoryUsage.toFixed(1)}%</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6}>
                    <Box sx={{ textAlign: 'center' }}>
                      <SpeedIcon color={getMetricColor(systemMetrics.cpuUsage, 80, true) as any} />
                      <Typography variant="caption" display="block">CPU</Typography>
                      <Typography variant="h6">{systemMetrics.cpuUsage.toFixed(1)}%</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6}>
                    <Box sx={{ textAlign: 'center' }}>
                      <NetworkIcon color={getMetricColor(systemMetrics.networkLatency, 50, true) as any} />
                      <Typography variant="caption" display="block">Latency</Typography>
                      <Typography variant="h6">{systemMetrics.networkLatency.toFixed(0)}ms</Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6}>
                    <Box sx={{ textAlign: 'center' }}>
                      <AssessmentIcon color={getMetricColor(systemMetrics.throughput, 800) as any} />
                      <Typography variant="caption" display="block">Throughput</Typography>
                      <Typography variant="h6">{systemMetrics.throughput.toFixed(0)}/s</Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Paper>

              {qualityGates.length > 0 && (
                <Paper sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Quality Gates
                  </Typography>
                  {qualityGates.map((gate, index) => (
                    <Box key={index} sx={{ mb: 1 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="body2">{gate.gate}</Typography>
                        <Chip
                          label={gate.passed ? 'PASS' : 'FAIL'}
                          size="small"
                          color={gate.passed ? 'success' : 'error'}
                          icon={gate.passed ? <VerifiedIcon /> : <BugReportIcon />}
                        />
                      </Box>
                      <Typography variant="caption" color="text.secondary">
                        Score: {gate.score.toFixed(1)} / {gate.threshold}
                      </Typography>
                    </Box>
                  ))}
                </Paper>
              )}
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      <Dialog open={showSaveDialog} onClose={() => setShowSaveDialog(false)}>
        <DialogTitle>Export Results</DialogTitle>
        <DialogContent>
          <FormControl fullWidth sx={{ mt: 1 }}>
            <InputLabel>Export Format</InputLabel>
            <Select
              value={exportFormat}
              onChange={(e) => setExportFormat(e.target.value)}
            >
              <MenuItem value="json">JSON</MenuItem>
              <MenuItem value="markdown">Markdown</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowSaveDialog(false)}>Cancel</Button>
          <Button onClick={exportResults} variant="contained">Export</Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}