import React, { useState, useEffect } from 'react'
import {
  Box,
  Typography,
  Button,
  Grid,
  Paper,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  CardActions,
  Chip,
  Divider
} from '@mui/material'

interface ApiStatus {
  status: string
  experiments_count: number
  uptime: number
  engine_type: string
}

interface Experiment {
  id: string
  name: string
  description: string
  created_at: number
  status: string
}

export function RobustDemo() {
  const [counter, setCounter] = useState(0)
  const [apiStatus, setApiStatus] = useState<ApiStatus | null>(null)
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [connected, setConnected] = useState(false)

  // Test backend connectivity
  useEffect(() => {
    const testConnection = async () => {
      try {
        const response = await fetch('http://localhost:8001/health')
        if (response.ok) {
          setConnected(true)
          setError(null)
        } else {
          throw new Error(`HTTP ${response.status}`)
        }
      } catch (err) {
        console.warn('Backend connection failed:', err)
        setConnected(false)
        setError('Backend server unavailable - running in offline mode')
      }
    }

    testConnection()
    const interval = setInterval(testConnection, 10000) // Check every 10 seconds
    return () => clearInterval(interval)
  }, [])

  // Load API status
  const loadApiStatus = async () => {
    if (!connected) return
    
    try {
      setLoading(true)
      const response = await fetch('http://localhost:8001/api/status')
      if (response.ok) {
        const status = await response.json()
        setApiStatus(status)
      }
    } catch (err) {
      console.warn('Failed to load API status:', err)
      setError('Failed to communicate with backend')
    } finally {
      setLoading(false)
    }
  }

  // Load experiments
  const loadExperiments = async () => {
    if (!connected) return
    
    try {
      setLoading(true)
      const response = await fetch('http://localhost:8001/api/experiments')
      if (response.ok) {
        const data = await response.json()
        setExperiments(data.experiments || [])
      }
    } catch (err) {
      console.warn('Failed to load experiments:', err)
      setError('Failed to load experiments from backend')
    } finally {
      setLoading(false)
    }
  }

  // Create new experiment
  const createExperiment = async () => {
    if (!connected) {
      setError('Cannot create experiment - backend not connected')
      return
    }
    
    try {
      setLoading(true)
      const response = await fetch('http://localhost:8001/api/experiments', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: `Demo Experiment ${experiments.length + 1}`,
          description: 'Test experiment created from robust demo'
        })
      })
      
      if (response.ok) {
        const newExperiment = await response.json()
        setExperiments(prev => [...prev, newExperiment])
        setError(null)
      } else {
        throw new Error(`HTTP ${response.status}`)
      }
    } catch (err) {
      console.error('Failed to create experiment:', err)
      setError('Failed to create experiment')
    } finally {
      setLoading(false)
    }
  }

  // Test causal intervention
  const testIntervention = async (variable: string, value: number) => {
    if (!connected) {
      setError('Cannot test intervention - backend not connected')
      return
    }
    
    try {
      setLoading(true)
      const response = await fetch('http://localhost:8001/api/interventions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ variable, value })
      })
      
      if (response.ok) {
        const result = await response.json()
        console.log('Intervention result:', result)
        setError(null)
        return result
      } else {
        throw new Error(`HTTP ${response.status}`)
      }
    } catch (err) {
      console.error('Intervention failed:', err)
      setError('Intervention request failed')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (connected) {
      loadApiStatus()
      loadExperiments()
    }
  }, [connected])

  const handleIncrement = () => {
    setCounter(prev => prev + 1)
  }

  const handleReset = () => {
    setCounter(0)
  }

  const formatUptime = (uptime: number) => {
    if (uptime < 60) return `${uptime.toFixed(1)}s`
    if (uptime < 3600) return `${(uptime / 60).toFixed(1)}m`
    return `${(uptime / 3600).toFixed(1)}h`
  }

  return (
    <Box sx={{ p: 4 }}>
      <Typography variant="h4" gutterBottom>
        Causal UI Gym - Robust Demo (Generation 2)
      </Typography>

      {/* Connection Status */}
      <Box sx={{ mb: 3 }}>
        <Alert 
          severity={connected ? 'success' : 'warning'}
          sx={{ mb: 2 }}
        >
          Backend Status: {connected ? '✅ Connected' : '⚠️ Disconnected'} 
          {apiStatus && ` • Engine: ${apiStatus.engine_type} • Uptime: ${formatUptime(apiStatus.uptime)}`}
        </Alert>
        
        {error && (
          <Alert severity="error" onClose={() => setError(null)}>
            {error}
          </Alert>
        )}
      </Box>

      <Grid container spacing={3}>
        {/* Frontend Demo */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Frontend Functionality
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                React components with error boundaries and robust state management.
              </Typography>
              <Box sx={{ mt: 2, mb: 2 }}>
                <Typography variant="h3" color="primary">
                  {counter}
                </Typography>
              </Box>
            </CardContent>
            <CardActions>
              <Button 
                variant="contained" 
                onClick={handleIncrement}
                disabled={loading}
              >
                Increment
              </Button>
              <Button 
                variant="outlined" 
                onClick={handleReset}
                disabled={loading}
              >
                Reset
              </Button>
            </CardActions>
          </Card>
        </Grid>

        {/* Backend Integration */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Backend Integration
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Resilient API communication with graceful degradation.
              </Typography>
              
              {apiStatus && (
                <Box sx={{ mt: 2 }}>
                  <Chip 
                    label={`Status: ${apiStatus.status}`} 
                    color="success" 
                    size="small" 
                    sx={{ mr: 1, mb: 1 }}
                  />
                  <Chip 
                    label={`Experiments: ${apiStatus.experiments_count}`} 
                    color="primary" 
                    size="small" 
                    sx={{ mr: 1, mb: 1 }}
                  />
                </Box>
              )}
            </CardContent>
            <CardActions>
              <Button 
                variant="contained" 
                onClick={loadApiStatus}
                disabled={loading || !connected}
                startIcon={loading ? <CircularProgress size={16} /> : null}
              >
                Refresh Status
              </Button>
              <Button 
                variant="outlined" 
                onClick={() => testIntervention('test_var', Math.random())}
                disabled={loading || !connected}
              >
                Test Intervention
              </Button>
            </CardActions>
          </Card>
        </Grid>

        {/* Experiments Management */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Experiment Management
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Create and manage causal reasoning experiments.
              </Typography>
              
              <Box sx={{ mt: 2, mb: 2 }}>
                <Button 
                  variant="contained" 
                  onClick={createExperiment}
                  disabled={loading || !connected}
                  startIcon={loading ? <CircularProgress size={16} /> : null}
                  sx={{ mr: 2 }}
                >
                  Create Experiment
                </Button>
                <Button 
                  variant="outlined" 
                  onClick={loadExperiments}
                  disabled={loading || !connected}
                >
                  Refresh List
                </Button>
              </Box>

              <Divider sx={{ my: 2 }} />

              {experiments.length === 0 ? (
                <Typography variant="body2" color="text.secondary">
                  No experiments found. Create one to get started.
                </Typography>
              ) : (
                <Grid container spacing={2}>
                  {experiments.slice(0, 3).map((exp) => (
                    <Grid item xs={12} sm={6} md={4} key={exp.id}>
                      <Paper sx={{ p: 2 }}>
                        <Typography variant="subtitle1" gutterBottom>
                          {exp.name}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {exp.description || 'No description'}
                        </Typography>
                        <Box sx={{ mt: 1 }}>
                          <Chip 
                            label={exp.status} 
                            size="small" 
                            color={exp.status === 'created' ? 'success' : 'default'}
                          />
                        </Box>
                      </Paper>
                    </Grid>
                  ))}
                </Grid>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Box sx={{ mt: 4 }}>
        <Typography variant="body2" color="text.secondary" align="center">
          Generation 2: MAKE IT ROBUST - Enhanced Error Handling, Logging, Monitoring & Health Checks
        </Typography>
      </Box>
    </Box>
  )
}