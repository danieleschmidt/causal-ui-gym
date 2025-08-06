import React, { useState, useCallback } from 'react'
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  Tabs,
  Tab,
  Paper,
  Alert
} from '@mui/material'
import { 
  CausalGraph,
  InterventionControl,
  MetricsDashboard,
  ExperimentBuilder,
  ErrorBoundary
} from '../src/components'
import {
  CausalDAG,
  CausalResult,
  ExperimentConfig,
  Intervention
} from '../src/types'

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
})

interface TabPanelProps {
  children?: React.ReactNode
  index: number
  value: number
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`demo-tabpanel-${index}`}
      aria-labelledby={`demo-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  )
}

const DEMO_DAG: CausalDAG = {
  name: 'Supply & Demand Economics',
  description: 'Classic economic model showing relationships between price, supply, demand, and revenue',
  nodes: [
    { id: 'price', label: 'Price ($)', position: { x: 200, y: 50 }, variable_type: 'continuous' },
    { id: 'demand', label: 'Demand', position: { x: 100, y: 150 }, variable_type: 'continuous' },
    { id: 'supply', label: 'Supply', position: { x: 300, y: 150 }, variable_type: 'continuous' },
    { id: 'revenue', label: 'Revenue', position: { x: 200, y: 250 }, variable_type: 'continuous' },
    { id: 'competitor_price', label: 'Competitor Price', position: { x: 50, y: 200 }, variable_type: 'continuous' }
  ],
  edges: [
    { source: 'price', target: 'demand', weight: -0.8, edge_type: 'causal' },
    { source: 'price', target: 'supply', weight: 0.6, edge_type: 'causal' },
    { source: 'demand', target: 'revenue', weight: 0.7, edge_type: 'causal' },
    { source: 'supply', target: 'revenue', weight: 0.3, edge_type: 'causal' },
    { source: 'competitor_price', target: 'demand', weight: 0.4, edge_type: 'causal' }
  ]
}

export default function CausalUIGymDemo() {
  const [tabValue, setTabValue] = useState(0)
  const [results, setResults] = useState<CausalResult[]>([])
  const [activeInterventions, setActiveInterventions] = useState<Intervention[]>([])
  const [isRunning, setIsRunning] = useState(false)

  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue)
  }

  const handleIntervention = useCallback(async (intervention: Intervention) => {
    console.log('Intervention triggered:', intervention)
    
    setActiveInterventions(prev => {
      const filtered = prev.filter(i => i.variable !== intervention.variable)
      return [...filtered, intervention]
    })

    // Simulate API call to backend
    setIsRunning(true)
    
    try {
      // Mock causal computation
      const mockResult: CausalResult = {
        id: `result-${Date.now()}`,
        dag_id: 'demo-dag',
        intervention,
        outcome_variable: 'revenue',
        metrics: [{
          metric_type: 'ate',
          value: Math.random() * 20 - 10, // Random ATE between -10 and 10
          confidence_interval: [Math.random() * 5 - 2.5, Math.random() * 5 + 2.5] as [number, number],
          standard_error: Math.random() * 2,
          p_value: Math.random() * 0.1,
          sample_size: 10000,
          computation_time: Math.random() * 50 + 10,
          metadata: {
            intervention_variable: intervention.variable,
            intervention_value: intervention.value,
            baseline_value: 0
          }
        }],
        outcome_distribution: Array.from({ length: 1000 }, () => 
          Math.random() * 100 + Number(intervention.value) * 5
        ),
        created_at: new Date()
      }

      // Simulate processing delay
      setTimeout(() => {
        setResults(prev => [...prev, mockResult])
        setIsRunning(false)
      }, 1500)
    } catch (error) {
      console.error('Error running intervention:', error)
      setIsRunning(false)
    }
  }, [])

  const handleRunExperiment = useCallback(async (config: ExperimentConfig): Promise<CausalResult[]> => {
    console.log('Running experiment:', config)
    
    setIsRunning(true)
    
    // Simulate experiment execution
    const experimentResults: CausalResult[] = config.interventions.map((intervention, index) => ({
      id: `exp-result-${index}`,
      dag_id: config.dag.name || 'experiment',
      intervention,
      outcome_variable: config.outcome_variables[0] || 'outcome',
      metrics: [{
        metric_type: 'ate',
        value: Math.random() * 10 - 5,
        confidence_interval: [Math.random() * 2 - 1, Math.random() * 2 + 1] as [number, number],
        standard_error: Math.random() * 1,
        p_value: Math.random() * 0.05,
        sample_size: config.sample_size || 10000,
        computation_time: Math.random() * 100,
      }],
      outcome_distribution: Array.from({ length: 500 }, () => Math.random() * 50),
      created_at: new Date()
    }))

    return new Promise((resolve) => {
      setTimeout(() => {
        setResults(prev => [...prev, ...experimentResults])
        setIsRunning(false)
        resolve(experimentResults)
      }, 3000)
    })
  }, [])

  const clearResults = () => {
    setResults([])
    setActiveInterventions([])
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ErrorBoundary>
        <AppBar position="static" elevation={2}>
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              🧠 Causal UI Gym - Interactive Demo
            </Typography>
            <Typography variant="body2">
              React + JAX Causal Reasoning Framework
            </Typography>
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ mt: 3, mb: 3 }}>
          <Alert severity="info" sx={{ mb: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              Welcome to Causal UI Gym!
            </Typography>
            This demo showcases interactive causal reasoning components. Try different interventions 
            to see how they affect outcomes in the economic model below.
          </Alert>

          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={tabValue} onChange={handleTabChange} aria-label="demo tabs">
              <Tab label="Interactive Graph" />
              <Tab label="Intervention Controls" />
              <Tab label="Metrics Dashboard" />
              <Tab label="Experiment Builder" />
            </Tabs>
          </Box>

          {/* Tab 1: Interactive Causal Graph */}
          <TabPanel value={tabValue} index={0}>
            <Typography variant="h5" gutterBottom>
              Interactive Causal Graph
            </Typography>
            <Typography variant="body1" paragraph>
              Click on nodes to perform interventions. The graph shows causal relationships 
              in a supply & demand economic model.
            </Typography>
            
            <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
              <CausalGraph 
                dag={DEMO_DAG}
                onIntervene={(nodeId, value) => {
                  const intervention: Intervention = {
                    variable: nodeId,
                    value,
                    intervention_type: 'do',
                    timestamp: new Date()
                  }
                  handleIntervention(intervention)
                }}
              />
            </Paper>

            <Typography variant="h6" gutterBottom>
              Active Interventions ({activeInterventions.length})
            </Typography>
            {activeInterventions.map((intervention, index) => (
              <Alert key={index} severity="success" sx={{ mb: 1 }}>
                <strong>do({intervention.variable} = {intervention.value})</strong> - 
                Intervention applied at {intervention.timestamp?.toLocaleTimeString()}
              </Alert>
            ))}
          </TabPanel>

          {/* Tab 2: Intervention Controls */}
          <TabPanel value={tabValue} index={1}>
            <Typography variant="h5" gutterBottom>
              Intervention Controls
            </Typography>
            <Typography variant="body1" paragraph>
              Use these controls to precisely set intervention values for each variable.
            </Typography>

            <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: 2 }}>
              {DEMO_DAG.nodes.map((node) => (
                <InterventionControl
                  key={node.id}
                  variable={node.id}
                  min={node.id === 'price' ? 0 : 0}
                  max={node.id === 'price' ? 200 : 100}
                  step={node.id === 'price' ? 5 : 1}
                  onIntervene={handleIntervention}
                  disabled={isRunning}
                />
              ))}
            </Box>
          </TabPanel>

          {/* Tab 3: Metrics Dashboard */}
          <TabPanel value={tabValue} index={2}>
            <Typography variant="h5" gutterBottom>
              Causal Metrics Dashboard
            </Typography>
            <Typography variant="body1" paragraph>
              View real-time metrics and analysis of your causal interventions.
            </Typography>

            <MetricsDashboard
              experimentId="demo-experiment"
              results={results}
              isLoading={isRunning}
              onRefresh={() => window.location.reload()}
            />

            {results.length > 0 && (
              <Box sx={{ mt: 2 }}>
                <Alert severity="warning">
                  <Typography variant="body2">
                    These are simulated results for demonstration purposes. In a real application,
                    results would be computed using the JAX backend with actual causal inference algorithms.
                  </Typography>
                </Alert>
              </Box>
            )}
          </TabPanel>

          {/* Tab 4: Experiment Builder */}
          <TabPanel value={tabValue} index={3}>
            <Typography variant="h5" gutterBottom>
              Experiment Builder
            </Typography>
            <Typography variant="body1" paragraph>
              Create and configure custom causal experiments with your own DAGs and interventions.
            </Typography>

            <ExperimentBuilder
              onCreateExperiment={(config) => {
                console.log('Experiment created:', config)
                alert('Experiment configuration saved! Check console for details.')
              }}
              onRunExperiment={handleRunExperiment}
            />
          </TabPanel>

          {/* Footer with stats */}
          <Paper elevation={1} sx={{ p: 2, mt: 4, bgcolor: 'grey.50' }}>
            <Typography variant="body2" textAlign="center">
              <strong>Demo Stats:</strong> {results.length} interventions performed • 
              {activeInterventions.length} active interventions • 
              {results.reduce((sum, r) => sum + (r.metrics.find(m => m.metric_type === 'ate')?.sample_size || 0), 0).toLocaleString()} total samples processed
            </Typography>
          </Paper>
        </Container>
      </ErrorBoundary>
    </ThemeProvider>
  )
}