import React, { useState, useCallback } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Alert,
  Stepper,
  Step,
  StepLabel,
  StepContent
} from '@mui/material'
import { Add as AddIcon, Delete as DeleteIcon, PlayArrow as PlayIcon } from '@mui/icons-material'
import { CausalDAG, CausalNode, CausalEdge, ExperimentConfig, Intervention } from '../types'
import { validateDAG, generateNodeLayout } from '../utils'
import { CausalGraph } from './CausalGraph'

interface ExperimentBuilderProps {
  onCreateExperiment: (config: ExperimentConfig) => void
  onRunExperiment?: (config: ExperimentConfig) => void
  className?: string
}

const steps = [
  'Define Variables',
  'Create Relationships', 
  'Configure Interventions',
  'Set Parameters',
  'Review & Launch'
]

export function ExperimentBuilder({
  onCreateExperiment,
  onRunExperiment,
  className
}: ExperimentBuilderProps) {
  const [activeStep, setActiveStep] = useState(0)
  const [experimentName, setExperimentName] = useState('')
  const [experimentDescription, setExperimentDescription] = useState('')
  
  // DAG building state
  const [nodes, setNodes] = useState<CausalNode[]>([])
  const [edges, setEdges] = useState<CausalEdge[]>([])
  const [newNodeName, setNewNodeName] = useState('')
  const [newNodeType, setNewNodeType] = useState<'continuous' | 'discrete' | 'binary'>('continuous')
  
  // Intervention configuration
  const [interventions, setInterventions] = useState<Intervention[]>([])
  const [outcomeVariables, setOutcomeVariables] = useState<string[]>([])
  
  // Experiment parameters
  const [sampleSize, setSampleSize] = useState(10000)
  const [randomSeed, setRandomSeed] = useState(42)
  
  // Validation
  const [validationErrors, setValidationErrors] = useState<string[]>([])

  const handleAddNode = useCallback(() => {
    if (!newNodeName.trim()) return
    
    if (nodes.some(node => node.id === newNodeName)) {
      setValidationErrors(['Node name already exists'])
      return
    }

    const newNode: CausalNode = {
      id: newNodeName,
      label: newNodeName,
      position: { x: 100 + nodes.length * 150, y: 100 + (nodes.length % 3) * 100 },
      variable_type: newNodeType,
      description: `Variable: ${newNodeName}`
    }

    setNodes(prev => [...prev, newNode])
    setNewNodeName('')
    setValidationErrors([])
  }, [newNodeName, newNodeType, nodes])

  const handleRemoveNode = useCallback((nodeId: string) => {
    setNodes(prev => prev.filter(node => node.id !== nodeId))
    setEdges(prev => prev.filter(edge => edge.source !== nodeId && edge.target !== nodeId))
    setInterventions(prev => prev.filter(intervention => intervention.variable !== nodeId))
    setOutcomeVariables(prev => prev.filter(variable => variable !== nodeId))
  }, [])

  const handleAddEdge = useCallback((sourceId: string, targetId: string) => {
    if (sourceId === targetId) return
    
    const existingEdge = edges.find(edge => 
      edge.source === sourceId && edge.target === targetId
    )
    
    if (existingEdge) return

    const newEdge: CausalEdge = {
      source: sourceId,
      target: targetId,
      weight: 1.0,
      edge_type: 'causal',
      confidence: 0.9
    }

    setEdges(prev => [...prev, newEdge])
  }, [edges])

  const handleRemoveEdge = useCallback((sourceId: string, targetId: string) => {
    setEdges(prev => prev.filter(edge => 
      !(edge.source === sourceId && edge.target === targetId)
    ))
  }, [])

  const handleAddIntervention = useCallback(() => {
    const newIntervention: Intervention = {
      variable: nodes[0]?.id || '',
      value: 1,
      intervention_type: 'do',
      description: 'New intervention'
    }
    setInterventions(prev => [...prev, newIntervention])
  }, [nodes])

  const handleUpdateIntervention = useCallback((index: number, field: keyof Intervention, value: any) => {
    setInterventions(prev => prev.map((intervention, i) => 
      i === index ? { ...intervention, [field]: value } : intervention
    ))
  }, [])

  const handleRemoveIntervention = useCallback((index: number) => {
    setInterventions(prev => prev.filter((_, i) => i !== index))
  }, [])

  const handleAutoLayout = useCallback(() => {
    const layoutedNodes = generateNodeLayout(nodes, edges)
    setNodes(layoutedNodes)
  }, [nodes, edges])

  const handleValidateAndNext = useCallback(() => {
    const dag: CausalDAG = {
      name: experimentName || 'Untitled Experiment',
      description: experimentDescription,
      nodes,
      edges
    }

    const validation = validateDAG(dag)
    
    if (!validation.is_valid) {
      setValidationErrors(validation.errors)
      return
    }

    setValidationErrors([])
    setActiveStep(prev => Math.min(prev + 1, steps.length - 1))
  }, [experimentName, experimentDescription, nodes, edges])

  const handleCreateExperiment = useCallback(() => {
    const dag: CausalDAG = {
      name: experimentName,
      description: experimentDescription,
      nodes,
      edges,
      created_at: new Date()
    }

    const config: ExperimentConfig = {
      name: experimentName,
      description: experimentDescription,
      dag,
      interventions,
      outcome_variables: outcomeVariables,
      sample_size: sampleSize,
      random_seed: randomSeed,
      status: 'created',
      created_at: new Date(),
      updated_at: new Date()
    }

    onCreateExperiment(config)

    if (onRunExperiment) {
      onRunExperiment(config)
    }
  }, [
    experimentName,
    experimentDescription,
    nodes,
    edges,
    interventions,
    outcomeVariables,
    sampleSize,
    randomSeed,
    onCreateExperiment,
    onRunExperiment
  ])

  const dag: CausalDAG = {
    name: experimentName || 'Draft Experiment',
    description: experimentDescription,
    nodes,
    edges
  }

  return (
    <Box className={`experiment-builder ${className || ''}`}>
      <Card>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Create New Experiment
          </Typography>

          <Stepper activeStep={activeStep} orientation="vertical">
            {/* Step 1: Define Variables */}
            <Step>
              <StepLabel>Define Variables</StepLabel>
              <StepContent>
                <Box sx={{ mb: 2 }}>
                  <TextField
                    fullWidth
                    label="Experiment Name"
                    value={experimentName}
                    onChange={(e) => setExperimentName(e.target.value)}
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    fullWidth
                    multiline
                    rows={2}
                    label="Description"
                    value={experimentDescription}
                    onChange={(e) => setExperimentDescription(e.target.value)}
                    sx={{ mb: 3 }}
                  />

                  <Box display="flex" gap={2} alignItems="center" sx={{ mb: 2 }}>
                    <TextField
                      label="Variable Name"
                      value={newNodeName}
                      onChange={(e) => setNewNodeName(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleAddNode()}
                    />
                    <FormControl sx={{ minWidth: 120 }}>
                      <InputLabel>Type</InputLabel>
                      <Select
                        value={newNodeType}
                        onChange={(e) => setNewNodeType(e.target.value as any)}
                      >
                        <MenuItem value="continuous">Continuous</MenuItem>
                        <MenuItem value="discrete">Discrete</MenuItem>
                        <MenuItem value="binary">Binary</MenuItem>
                      </Select>
                    </FormControl>
                    <Button
                      variant="contained"
                      startIcon={<AddIcon />}
                      onClick={handleAddNode}
                      disabled={!newNodeName.trim()}
                    >
                      Add Variable
                    </Button>
                  </Box>

                  <Box sx={{ mb: 2 }}>
                    {nodes.map((node) => (
                      <Chip
                        key={node.id}
                        label={`${node.label} (${node.variable_type})`}
                        onDelete={() => handleRemoveNode(node.id)}
                        sx={{ mr: 1, mb: 1 }}
                      />
                    ))}
                  </Box>

                  {validationErrors.length > 0 && (
                    <Alert severity="error" sx={{ mb: 2 }}>
                      {validationErrors.map((error, index) => (
                        <div key={index}>{error}</div>
                      ))}
                    </Alert>
                  )}

                  <Button
                    variant="contained"
                    onClick={() => setActiveStep(1)}
                    disabled={nodes.length < 2}
                  >
                    Next: Define Relationships
                  </Button>
                </Box>
              </StepContent>
            </Step>

            {/* Step 2: Create Relationships */}
            <Step>
              <StepLabel>Create Relationships</StepLabel>
              <StepContent>
                <Box sx={{ mb: 2 }}>
                  <Box display="flex" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
                    <Typography variant="h6">Causal Graph</Typography>
                    <Button variant="outlined" onClick={handleAutoLayout}>
                      Auto Layout
                    </Button>
                  </Box>

                  <Box sx={{ height: 400, border: 1, borderColor: 'divider', mb: 2 }}>
                    <CausalGraph
                      dag={dag}
                      onIntervene={(nodeId, value) => {
                        // This will be handled in the intervention step
                      }}
                    />
                  </Box>

                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    Click on nodes in the graph above to create causal relationships. 
                    Current edges: {edges.length}
                  </Typography>

                  <Box display="flex" gap={2}>
                    <Button
                      variant="outlined"
                      onClick={() => setActiveStep(0)}
                    >
                      Back
                    </Button>
                    <Button
                      variant="contained"
                      onClick={handleValidateAndNext}
                    >
                      Next: Configure Interventions
                    </Button>
                  </Box>
                </Box>
              </StepContent>
            </Step>

            {/* Step 3: Configure Interventions */}
            <Step>
              <StepLabel>Configure Interventions</StepLabel>
              <StepContent>
                <Box sx={{ mb: 2 }}>
                  <Box display="flex" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
                    <Typography variant="h6">Interventions</Typography>
                    <Button
                      variant="outlined"
                      startIcon={<AddIcon />}
                      onClick={handleAddIntervention}
                      disabled={nodes.length === 0}
                    >
                      Add Intervention
                    </Button>
                  </Box>

                  {interventions.map((intervention, index) => (
                    <Card key={index} variant="outlined" sx={{ mb: 2 }}>
                      <CardContent>
                        <Box display="flex" gap={2} alignItems="center">
                          <FormControl sx={{ minWidth: 150 }}>
                            <InputLabel>Variable</InputLabel>
                            <Select
                              value={intervention.variable}
                              onChange={(e) => handleUpdateIntervention(index, 'variable', e.target.value)}
                            >
                              {nodes.map((node) => (
                                <MenuItem key={node.id} value={node.id}>
                                  {node.label}
                                </MenuItem>
                              ))}
                            </Select>
                          </FormControl>
                          <TextField
                            type="number"
                            label="Value"
                            value={intervention.value}
                            onChange={(e) => handleUpdateIntervention(index, 'value', parseFloat(e.target.value))}
                          />
                          <TextField
                            label="Description"
                            value={intervention.description || ''}
                            onChange={(e) => handleUpdateIntervention(index, 'description', e.target.value)}
                            sx={{ flexGrow: 1 }}
                          />
                          <Button
                            color="error"
                            onClick={() => handleRemoveIntervention(index)}
                          >
                            <DeleteIcon />
                          </Button>
                        </Box>
                      </CardContent>
                    </Card>
                  ))}

                  <Box display="flex" gap={2}>
                    <Button
                      variant="outlined"
                      onClick={() => setActiveStep(1)}
                    >
                      Back
                    </Button>
                    <Button
                      variant="contained"
                      onClick={() => setActiveStep(3)}
                      disabled={interventions.length === 0}
                    >
                      Next: Set Parameters
                    </Button>
                  </Box>
                </Box>
              </StepContent>
            </Step>

            {/* Step 4: Set Parameters */}
            <Step>
              <StepLabel>Set Parameters</StepLabel>
              <StepContent>
                <Box sx={{ mb: 2 }}>
                  <TextField
                    type="number"
                    label="Sample Size"
                    value={sampleSize}
                    onChange={(e) => setSampleSize(parseInt(e.target.value))}
                    fullWidth
                    sx={{ mb: 2 }}
                  />
                  <TextField
                    type="number"
                    label="Random Seed"
                    value={randomSeed}
                    onChange={(e) => setRandomSeed(parseInt(e.target.value))}
                    fullWidth
                    sx={{ mb: 3 }}
                  />

                  <Box display="flex" gap={2}>
                    <Button
                      variant="outlined"
                      onClick={() => setActiveStep(2)}
                    >
                      Back
                    </Button>
                    <Button
                      variant="contained"
                      onClick={() => setActiveStep(4)}
                    >
                      Next: Review & Launch
                    </Button>
                  </Box>
                </Box>
              </StepContent>
            </Step>

            {/* Step 5: Review & Launch */}
            <Step>
              <StepLabel>Review & Launch</StepLabel>
              <StepContent>
                <Box sx={{ mb: 2 }}>
                  <Alert severity="info" sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Experiment Summary
                    </Typography>
                    <Typography variant="body2">
                      • Variables: {nodes.length}<br />
                      • Relationships: {edges.length}<br />
                      • Interventions: {interventions.length}<br />
                      • Sample Size: {sampleSize.toLocaleString()}
                    </Typography>
                  </Alert>

                  <Box display="flex" gap={2}>
                    <Button
                      variant="outlined"
                      onClick={() => setActiveStep(3)}
                    >
                      Back
                    </Button>
                    <Button
                      variant="contained"
                      startIcon={<PlayIcon />}
                      onClick={handleCreateExperiment}
                      color="success"
                      size="large"
                    >
                      Create & Run Experiment
                    </Button>
                  </Box>
                </Box>
              </StepContent>
            </Step>
          </Stepper>
        </CardContent>
      </Card>
    </Box>
  )
}