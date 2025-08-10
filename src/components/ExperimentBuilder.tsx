import React, { useState, useCallback, useEffect, useMemo } from 'react'
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
  StepContent,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Slider,
  Switch,
  FormControlLabel,
  List,
  ListItem,
  ListItemText,
  ListItemSecondary,
  Divider,
  Tabs,
  Tab,
  Grid,
  Paper,
  Tooltip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  CircularProgress,
  LinearProgress
} from '@mui/material'
import { 
  Add as AddIcon, 
  Delete as DeleteIcon, 
  PlayArrow as PlayIcon,
  ExpandMore as ExpandMoreIcon,
  AutoFixHigh as AutoFixHighIcon,
  Preview as PreviewIcon,
  Science as ScienceIcon,
  Timeline as TimelineIcon,
  Psychology as PsychologyIcon,
  Storage as StorageIcon,
  Speed as SpeedIcon,
  TrendingUp as TrendingUpIcon,
  Assessment as AssessmentIcon
} from '@mui/icons-material'
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
  'LLM Agent Setup',
  'Advanced Parameters',
  'Review & Launch'
]

interface LLMAgent {
  id: string
  name: string
  model: string
  provider: 'openai' | 'anthropic' | 'huggingface'
  temperature: number
  maxTokens: number
  systemPrompt: string
  enabled: boolean
}

interface AdvancedParameters {
  enableRealTimeMetrics: boolean
  enableCausalFlowAnimation: boolean
  computationBatchSize: number
  confidenceLevel: number
  parallelComputations: number
  cacheResults: boolean
  exportFormat: 'json' | 'csv' | 'latex'
}

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
  
  // LLM Agents configuration
  const [llmAgents, setLlmAgents] = useState<LLMAgent[]>([
    {
      id: 'gpt4',
      name: 'GPT-4',
      model: 'gpt-4',
      provider: 'openai',
      temperature: 0.7,
      maxTokens: 2048,
      systemPrompt: 'You are an expert in causal reasoning and statistical inference.',
      enabled: true
    },
    {
      id: 'claude3',
      name: 'Claude-3',
      model: 'claude-3-sonnet',
      provider: 'anthropic',
      temperature: 0.5,
      maxTokens: 2048,
      systemPrompt: 'You are an expert in causal reasoning and statistical inference.',
      enabled: false
    }
  ])
  
  // Advanced parameters
  const [advancedParams, setAdvancedParams] = useState<AdvancedParameters>({
    enableRealTimeMetrics: true,
    enableCausalFlowAnimation: true,
    computationBatchSize: 1000,
    confidenceLevel: 0.95,
    parallelComputations: 4,
    cacheResults: true,
    exportFormat: 'json'
  })
  
  // UI state
  const [tabValue, setTabValue] = useState(0)
  const [previewDialogOpen, setPreviewDialogOpen] = useState(false)
  const [isGeneratingVariables, setIsGeneratingVariables] = useState(false)
  
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
      description: 'New intervention',
      timestamp: Date.now()
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
      llm_agents: llmAgents.filter(agent => agent.enabled),
      advanced_parameters: advancedParams,
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
    llmAgents,
    advancedParams,
    onCreateExperiment,
    onRunExperiment
  ])

  const dag: CausalDAG = {
    name: experimentName || 'Draft Experiment',
    description: experimentDescription,
    nodes,
    edges
  }

  // Additional helper functions
  const handleAutoGenerateVariables = useCallback(async () => {
    setIsGeneratingVariables(true)
    // This would integrate with an AI service to suggest variables
    setTimeout(() => {
      const suggestedVariables = [
        { name: 'treatment', type: 'binary' as const },
        { name: 'outcome', type: 'continuous' as const },
        { name: 'confounder', type: 'continuous' as const }
      ]
      
      suggestedVariables.forEach((variable, index) => {
        const newNode: CausalNode = {
          id: variable.name,
          label: variable.name.charAt(0).toUpperCase() + variable.name.slice(1),
          position: { x: 100 + index * 150, y: 100 + (index % 3) * 100 },
          variable_type: variable.type,
          description: `AI-suggested variable: ${variable.name}`
        }
        setNodes(prev => [...prev, newNode])
      })
      
      setIsGeneratingVariables(false)
    }, 2000)
  }, [])
  
  const handleSuggestRelationships = useCallback(() => {
    // AI-powered relationship suggestions
    if (nodes.length >= 2) {
      const suggestedEdges: CausalEdge[] = [
        {
          source: nodes[0].id,
          target: nodes[1].id,
          weight: 0.7,
          edge_type: 'causal',
          confidence: 0.85
        }
      ]
      setEdges(prev => [...prev, ...suggestedEdges])
    }
  }, [nodes])
  
  const handleAddLLMAgent = useCallback(() => {
    const newAgent: LLMAgent = {
      id: `agent_${Date.now()}`,
      name: 'Custom Agent',
      model: 'gpt-3.5-turbo',
      provider: 'openai',
      temperature: 0.5,
      maxTokens: 1024,
      systemPrompt: 'You are an expert in causal reasoning.',
      enabled: true
    }
    setLlmAgents(prev => [...prev, newAgent])
  }, [])
  
  const calculateEstimatedRuntime = () => {
    const baseTime = sampleSize / 1000 // Base computation time
    const interventionMultiplier = interventions.length * 0.5
    const agentMultiplier = llmAgents.filter(a => a.enabled).length * 2
    const totalMinutes = Math.ceil(baseTime + interventionMultiplier + agentMultiplier)
    return `${totalMinutes} minutes`
  }
  
  const calculateReadinessScore = () => {
    let score = 0
    if (nodes.length >= 2) score += 20
    if (edges.length >= 1) score += 20
    if (interventions.length >= 1) score += 20
    if (llmAgents.some(a => a.enabled)) score += 20
    if (experimentName.length > 0) score += 10
    if (experimentDescription.length > 0) score += 10
    return score
  }
  
  const getReadinessWarnings = () => {
    const warnings = []
    if (nodes.length < 3) warnings.push('Consider adding more variables for richer analysis')
    if (interventions.length < 2) warnings.push('Multiple interventions provide better insights')
    if (!advancedParams.enableRealTimeMetrics) warnings.push('Real-time metrics enhance analysis')
    return warnings
  }

  return (
    <Box className={`experiment-builder ${className || ''}`}>
      <Card>
        <CardContent>
          <Typography variant="h5" gutterBottom startIcon={<ScienceIcon />}>
            Create New Causal Experiment
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

                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Button
                      variant="outlined"
                      startIcon={<AutoFixHighIcon />}
                      onClick={handleAutoGenerateVariables}
                      disabled={isGeneratingVariables}
                    >
                      {isGeneratingVariables ? <CircularProgress size={16} /> : 'AI Suggestions'}
                    </Button>
                    <Button
                      variant="contained"
                      onClick={() => setActiveStep(1)}
                      disabled={nodes.length < 2}
                    >
                      Next: Define Relationships
                    </Button>
                  </Box>
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
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Button variant="outlined" onClick={handleAutoLayout}>
                        Auto Layout
                      </Button>
                      <Button 
                        variant="outlined" 
                        startIcon={<PreviewIcon />}
                        onClick={() => setPreviewDialogOpen(true)}
                      >
                        Preview
                      </Button>
                      <Button 
                        variant="outlined"
                        startIcon={<ScienceIcon />}
                        onClick={handleSuggestRelationships}
                      >
                        AI Suggestions
                      </Button>
                    </Box>
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
                      Next: LLM Agents
                    </Button>
                  </Box>
                </Box>
              </StepContent>
            </Step>

            {/* Step 4: LLM Agent Setup */}
            <Step>
              <StepLabel>LLM Agent Setup</StepLabel>
              <StepContent>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Configure AI Agents for Causal Reasoning
                  </Typography>
                  
                  {llmAgents.map((agent, index) => (
                    <Accordion key={agent.id} sx={{ mb: 1 }}>
                      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                          <Switch
                            checked={agent.enabled}
                            onChange={(e) => {
                              const newAgents = [...llmAgents]
                              newAgents[index].enabled = e.target.checked
                              setLlmAgents(newAgents)
                            }}
                            onClick={(e) => e.stopPropagation()}
                          />
                          <Typography sx={{ ml: 2 }}>
                            {agent.name} ({agent.model})
                          </Typography>
                        </Box>
                      </AccordionSummary>
                      <AccordionDetails>
                        <Grid container spacing={2}>
                          <Grid item xs={6}>
                            <TextField
                              fullWidth
                              label="Model"
                              value={agent.model}
                              onChange={(e) => {
                                const newAgents = [...llmAgents]
                                newAgents[index].model = e.target.value
                                setLlmAgents(newAgents)
                              }}
                            />
                          </Grid>
                          <Grid item xs={6}>
                            <FormControl fullWidth>
                              <InputLabel>Provider</InputLabel>
                              <Select
                                value={agent.provider}
                                onChange={(e) => {
                                  const newAgents = [...llmAgents]
                                  newAgents[index].provider = e.target.value as any
                                  setLlmAgents(newAgents)
                                }}
                              >
                                <MenuItem value="openai">OpenAI</MenuItem>
                                <MenuItem value="anthropic">Anthropic</MenuItem>
                                <MenuItem value="huggingface">HuggingFace</MenuItem>
                              </Select>
                            </FormControl>
                          </Grid>
                          <Grid item xs={12}>
                            <Typography gutterBottom>Temperature: {agent.temperature}</Typography>
                            <Slider
                              value={agent.temperature}
                              onChange={(_, value) => {
                                const newAgents = [...llmAgents]
                                newAgents[index].temperature = value as number
                                setLlmAgents(newAgents)
                              }}
                              min={0}
                              max={1}
                              step={0.1}
                              marks
                              valueLabelDisplay="auto"
                            />
                          </Grid>
                          <Grid item xs={12}>
                            <TextField
                              fullWidth
                              multiline
                              rows={3}
                              label="System Prompt"
                              value={agent.systemPrompt}
                              onChange={(e) => {
                                const newAgents = [...llmAgents]
                                newAgents[index].systemPrompt = e.target.value
                                setLlmAgents(newAgents)
                              }}
                            />
                          </Grid>
                        </Grid>
                      </AccordionDetails>
                    </Accordion>
                  ))}
                  
                  <Button
                    variant="outlined"
                    startIcon={<AddIcon />}
                    onClick={handleAddLLMAgent}
                    sx={{ mt: 2 }}
                  >
                    Add Custom Agent
                  </Button>

                  <Box display="flex" gap={2} sx={{ mt: 3 }}>
                    <Button
                      variant="outlined"
                      onClick={() => setActiveStep(2)}
                    >
                      Back
                    </Button>
                    <Button
                      variant="contained"
                      onClick={() => setActiveStep(4)}
                      disabled={!llmAgents.some(agent => agent.enabled)}
                    >
                      Next: Advanced Parameters
                    </Button>
                  </Box>
                </Box>
              </StepContent>
            </Step>

            {/* Step 5: Advanced Parameters */}
            <Step>
              <StepLabel>Advanced Parameters</StepLabel>
              <StepContent>
                <Box sx={{ mb: 2 }}>
                  <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)}>
                    <Tab label="Computation" icon={<SpeedIcon />} />
                    <Tab label="Visualization" icon={<TimelineIcon />} />
                    <Tab label="Export" icon={<StorageIcon />} />
                  </Tabs>
                  
                  {tabValue === 0 && (
                    <Box sx={{ mt: 2 }}>
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <TextField
                            type="number"
                            fullWidth
                            label="Sample Size"
                            value={sampleSize}
                            onChange={(e) => setSampleSize(parseInt(e.target.value))}
                          />
                        </Grid>
                        <Grid item xs={6}>
                          <TextField
                            type="number"
                            fullWidth
                            label="Random Seed"
                            value={randomSeed}
                            onChange={(e) => setRandomSeed(parseInt(e.target.value))}
                          />
                        </Grid>
                        <Grid item xs={6}>
                          <TextField
                            type="number"
                            fullWidth
                            label="Batch Size"
                            value={advancedParams.computationBatchSize}
                            onChange={(e) => setAdvancedParams(prev => ({ 
                              ...prev, 
                              computationBatchSize: parseInt(e.target.value) 
                            }))}
                          />
                        </Grid>
                        <Grid item xs={6}>
                          <TextField
                            type="number"
                            fullWidth
                            label="Parallel Computations"
                            value={advancedParams.parallelComputations}
                            onChange={(e) => setAdvancedParams(prev => ({ 
                              ...prev, 
                              parallelComputations: parseInt(e.target.value) 
                            }))}
                          />
                        </Grid>
                        <Grid item xs={12}>
                          <Typography gutterBottom>
                            Confidence Level: {advancedParams.confidenceLevel}
                          </Typography>
                          <Slider
                            value={advancedParams.confidenceLevel}
                            onChange={(_, value) => setAdvancedParams(prev => ({ 
                              ...prev, 
                              confidenceLevel: value as number 
                            }))}
                            min={0.8}
                            max={0.99}
                            step={0.01}
                            marks={[{value: 0.90, label: '90%'}, {value: 0.95, label: '95%'}, {value: 0.99, label: '99%'}]}
                            valueLabelDisplay="auto"
                            valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
                          />
                        </Grid>
                        <Grid item xs={12}>
                          <FormControlLabel
                            control={
                              <Switch
                                checked={advancedParams.cacheResults}
                                onChange={(e) => setAdvancedParams(prev => ({ 
                                  ...prev, 
                                  cacheResults: e.target.checked 
                                }))}
                              />
                            }
                            label="Cache Computation Results"
                          />
                        </Grid>
                      </Grid>
                    </Box>
                  )}
                  
                  {tabValue === 1 && (
                    <Box sx={{ mt: 2 }}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={advancedParams.enableRealTimeMetrics}
                            onChange={(e) => setAdvancedParams(prev => ({ 
                              ...prev, 
                              enableRealTimeMetrics: e.target.checked 
                            }))}
                          />
                        }
                        label="Enable Real-time Metrics Dashboard"
                      />
                      <FormControlLabel
                        control={
                          <Switch
                            checked={advancedParams.enableCausalFlowAnimation}
                            onChange={(e) => setAdvancedParams(prev => ({ 
                              ...prev, 
                              enableCausalFlowAnimation: e.target.checked 
                            }))}
                          />
                        }
                        label="Enable Causal Flow Animations"
                      />
                    </Box>
                  )}
                  
                  {tabValue === 2 && (
                    <Box sx={{ mt: 2 }}>
                      <FormControl fullWidth>
                        <InputLabel>Export Format</InputLabel>
                        <Select
                          value={advancedParams.exportFormat}
                          onChange={(e) => setAdvancedParams(prev => ({ 
                            ...prev, 
                            exportFormat: e.target.value as any 
                          }))}
                        >
                          <MenuItem value="json">JSON</MenuItem>
                          <MenuItem value="csv">CSV</MenuItem>
                          <MenuItem value="latex">LaTeX</MenuItem>
                        </Select>
                      </FormControl>
                    </Box>
                  )}

                  <Box display="flex" gap={2} sx={{ mt: 3 }}>
                    <Button
                      variant="outlined"
                      onClick={() => setActiveStep(3)}
                    >
                      Back
                    </Button>
                    <Button
                      variant="contained"
                      onClick={() => setActiveStep(5)}
                    >
                      Next: Review & Launch
                    </Button>
                  </Box>
                </Box>
              </StepContent>
            </Step>

            {/* Step 6: Review & Launch */}
            <Step>
              <StepLabel>Review & Launch</StepLabel>
              <StepContent>
                <Box sx={{ mb: 2 }}>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={8}>
                      <Paper sx={{ p: 2, mb: 2 }}>
                        <Typography variant="h6" gutterBottom startIcon={<AssessmentIcon />}>
                          Experiment Summary
                        </Typography>
                        <List dense>
                          <ListItem>
                            <ListItemText primary="Variables" secondary={nodes.length} />
                          </ListItem>
                          <ListItem>
                            <ListItemText primary="Causal Relationships" secondary={edges.length} />
                          </ListItem>
                          <ListItem>
                            <ListItemText primary="Interventions Planned" secondary={interventions.length} />
                          </ListItem>
                          <ListItem>
                            <ListItemText 
                              primary="Active LLM Agents" 
                              secondary={llmAgents.filter(a => a.enabled).map(a => a.name).join(', ')}
                            />
                          </ListItem>
                          <ListItem>
                            <ListItemText primary="Sample Size" secondary={sampleSize.toLocaleString()} />
                          </ListItem>
                          <ListItem>
                            <ListItemText primary="Confidence Level" secondary={`${(advancedParams.confidenceLevel * 100).toFixed(0)}%`} />
                          </ListItem>
                          <Divider />
                          <ListItem>
                            <ListItemText 
                              primary="Estimated Runtime" 
                              secondary={calculateEstimatedRuntime()}
                            />
                          </ListItem>
                          <ListItem>
                            <ListItemText 
                              primary="Features Enabled" 
                              secondary={[
                                advancedParams.enableRealTimeMetrics && 'Real-time Metrics',
                                advancedParams.enableCausalFlowAnimation && 'Flow Animation',
                                advancedParams.cacheResults && 'Result Caching'
                              ].filter(Boolean).join(', ')}
                            />
                          </ListItem>
                        </List>
                      </Paper>
                    </Grid>
                    
                    <Grid item xs={12} md={4}>
                      <Paper sx={{ p: 2, mb: 2, textAlign: 'center' }}>
                        <Typography variant="h6" gutterBottom>
                          Experiment Readiness
                        </Typography>
                        <LinearProgress 
                          variant="determinate" 
                          value={calculateReadinessScore()} 
                          sx={{ mb: 2 }}
                        />
                        <Typography variant="body2" color="text.secondary">
                          {calculateReadinessScore()}% Complete
                        </Typography>
                        
                        {getReadinessWarnings().length > 0 && (
                          <Alert severity="warning" sx={{ mt: 2, textAlign: 'left' }}>
                            <Typography variant="subtitle2">Recommendations:</Typography>
                            {getReadinessWarnings().map((warning, idx) => (
                              <Typography key={idx} variant="body2">• {warning}</Typography>
                            ))}
                          </Alert>
                        )}
                      </Paper>
                    </Grid>
                  </Grid>

                  <Box display="flex" gap={2} justifyContent="center">
                    <Button
                      variant="outlined"
                      onClick={() => setActiveStep(4)}
                    >
                      Back
                    </Button>
                    <Button
                      variant="outlined"
                      startIcon={<PreviewIcon />}
                      onClick={() => setPreviewDialogOpen(true)}
                    >
                      Final Preview
                    </Button>
                    <Button
                      variant="contained"
                      startIcon={<PlayIcon />}
                      onClick={handleCreateExperiment}
                      color="success"
                      size="large"
                      disabled={calculateReadinessScore() < 80}
                    >
                      Launch Experiment
                    </Button>
                  </Box>
                </Box>
              </StepContent>
            </Step>
          </Stepper>
        </CardContent>
      </Card>
      
      {/* Preview Dialog */}
      <Dialog 
        open={previewDialogOpen} 
        onClose={() => setPreviewDialogOpen(false)}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>
          Experiment Preview: {experimentName || 'Draft Experiment'}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ height: 500 }}>
            <CausalGraph
              dag={dag}
              interventions={interventions}
              showValidation={true}
              showCausalFlow={advancedParams.enableCausalFlowAnimation}
              enableRealTimeMetrics={advancedParams.enableRealTimeMetrics}
              animationSpeed={1000}
            />
          </Box>
          
          <Typography variant="h6" sx={{ mt: 2, mb: 1 }}>Configuration Summary</Typography>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Typography variant="body2">
                <strong>Variables:</strong> {nodes.map(n => n.label).join(', ')}
              </Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="body2">
                <strong>LLM Agents:</strong> {llmAgents.filter(a => a.enabled).map(a => a.name).join(', ')}
              </Typography>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setPreviewDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}