import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react'
import {
  Box,
  Tooltip,
  Chip,
  Alert,
  Typography,
  ButtonGroup,
  Button,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Switch,
  FormControlLabel,
  Slider,
  Paper,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  IconButton,
  Badge
} from '@mui/material'
import {
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  CenterFocusStrong as CenterIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  ExpandMore as ExpandMoreIcon,
  Settings as SettingsIcon,
  Timeline as TimelineIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon
} from '@mui/icons-material'
import { CausalDAG, CausalNode, CausalEdge } from '../types'
import { validateDAG } from '../utils'
import { useErrorHandler } from './ErrorBoundary'

interface CausalGraphProps {
  dag: CausalDAG
  onIntervene?: (nodeId: string, value: number) => void
  onNodeSelect?: (nodeId: string | null) => void
  selectedNode?: string | null
  interventions?: Array<{ variable: string; value: number; timestamp?: number }>
  showValidation?: boolean
  interactive?: boolean
  className?: string
  animationSpeed?: number
  showCausalFlow?: boolean
  enableRealTimeMetrics?: boolean
  onMetricsUpdate?: (metrics: any) => void
}

interface GraphState {
  scale: number
  panX: number
  panY: number
  isDragging: boolean
  showCausalFlow: boolean
  animationSpeed: number
  showEdgeWeights: boolean
  showInterventionHistory: boolean
  layoutType: 'force' | 'hierarchical' | 'circular'
  highlightBackdoorPaths: boolean
}

interface CausalFlowStep {
  id: string
  sourceNode: string
  targetNode: string
  value: number
  timestamp: number
  active: boolean
}

interface LayoutNode extends CausalNode {
  isIntervened: boolean
  interventionValue?: number
}

export function CausalGraph({ 
  dag, 
  onIntervene, 
  onNodeSelect,
  selectedNode,
  interventions = [],
  showValidation = true,
  interactive = true,
  className,
  animationSpeed = 1000,
  showCausalFlow = false,
  enableRealTimeMetrics = false,
  onMetricsUpdate
}: CausalGraphProps) {
  const { handleError } = useErrorHandler()
  const svgRef = useRef<SVGSVGElement>(null)
  const animationFrameRef = useRef<number>()
  const [graphState, setGraphState] = useState<GraphState>({
    scale: 1,
    panX: 0,
    panY: 0,
    isDragging: false,
    showCausalFlow: showCausalFlow,
    animationSpeed: animationSpeed,
    showEdgeWeights: true,
    showInterventionHistory: false,
    layoutType: 'force',
    highlightBackdoorPaths: false
  })
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; nodeId: string } | null>(null)
  const [interventionDialog, setInterventionDialog] = useState<{ nodeId: string } | null>(null)
  const [interventionValue, setInterventionValue] = useState<string>('1')
  const [validationResult, setValidationResult] = useState<any>(null)
  const [causalFlowSteps, setCausalFlowSteps] = useState<CausalFlowStep[]>([])
  const [activeAnimation, setActiveAnimation] = useState<boolean>(false)
  const [backdoorPaths, setBackdoorPaths] = useState<string[][]>([])
  const [settingsOpen, setSettingsOpen] = useState<boolean>(false)
  const [realtimeMetrics, setRealtimeMetrics] = useState<any>({})

  // Validate DAG
  useEffect(() => {
    try {
      const result = validateDAG(dag)
      setValidationResult(result)
      if (!result.is_valid && showValidation) {
        console.warn('DAG validation failed:', result.errors)
      }
    } catch (error) {
      handleError(error as Error, 'CausalGraph.validation')
    }
  }, [dag, showValidation, handleError])

  // Enhanced layout computation with automatic positioning
  const computedLayout = useMemo(() => {
    try {
      const nodes: LayoutNode[] = dag.nodes.map((node, index) => {
        const intervention = interventions.find(i => i.variable === node.id)
        
        // Auto-position nodes if no position specified
        let position = node.position
        if (!position || (position.x === 0 && position.y === 0)) {
          const angle = (index / dag.nodes.length) * 2 * Math.PI
          const radius = Math.min(200, 50 + dag.nodes.length * 15)
          position = {
            x: 300 + radius * Math.cos(angle),
            y: 200 + radius * Math.sin(angle)
          }
        }
        
        return {
          ...node,
          position,
          isIntervened: !!intervention,
          interventionValue: intervention?.value
        }
      })

      return { nodes, edges: dag.edges }
    } catch (error) {
      handleError(error as Error, 'CausalGraph.computeLayout')
      return { nodes: dag.nodes.map((n, i) => ({ ...n, isIntervened: false, position: { x: 100 + i * 100, y: 200 } })), edges: dag.edges }
    }
  }, [dag, interventions, handleError])

  const handleNodeClick = useCallback((nodeId: string, event: React.MouseEvent) => {
    if (!interactive) return
    
    try {
      if (event.ctrlKey || event.metaKey) {
        setContextMenu({ x: event.clientX, y: event.clientY, nodeId })
      } else {
        if (onIntervene) {
          onIntervene(nodeId, 1)
        }
        if (onNodeSelect) {
          onNodeSelect(nodeId === selectedNode ? null : nodeId)
        }
      }
    } catch (error) {
      handleError(error as Error, 'CausalGraph.nodeClick')
    }
  }, [interactive, onIntervene, onNodeSelect, selectedNode, handleError])

  const handleZoom = (direction: 'in' | 'out' | 'reset') => {
    setGraphState(prev => ({
      ...prev,
      scale: direction === 'in' ? Math.min(prev.scale * 1.2, 3) :
             direction === 'out' ? Math.max(prev.scale / 1.2, 0.3) :
             1,
      panX: direction === 'reset' ? 0 : prev.panX,
      panY: direction === 'reset' ? 0 : prev.panY
    }))
  }

  // Enhanced intervention handling with animation
  const handleInterventionSubmit = () => {
    if (!interventionDialog || !onIntervene) return
    
    try {
      const value = parseFloat(interventionValue)
      if (isNaN(value)) {
        alert('Please enter a valid number')
        return
      }
      
      // Trigger causal flow animation if enabled
      if (graphState.showCausalFlow) {
        triggerCausalFlow(interventionDialog.nodeId, value)
      }
      
      onIntervene(interventionDialog.nodeId, value)
      setInterventionDialog(null)
      setInterventionValue('1')
      
      // Update real-time metrics if enabled
      if (enableRealTimeMetrics && onMetricsUpdate) {
        updateMetrics(interventionDialog.nodeId, value)
      }
    } catch (error) {
      handleError(error as Error, 'CausalGraph.intervention')
    }
  }
  
  // Causal flow animation trigger
  const triggerCausalFlow = (sourceNode: string, value: number) => {
    const affectedEdges = dag.edges.filter(edge => edge.source === sourceNode)
    const flowSteps: CausalFlowStep[] = affectedEdges.map((edge, index) => ({
      id: `flow-${Date.now()}-${index}`,
      sourceNode: edge.source,
      targetNode: edge.target,
      value,
      timestamp: Date.now() + index * 200,
      active: false
    }))
    
    setCausalFlowSteps(prev => [...prev, ...flowSteps])
    setActiveAnimation(true)
    
    // Animate flow steps
    flowSteps.forEach((step, index) => {
      setTimeout(() => {
        setCausalFlowSteps(prev => 
          prev.map(s => s.id === step.id ? { ...s, active: true } : s)
        )
        
        // Deactivate after animation duration
        setTimeout(() => {
          setCausalFlowSteps(prev => 
            prev.filter(s => s.id !== step.id)
          )
        }, graphState.animationSpeed)
      }, index * 200)
    })
  }
  
  // Update real-time metrics
  const updateMetrics = (nodeId: string, value: number) => {
    const timestamp = Date.now()
    const newMetrics = {
      ...realtimeMetrics,
      lastIntervention: { nodeId, value, timestamp },
      interventionCount: (realtimeMetrics.interventionCount || 0) + 1,
      avgInterventionValue: (
        (realtimeMetrics.avgInterventionValue || 0) * (realtimeMetrics.interventionCount || 0) + value
      ) / ((realtimeMetrics.interventionCount || 0) + 1)
    }
    
    setRealtimeMetrics(newMetrics)
    if (onMetricsUpdate) {
      onMetricsUpdate(newMetrics)
    }
  }

  const svgWidth = 600
  const svgHeight = 400

  return (
    <Box className={`causal-graph ${className || ''}`} sx={{ position: 'relative' }}>
      {/* Validation Alerts */}
      {showValidation && validationResult && !validationResult.is_valid && (
        <Alert severity="error" sx={{ mb: 2 }}>
          <Typography variant="subtitle2">DAG Validation Errors:</Typography>
          {validationResult.errors.map((error: string, index: number) => (
            <Typography key={index} variant="body2">• {error}</Typography>
          ))}
        </Alert>
      )}

      {/* Enhanced Graph Controls */}
      <Box sx={{ position: 'absolute', top: 10, right: 10, zIndex: 10, display: 'flex', flexDirection: 'column', gap: 1 }}>
        <ButtonGroup size="small" orientation="vertical">
          <Tooltip title="Zoom In">
            <Button onClick={() => handleZoom('in')}><ZoomInIcon /></Button>
          </Tooltip>
          <Tooltip title="Zoom Out">
            <Button onClick={() => handleZoom('out')}><ZoomOutIcon /></Button>
          </Tooltip>
          <Tooltip title="Reset View">
            <Button onClick={() => handleZoom('reset')}><CenterIcon /></Button>
          </Tooltip>
        </ButtonGroup>
        
        <ButtonGroup size="small" orientation="vertical">
          <Tooltip title={graphState.showCausalFlow ? "Disable Causal Flow" : "Enable Causal Flow"}>
            <IconButton 
              size="small" 
              onClick={() => setGraphState(prev => ({ ...prev, showCausalFlow: !prev.showCausalFlow }))}
              color={graphState.showCausalFlow ? 'primary' : 'default'}
            >
              <TimelineIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Settings">
            <IconButton size="small" onClick={() => setSettingsOpen(!settingsOpen)}>
              <SettingsIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Toggle Edge Weights">
            <IconButton 
              size="small" 
              onClick={() => setGraphState(prev => ({ ...prev, showEdgeWeights: !prev.showEdgeWeights }))}
              color={graphState.showEdgeWeights ? 'primary' : 'default'}
            >
              {graphState.showEdgeWeights ? <VisibilityIcon /> : <VisibilityOffIcon />}
            </IconButton>
          </Tooltip>
        </ButtonGroup>
        
        {enableRealTimeMetrics && (
          <Badge badgeContent={realtimeMetrics.interventionCount || 0} color="secondary">
            <Paper sx={{ p: 1, minWidth: 120 }}>
              <Typography variant="caption" display="block">Interventions</Typography>
              <Typography variant="body2">
                Avg: {(realtimeMetrics.avgInterventionValue || 0).toFixed(2)}
              </Typography>
            </Paper>
          </Badge>
        )}
      </Box>

      {/* Enhanced Graph Info */}
      <Box sx={{ position: 'absolute', top: 10, left: 10, zIndex: 10, display: 'flex', flexDirection: 'column', gap: 1 }}>
        <Paper sx={{ px: 1, py: 0.5, borderRadius: 1 }}>
          <Typography variant="caption" display="block">
            {dag.nodes.length} nodes • {dag.edges.length} edges
            {interventions.length > 0 && ` • ${interventions.length} interventions`}
          </Typography>
          {activeAnimation && (
            <Typography variant="caption" color="primary" display="block">
              🔄 Causal flow active
            </Typography>
          )}
        </Paper>
        
        {/* Settings Panel */}
        {settingsOpen && (
          <Paper sx={{ p: 2, minWidth: 200 }}>
            <Typography variant="subtitle2" gutterBottom>Graph Settings</Typography>
            <FormControlLabel
              control={
                <Switch 
                  checked={graphState.highlightBackdoorPaths}
                  onChange={(e) => setGraphState(prev => ({ ...prev, highlightBackdoorPaths: e.target.checked }))}
                  size="small"
                />
              }
              label="Highlight Backdoor Paths"
            />
            <Box sx={{ mt: 1 }}>
              <Typography variant="caption">Animation Speed</Typography>
              <Slider
                value={graphState.animationSpeed}
                onChange={(_, value) => setGraphState(prev => ({ ...prev, animationSpeed: value as number }))}
                min={100}
                max={3000}
                step={100}
                size="small"
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${value}ms`}
              />
            </Box>
          </Paper>
        )}
      </Box>

      {/* Main SVG Graph */}
      <svg 
        width={svgWidth} 
        height={svgHeight} 
        viewBox={`0 0 ${svgWidth} ${svgHeight}`}
        style={{ 
          border: '1px solid #e0e0e0', 
          borderRadius: '8px',
          transform: `scale(${graphState.scale}) translate(${graphState.panX}px, ${graphState.panY}px)`
        }}
      >
        {/* Background */}
        <rect width="100%" height="100%" fill="#fafafa" />

        {/* Grid Pattern */}
        <defs>
          <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#e8e8e8" strokeWidth="0.5"/>
          </pattern>
          
          <marker
            id="arrowhead"
            markerWidth="10"
            markerHeight="7"
            refX="9"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
          </marker>
          
          <marker
            id="arrowhead-selected"
            markerWidth="12"
            markerHeight="8"
            refX="10"
            refY="4"
            orient="auto"
          >
            <polygon points="0 0, 12 4, 0 8" fill="#1976d2" />
          </marker>
        </defs>
        
        <rect width="100%" height="100%" fill="url(#grid)" />
        
        {/* Enhanced edge rendering with animations */}
        {computedLayout.edges.map((edge, index) => {
          const sourceNode = computedLayout.nodes.find(n => n.id === edge.source)
          const targetNode = computedLayout.nodes.find(n => n.id === edge.target)
          
          if (!sourceNode || !targetNode) return null
          
          const strokeWidth = Math.abs(edge.weight || 1) * 2 + 1
          const strokeColor = edge.weight && edge.weight < 0 ? '#d32f2f' : '#666'
          
          // Check if this edge has active causal flow
          const activeFlow = causalFlowSteps.find(step => 
            step.sourceNode === edge.source && step.targetNode === edge.target && step.active
          )
          
          // Check if this is a backdoor path that should be highlighted
          const isBackdoorPath = graphState.highlightBackdoorPaths && 
            backdoorPaths.some(path => 
              path.includes(edge.source) && path.includes(edge.target)
            )
          
          return (
            <g key={`edge-${index}`}>
              <line
                x1={sourceNode.position.x}
                y1={sourceNode.position.y}
                x2={targetNode.position.x}
                y2={targetNode.position.y}
                stroke={activeFlow ? '#ff5722' : isBackdoorPath ? '#e91e63' : strokeColor}
                strokeWidth={activeFlow ? strokeWidth + 2 : strokeWidth}
                opacity={activeFlow ? 1 : isBackdoorPath ? 0.8 : 0.7}
                markerEnd={activeFlow ? "url(#arrowhead-selected)" : "url(#arrowhead)"}
                style={{ cursor: interactive ? 'pointer' : 'default' }}
                strokeDasharray={isBackdoorPath ? '5,5' : 'none'}
              >
                {activeFlow && (
                  <animate
                    attributeName="stroke-opacity"
                    values="0.5;1;0.5"
                    dur={`${graphState.animationSpeed}ms`}
                    repeatCount="1"
                  />
                )}
              </line>
              
              {/* Animated flow particle */}
              {activeFlow && (
                <circle
                  r="3"
                  fill="#ff5722"
                  opacity="0.8"
                >
                  <animateMotion
                    dur={`${graphState.animationSpeed}ms`}
                    repeatCount="1"
                  >
                    <mpath href={`#path-${index}`} />
                  </animateMotion>
                </circle>
              )}
              
              {/* Hidden path for animation */}
              <path
                id={`path-${index}`}
                d={`M ${sourceNode.position.x} ${sourceNode.position.y} L ${targetNode.position.x} ${targetNode.position.y}`}
                style={{ display: 'none' }}
              />
              
              {/* Edge weight label */}
              {graphState.showEdgeWeights && edge.weight !== undefined && Math.abs(edge.weight) > 0.1 && (
                <text
                  x={(sourceNode.position.x + targetNode.position.x) / 2}
                  y={(sourceNode.position.y + targetNode.position.y) / 2 - 8}
                  textAnchor="middle"
                  fill={activeFlow ? '#ff5722' : '#666'}
                  fontSize="10"
                  fontWeight={activeFlow ? 'bold' : 'normal'}
                  className="pointer-events-none select-none"
                >
                  {edge.weight.toFixed(1)}
                  {activeFlow && ` (${activeFlow.value.toFixed(1)})`}
                </text>
              )}
            </g>
          )
        })}
        
        {/* Enhanced node rendering with advanced features */}
        {computedLayout.nodes.map((node) => {
          const isSelected = selectedNode === node.id
          const isHovered = hoveredNode === node.id
          const isIntervened = node.isIntervened
          
          // Check if node is part of active causal flow
          const hasActiveFlow = causalFlowSteps.some(step => 
            (step.sourceNode === node.id || step.targetNode === node.id) && step.active
          )
          
          const nodeRadius = isHovered ? 25 : isSelected ? 23 : hasActiveFlow ? 22 : 20
          const fillColor = isIntervened ? '#ff5722' : 
                           hasActiveFlow ? '#4caf50' : 
                           isSelected ? '#1976d2' : '#4f46e5'
          const strokeColor = isHovered ? '#333' : hasActiveFlow ? '#2e7d32' : '#1e1b4b'
          const strokeWidth = isSelected ? 3 : hasActiveFlow ? 2.5 : 2
          
          return (
            <g key={node.id}>
              <circle
                cx={node.position.x}
                cy={node.position.y}
                r={nodeRadius}
                fill={fillColor}
                stroke={strokeColor}
                strokeWidth={strokeWidth}
                style={{ cursor: interactive ? 'pointer' : 'default' }}
                onMouseEnter={() => interactive && setHoveredNode(node.id)}
                onMouseLeave={() => setHoveredNode(null)}
                onClick={(e) => handleNodeClick(node.id, e.nativeEvent)}
              >
                {hasActiveFlow && (
                  <animate
                    attributeName="r"
                    values={`${nodeRadius};${nodeRadius + 3};${nodeRadius}`}
                    dur={`${graphState.animationSpeed / 2}ms`}
                    repeatCount="2"
                  />
                )}
              </circle>
              
              {/* Node pulse effect for active flow */}
              {hasActiveFlow && (
                <circle
                  cx={node.position.x}
                  cy={node.position.y}
                  r={nodeRadius}
                  fill="none"
                  stroke={fillColor}
                  strokeWidth="1"
                  opacity="0"
                >
                  <animate
                    attributeName="r"
                    values={`${nodeRadius};${nodeRadius + 15}`}
                    dur={`${graphState.animationSpeed}ms`}
                    repeatCount="1"
                  />
                  <animate
                    attributeName="opacity"
                    values="0.8;0"
                    dur={`${graphState.animationSpeed}ms`}
                    repeatCount="1"
                  />
                </circle>
              )}
              
              <text
                x={node.position.x}
                y={node.position.y + 4}
                textAnchor="middle"
                fill="white"
                fontSize="12"
                fontWeight={isSelected ? 'bold' : 'normal'}
                className="pointer-events-none select-none"
              >
                {node.label}
              </text>
              
              {/* Enhanced intervention indicator */}
              {isIntervened && (
                <g>
                  <rect
                    x={node.position.x + 15}
                    y={node.position.y - 25}
                    width={40}
                    height={16}
                    fill={fillColor}
                    rx={8}
                    opacity={0.9}
                  />
                  <text
                    x={node.position.x + 35}
                    y={node.position.y - 15}
                    textAnchor="middle"
                    fill="white"
                    fontSize="9"
                    fontWeight="bold"
                    className="pointer-events-none select-none"
                  >
                    do({node.interventionValue})
                  </text>
                </g>
              )}
              
              {/* Node value indicator */}
              {node.variable_type && (
                <text
                  x={node.position.x}
                  y={node.position.y + 35}
                  textAnchor="middle"
                  fill="#666"
                  fontSize="8"
                  className="pointer-events-none select-none"
                >
                  {node.variable_type}
                </text>
              )}
            </g>
          )
        })}
      </svg>

      {/* Enhanced hover information */}
      {hoveredNode && (
        <Box sx={{ position: 'absolute', bottom: 10, left: 10, zIndex: 10 }}>
          <Paper sx={{ p: 1 }}>
            <Typography variant="subtitle2">{hoveredNode}</Typography>
            <Typography variant="caption" display="block">
              Type: {dag.nodes.find(n => n.id === hoveredNode)?.variable_type || 'unknown'}
            </Typography>
            {interventions.find(i => i.variable === hoveredNode) && (
              <Typography variant="caption" color="primary" display="block">
                Intervened: {interventions.find(i => i.variable === hoveredNode)?.value}
              </Typography>
            )}
            <Typography variant="caption" display="block">
              Parents: {dag.edges.filter(e => e.target === hoveredNode).length}
            </Typography>
            <Typography variant="caption" display="block">
              Children: {dag.edges.filter(e => e.source === hoveredNode).length}
            </Typography>
          </Paper>
        </Box>
      )}

      {/* Context Menu */}
      <Menu
        anchorReference="anchorPosition"
        anchorPosition={contextMenu ? { top: contextMenu.y, left: contextMenu.x } : undefined}
        open={Boolean(contextMenu)}
        onClose={() => setContextMenu(null)}
      >
        <MenuItem onClick={() => {
          if (contextMenu) {
            setInterventionDialog({ nodeId: contextMenu.nodeId })
          }
          setContextMenu(null)
        }}>
          Set Intervention Value...
        </MenuItem>
        <MenuItem onClick={() => {
          if (contextMenu && graphState.showCausalFlow) {
            triggerCausalFlow(contextMenu.nodeId, 1.0)
          }
          setContextMenu(null)
        }}>
          Simulate Causal Flow
        </MenuItem>
        <MenuItem onClick={() => {
          if (contextMenu) {
            // Find backdoor paths for this node
            const targetNodes = dag.nodes.filter(n => n.id !== contextMenu.nodeId)
            // This would integrate with causal engine for proper backdoor identification
            console.log(`Finding backdoor paths for ${contextMenu.nodeId}`)
          }
          setContextMenu(null)
        }}>
          Find Backdoor Paths
        </MenuItem>
      </Menu>

      {/* Intervention Dialog */}
      <Dialog
        open={Boolean(interventionDialog)}
        onClose={() => setInterventionDialog(null)}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle>
          Set Intervention for {interventionDialog?.nodeId}
        </DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Intervention Value"
            type="number"
            fullWidth
            value={interventionValue}
            onChange={(e) => setInterventionValue(e.target.value)}
            helperText="Enter the value to set this variable to"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setInterventionDialog(null)}>Cancel</Button>
          <Button onClick={handleInterventionSubmit} variant="contained">Apply Intervention</Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}