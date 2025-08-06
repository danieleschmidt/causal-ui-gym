import React, { useState, useEffect, useMemo, useCallback } from 'react'
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
  TextField
} from '@mui/material'
import {
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  CenterFocusStrong as CenterIcon
} from '@mui/icons-material'
import { CausalDAG, CausalNode, CausalEdge } from '../types'
import { validateDAG } from '../utils'
import { useErrorHandler } from './ErrorBoundary'

interface CausalGraphProps {
  dag: CausalDAG
  onIntervene?: (nodeId: string, value: number) => void
  onNodeSelect?: (nodeId: string | null) => void
  selectedNode?: string | null
  interventions?: Array<{ variable: string; value: number }>
  showValidation?: boolean
  interactive?: boolean
  className?: string
}

interface GraphState {
  scale: number
  panX: number
  panY: number
  isDragging: boolean
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
  className 
}: CausalGraphProps) {
  const { handleError } = useErrorHandler()
  const [graphState, setGraphState] = useState<GraphState>({
    scale: 1,
    panX: 0,
    panY: 0,
    isDragging: false
  })
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; nodeId: string } | null>(null)
  const [interventionDialog, setInterventionDialog] = useState<{ nodeId: string } | null>(null)
  const [interventionValue, setInterventionValue] = useState<string>('1')
  const [validationResult, setValidationResult] = useState<any>(null)

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

  // Compute enhanced layout
  const computedLayout = useMemo(() => {
    try {
      const nodes: LayoutNode[] = dag.nodes.map(node => {
        const intervention = interventions.find(i => i.variable === node.id)
        return {
          ...node,
          isIntervened: !!intervention,
          interventionValue: intervention?.value
        }
      })

      return { nodes, edges: dag.edges }
    } catch (error) {
      handleError(error as Error, 'CausalGraph.computeLayout')
      return { nodes: dag.nodes.map(n => ({ ...n, isIntervened: false })), edges: dag.edges }
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

  const handleInterventionSubmit = () => {
    if (!interventionDialog || !onIntervene) return
    
    try {
      const value = parseFloat(interventionValue)
      if (isNaN(value)) {
        alert('Please enter a valid number')
        return
      }
      
      onIntervene(interventionDialog.nodeId, value)
      setInterventionDialog(null)
      setInterventionValue('1')
    } catch (error) {
      handleError(error as Error, 'CausalGraph.intervention')
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

      {/* Graph Controls */}
      <Box sx={{ position: 'absolute', top: 10, right: 10, zIndex: 10 }}>
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
      </Box>

      {/* Graph Info */}
      <Box sx={{ position: 'absolute', top: 10, left: 10, zIndex: 10 }}>
        <Typography variant="caption" sx={{ bgcolor: 'background.paper', px: 1, py: 0.5, borderRadius: 1 }}>
          {dag.nodes.length} nodes • {dag.edges.length} edges
          {interventions.length > 0 && ` • ${interventions.length} interventions`}
        </Typography>
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
        
        {/* Render edges */}
        {computedLayout.edges.map((edge, index) => {
          const sourceNode = computedLayout.nodes.find(n => n.id === edge.source)
          const targetNode = computedLayout.nodes.find(n => n.id === edge.target)
          
          if (!sourceNode || !targetNode) return null
          
          const strokeWidth = Math.abs(edge.weight || 1) * 2 + 1
          const strokeColor = edge.weight && edge.weight < 0 ? '#d32f2f' : '#666'
          
          return (
            <g key={`edge-${index}`}>
              <line
                x1={sourceNode.position.x}
                y1={sourceNode.position.y}
                x2={targetNode.position.x}
                y2={targetNode.position.y}
                stroke={strokeColor}
                strokeWidth={strokeWidth}
                opacity={0.7}
                markerEnd="url(#arrowhead)"
                style={{ cursor: interactive ? 'pointer' : 'default' }}
              />
              
              {/* Edge weight label */}
              {edge.weight !== undefined && Math.abs(edge.weight) > 0.1 && (
                <text
                  x={(sourceNode.position.x + targetNode.position.x) / 2}
                  y={(sourceNode.position.y + targetNode.position.y) / 2 - 8}
                  textAnchor="middle"
                  fill="#666"
                  fontSize="10"
                  className="pointer-events-none select-none"
                >
                  {edge.weight.toFixed(1)}
                </text>
              )}
            </g>
          )
        })}
        
        {/* Render nodes */}
        {computedLayout.nodes.map((node) => {
          const isSelected = selectedNode === node.id
          const isHovered = hoveredNode === node.id
          const isIntervened = node.isIntervened
          
          const nodeRadius = isHovered ? 25 : isSelected ? 23 : 20
          const fillColor = isIntervened ? '#ff5722' : isSelected ? '#1976d2' : '#4f46e5'
          const strokeColor = isHovered ? '#333' : '#1e1b4b'
          const strokeWidth = isSelected ? 3 : 2
          
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
              />
              
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
              
              {/* Intervention indicator */}
              {isIntervened && (
                <text
                  x={node.position.x + 18}
                  y={node.position.y - 15}
                  textAnchor="middle"
                  fill={fillColor}
                  fontSize="10"
                  fontWeight="bold"
                  className="pointer-events-none select-none"
                >
                  do({node.interventionValue})
                </text>
              )}
            </g>
          )
        })}
      </svg>

      {/* Hover information */}
      {hoveredNode && (
        <Box sx={{ position: 'absolute', bottom: 10, left: 10, zIndex: 10 }}>
          <Chip 
            label={`${hoveredNode} (${dag.nodes.find(n => n.id === hoveredNode)?.variable_type || 'unknown'})`}
            size="small"
          />
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