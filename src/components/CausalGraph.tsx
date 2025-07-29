import React from 'react'
import { CausalDAG, CausalNode, CausalEdge } from '../types'

interface CausalGraphProps {
  dag: CausalDAG
  onIntervene?: (nodeId: string, value: number) => void
  className?: string
}

export function CausalGraph({ dag, onIntervene, className }: CausalGraphProps) {
  const handleNodeClick = (nodeId: string) => {
    if (onIntervene) {
      // Placeholder intervention value
      onIntervene(nodeId, 1)
    }
  }

  return (
    <div className={`causal-graph ${className || ''}`}>
      <svg width="400" height="300" viewBox="0 0 400 300">
        {/* Render edges */}
        {dag.edges.map((edge, index) => {
          const sourceNode = dag.nodes.find(n => n.id === edge.source)
          const targetNode = dag.nodes.find(n => n.id === edge.target)
          
          if (!sourceNode || !targetNode) return null
          
          return (
            <line
              key={`edge-${index}`}
              x1={sourceNode.position.x}
              y1={sourceNode.position.y}
              x2={targetNode.position.x}
              y2={targetNode.position.y}
              stroke="#666"
              strokeWidth="2"
              markerEnd="url(#arrowhead)"
            />
          )
        })}
        
        {/* Render nodes */}
        {dag.nodes.map((node) => (
          <g key={node.id}>
            <circle
              cx={node.position.x}
              cy={node.position.y}
              r="20"
              fill="#4f46e5"
              stroke="#1e1b4b"
              strokeWidth="2"
              className="cursor-pointer hover:fill-indigo-400"
              onClick={() => handleNodeClick(node.id)}
            />
            <text
              x={node.position.x}
              y={node.position.y + 5}
              textAnchor="middle"
              fill="white"
              fontSize="12"
              className="pointer-events-none select-none"
            >
              {node.label}
            </text>
          </g>
        ))}
        
        {/* Arrow marker definition */}
        <defs>
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
        </defs>
      </svg>
    </div>
  )
}