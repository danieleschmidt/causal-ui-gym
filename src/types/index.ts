// Type definitions for Causal UI Gym
export interface CausalNode {
  id: string
  label: string
  position: { x: number; y: number }
}

export interface CausalEdge {
  source: string
  target: string
  weight?: number
}

export interface CausalDAG {
  nodes: CausalNode[]
  edges: CausalEdge[]
}