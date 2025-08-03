// Type definitions for Causal UI Gym

export interface NodePosition {
  x: number
  y: number
}

export interface CausalNode {
  id: string
  label: string
  position: NodePosition
  variable_type?: 'continuous' | 'discrete' | 'binary'
  description?: string
}

export interface CausalEdge {
  source: string
  target: string
  weight?: number
  edge_type?: 'causal' | 'correlational'
  confidence?: number
}

export interface CausalDAG {
  id?: string
  name: string
  description?: string
  nodes: CausalNode[]
  edges: CausalEdge[]
  metadata?: Record<string, any>
  created_at?: Date
}

export interface Intervention {
  id?: string
  variable: string
  value: number | string
  intervention_type?: 'do' | 'soft' | 'conditional'
  timestamp?: Date
  description?: string
}

export interface CausalMetric {
  metric_type: 'ate' | 'ite' | 'cate' | 'backdoor' | 'frontdoor'
  value: number
  confidence_interval?: [number, number]
  standard_error?: number
  p_value?: number
  sample_size: number
  computation_time: number
  metadata?: Record<string, any>
}

export interface CausalResult {
  id: string
  dag_id: string
  intervention: Intervention
  outcome_variable: string
  metrics: CausalMetric[]
  outcome_distribution?: number[]
  created_at: Date
}

export interface ExperimentConfig {
  id?: string
  name: string
  description?: string
  dag: CausalDAG
  interventions: Intervention[]
  outcome_variables: string[]
  sample_size?: number
  random_seed?: number
  status?: 'created' | 'running' | 'completed' | 'failed' | 'cancelled'
  created_at?: Date
  updated_at?: Date
}

export interface LLMAgent {
  id: string
  name: string
  provider: 'openai' | 'anthropic' | 'other'
  model: string
  status: 'available' | 'busy' | 'error'
}

export interface BeliefState {
  agent_id: string
  variable_pair: [string, string]
  belief_strength: number
  confidence: number
  reasoning?: string
  timestamp: Date
}

export interface ValidationResult {
  is_valid: boolean
  errors: string[]
  warnings: string[]
  assumptions: Record<string, boolean>
}