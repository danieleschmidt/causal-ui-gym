/**
 * Test data fixtures for Causal UI Gym tests
 */

export interface CausalNode {
  id: string
  label: string
  type: 'continuous' | 'discrete' | 'binary'
  description?: string
}

export interface CausalEdge {
  from: string
  to: string
  relationship: 'positive' | 'negative' | 'unknown'
  strength?: number
}

export interface CausalModel {
  nodes: CausalNode[]
  edges: CausalEdge[]
  metadata?: {
    name: string
    description: string
    domain: string
  }
}

export interface Intervention {
  id: string
  variable: string
  value: number | string
  timestamp: string
  experimentId: string
}

export interface LLMResponse {
  id: string
  question: string
  prediction: string
  confidence: number
  reasoning: string
  timestamp: string
  model: string
}

export interface ExperimentMetrics {
  experimentId: string
  ateError: number
  causalAccuracy: number
  responseTime: number
  interventionSuccess: boolean
  timestamp: string
}

// Basic causal models for testing
export const SIMPLE_CAUSAL_MODEL: CausalModel = {
  nodes: [
    { id: 'price', label: 'Price', type: 'continuous' },
    { id: 'demand', label: 'Demand', type: 'continuous' },
    { id: 'revenue', label: 'Revenue', type: 'continuous' },
  ],
  edges: [
    { from: 'price', to: 'demand', relationship: 'negative', strength: 0.8 },
    { from: 'price', to: 'revenue', relationship: 'positive', strength: 0.6 },
    { from: 'demand', to: 'revenue', relationship: 'positive', strength: 0.9 },
  ],
  metadata: {
    name: 'Simple Pricing Model',
    description: 'Basic price-demand-revenue relationship',
    domain: 'economics',
  },
}

export const CONFOUNDED_MODEL: CausalModel = {
  nodes: [
    { id: 'treatment', label: 'Treatment', type: 'binary' },
    { id: 'outcome', label: 'Outcome', type: 'continuous' },
    { id: 'confounder', label: 'Confounder', type: 'continuous' },
    { id: 'mediator', label: 'Mediator', type: 'continuous' },
  ],
  edges: [
    { from: 'confounder', to: 'treatment', relationship: 'positive', strength: 0.7 },
    { from: 'confounder', to: 'outcome', relationship: 'positive', strength: 0.5 },
    { from: 'treatment', to: 'mediator', relationship: 'positive', strength: 0.8 },
    { from: 'mediator', to: 'outcome', relationship: 'positive', strength: 0.9 },
    { from: 'treatment', to: 'outcome', relationship: 'positive', strength: 0.3 },
  ],
  metadata: {
    name: 'Confounded Treatment Model',
    description: 'Model with confounding and mediation',
    domain: 'medical',
  },
}

export const LARGE_CAUSAL_MODEL: CausalModel = {
  nodes: Array.from({ length: 20 }, (_, i) => ({
    id: `var_${i}`,
    label: `Variable ${i}`,
    type: 'continuous' as const,
  })),
  edges: Array.from({ length: 30 }, (_, i) => ({
    from: `var_${i % 19}`,
    to: `var_${(i % 19) + 1}`,
    relationship: Math.random() > 0.5 ? 'positive' : 'negative' as const,
    strength: Math.random(),
  })),
  metadata: {
    name: 'Large Network Model',
    description: 'Complex network with 20 variables and 30 edges',
    domain: 'complex_systems',
  },
}

// Sample interventions
export const SAMPLE_INTERVENTIONS: Intervention[] = [
  {
    id: 'int_001',
    variable: 'price',
    value: 50,
    timestamp: '2025-01-01T10:00:00Z',
    experimentId: 'exp_001',
  },
  {
    id: 'int_002',
    variable: 'price',
    value: 75,
    timestamp: '2025-01-01T10:05:00Z',
    experimentId: 'exp_001',
  },
  {
    id: 'int_003',
    variable: 'demand',
    value: 100,
    timestamp: '2025-01-01T10:10:00Z',
    experimentId: 'exp_001',
  },
]

// Sample LLM responses
export const SAMPLE_LLM_RESPONSES: LLMResponse[] = [
  {
    id: 'llm_001',
    question: 'What happens to revenue if price increases by 20%?',
    prediction: 'Revenue will increase by approximately 15% due to higher price, despite some reduction in demand.',
    confidence: 0.85,
    reasoning: 'Higher prices directly increase revenue per unit, and the price elasticity of demand appears moderate.',
    timestamp: '2025-01-01T10:01:00Z',
    model: 'gpt-4',
  },
  {
    id: 'llm_002',
    question: 'How does demand respond to price changes?',
    prediction: 'Demand decreases as price increases, following a negative relationship with elasticity around -0.8.',
    confidence: 0.92,
    reasoning: 'This follows basic economic principles of demand curves and price sensitivity.',
    timestamp: '2025-01-01T10:06:00Z',
    model: 'claude-3',
  },
  {
    id: 'llm_003',
    question: 'What is the optimal price for maximum revenue?',
    prediction: 'The optimal price appears to be around $65-70 based on the demand curve characteristics.',
    confidence: 0.73,
    reasoning: 'This maximizes the revenue function given the price-demand relationship.',
    timestamp: '2025-01-01T10:11:00Z',
    model: 'llama-3',
  },
]

// Sample metrics data
export const SAMPLE_METRICS: ExperimentMetrics[] = [
  {
    experimentId: 'exp_001',
    ateError: 0.12,
    causalAccuracy: 0.89,
    responseTime: 234,
    interventionSuccess: true,
    timestamp: '2025-01-01T10:01:00Z',
  },
  {
    experimentId: 'exp_001',
    ateError: 0.08,
    causalAccuracy: 0.92,
    responseTime: 198,
    interventionSuccess: true,
    timestamp: '2025-01-01T10:06:00Z',
  },
  {
    experimentId: 'exp_001',
    ateError: 0.15,
    causalAccuracy: 0.87,
    responseTime: 267,
    interventionSuccess: true,
    timestamp: '2025-01-01T10:11:00Z',
  },
]

// Test data generators
export const generateRandomCausalModel = (
  nodeCount: number,
  edgeCount: number
): CausalModel => {
  const nodes: CausalNode[] = Array.from({ length: nodeCount }, (_, i) => ({
    id: `node_${i}`,
    label: `Node ${i}`,
    type: ['continuous', 'discrete', 'binary'][Math.floor(Math.random() * 3)] as any,
  }))

  const edges: CausalEdge[] = []
  for (let i = 0; i < edgeCount; i++) {
    const from = nodes[Math.floor(Math.random() * nodeCount)]
    const to = nodes[Math.floor(Math.random() * nodeCount)]
    
    // Avoid self-loops
    if (from.id !== to.id) {
      edges.push({
        from: from.id,
        to: to.id,
        relationship: Math.random() > 0.5 ? 'positive' : 'negative',
        strength: Math.random(),
      })
    }
  }

  return {
    nodes,
    edges,
    metadata: {
      name: `Random Model ${nodeCount}x${edgeCount}`,
      description: `Randomly generated model with ${nodeCount} nodes and ${edgeCount} edges`,
      domain: 'synthetic',
    },
  }
}

export const generateInterventionSequence = (
  variables: string[],
  count: number
): Intervention[] => {
  return Array.from({ length: count }, (_, i) => ({
    id: `int_${i.toString().padStart(3, '0')}`,
    variable: variables[Math.floor(Math.random() * variables.length)],
    value: Math.random() * 100,
    timestamp: new Date(Date.now() + i * 60000).toISOString(),
    experimentId: 'test_exp',
  }))
}

export const generateLLMResponseSequence = (
  questions: string[],
  models: string[] = ['gpt-4', 'claude-3', 'llama-3']
): LLMResponse[] => {
  return questions.map((question, i) => ({
    id: `llm_${i.toString().padStart(3, '0')}`,
    question,
    prediction: `Prediction for question ${i + 1}`,
    confidence: 0.5 + Math.random() * 0.5, // 0.5 to 1.0
    reasoning: `Reasoning based on causal analysis for question ${i + 1}`,
    timestamp: new Date(Date.now() + i * 30000).toISOString(),
    model: models[i % models.length],
  }))
}

export const generateMetricsTimeSeries = (
  experimentId: string,
  points: number
): ExperimentMetrics[] => {
  return Array.from({ length: points }, (_, i) => ({
    experimentId,
    ateError: Math.random() * 0.3, // 0 to 0.3
    causalAccuracy: 0.7 + Math.random() * 0.3, // 0.7 to 1.0
    responseTime: 100 + Math.random() * 300, // 100 to 400ms
    interventionSuccess: Math.random() > 0.1, // 90% success rate
    timestamp: new Date(Date.now() + i * 120000).toISOString(), // 2-minute intervals
  }))
}

// API mock responses
export const API_MOCK_RESPONSES = {
  health: {
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: '1.0.0',
  },
  
  createExperiment: (model: CausalModel) => ({
    id: `exp_${Math.random().toString(36).substr(2, 9)}`,
    name: model.metadata?.name || 'Untitled Experiment',
    model,
    status: 'active',
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  }),
  
  performIntervention: (intervention: Omit<Intervention, 'id' | 'timestamp'>) => ({
    ...intervention,
    id: `int_${Math.random().toString(36).substr(2, 9)}`,
    timestamp: new Date().toISOString(),
    result: {
      [intervention.variable]: intervention.value,
      // Mock downstream effects
      affected_variables: ['revenue', 'demand'].map(v => ({
        variable: v,
        value: Math.random() * 100,
        change: (Math.random() - 0.5) * 20,
      })),
    },
  }),
  
  queryLLM: (question: string, model: string = 'gpt-4') => ({
    id: `llm_${Math.random().toString(36).substr(2, 9)}`,
    question,
    prediction: `Mock prediction for: ${question}`,
    confidence: 0.5 + Math.random() * 0.5,
    reasoning: 'Mock reasoning based on causal relationships',
    timestamp: new Date().toISOString(),
    model,
  }),
}

// Error scenarios for testing
export const ERROR_SCENARIOS = {
  invalidCausalModel: {
    nodes: [
      { id: '', label: '', type: 'invalid' as any }, // Invalid empty values
    ],
    edges: [
      { from: 'A', to: 'B' }, // Missing relationship
    ],
  },
  
  cyclicModel: {
    nodes: [
      { id: 'A', label: 'Node A', type: 'continuous' as const },
      { id: 'B', label: 'Node B', type: 'continuous' as const },
      { id: 'C', label: 'Node C', type: 'continuous' as const },
    ],
    edges: [
      { from: 'A', to: 'B', relationship: 'positive' as const },
      { from: 'B', to: 'C', relationship: 'positive' as const },
      { from: 'C', to: 'A', relationship: 'positive' as const }, // Creates cycle
    ],
  },
  
  invalidIntervention: {
    id: 'invalid',
    variable: 'nonexistent_variable',
    value: 'not_a_number' as any,
    timestamp: 'invalid_timestamp',
    experimentId: '',
  },
}

// Performance test data
export const PERFORMANCE_TEST_DATA = {
  smallModel: generateRandomCausalModel(5, 8),
  mediumModel: generateRandomCausalModel(20, 35),
  largeModel: generateRandomCausalModel(100, 200),
  massiveModel: generateRandomCausalModel(500, 1000),
  
  interventionBatches: {
    small: generateInterventionSequence(['A', 'B', 'C'], 10),
    medium: generateInterventionSequence(['A', 'B', 'C', 'D', 'E'], 100),
    large: generateInterventionSequence(
      Array.from({ length: 20 }, (_, i) => `var_${i}`),
      1000
    ),
  },
}