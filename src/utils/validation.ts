import { CausalDAG, CausalNode, CausalEdge, Intervention, ExperimentConfig } from '../types'

export interface ValidationRule<T> {
  name: string
  description: string
  validate: (value: T) => boolean
  errorMessage: string
  severity: 'error' | 'warning'
}

export interface ValidationResult {
  isValid: boolean
  errors: ValidationError[]
  warnings: ValidationError[]
  score: number // 0-100 quality score
}

export interface ValidationError {
  field: string
  message: string
  rule: string
  severity: 'error' | 'warning'
  suggestions?: string[]
}

// DAG Validation Rules
const dagValidationRules: ValidationRule<CausalDAG>[] = [
  {
    name: 'has_nodes',
    description: 'DAG must contain at least one node',
    validate: (dag) => dag.nodes && dag.nodes.length > 0,
    errorMessage: 'DAG must contain at least one node',
    severity: 'error'
  },
  {
    name: 'min_nodes_for_causality',
    description: 'DAG should have at least 2 nodes for meaningful causal analysis',
    validate: (dag) => dag.nodes && dag.nodes.length >= 2,
    errorMessage: 'DAG should have at least 2 nodes for causal analysis',
    severity: 'warning'
  },
  {
    name: 'unique_node_ids',
    description: 'All node IDs must be unique',
    validate: (dag) => {
      const ids = dag.nodes.map(n => n.id)
      return new Set(ids).size === ids.length
    },
    errorMessage: 'Node IDs must be unique',
    severity: 'error'
  },
  {
    name: 'valid_node_names',
    description: 'Node names should follow naming conventions',
    validate: (dag) => dag.nodes.every(node => 
      /^[a-zA-Z][a-zA-Z0-9_]*$/.test(node.id) && node.id.length <= 50
    ),
    errorMessage: 'Node names should start with a letter and contain only letters, numbers, and underscores (max 50 chars)',
    severity: 'warning'
  },
  {
    name: 'edges_reference_existing_nodes',
    description: 'All edges must reference existing nodes',
    validate: (dag) => {
      const nodeIds = new Set(dag.nodes.map(n => n.id))
      return dag.edges.every(edge => 
        nodeIds.has(edge.source) && nodeIds.has(edge.target)
      )
    },
    errorMessage: 'All edges must reference existing nodes',
    severity: 'error'
  },
  {
    name: 'no_self_loops',
    description: 'Nodes should not have edges to themselves',
    validate: (dag) => dag.edges.every(edge => edge.source !== edge.target),
    errorMessage: 'Self-loops are not allowed in causal DAGs',
    severity: 'error'
  },
  {
    name: 'reasonable_complexity',
    description: 'DAG complexity should be reasonable for analysis',
    validate: (dag) => dag.nodes.length <= 50 && dag.edges.length <= 200,
    errorMessage: 'DAG is too complex (max 50 nodes, 200 edges)',
    severity: 'warning'
  },
  {
    name: 'connected_components',
    description: 'DAG should be weakly connected for meaningful analysis',
    validate: (dag) => {
      // Simple connectivity check - at least some edges exist
      return dag.edges.length > 0 || dag.nodes.length === 1
    },
    errorMessage: 'DAG should have at least one edge connecting nodes',
    severity: 'warning'
  }
]

// Node Validation Rules
const nodeValidationRules: ValidationRule<CausalNode>[] = [
  {
    name: 'valid_position',
    description: 'Node position should be within reasonable bounds',
    validate: (node) => 
      node.position && 
      node.position.x >= -1000 && node.position.x <= 2000 &&
      node.position.y >= -1000 && node.position.y <= 2000,
    errorMessage: 'Node position should be within reasonable bounds (-1000 to 2000)',
    severity: 'warning'
  },
  {
    name: 'has_label',
    description: 'Node should have a descriptive label',
    validate: (node) => node.label && node.label.trim().length > 0,
    errorMessage: 'Node should have a non-empty label',
    severity: 'warning'
  },
  {
    name: 'valid_variable_type',
    description: 'Node variable type should be specified',
    validate: (node) => 
      !node.variable_type || 
      ['continuous', 'discrete', 'binary'].includes(node.variable_type),
    errorMessage: 'Variable type must be continuous, discrete, or binary',
    severity: 'error'
  }
]

// Edge Validation Rules  
const edgeValidationRules: ValidationRule<CausalEdge>[] = [
  {
    name: 'valid_weight',
    description: 'Edge weight should be reasonable',
    validate: (edge) => 
      !edge.weight || (edge.weight >= -10 && edge.weight <= 10 && edge.weight !== 0),
    errorMessage: 'Edge weight should be between -10 and 10 and not zero',
    severity: 'warning'
  },
  {
    name: 'valid_edge_type',
    description: 'Edge type should be specified correctly',
    validate: (edge) =>
      !edge.edge_type || ['causal', 'correlational'].includes(edge.edge_type),
    errorMessage: 'Edge type must be either causal or correlational',
    severity: 'warning'
  },
  {
    name: 'reasonable_confidence',
    description: 'Edge confidence should be between 0 and 1',
    validate: (edge) =>
      !edge.confidence || (edge.confidence >= 0 && edge.confidence <= 1),
    errorMessage: 'Edge confidence should be between 0 and 1',
    severity: 'warning'
  }
]

// Intervention Validation Rules
const interventionValidationRules: ValidationRule<Intervention>[] = [
  {
    name: 'has_variable',
    description: 'Intervention must specify a variable',
    validate: (intervention) => intervention.variable && intervention.variable.trim().length > 0,
    errorMessage: 'Intervention must specify a target variable',
    severity: 'error'
  },
  {
    name: 'has_value',
    description: 'Intervention must specify a value',
    validate: (intervention) => intervention.value !== null && intervention.value !== undefined,
    errorMessage: 'Intervention must specify a value',
    severity: 'error'
  },
  {
    name: 'reasonable_value',
    description: 'Intervention value should be reasonable',
    validate: (intervention) => {
      if (typeof intervention.value === 'number') {
        return !isNaN(intervention.value) && isFinite(intervention.value) &&
               intervention.value >= -1000000 && intervention.value <= 1000000
      }
      return true
    },
    errorMessage: 'Intervention value should be a finite number within reasonable bounds',
    severity: 'warning'
  },
  {
    name: 'valid_type',
    description: 'Intervention type should be valid',
    validate: (intervention) =>
      !intervention.intervention_type || 
      ['do', 'soft', 'conditional'].includes(intervention.intervention_type),
    errorMessage: 'Intervention type must be do, soft, or conditional',
    severity: 'warning'
  }
]

// Experiment Config Validation Rules
const experimentValidationRules: ValidationRule<ExperimentConfig>[] = [
  {
    name: 'has_name',
    description: 'Experiment must have a name',
    validate: (config) => config.name && config.name.trim().length > 0,
    errorMessage: 'Experiment must have a non-empty name',
    severity: 'error'
  },
  {
    name: 'reasonable_name_length',
    description: 'Experiment name should be reasonable length',
    validate: (config) => config.name && config.name.length <= 200,
    errorMessage: 'Experiment name should not exceed 200 characters',
    severity: 'warning'
  },
  {
    name: 'has_interventions',
    description: 'Experiment should have at least one intervention',
    validate: (config) => config.interventions && config.interventions.length > 0,
    errorMessage: 'Experiment should have at least one intervention',
    severity: 'warning'
  },
  {
    name: 'has_outcome_variables',
    description: 'Experiment should specify outcome variables',
    validate: (config) => config.outcome_variables && config.outcome_variables.length > 0,
    errorMessage: 'Experiment should specify at least one outcome variable',
    severity: 'warning'
  },
  {
    name: 'reasonable_sample_size',
    description: 'Sample size should be reasonable',
    validate: (config) => 
      !config.sample_size || 
      (config.sample_size >= 100 && config.sample_size <= 1000000),
    errorMessage: 'Sample size should be between 100 and 1,000,000',
    severity: 'warning'
  },
  {
    name: 'valid_outcome_variables',
    description: 'Outcome variables should exist in DAG',
    validate: (config) => {
      if (!config.outcome_variables || !config.dag) return true
      const nodeIds = new Set(config.dag.nodes.map(n => n.id))
      return config.outcome_variables.every(variable => nodeIds.has(variable))
    },
    errorMessage: 'All outcome variables must exist in the DAG',
    severity: 'error'
  },
  {
    name: 'valid_intervention_variables',
    description: 'Intervention variables should exist in DAG',
    validate: (config) => {
      if (!config.interventions || !config.dag) return true
      const nodeIds = new Set(config.dag.nodes.map(n => n.id))
      return config.interventions.every(intervention => nodeIds.has(intervention.variable))
    },
    errorMessage: 'All intervention variables must exist in the DAG',
    severity: 'error'
  }
]

// Main validation functions
export function validateDAG(dag: CausalDAG): ValidationResult {
  const errors: ValidationError[] = []
  const warnings: ValidationError[] = []

  // Check DAG-level rules
  for (const rule of dagValidationRules) {
    if (!rule.validate(dag)) {
      const error: ValidationError = {
        field: 'dag',
        message: rule.errorMessage,
        rule: rule.name,
        severity: rule.severity,
        suggestions: generateSuggestions(rule.name, dag)
      }
      
      if (rule.severity === 'error') {
        errors.push(error)
      } else {
        warnings.push(error)
      }
    }
  }

  // Check node-level rules
  dag.nodes.forEach((node, index) => {
    for (const rule of nodeValidationRules) {
      if (!rule.validate(node)) {
        const error: ValidationError = {
          field: `nodes[${index}]`,
          message: `Node '${node.id}': ${rule.errorMessage}`,
          rule: rule.name,
          severity: rule.severity,
          suggestions: generateNodeSuggestions(rule.name, node)
        }
        
        if (rule.severity === 'error') {
          errors.push(error)
        } else {
          warnings.push(error)
        }
      }
    }
  })

  // Check edge-level rules
  dag.edges.forEach((edge, index) => {
    for (const rule of edgeValidationRules) {
      if (!rule.validate(edge)) {
        const error: ValidationError = {
          field: `edges[${index}]`,
          message: `Edge '${edge.source}→${edge.target}': ${rule.errorMessage}`,
          rule: rule.name,
          severity: rule.severity,
          suggestions: generateEdgeSuggestions(rule.name, edge)
        }
        
        if (rule.severity === 'error') {
          errors.push(error)
        } else {
          warnings.push(error)
        }
      }
    }
  })

  // Calculate quality score
  const totalRules = dagValidationRules.length + 
                    (dag.nodes.length * nodeValidationRules.length) +
                    (dag.edges.length * edgeValidationRules.length)
  const failedRules = errors.length + warnings.length
  const score = Math.max(0, Math.round(100 * (1 - failedRules / totalRules)))

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
    score
  }
}

export function validateIntervention(intervention: Intervention, dag?: CausalDAG): ValidationResult {
  const errors: ValidationError[] = []
  const warnings: ValidationError[] = []

  // Check intervention rules
  for (const rule of interventionValidationRules) {
    if (!rule.validate(intervention)) {
      const error: ValidationError = {
        field: 'intervention',
        message: rule.errorMessage,
        rule: rule.name,
        severity: rule.severity,
        suggestions: generateInterventionSuggestions(rule.name, intervention)
      }
      
      if (rule.severity === 'error') {
        errors.push(error)
      } else {
        warnings.push(error)
      }
    }
  }

  // Check if intervention variable exists in DAG
  if (dag && !dag.nodes.some(node => node.id === intervention.variable)) {
    errors.push({
      field: 'intervention.variable',
      message: `Variable '${intervention.variable}' does not exist in the DAG`,
      rule: 'variable_exists_in_dag',
      severity: 'error',
      suggestions: [
        `Add a node with ID '${intervention.variable}' to the DAG`,
        'Choose a different variable that exists in the DAG',
        'Check for typos in the variable name'
      ]
    })
  }

  const score = Math.max(0, Math.round(100 * (1 - (errors.length + warnings.length) / interventionValidationRules.length)))

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
    score
  }
}

export function validateExperimentConfig(config: ExperimentConfig): ValidationResult {
  const errors: ValidationError[] = []
  const warnings: ValidationError[] = []

  // Check experiment-level rules
  for (const rule of experimentValidationRules) {
    if (!rule.validate(config)) {
      const error: ValidationError = {
        field: 'config',
        message: rule.errorMessage,
        rule: rule.name,
        severity: rule.severity,
        suggestions: generateExperimentSuggestions(rule.name, config)
      }
      
      if (rule.severity === 'error') {
        errors.push(error)
      } else {
        warnings.push(error)
      }
    }
  }

  // Validate the DAG
  if (config.dag) {
    const dagValidation = validateDAG(config.dag)
    errors.push(...dagValidation.errors)
    warnings.push(...dagValidation.warnings)
  }

  // Validate interventions
  config.interventions.forEach((intervention, index) => {
    const interventionValidation = validateIntervention(intervention, config.dag)
    interventionValidation.errors.forEach(error => {
      errors.push({
        ...error,
        field: `interventions[${index}].${error.field}`
      })
    })
    interventionValidation.warnings.forEach(warning => {
      warnings.push({
        ...warning,
        field: `interventions[${index}].${warning.field}`
      })
    })
  })

  const totalIssues = errors.length + warnings.length
  const score = Math.max(0, Math.round(100 * Math.exp(-totalIssues / 10)))

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
    score
  }
}

// Suggestion generators
function generateSuggestions(ruleName: string, dag: CausalDAG): string[] {
  switch (ruleName) {
    case 'has_nodes':
      return ['Add at least one node to the DAG', 'Create variables for your causal model']
    case 'min_nodes_for_causality':
      return ['Add another node to enable causal analysis', 'Define both treatment and outcome variables']
    case 'unique_node_ids':
      return ['Rename duplicate nodes with unique IDs', 'Use descriptive, unique variable names']
    case 'no_self_loops':
      return ['Remove edges from nodes to themselves', 'Self-causation is not allowed in DAGs']
    case 'connected_components':
      return ['Add edges to connect isolated nodes', 'Ensure all variables have causal relationships']
    default:
      return ['Review the DAG structure', 'Consult causal modeling best practices']
  }
}

function generateNodeSuggestions(ruleName: string, node: CausalNode): string[] {
  switch (ruleName) {
    case 'valid_position':
      return ['Adjust node position to be within visible canvas', 'Use auto-layout to position nodes']
    case 'has_label':
      return ['Add a descriptive label to the node', 'Use clear, meaningful variable names']
    case 'valid_variable_type':
      return ['Specify whether variable is continuous, discrete, or binary', 'Choose appropriate variable type for your data']
    default:
      return ['Review node properties', 'Ensure all node fields are properly set']
  }
}

function generateEdgeSuggestions(ruleName: string, edge: CausalEdge): string[] {
  switch (ruleName) {
    case 'valid_weight':
      return ['Set edge weight between -10 and 10', 'Use non-zero weights for meaningful relationships']
    case 'valid_edge_type':
      return ['Specify whether edge represents causal or correlational relationship']
    case 'reasonable_confidence':
      return ['Set confidence level between 0 and 1', 'Use confidence to indicate certainty of relationship']
    default:
      return ['Review edge properties', 'Ensure edge represents valid causal relationship']
  }
}

function generateInterventionSuggestions(ruleName: string, intervention: Intervention): string[] {
  switch (ruleName) {
    case 'has_variable':
      return ['Specify which variable to intervene on', 'Choose a variable from your DAG']
    case 'has_value':
      return ['Set the intervention value', 'Specify what value to set the variable to']
    case 'reasonable_value':
      return ['Use finite, reasonable intervention values', 'Consider the natural range of your variable']
    case 'valid_type':
      return ['Use intervention type: do, soft, or conditional', 'Choose appropriate intervention type for your analysis']
    default:
      return ['Review intervention parameters', 'Ensure intervention is properly specified']
  }
}

function generateExperimentSuggestions(ruleName: string, config: ExperimentConfig): string[] {
  switch (ruleName) {
    case 'has_name':
      return ['Give your experiment a descriptive name', 'Use a name that describes the research question']
    case 'has_interventions':
      return ['Add at least one intervention to test', 'Define what you want to manipulate in the experiment']
    case 'has_outcome_variables':
      return ['Specify which variables you want to measure', 'Define the outcomes of interest']
    case 'reasonable_sample_size':
      return ['Use sample size between 100 and 1,000,000', 'Consider statistical power requirements']
    default:
      return ['Review experiment configuration', 'Ensure all required fields are set']
  }
}

// Utility function for batch validation
export function validateBatch<T>(
  items: T[],
  validator: (item: T) => ValidationResult
): ValidationResult {
  const allErrors: ValidationError[] = []
  const allWarnings: ValidationError[] = []
  let totalScore = 0

  items.forEach((item, index) => {
    const result = validator(item)
    
    result.errors.forEach(error => {
      allErrors.push({
        ...error,
        field: `[${index}].${error.field}`
      })
    })
    
    result.warnings.forEach(warning => {
      allWarnings.push({
        ...warning,
        field: `[${index}].${warning.field}`
      })
    })
    
    totalScore += result.score
  })

  return {
    isValid: allErrors.length === 0,
    errors: allErrors,
    warnings: allWarnings,
    score: items.length > 0 ? Math.round(totalScore / items.length) : 100
  }
}