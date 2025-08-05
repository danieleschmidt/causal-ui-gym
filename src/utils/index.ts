import { CausalDAG, CausalNode, CausalEdge, ValidationResult } from '../types'

export const validateDAG = (dag: CausalDAG): ValidationResult => {
  const errors: string[] = []
  const warnings: string[] = []
  const assumptions: Record<string, boolean> = {}

  if (!dag.nodes || dag.nodes.length === 0) {
    errors.push('DAG must contain at least one node')
  }

  const nodeIds = new Set(dag.nodes.map(n => n.id))
  
  for (const edge of dag.edges) {
    if (!nodeIds.has(edge.source)) {
      errors.push(`Edge source '${edge.source}' not found in nodes`)
    }
    if (!nodeIds.has(edge.target)) {
      errors.push(`Edge target '${edge.target}' not found in nodes`)
    }
  }

  assumptions.is_acyclic = !hasCycles(dag)
  assumptions.has_valid_nodes = nodeIds.size === dag.nodes.length
  assumptions.has_valid_edges = dag.edges.every(e => 
    nodeIds.has(e.source) && nodeIds.has(e.target)
  )

  if (!assumptions.is_acyclic) {
    errors.push('DAG contains cycles')
  }

  return {
    is_valid: errors.length === 0,
    errors,
    warnings,
    assumptions
  }
}

export const hasCycles = (dag: CausalDAG): boolean => {
  const visited = new Set<string>()
  const recursionStack = new Set<string>()
  
  const adjacencyList = new Map<string, string[]>()
  dag.nodes.forEach(node => adjacencyList.set(node.id, []))
  dag.edges.forEach(edge => {
    adjacencyList.get(edge.source)?.push(edge.target)
  })

  const dfs = (nodeId: string): boolean => {
    visited.add(nodeId)
    recursionStack.add(nodeId)

    const neighbors = adjacencyList.get(nodeId) || []
    for (const neighbor of neighbors) {
      if (!visited.has(neighbor) && dfs(neighbor)) {
        return true
      } else if (recursionStack.has(neighbor)) {
        return true
      }
    }

    recursionStack.delete(nodeId)
    return false
  }

  for (const node of dag.nodes) {
    if (!visited.has(node.id) && dfs(node.id)) {
      return true
    }
  }

  return false
}

export const generateNodeLayout = (nodes: CausalNode[], edges: CausalEdge[]): CausalNode[] => {
  const updatedNodes = [...nodes]
  const nodeCount = nodes.length
  
  if (nodeCount === 0) return updatedNodes

  const radius = Math.max(100, nodeCount * 20)
  const angleStep = (2 * Math.PI) / nodeCount

  updatedNodes.forEach((node, index) => {
    const angle = index * angleStep
    node.position = {
      x: 200 + radius * Math.cos(angle),
      y: 150 + radius * Math.sin(angle)
    }
  })

  return updatedNodes
}

export const calculateMetrics = async (
  dag: CausalDAG,
  interventionVariable: string,
  outcomeVariable: string,
  interventionValue: number,
  baselineValue: number = 0
) => {
  const apiResponse = await fetch('/api/interventions/compute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dag,
      intervention: {
        variable: interventionVariable,
        value: interventionValue
      },
      outcome_variable: outcomeVariable,
      baseline_value: baselineValue
    })
  })

  if (!apiResponse.ok) {
    throw new Error(`Failed to compute metrics: ${apiResponse.statusText}`)
  }

  return await apiResponse.json()
}

export const formatMetricValue = (
  value: number, 
  type: 'percentage' | 'decimal' | 'integer' = 'decimal',
  precision: number = 3
): string => {
  switch (type) {
    case 'percentage':
      return `${(value * 100).toFixed(precision)}%`
    case 'integer':
      return Math.round(value).toString()
    case 'decimal':
    default:
      return value.toFixed(precision)
  }
}

export const downloadResults = (data: any, filename: string, format: 'json' | 'csv' = 'json') => {
  let content: string
  let mimeType: string

  if (format === 'csv') {
    if (Array.isArray(data)) {
      const headers = Object.keys(data[0]).join(',')
      const rows = data.map(row => Object.values(row).join(','))
      content = [headers, ...rows].join('\n')
    } else {
      content = Object.entries(data).map(([key, value]) => `${key},${value}`).join('\n')
    }
    mimeType = 'text/csv'
  } else {
    content = JSON.stringify(data, null, 2)
    mimeType = 'application/json'
  }

  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `${filename}.${format}`
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: NodeJS.Timeout
  return (...args: Parameters<T>) => {
    clearTimeout(timeout)
    timeout = setTimeout(() => func(...args), wait)
  }
}