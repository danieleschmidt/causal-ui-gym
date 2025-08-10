import { describe, it, expect } from 'vitest'
import { validateDAG, validateIntervention, sanitizeInput } from './validation'
import { CausalDAG, Intervention } from '../types'

describe('Validation Utils', () => {
  describe('validateDAG', () => {
    it('should validate a simple DAG', () => {
      const dag: CausalDAG = {
        name: 'Test DAG',
        description: 'A test DAG',
        nodes: [
          { id: 'A', label: 'Node A', position: { x: 0, y: 0 }, variable_type: 'continuous' },
          { id: 'B', label: 'Node B', position: { x: 100, y: 0 }, variable_type: 'continuous' }
        ],
        edges: [
          { source: 'A', target: 'B', weight: 1.0, edge_type: 'causal' }
        ]
      }

      const result = validateDAG(dag)
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
      expect(result.score).toBeGreaterThan(0)
    })

    it('should reject DAG with no nodes', () => {
      const dag: CausalDAG = {
        name: 'Empty DAG',
        description: 'An empty DAG',
        nodes: [],
        edges: []
      }

      const result = validateDAG(dag)
      expect(result.isValid).toBe(false)
      expect(result.errors.length).toBeGreaterThan(0)
      expect(result.errors[0].rule).toBe('has_nodes')
    })

    it('should detect duplicate node IDs', () => {
      const dag: CausalDAG = {
        name: 'Duplicate DAG',
        description: 'DAG with duplicate node IDs',
        nodes: [
          { id: 'A', label: 'Node A1', position: { x: 0, y: 0 }, variable_type: 'continuous' },
          { id: 'A', label: 'Node A2', position: { x: 100, y: 0 }, variable_type: 'continuous' }
        ],
        edges: []
      }

      const result = validateDAG(dag)
      expect(result.isValid).toBe(false)
      expect(result.errors.some(e => e.rule === 'unique_node_ids')).toBe(true)
    })

    it('should reject edges referencing non-existent nodes', () => {
      const dag: CausalDAG = {
        name: 'Invalid Edge DAG',
        description: 'DAG with invalid edge',
        nodes: [
          { id: 'A', label: 'Node A', position: { x: 0, y: 0 }, variable_type: 'continuous' }
        ],
        edges: [
          { source: 'A', target: 'NonExistent', weight: 1.0, edge_type: 'causal' }
        ]
      }

      const result = validateDAG(dag)
      expect(result.isValid).toBe(false)
      expect(result.errors.some(e => e.rule === 'edges_reference_existing_nodes')).toBe(true)
    })

    it('should reject self-loops', () => {
      const dag: CausalDAG = {
        name: 'Self Loop DAG',
        description: 'DAG with self-loop',
        nodes: [
          { id: 'A', label: 'Node A', position: { x: 0, y: 0 }, variable_type: 'continuous' }
        ],
        edges: [
          { source: 'A', target: 'A', weight: 1.0, edge_type: 'causal' }
        ]
      }

      const result = validateDAG(dag)
      expect(result.isValid).toBe(false)
      expect(result.errors.some(e => e.rule === 'no_self_loops')).toBe(true)
    })
  })

  describe('validateIntervention', () => {
    it('should validate a simple intervention', () => {
      const intervention: Intervention = {
        variable: 'A',
        value: 1.5,
        intervention_type: 'do',
        description: 'Test intervention',
        timestamp: Date.now()
      }

      const result = validateIntervention(intervention)
      expect(result.isValid).toBe(true)
      expect(result.errors).toHaveLength(0)
    })

    it('should require variable name', () => {
      const intervention: Intervention = {
        variable: '',
        value: 1.5,
        intervention_type: 'do',
        description: 'Test intervention',
        timestamp: Date.now()
      }

      const result = validateIntervention(intervention)
      expect(result.isValid).toBe(false)
      expect(result.errors.some(e => e.rule === 'has_variable')).toBe(true)
    })

    it('should require intervention value', () => {
      const intervention: Intervention = {
        variable: 'A',
        value: null as any,
        intervention_type: 'do',
        description: 'Test intervention',
        timestamp: Date.now()
      }

      const result = validateIntervention(intervention)
      expect(result.isValid).toBe(false)
      expect(result.errors.some(e => e.rule === 'has_value')).toBe(true)
    })

    it('should validate intervention variable exists in DAG', () => {
      const dag: CausalDAG = {
        name: 'Test DAG',
        description: 'Test DAG',
        nodes: [
          { id: 'A', label: 'Node A', position: { x: 0, y: 0 }, variable_type: 'continuous' }
        ],
        edges: []
      }

      const intervention: Intervention = {
        variable: 'B', // Non-existent variable
        value: 1.5,
        intervention_type: 'do',
        description: 'Test intervention',
        timestamp: Date.now()
      }

      const result = validateIntervention(intervention, dag)
      expect(result.isValid).toBe(false)
      expect(result.errors.some(e => e.rule === 'variable_exists_in_dag')).toBe(true)
    })
  })

  describe('sanitizeInput', () => {
    it('should allow clean input', () => {
      const result = sanitizeInput('Hello world')
      expect(result.isSecure).toBe(true)
      expect(result.sanitized).toBe('Hello world')
      expect(result.threats).toHaveLength(0)
    })

    it('should detect XSS patterns', () => {
      const result = sanitizeInput('<script>alert("xss")</script>')
      expect(result.isSecure).toBe(false)
      expect(result.threats.some(t => t.type === 'xss')).toBe(true)
      expect(result.sanitized).not.toContain('<script>')
      expect(result.riskLevel).toBe('high')
    })

    it('should detect javascript: URLs', () => {
      const result = sanitizeInput('javascript:alert("xss")')
      expect(result.isSecure).toBe(false)
      expect(result.threats.some(t => t.type === 'xss')).toBe(true)
      expect(result.sanitized).not.toContain('javascript:')
    })

    it('should handle excessive length', () => {
      const longString = 'a'.repeat(20000)
      const result = sanitizeInput(longString, { maxLength: 100 })
      expect(result.isSecure).toBe(false)
      expect(result.threats.some(t => t.type === 'excessive_length')).toBe(true)
      expect(result.sanitized.length).toBe(100)
    })

    it('should detect path traversal', () => {
      const result = sanitizeInput('../../../etc/passwd')
      expect(result.isSecure).toBe(false)
      expect(result.threats.some(t => t.type === 'path_traversal')).toBe(true)
      expect(result.sanitized).toBe('') // Should strip everything after path traversal pattern
    })

    it('should respect allowed characters', () => {
      const result = sanitizeInput('Hello123!@#', { allowedChars: /[a-zA-Z0-9]/ })
      expect(result.sanitized).toBe('Hello123')
      expect(result.threats.some(t => t.type === 'malformed_data')).toBe(true)
    })

    it('should strip HTML tags', () => {
      const result = sanitizeInput('<p>Hello <b>world</b></p>', { stripTags: true })
      expect(result.sanitized).toBe('Hello world')
    })

    it('should escape HTML', () => {
      const result = sanitizeInput('<p>Hello & goodbye</p>', { escapeHtml: true })
      expect(result.sanitized).toBe('&lt;p&gt;Hello &amp; goodbye&lt;&#x2F;p&gt;')
    })
  })
})