/**
 * API Contract Testing for Causal UI Gym
 * Ensures API compatibility between frontend and JAX backend
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import { Pact } from '@pact-foundation/pact'
import path from 'path'

// Contract testing configuration
const provider = new Pact({
  consumer: 'causal-ui-frontend',
  provider: 'jax-causal-backend',
  port: 1234,
  log: path.resolve(process.cwd(), 'tests/contract/logs', 'pact.log'),
  dir: path.resolve(process.cwd(), 'tests/contract/pacts'),
  logLevel: 'INFO',
  spec: 2,
})

describe('JAX Backend API Contracts', () => {
  beforeAll(async () => {
    await provider.setup()
  })

  afterAll(async () => {
    await provider.finalize()
  })

  describe('Causal Graph Operations', () => {
    it('should create causal DAG from node data', async () => {
      // Define the expected interaction
      await provider
        .given('valid node and edge data')
        .uponReceiving('a request to create causal DAG')
        .withRequest({
          method: 'POST',
          path: '/api/causal/dag',
          headers: {
            'Content-Type': 'application/json',
          },
          body: {
            nodes: [
              { id: 'X', label: 'Treatment' },
              { id: 'Y', label: 'Outcome' },
              { id: 'Z', label: 'Confounder' }
            ],
            edges: [
              { from: 'X', to: 'Y' },
              { from: 'Z', to: 'X' },
              { from: 'Z', to: 'Y' }
            ]
          }
        })
        .willRespondWith({
          status: 200,
          headers: {
            'Content-Type': 'application/json',
          },
          body: {
            dag_id: 'dag_123',
            nodes: 3,
            edges: 3,
            is_valid: true,
            adjacency_matrix: [[0, 1, 0], [0, 0, 0], [1, 1, 0]]
          }
        })

      // Execute the actual request
      const response = await fetch('http://localhost:1234/api/causal/dag', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          nodes: [
            { id: 'X', label: 'Treatment' },
            { id: 'Y', label: 'Outcome' },
            { id: 'Z', label: 'Confounder' }
          ],
          edges: [
            { from: 'X', to: 'Y' },
            { from: 'Z', to: 'X' },
            { from: 'Z', to: 'Y' }
          ]
        })
      })

      const data = await response.json()
      expect(response.status).toBe(200)
      expect(data.is_valid).toBe(true)
      expect(data.nodes).toBe(3)
    })

    it('should compute do-calculus intervention', async () => {
      await provider
        .given('valid DAG exists')
        .uponReceiving('a request to compute intervention')
        .withRequest({
          method: 'POST',
          path: '/api/causal/intervention',
          headers: {
            'Content-Type': 'application/json',
          },
          body: {
            dag_id: 'dag_123',
            intervention: { variable: 'X', value: 1.0 },
            target: 'Y'
          }
        })
        .willRespondWith({
          status: 200,
          headers: {
            'Content-Type': 'application/json',
          },
          body: {
            intervention_id: 'int_456',
            result: {
              ate: 0.25,
              confidence_interval: [0.15, 0.35],
              p_value: 0.003,
              effect_size: 'medium'
            },
            computation_time_ms: 150
          }
        })

      const response = await fetch('http://localhost:1234/api/causal/intervention', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          dag_id: 'dag_123',
          intervention: { variable: 'X', value: 1.0 },
          target: 'Y'
        })
      })

      const data = await response.json()
      expect(response.status).toBe(200)
      expect(data.result.ate).toBeTypeOf('number')
      expect(data.computation_time_ms).toBeLessThan(1000)
    })
  })

  describe('Data Processing Contracts', () => {
    it('should validate causal data format', async () => {
      await provider
        .given('causal data validation service is available')
        .uponReceiving('a request to validate data format')
        .withRequest({
          method: 'POST',
          path: '/api/data/validate',
          headers: {
            'Content-Type': 'application/json',
          },
          body: {
            data: [
              { X: 1, Y: 2, Z: 0 },
              { X: 0, Y: 1, Z: 1 }
            ],
            schema: {
              variables: ['X', 'Y', 'Z'],
              types: ['continuous', 'continuous', 'binary']
            }
          }
        })
        .willRespondWith({
          status: 200,
          headers: {
            'Content-Type': 'application/json',
          },
          body: {
            is_valid: true,
            sample_size: 2,
            missing_values: 0,
            validation_errors: []
          }
        })

      const response = await fetch('http://localhost:1234/api/data/validate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          data: [
            { X: 1, Y: 2, Z: 0 },
            { X: 0, Y: 1, Z: 1 }
          ],
          schema: {
            variables: ['X', 'Y', 'Z'],
            types: ['continuous', 'continuous', 'binary']
          }
        })
      })

      const data = await response.json()
      expect(response.status).toBe(200)
      expect(data.is_valid).toBe(true)
    })
  })

  describe('Error Handling Contracts', () => {
    it('should handle invalid DAG structure', async () => {
      await provider
        .given('invalid DAG data is provided')
        .uponReceiving('a request with cyclic graph')
        .withRequest({
          method: 'POST',
          path: '/api/causal/dag',
          headers: {
            'Content-Type': 'application/json',
          },
          body: {
            nodes: [{ id: 'A' }, { id: 'B' }],
            edges: [{ from: 'A', to: 'B' }, { from: 'B', to: 'A' }]
          }
        })
        .willRespondWith({
          status: 400,
          headers: {
            'Content-Type': 'application/json',
          },
          body: {
            error: 'Invalid DAG',
            message: 'Graph contains cycles',
            error_code: 'CYCLIC_GRAPH',
            details: {
              cycles_detected: [['A', 'B', 'A']]
            }
          }
        })

      const response = await fetch('http://localhost:1234/api/causal/dag', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          nodes: [{ id: 'A' }, { id: 'B' }],
          edges: [{ from: 'A', to: 'B' }, { from: 'B', to: 'A' }]
        })
      })

      const data = await response.json()
      expect(response.status).toBe(400)
      expect(data.error_code).toBe('CYCLIC_GRAPH')
    })
  })
})

// Provider verification (run separately)
export const verifyProvider = async () => {
  const opts = {
    provider: 'jax-causal-backend',
    providerBaseUrl: 'http://localhost:8000',
    pactUrls: [
      path.resolve(process.cwd(), 'tests/contract/pacts/causal-ui-frontend-jax-causal-backend.json')
    ],
    publishVerificationResult: true,
    providerVersion: process.env.GIT_COMMIT || '1.0.0',
  }

  return await new Pact().verifyProvider(opts)
}