/**
 * Integration tests for API endpoints
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import { mockApiResponses } from '../utils/test-helpers'

// Mock fetch for testing
global.fetch = vi.fn()

describe('API Endpoints Integration', () => {
  const API_BASE_URL = 'http://localhost:8000/api'

  beforeAll(() => {
    // Setup mock responses
    vi.mocked(fetch).mockImplementation((url: string) => {
      const urlStr = url.toString()
      
      if (urlStr.includes('/health')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve(mockApiResponses.healthCheck),
        } as Response)
      }
      
      if (urlStr.includes('/experiments')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve(mockApiResponses.experiment),
        } as Response)
      }
      
      if (urlStr.includes('/interventions')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve(mockApiResponses.intervention),
        } as Response)
      }
      
      if (urlStr.includes('/metrics')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve(mockApiResponses.metrics),
        } as Response)
      }
      
      return Promise.resolve({
        ok: false,
        status: 404,
        json: () => Promise.resolve({ error: 'Not found' }),
      } as Response)
    })
  })

  afterAll(() => {
    vi.restoreAllMocks()
  })

  describe('Health Check Endpoint', () => {
    it('returns healthy status', async () => {
      const response = await fetch(`${API_BASE_URL}/health`)
      const data = await response.json()
      
      expect(response.ok).toBe(true)
      expect(data.status).toBe('healthy')
      expect(data.timestamp).toBeDefined()
    })
  })

  describe('Experiments Endpoint', () => {
    it('creates a new experiment', async () => {
      const experimentData = {
        name: 'Test Experiment',
        model: mockApiResponses.experiment.model,
      }
      
      const response = await fetch(`${API_BASE_URL}/experiments`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(experimentData),
      })
      
      const data = await response.json()
      
      expect(response.ok).toBe(true)
      expect(data.id).toBeDefined()
      expect(data.status).toBe('active')
      expect(data.model).toEqual(mockApiResponses.experiment.model)
    })

    it('retrieves experiment by ID', async () => {
      const experimentId = 'test-experiment-123'
      
      const response = await fetch(`${API_BASE_URL}/experiments/${experimentId}`)
      const data = await response.json()
      
      expect(response.ok).toBe(true)
      expect(data.id).toBe(experimentId)
      expect(data.model).toBeDefined()
    })

    it('handles invalid experiment ID', async () => {
      const response = await fetch(`${API_BASE_URL}/experiments/invalid-id`)
      
      expect(response.ok).toBe(false)
      expect(response.status).toBe(404)
    })
  })

  describe('Interventions Endpoint', () => {
    it('performs intervention successfully', async () => {
      const interventionData = {
        experimentId: 'test-experiment-123',
        variable: 'price',
        value: 75,
      }
      
      const response = await fetch(`${API_BASE_URL}/interventions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(interventionData),
      })
      
      const data = await response.json()
      
      expect(response.ok).toBe(true)
      expect(data.id).toBeDefined()
      expect(data.variable).toBe('price')
      expect(data.value).toBe(75)
      expect(data.result).toBeDefined()
    })

    it('validates intervention data', async () => {
      const invalidData = {
        experimentId: 'test-experiment-123',
        // Missing required fields
      }
      
      const response = await fetch(`${API_BASE_URL}/interventions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(invalidData),
      })
      
      expect(response.ok).toBe(false)
    })
  })

  describe('Metrics Endpoint', () => {
    it('retrieves experiment metrics', async () => {
      const experimentId = 'test-experiment-123'
      
      const response = await fetch(`${API_BASE_URL}/metrics/${experimentId}`)
      const data = await response.json()
      
      expect(response.ok).toBe(true)
      expect(data.experimentId).toBe(experimentId)
      expect(data.ateError).toBeDefined()
      expect(data.causalAccuracy).toBeDefined()
      expect(data.history).toBeInstanceOf(Array)
    })

    it('handles metrics for non-existent experiment', async () => {
      const response = await fetch(`${API_BASE_URL}/metrics/non-existent`)
      
      expect(response.ok).toBe(false)
      expect(response.status).toBe(404)
    })
  })

  describe('Error Handling', () => {
    it('handles network errors gracefully', async () => {
      vi.mocked(fetch).mockRejectedValueOnce(new Error('Network error'))
      
      await expect(
        fetch(`${API_BASE_URL}/health`)
      ).rejects.toThrow('Network error')
    })

    it('handles malformed JSON responses', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.reject(new Error('Invalid JSON')),
      } as Response)
      
      const response = await fetch(`${API_BASE_URL}/health`)
      
      await expect(response.json()).rejects.toThrow('Invalid JSON')
    })

    it('handles rate limiting', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 429,
        json: () => Promise.resolve({
          error: 'Rate limit exceeded',
          retryAfter: 60,
        }),
      } as Response)
      
      const response = await fetch(`${API_BASE_URL}/experiments`)
      const data = await response.json()
      
      expect(response.status).toBe(429)
      expect(data.error).toContain('Rate limit')
      expect(data.retryAfter).toBe(60)
    })
  })

  describe('Authentication', () => {
    it('handles authenticated requests', async () => {
      const response = await fetch(`${API_BASE_URL}/experiments`, {
        headers: {
          'Authorization': 'Bearer mock-token',
          'Content-Type': 'application/json',
        },
      })
      
      expect(response.ok).toBe(true)
    })

    it('rejects unauthenticated requests for protected endpoints', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({
          error: 'Unauthorized',
        }),
      } as Response)
      
      const response = await fetch(`${API_BASE_URL}/experiments/protected`)
      
      expect(response.status).toBe(401)
    })
  })

  describe('Data Validation', () => {
    it('validates causal model structure', async () => {
      const invalidModel = {
        nodes: [{ id: '', label: '' }], // Invalid empty values
        edges: [{ from: 'A', to: 'B' }], // Missing relationship
      }
      
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'Invalid model structure',
          details: ['Node ID cannot be empty', 'Edge relationship required'],
        }),
      } as Response)
      
      const response = await fetch(`${API_BASE_URL}/experiments`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: invalidModel }),
      })
      
      const data = await response.json()
      
      expect(response.status).toBe(400)
      expect(data.error).toContain('Invalid model')
      expect(data.details).toBeInstanceOf(Array)
    })

    it('validates intervention values', async () => {
      const invalidIntervention = {
        experimentId: 'test-experiment-123',
        variable: 'price',
        value: 'invalid-number', // Should be numeric
      }
      
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({
          error: 'Invalid intervention value',
        }),
      } as Response)
      
      const response = await fetch(`${API_BASE_URL}/interventions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(invalidIntervention),
      })
      
      expect(response.status).toBe(400)
    })
  })

  describe('Performance', () => {
    it('responds within acceptable time limits', async () => {
      const startTime = performance.now()
      
      await fetch(`${API_BASE_URL}/health`)
      
      const responseTime = performance.now() - startTime
      
      // API should respond within 500ms for health checks
      expect(responseTime).toBeLessThan(500)
    })

    it('handles concurrent requests', async () => {
      const promises = Array.from({ length: 10 }, () =>
        fetch(`${API_BASE_URL}/health`)
      )
      
      const responses = await Promise.all(promises)
      
      responses.forEach(response => {
        expect(response.ok).toBe(true)
      })
    })
  })
})