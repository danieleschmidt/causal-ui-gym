/**
 * Unit tests for metrics utilities
 */

import { describe, it, expect, vi } from 'vitest'
import { calculateATE, calculateCausalAccuracy, measureResponseTime } from '../../../src/utils/metrics'

describe('Metrics Utilities', () => {
  describe('calculateATE', () => {
    it('calculates average treatment effect correctly', () => {
      const treatmentData = [100, 110, 120, 105, 115]
      const controlData = [80, 85, 90, 82, 88]
      
      const ate = calculateATE(treatmentData, controlData)
      
      // Expected ATE: mean(treatment) - mean(control) = 110 - 85 = 25
      expect(ate).toBeCloseTo(25, 1)
    })

    it('handles empty arrays gracefully', () => {
      expect(() => calculateATE([], [])).toThrow('Data arrays cannot be empty')
    })

    it('handles unequal array lengths', () => {
      const treatmentData = [100, 110]
      const controlData = [80, 85, 90]
      
      const ate = calculateATE(treatmentData, controlData)
      
      // Should still calculate correctly with different lengths
      expect(typeof ate).toBe('number')
      expect(ate).toBeCloseTo(21.67, 1) // 105 - 85 = 20
    })

    it('handles negative effects', () => {
      const treatmentData = [70, 75, 80]
      const controlData = [100, 105, 110]
      
      const ate = calculateATE(treatmentData, controlData)
      
      expect(ate).toBeLessThan(0)
      expect(ate).toBeCloseTo(-30, 1) // 75 - 105 = -30
    })

    it('returns zero for identical groups', () => {
      const data = [100, 100, 100]
      
      const ate = calculateATE(data, data)
      
      expect(ate).toBe(0)
    })
  })

  describe('calculateCausalAccuracy', () => {
    it('calculates accuracy for perfect predictions', () => {
      const predictions = [
        { from: 'A', to: 'B', relationship: 'positive' },
        { from: 'B', to: 'C', relationship: 'negative' },
        { from: 'A', to: 'C', relationship: 'positive' },
      ]
      
      const groundTruth = [
        { from: 'A', to: 'B', relationship: 'positive' },
        { from: 'B', to: 'C', relationship: 'negative' },
        { from: 'A', to: 'C', relationship: 'positive' },
      ]
      
      const accuracy = calculateCausalAccuracy(predictions, groundTruth)
      
      expect(accuracy).toBe(1.0)
    })

    it('calculates accuracy for partial predictions', () => {
      const predictions = [
        { from: 'A', to: 'B', relationship: 'positive' },
        { from: 'B', to: 'C', relationship: 'positive' }, // Wrong
        { from: 'A', to: 'C', relationship: 'positive' },
      ]
      
      const groundTruth = [
        { from: 'A', to: 'B', relationship: 'positive' },
        { from: 'B', to: 'C', relationship: 'negative' },
        { from: 'A', to: 'C', relationship: 'positive' },
      ]
      
      const accuracy = calculateCausalAccuracy(predictions, groundTruth)
      
      expect(accuracy).toBeCloseTo(0.67, 2) // 2/3 correct
    })

    it('handles missing predictions', () => {
      const predictions = [
        { from: 'A', to: 'B', relationship: 'positive' },
      ]
      
      const groundTruth = [
        { from: 'A', to: 'B', relationship: 'positive' },
        { from: 'B', to: 'C', relationship: 'negative' },
      ]
      
      const accuracy = calculateCausalAccuracy(predictions, groundTruth)
      
      expect(accuracy).toBe(0.5) // 1/2 correct (missing predictions count as wrong)
    })

    it('handles extra predictions', () => {
      const predictions = [
        { from: 'A', to: 'B', relationship: 'positive' },
        { from: 'B', to: 'C', relationship: 'negative' },
        { from: 'D', to: 'E', relationship: 'positive' }, // Extra
      ]
      
      const groundTruth = [
        { from: 'A', to: 'B', relationship: 'positive' },
        { from: 'B', to: 'C', relationship: 'negative' },
      ]
      
      const accuracy = calculateCausalAccuracy(predictions, groundTruth)
      
      expect(accuracy).toBeCloseTo(0.67, 2) // 2/3 (extra predictions penalized)
    })

    it('returns 0 for completely wrong predictions', () => {
      const predictions = [
        { from: 'A', to: 'B', relationship: 'negative' },
        { from: 'B', to: 'C', relationship: 'positive' },
      ]
      
      const groundTruth = [
        { from: 'A', to: 'B', relationship: 'positive' },
        { from: 'B', to: 'C', relationship: 'negative' },
      ]
      
      const accuracy = calculateCausalAccuracy(predictions, groundTruth)
      
      expect(accuracy).toBe(0)
    })
  })

  describe('measureResponseTime', () => {
    it('measures function execution time', async () => {
      const slowFunction = async () => {
        await new Promise(resolve => setTimeout(resolve, 100))
        return 'result'
      }
      
      const { result, responseTime } = await measureResponseTime(slowFunction)
      
      expect(result).toBe('result')
      expect(responseTime).toBeGreaterThan(90)
      expect(responseTime).toBeLessThan(200)
    })

    it('measures synchronous function time', async () => {
      const fastFunction = () => {
        let sum = 0
        for (let i = 0; i < 1000; i++) {
          sum += i
        }
        return sum
      }
      
      const { result, responseTime } = await measureResponseTime(fastFunction)
      
      expect(result).toBe(499500)
      expect(responseTime).toBeGreaterThan(0)
      expect(responseTime).toBeLessThan(10)
    })

    it('handles function errors correctly', async () => {
      const errorFunction = async () => {
        throw new Error('Test error')
      }
      
      await expect(measureResponseTime(errorFunction)).rejects.toThrow('Test error')
    })

    it('measures multiple executions', async () => {
      const testFunction = () => Math.random()
      
      const measurements = await Promise.all([
        measureResponseTime(testFunction),
        measureResponseTime(testFunction),
        measureResponseTime(testFunction),
      ])
      
      expect(measurements).toHaveLength(3)
      measurements.forEach(({ result, responseTime }) => {
        expect(typeof result).toBe('number')
        expect(responseTime).toBeGreaterThan(0)
      })
    })
  })

  describe('Performance Tests', () => {
    it('calculates ATE for large datasets efficiently', () => {
      const size = 10000
      const treatmentData = Array.from({ length: size }, () => Math.random() * 100)
      const controlData = Array.from({ length: size }, () => Math.random() * 100)
      
      const startTime = performance.now()
      const ate = calculateATE(treatmentData, controlData)
      const executionTime = performance.now() - startTime
      
      expect(typeof ate).toBe('number')
      expect(executionTime).toBeLessThan(100) // Should complete in under 100ms
    })

    it('calculates accuracy for large prediction sets efficiently', () => {
      const size = 1000
      const predictions = Array.from({ length: size }, (_, i) => ({
        from: `node_${i}`,
        to: `node_${i + 1}`,
        relationship: Math.random() > 0.5 ? 'positive' : 'negative' as const,
      }))
      
      const groundTruth = Array.from({ length: size }, (_, i) => ({
        from: `node_${i}`,
        to: `node_${i + 1}`,
        relationship: Math.random() > 0.5 ? 'positive' : 'negative' as const,
      }))
      
      const startTime = performance.now()
      const accuracy = calculateCausalAccuracy(predictions, groundTruth)
      const executionTime = performance.now() - startTime
      
      expect(typeof accuracy).toBe('number')
      expect(accuracy).toBeGreaterThanOrEqual(0)
      expect(accuracy).toBeLessThanOrEqual(1)
      expect(executionTime).toBeLessThan(50) // Should complete in under 50ms
    })
  })

  describe('Edge Cases', () => {
    it('handles NaN values in ATE calculation', () => {
      const treatmentData = [100, NaN, 120]
      const controlData = [80, 85, 90]
      
      expect(() => calculateATE(treatmentData, controlData)).toThrow(/invalid data/i)
    })

    it('handles infinite values in ATE calculation', () => {
      const treatmentData = [100, Infinity, 120]
      const controlData = [80, 85, 90]
      
      expect(() => calculateATE(treatmentData, controlData)).toThrow(/invalid data/i)
    })

    it('handles circular relationships in accuracy calculation', () => {
      const predictions = [
        { from: 'A', to: 'B', relationship: 'positive' },
        { from: 'B', to: 'A', relationship: 'negative' },
      ]
      
      const groundTruth = [
        { from: 'A', to: 'B', relationship: 'positive' },
        { from: 'B', to: 'A', relationship: 'negative' },
      ]
      
      const accuracy = calculateCausalAccuracy(predictions, groundTruth)
      
      expect(accuracy).toBe(1.0)
    })
  })
})