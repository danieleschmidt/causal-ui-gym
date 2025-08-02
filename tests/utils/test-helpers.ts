/**
 * Test utilities and helpers for Causal UI Gym
 */

import { render, RenderOptions } from '@testing-library/react'
import { ReactElement } from 'react'
import { ThemeProvider, createTheme } from '@mui/material/styles'

// Mock theme for testing
const mockTheme = createTheme({
  palette: {
    mode: 'light',
  },
})

/**
 * Custom render function that includes providers
 */
const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) => {
  const Wrapper = ({ children }: { children: React.ReactNode }) => (
    <ThemeProvider theme={mockTheme}>{children}</ThemeProvider>
  )

  return render(ui, { wrapper: Wrapper, ...options })
}

/**
 * Mock causal model for testing
 */
export const mockCausalModel = {
  nodes: [
    { id: 'price', label: 'Price', type: 'continuous' as const },
    { id: 'demand', label: 'Demand', type: 'continuous' as const },
    { id: 'revenue', label: 'Revenue', type: 'continuous' as const },
  ],
  edges: [
    { from: 'price', to: 'demand', relationship: 'negative' as const },
    { from: 'price', to: 'revenue', relationship: 'positive' as const },
    { from: 'demand', to: 'revenue', relationship: 'positive' as const },
  ],
}

/**
 * Mock intervention data
 */
export const mockIntervention = {
  variable: 'price',
  value: 50,
  timestamp: new Date().toISOString(),
}

/**
 * Mock LLM response
 */
export const mockLLMResponse = {
  prediction: 'Revenue will increase by approximately 25%',
  confidence: 0.85,
  reasoning: 'Higher price leads to higher revenue despite reduced demand',
  timestamp: new Date().toISOString(),
}

/**
 * Mock metrics data
 */
export const mockMetrics = {
  ateError: 0.12,
  causalAccuracy: 0.89,
  interventionSuccess: true,
  responseTime: 234,
}

/**
 * Wait for a specific condition to be true
 */
export const waitFor = async (
  condition: () => boolean,
  timeout: number = 5000
): Promise<void> => {
  const start = Date.now()
  while (!condition() && Date.now() - start < timeout) {
    await new Promise(resolve => setTimeout(resolve, 100))
  }
  if (!condition()) {
    throw new Error(`Condition not met within ${timeout}ms`)
  }
}

/**
 * Generate random test data
 */
export const generateTestData = {
  nodes: (count: number) =>
    Array.from({ length: count }, (_, i) => ({
      id: `node_${i}`,
      label: `Node ${i}`,
      type: 'continuous' as const,
    })),
  
  edges: (nodeCount: number) =>
    Array.from({ length: nodeCount - 1 }, (_, i) => ({
      from: `node_${i}`,
      to: `node_${i + 1}`,
      relationship: 'positive' as const,
    })),

  interventions: (count: number) =>
    Array.from({ length: count }, (_, i) => ({
      variable: `node_${i % 3}`,
      value: Math.random() * 100,
      timestamp: new Date(Date.now() - i * 1000).toISOString(),
    })),
}

/**
 * Mock API responses for testing
 */
export const mockApiResponses = {
  healthCheck: { status: 'healthy', timestamp: new Date().toISOString() },
  experiment: {
    id: 'test-experiment-123',
    model: mockCausalModel,
    status: 'active',
    createdAt: new Date().toISOString(),
  },
  intervention: {
    id: 'intervention-456',
    ...mockIntervention,
    result: { revenue: 125, demand: 75 },
  },
  metrics: {
    experimentId: 'test-experiment-123',
    ...mockMetrics,
    history: [mockMetrics],
  },
  llmResponse: {
    id: 'llm-response-789',
    ...mockLLMResponse,
  },
}

/**
 * Performance test utilities
 */
export const performanceHelpers = {
  measureRenderTime: async (renderFn: () => void): Promise<number> => {
    const start = performance.now()
    renderFn()
    return performance.now() - start
  },

  simulateSlowNetwork: (delay: number = 2000) => {
    return new Promise(resolve => setTimeout(resolve, delay))
  },

  createLargeDataset: (size: number) => ({
    nodes: generateTestData.nodes(size),
    edges: generateTestData.edges(size),
    interventions: generateTestData.interventions(size),
  }),
}

/**
 * Accessibility test helpers
 */
export const a11yHelpers = {
  checkKeyboardNavigation: async (element: HTMLElement) => {
    element.focus()
    return document.activeElement === element
  },

  checkAriaLabels: (element: HTMLElement) => {
    return {
      hasAriaLabel: element.hasAttribute('aria-label'),
      hasAriaDescribedBy: element.hasAttribute('aria-describedby'),
      hasRole: element.hasAttribute('role'),
    }
  },

  simulateKeyPress: (element: HTMLElement, key: string) => {
    const event = new KeyboardEvent('keydown', { key })
    element.dispatchEvent(event)
  },
}

/**
 * Visual regression test helpers
 */
export const visualHelpers = {
  compareScreenshots: async (
    element: HTMLElement,
    baselineName: string
  ): Promise<boolean> => {
    // Implementation would depend on the screenshot testing library
    // This is a placeholder for the actual implementation
    console.log(`Comparing ${baselineName} with current state`)
    return true
  },

  captureElement: async (element: HTMLElement): Promise<string> => {
    // Implementation for capturing element as image data
    // This is a placeholder for the actual implementation
    return 'data:image/png;base64,mock-image-data'
  },
}

// Re-export testing library utilities with our custom render
export * from '@testing-library/react'
export { customRender as render }