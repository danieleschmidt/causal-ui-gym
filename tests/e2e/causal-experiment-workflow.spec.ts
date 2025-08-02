/**
 * End-to-end tests for complete causal experiment workflows
 */

import { test, expect, type Page } from '@playwright/test'

test.describe('Causal Experiment Workflow', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the application
    await page.goto('/')
    
    // Wait for the application to load
    await expect(page.getByText('Causal UI Gym')).toBeVisible()
  })

  test('creates and runs a complete pricing experiment', async ({ page }) => {
    // Step 1: Create a new experiment
    await page.click('[data-testid="create-experiment"]')
    await page.fill('[data-testid="experiment-name"]', 'Pricing Strategy Test')
    await page.selectOption('[data-testid="experiment-template"]', 'pricing')
    await page.click('[data-testid="create-button"]')

    // Step 2: Verify experiment creation
    await expect(page.getByText('Pricing Strategy Test')).toBeVisible()
    await expect(page.getByTestId('causal-graph')).toBeVisible()

    // Step 3: Verify initial causal model
    await expect(page.getByText('Price')).toBeVisible()
    await expect(page.getByText('Demand')).toBeVisible()
    await expect(page.getByText('Revenue')).toBeVisible()

    // Step 4: Perform first intervention
    await page.click('[data-testid="intervention-price"]')
    await page.fill('[data-testid="price-value"]', '50')
    await page.click('[data-testid="apply-intervention"]')

    // Step 5: Verify intervention results
    await expect(page.getByTestId('intervention-result')).toBeVisible()
    await expect(page.getByText(/revenue.*updated/i)).toBeVisible()

    // Step 6: Check metrics dashboard
    await page.click('[data-testid="metrics-tab"]')
    await expect(page.getByTestId('ate-error')).toBeVisible()
    await expect(page.getByTestId('causal-accuracy')).toBeVisible()

    // Step 7: Ask LLM a causal question
    await page.click('[data-testid="ask-llm"]')
    await page.fill(
      '[data-testid="causal-question"]',
      'What happens to revenue if we increase price by 20%?'
    )
    await page.click('[data-testid="submit-question"]')

    // Step 8: Verify LLM response
    await expect(page.getByTestId('llm-response')).toBeVisible()
    await expect(page.getByText(/revenue/i)).toBeVisible()

    // Step 9: Compare with ground truth
    await expect(page.getByTestId('accuracy-score')).toBeVisible()
    
    const accuracyText = await page.getByTestId('accuracy-score').textContent()
    const accuracy = parseFloat(accuracyText?.match(/(\d+\.?\d*)%/)?.[1] || '0')
    expect(accuracy).toBeGreaterThan(0)
    expect(accuracy).toBeLessThanOrEqual(100)
  })

  test('handles multiple interventions in sequence', async ({ page }) => {
    // Create experiment
    await createBasicExperiment(page, 'Multi-Intervention Test')

    // Perform multiple interventions
    const interventions = [
      { variable: 'price', value: '30' },
      { variable: 'price', value: '60' },
      { variable: 'price', value: '90' },
    ]

    for (const intervention of interventions) {
      await page.click(`[data-testid="intervention-${intervention.variable}"]`)
      await page.fill('[data-testid="intervention-value"]', intervention.value)
      await page.click('[data-testid="apply-intervention"]')
      
      // Wait for intervention to complete
      await expect(page.getByTestId('intervention-complete')).toBeVisible()
    }

    // Verify intervention history
    await page.click('[data-testid="history-tab"]')
    await expect(page.getByTestId('intervention-history')).toBeVisible()
    
    const historyItems = page.getByTestId('history-item')
    await expect(historyItems).toHaveCount(3)

    // Verify metrics tracking
    await page.click('[data-testid="metrics-tab"]')
    const metricsChart = page.getByTestId('metrics-chart')
    await expect(metricsChart).toBeVisible()
  })

  test('supports real-time collaboration', async ({ page, context }) => {
    // Create a second page to simulate another user
    const page2 = await context.newPage()
    
    // Both users navigate to the same experiment
    await createBasicExperiment(page, 'Collaboration Test')
    const experimentUrl = page.url()
    await page2.goto(experimentUrl)

    // User 1 performs an intervention
    await page.click('[data-testid="intervention-price"]')
    await page.fill('[data-testid="intervention-value"]', '75')
    await page.click('[data-testid="apply-intervention"]')

    // User 2 should see the update in real-time
    await expect(page2.getByTestId('intervention-result')).toBeVisible()
    await expect(page2.getByText(/price.*75/i)).toBeVisible()

    // User 2 adds a comment
    await page2.click('[data-testid="add-comment"]')
    await page2.fill('[data-testid="comment-text"]', 'Interesting price point!')
    await page2.click('[data-testid="submit-comment"]')

    // User 1 should see the comment
    await expect(page.getByText('Interesting price point!')).toBeVisible()

    await page2.close()
  })

  test('exports experiment data', async ({ page }) => {
    await createBasicExperiment(page, 'Export Test')

    // Perform some interventions to generate data
    await page.click('[data-testid="intervention-price"]')
    await page.fill('[data-testid="intervention-value"]', '45')
    await page.click('[data-testid="apply-intervention"]')

    // Export data
    await page.click('[data-testid="export-menu"]')
    
    // Test CSV export
    const [csvDownload] = await Promise.all([
      page.waitForEvent('download'),
      page.click('[data-testid="export-csv"]')
    ])
    
    expect(csvDownload.suggestedFilename()).toMatch(/.*\.csv$/)

    // Test JSON export
    const [jsonDownload] = await Promise.all([
      page.waitForEvent('download'),
      page.click('[data-testid="export-json"]')
    ])
    
    expect(jsonDownload.suggestedFilename()).toMatch(/.*\.json$/)
  })

  test('handles error states gracefully', async ({ page }) => {
    await createBasicExperiment(page, 'Error Handling Test')

    // Test invalid intervention value
    await page.click('[data-testid="intervention-price"]')
    await page.fill('[data-testid="intervention-value"]', 'invalid-number')
    await page.click('[data-testid="apply-intervention"]')

    // Should show error message
    await expect(page.getByTestId('error-message')).toBeVisible()
    await expect(page.getByText(/invalid.*value/i)).toBeVisible()

    // Test network error simulation
    await page.route('**/api/interventions', route => {
      route.abort('failed')
    })

    await page.fill('[data-testid="intervention-value"]', '50')
    await page.click('[data-testid="apply-intervention"]')

    // Should show network error
    await expect(page.getByText(/network.*error/i)).toBeVisible()

    // Should have retry option
    await expect(page.getByTestId('retry-button')).toBeVisible()
  })

  test('performs accessibility compliance', async ({ page }) => {
    await createBasicExperiment(page, 'Accessibility Test')

    // Test keyboard navigation
    await page.keyboard.press('Tab')
    await page.keyboard.press('Tab')
    await page.keyboard.press('Enter')

    // Should be able to navigate with keyboard
    const focusedElement = page.locator(':focus')
    await expect(focusedElement).toBeVisible()

    // Test screen reader content
    const mainContent = page.getByRole('main')
    await expect(mainContent).toBeVisible()

    // Check for proper heading structure
    const h1 = page.getByRole('heading', { level: 1 })
    await expect(h1).toBeVisible()

    // Check for alt text on images
    const images = page.getByRole('img')
    const imageCount = await images.count()
    
    for (let i = 0; i < imageCount; i++) {
      const image = images.nth(i)
      const altText = await image.getAttribute('alt')
      expect(altText).toBeTruthy()
    }
  })

  test('handles large datasets efficiently', async ({ page }) => {
    // Create experiment with large causal model
    await page.click('[data-testid="create-experiment"]')
    await page.fill('[data-testid="experiment-name"]', 'Large Dataset Test')
    await page.selectOption('[data-testid="experiment-template"]', 'large-model')
    await page.click('[data-testid="create-button"]')

    // Wait for large model to load
    await expect(page.getByTestId('loading-indicator')).toBeHidden()
    await expect(page.getByTestId('causal-graph')).toBeVisible()

    // Verify performance
    const startTime = Date.now()
    await page.click('[data-testid="intervention-node-50"]')
    await page.fill('[data-testid="intervention-value"]', '100')
    await page.click('[data-testid="apply-intervention"]')
    
    await expect(page.getByTestId('intervention-result')).toBeVisible()
    const endTime = Date.now()

    // Should complete within reasonable time (5 seconds)
    expect(endTime - startTime).toBeLessThan(5000)
  })

  test('supports different experiment templates', async ({ page }) => {
    const templates = [
      { name: 'Supply-Demand Economics', template: 'supply-demand' },
      { name: 'Medical Diagnosis', template: 'medical' },
      { name: 'Marketing Attribution', template: 'marketing' },
    ]

    for (const { name, template } of templates) {
      await page.click('[data-testid="create-experiment"]')
      await page.fill('[data-testid="experiment-name"]', name)
      await page.selectOption('[data-testid="experiment-template"]', template)
      await page.click('[data-testid="create-button"]')

      // Verify template-specific elements
      await expect(page.getByTestId('causal-graph')).toBeVisible()
      await expect(page.getByTestId(`${template}-template`)).toBeVisible()

      // Go back to create another
      await page.click('[data-testid="back-to-dashboard"]')
    }
  })
})

// Helper function to create a basic experiment
async function createBasicExperiment(page: Page, name: string) {
  await page.click('[data-testid="create-experiment"]')
  await page.fill('[data-testid="experiment-name"]', name)
  await page.selectOption('[data-testid="experiment-template"]', 'pricing')
  await page.click('[data-testid="create-button"]')
  
  await expect(page.getByText(name)).toBeVisible()
  await expect(page.getByTestId('causal-graph')).toBeVisible()
}

test.describe('Visual Regression Tests', () => {
  test('causal graph renders consistently', async ({ page }) => {
    await createBasicExperiment(page, 'Visual Test')
    
    const graph = page.getByTestId('causal-graph')
    await expect(graph).toHaveScreenshot('causal-graph-baseline.png')
  })

  test('intervention panel layout is stable', async ({ page }) => {
    await createBasicExperiment(page, 'Layout Test')
    
    const panel = page.getByTestId('intervention-panel')
    await expect(panel).toHaveScreenshot('intervention-panel-baseline.png')
  })

  test('metrics dashboard displays correctly', async ({ page }) => {
    await createBasicExperiment(page, 'Metrics Test')
    
    // Perform intervention to generate metrics
    await page.click('[data-testid="intervention-price"]')
    await page.fill('[data-testid="intervention-value"]', '50')
    await page.click('[data-testid="apply-intervention"]')
    
    await page.click('[data-testid="metrics-tab"]')
    const dashboard = page.getByTestId('metrics-dashboard')
    await expect(dashboard).toHaveScreenshot('metrics-dashboard-baseline.png')
  })
})