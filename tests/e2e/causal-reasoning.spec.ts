import { test, expect } from '@playwright/test';

test.describe('Causal Reasoning UI Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should render causal graph correctly', async ({ page }) => {
    // Wait for the causal graph component to load
    await page.waitForSelector('[data-testid="causal-graph"]');
    
    // Check that nodes are rendered
    const nodes = page.locator('[data-testid="causal-node"]');
    await expect(nodes).toHaveCount(3); // price, demand, revenue
    
    // Check that edges are rendered
    const edges = page.locator('[data-testid="causal-edge"]');
    await expect(edges).toHaveCount(3);
  });

  test('should perform intervention correctly', async ({ page }) => {
    // Wait for intervention controls
    await page.waitForSelector('[data-testid="intervention-control"]');
    
    // Set price intervention
    await page.fill('[data-testid="price-input"]', '29.99');
    await page.click('[data-testid="apply-intervention"]');
    
    // Verify intervention was applied
    await expect(page.locator('[data-testid="intervention-status"]'))
      .toContainText('Intervention: price = 29.99');
    
    // Check that causal metrics updated
    await page.waitForSelector('[data-testid="ate-metric"]');
    const ateValue = await page.textContent('[data-testid="ate-metric"]');
    expect(parseFloat(ateValue || '0')).not.toBe(0);
  });

  test('should track LLM belief updates', async ({ page }) => {
    // Mock API responses for LLM interaction
    await page.route('/api/llm/belief-update', route => {
      route.fulfill({
        json: {
          agent: 'gpt-4',
          beliefs: {
            causal_strength: 0.75,
            confidence: 0.82
          },
          reasoning: 'Price increase should decrease demand...'
        }
      });
    });

    await page.click('[data-testid="ask-llm-button"]');
    
    // Wait for belief tracker to update
    await page.waitForSelector('[data-testid="belief-tracker"]');
    
    // Verify belief values
    await expect(page.locator('[data-testid="causal-strength"]'))
      .toContainText('0.75');
    await expect(page.locator('[data-testid="confidence"]'))
      .toContainText('0.82');
  });

  test('should handle causal reasoning errors gracefully', async ({ page }) => {
    // Mock API error
    await page.route('/api/causal/compute', route => {
      route.fulfill({
        status: 500,
        json: { error: 'Causal computation failed' }
      });
    });

    await page.fill('[data-testid="price-input"]', 'invalid');
    await page.click('[data-testid="apply-intervention"]');
    
    // Check error handling
    await expect(page.locator('[data-testid="error-message"]'))
      .toBeVisible();
    await expect(page.locator('[data-testid="error-message"]'))
      .toContainText('computation failed');
  });

  test('should export causal analysis results', async ({ page }) => {
    // Perform some interventions first
    await page.fill('[data-testid="price-input"]', '25.99');
    await page.click('[data-testid="apply-intervention"]');
    
    // Wait for results
    await page.waitForSelector('[data-testid="results-ready"]');
    
    // Start download
    const [download] = await Promise.all([
      page.waitForEvent('download'),
      page.click('[data-testid="export-results"]')
    ]);
    
    // Verify download
    expect(download.suggestedFilename()).toBe('causal-analysis-results.json');
  });

  test('should support accessibility requirements', async ({ page }) => {
    // Check for proper ARIA labels
    await expect(page.locator('[data-testid="causal-graph"]'))
      .toHaveAttribute('aria-label');
    
    // Test keyboard navigation
    await page.keyboard.press('Tab');
    await expect(page.locator(':focus')).toBeVisible();
    
    // Check color contrast (basic check)
    const graphElement = page.locator('[data-testid="causal-graph"]');
    const styles = await graphElement.evaluate(el => getComputedStyle(el));
    
    // Ensure text is readable (this would need more sophisticated contrast checking)
    expect(styles.color).not.toBe(styles.backgroundColor);
  });
});