# Testing Documentation

This directory contains all test files for the Causal UI Gym project.

## Directory Structure

```
tests/
├── unit/           # Unit tests for individual components and functions
├── integration/    # Integration tests for component interactions
├── e2e/           # End-to-end tests using Playwright
├── fixtures/      # Test data and mock files
├── utils/         # Testing utilities and helpers
└── README.md      # This file
```

## Test Types

### Unit Tests (`tests/unit/`)
- Component tests using Vitest and React Testing Library
- Function tests for utility modules
- JAX computation tests

### Integration Tests (`tests/integration/`)
- API endpoint tests
- Component integration tests
- Causal computation pipeline tests

### End-to-End Tests (`tests/e2e/`)
- Full user workflow tests using Playwright
- Visual regression tests
- Cross-browser compatibility tests

## Running Tests

```bash
# Run all tests
npm test

# Run unit tests only
npm run test:unit

# Run integration tests
npm run test:integration

# Run e2e tests
npm run test:e2e

# Run with UI
npm run test:ui

# Run with coverage
npm run test:coverage
```

## Writing Tests

### Unit Test Example
```typescript
// tests/unit/components/CausalGraph.test.tsx
import { render, screen } from '@testing-library/react'
import { CausalGraph } from '../../src/components/CausalGraph'

describe('CausalGraph', () => {
  it('renders nodes correctly', () => {
    const nodes = [{ id: 'A', label: 'Variable A' }]
    render(<CausalGraph nodes={nodes} edges={[]} />)
    expect(screen.getByText('Variable A')).toBeInTheDocument()
  })
})
```

### E2E Test Example
```typescript
// tests/e2e/causal-experiments.spec.ts
import { test, expect } from '@playwright/test'

test('causal intervention updates visualization', async ({ page }) => {
  await page.goto('/experiments/supply-demand')
  await page.click('[data-testid="price-intervention"]')
  await expect(page.locator('[data-testid="demand-chart"]')).toBeVisible()
})
```

## Testing Guidelines

1. **Test file naming**: Use `.test.ts` or `.spec.ts` extensions
2. **Test organization**: Group related tests using `describe` blocks
3. **Assertions**: Use descriptive assertion messages
4. **Test data**: Store reusable test data in `fixtures/`
5. **Mock external dependencies**: Use appropriate mocking strategies
6. **Visual tests**: Include screenshots for visual regression testing

## Configuration Files

- `vitest.config.ts` - Unit and integration test configuration
- `playwright.config.ts` - E2E test configuration (to be created)
- `tests/setup.ts` - Global test setup (to be created)