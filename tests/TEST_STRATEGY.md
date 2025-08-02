# Testing Strategy for Causal UI Gym

## Overview

This document outlines the comprehensive testing strategy for the Causal UI Gym project, covering all aspects from unit tests to performance validation.

## Testing Pyramid

```
    ┌─────────────────┐
    │   E2E Tests     │  ← Few, High-level, Slow
    │  (Playwright)   │
    ├─────────────────┤
    │ Integration     │  ← Some, Medium-level
    │   Tests         │
    ├─────────────────┤
    │   Unit Tests    │  ← Many, Fast, Isolated
    │  (Vitest/Jest)  │
    └─────────────────┘
```

## Test Categories

### 1. Unit Tests (`tests/unit/`)

**Purpose**: Test individual components, functions, and utilities in isolation.

**Framework**: Vitest + React Testing Library
**Coverage Target**: 90%+
**Execution Time**: <2 seconds total

**What we test**:
- React component rendering and behavior
- Utility function correctness
- Causal computation algorithms
- State management logic
- Error handling

**Example**:
```typescript
// tests/unit/components/CausalGraph.test.tsx
import { render, screen } from '@testing-library/react'
import { CausalGraph } from '../../../src/components/CausalGraph'

test('renders causal nodes correctly', () => {
  const nodes = [{ id: 'A', label: 'Node A', type: 'continuous' }]
  render(<CausalGraph nodes={nodes} edges={[]} />)
  expect(screen.getByText('Node A')).toBeInTheDocument()
})
```

### 2. Integration Tests (`tests/integration/`)

**Purpose**: Test component interactions and API endpoints.

**Framework**: Vitest + MSW (Mock Service Worker)
**Coverage Target**: 80%+
**Execution Time**: <10 seconds total

**What we test**:
- API endpoint functionality
- Component integration flows
- Database operations (mocked)
- External service interactions

**Example**:
```typescript
// tests/integration/causal-pipeline.test.ts
test('complete causal computation pipeline', async () => {
  const model = createTestModel()
  const intervention = { variable: 'price', value: 50 }
  
  const result = await computeCausalEffect(model, intervention)
  expect(result.ateError).toBeLessThan(0.1)
})
```

### 3. End-to-End Tests (`tests/e2e/`)

**Purpose**: Test complete user workflows and scenarios.

**Framework**: Playwright
**Coverage Target**: Critical user paths
**Execution Time**: <60 seconds total

**What we test**:
- Complete experiment creation workflow
- LLM interaction and response handling
- Multi-user collaboration features
- Cross-browser compatibility
- Visual regression testing

**Example**:
```typescript
// tests/e2e/experiment-workflow.spec.ts
test('creates and runs pricing experiment', async ({ page }) => {
  await page.goto('/')
  await page.click('[data-testid="create-experiment"]')
  // ... complete workflow
  await expect(page.getByText('Experiment completed')).toBeVisible()
})
```

### 4. Performance Tests (`tests/performance/`)

**Purpose**: Validate system performance under various loads.

**Framework**: Python pytest + JAX benchmarks
**Execution Time**: <120 seconds total

**What we test**:
- Causal computation performance
- Large dataset handling
- Memory usage patterns
- Response time benchmarks
- Concurrent user scenarios

**Example**:
```python
# tests/performance/causal-computation.test.py
def test_large_dag_performance():
    dag = create_large_dag(1000)  # 1000 nodes
    start_time = time.time()
    result = compute_intervention(dag, intervention)
    execution_time = time.time() - start_time
    assert execution_time < 0.5  # 500ms threshold
```

### 5. Visual Regression Tests

**Purpose**: Ensure UI consistency across changes.

**Framework**: Playwright screenshots
**Coverage**: Key UI components
**Execution Time**: <30 seconds total

**What we test**:
- Causal graph rendering consistency
- Dashboard layout stability
- Component visual states
- Responsive design integrity

## Test Data Management

### Fixtures (`tests/fixtures/`)

Centralized test data to ensure consistency:

```typescript
// tests/fixtures/test-data.ts
export const SIMPLE_CAUSAL_MODEL = {
  nodes: [
    { id: 'price', label: 'Price', type: 'continuous' },
    { id: 'demand', label: 'Demand', type: 'continuous' },
  ],
  edges: [
    { from: 'price', to: 'demand', relationship: 'negative' }
  ]
}
```

### Test Utilities (`tests/utils/`)

Reusable testing helpers:

```typescript
// tests/utils/test-helpers.ts
export const mockCausalModel = { /* ... */ }
export const performanceHelpers = { /* ... */ }
export const a11yHelpers = { /* ... */ }
```

## Quality Gates

### Coverage Requirements

| Test Type | Minimum Coverage | Target Coverage |
|-----------|------------------|-----------------|
| Unit Tests | 80% | 90%+ |
| Integration Tests | 70% | 80%+ |
| E2E Tests | Critical Paths | All User Journeys |

### Performance Benchmarks

| Operation | Target Time | Maximum Time |
|-----------|-------------|--------------|
| Component Render | <10ms | <50ms |
| Causal Computation | <100ms | <500ms |
| API Response | <200ms | <1000ms |
| Page Load | <2s | <5s |

### Accessibility Standards

- WCAG 2.1 AA compliance
- Keyboard navigation support
- Screen reader compatibility
- Color contrast validation

## Test Execution Strategy

### Development Workflow

```bash
# Quick feedback loop during development
npm run test:watch

# Pre-commit validation
npm run validate  # lint + typecheck + test + security-check

# Full test suite before PR
npm run test:coverage
npm run test:e2e
npm run test:performance
```

### CI/CD Pipeline

```yaml
# .github/workflows/test.yml
jobs:
  unit-tests:
    run: npm run test:unit
  
  integration-tests:
    run: npm run test:integration
    
  e2e-tests:
    run: npm run test:e2e
    
  performance-tests:
    run: npm run test:performance
```

### Test Environments

1. **Local Development**
   - Fast unit/integration tests
   - Mock external services
   - Hot reload for test files

2. **CI Environment**
   - Full test suite execution
   - Real browser testing
   - Performance benchmarking

3. **Staging Environment**
   - End-to-end validation
   - Load testing
   - Integration with real services

## Test Configuration

### Vitest Configuration (`vitest.config.ts`)

```typescript
export default defineConfig({
  test: {
    environment: 'jsdom',
    coverage: {
      thresholds: {
        global: { branches: 80, functions: 80, lines: 80, statements: 80 }
      }
    }
  }
})
```

### Playwright Configuration (`playwright.config.ts`)

```typescript
export default defineConfig({
  testDir: './tests/e2e',
  timeout: 30000,
  retries: 2,
  use: {
    baseURL: 'http://localhost:5173',
    screenshot: 'only-on-failure'
  }
})
```

## Mock and Stub Strategy

### API Mocking

```typescript
// Use MSW for consistent API mocking
import { rest } from 'msw'

export const handlers = [
  rest.post('/api/experiments', (req, res, ctx) => {
    return res(ctx.json({ id: 'mock-experiment-id' }))
  })
]
```

### External Service Mocking

- LLM API responses
- Database operations
- File system operations
- Network requests

## Debugging and Troubleshooting

### Test Debugging

```bash
# Debug specific test
npm run test -- --reporter=verbose CausalGraph.test.tsx

# Debug with UI
npm run test:ui

# Debug E2E tests
npm run test:e2e -- --headed --debug
```

### Common Issues

1. **Flaky Tests**: Use proper waiting strategies
2. **Memory Leaks**: Clean up after each test
3. **Timing Issues**: Use deterministic waits
4. **Environment Differences**: Containerize test environments

## Continuous Improvement

### Metrics Tracking

- Test execution time trends
- Coverage percentage over time
- Flaky test identification
- Performance regression detection

### Regular Reviews

- Monthly test strategy review
- Quarterly performance benchmark updates
- Annual framework/tooling evaluation

### Knowledge Sharing

- Test writing workshops
- Best practices documentation
- Code review guidelines
- Testing community of practice

## Future Enhancements

### Planned Improvements

1. **Mutation Testing**: Using Stryker for test quality
2. **Property-Based Testing**: For causal algorithms
3. **Chaos Engineering**: Resilience testing
4. **AI-Assisted Testing**: Automated test generation

### Research Areas

- Causal reasoning test oracles
- LLM evaluation methodologies
- Interactive system testing
- Scientific computing validation

---

*This testing strategy is a living document that evolves with the project. Regular updates ensure it remains relevant and effective.*