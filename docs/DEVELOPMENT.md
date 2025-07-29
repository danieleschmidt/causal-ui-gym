# Development Guide

This guide will help you set up your development environment and understand the project structure.

## Prerequisites

- Node.js 18+ and npm
- Python 3.9+ and pip
- Git

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/causal-ui-gym.git
cd causal-ui-gym

# Install dependencies
npm install
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Start development servers
npm run dev              # Frontend on :5173
# python -m causal_ui_gym.server  # Backend on :8000 (when implemented)
```

## Project Structure

```
causal-ui-gym/
├── src/                    # Frontend source code
│   ├── components/         # React components
│   ├── types/             # TypeScript type definitions
│   ├── utils/             # Utility functions
│   └── test/              # Test setup files
├── tests/                 # Test files and fixtures
├── docs/                  # Documentation
└── backend/               # Python JAX backend (to be implemented)
```

## Development Scripts

```bash
# Development
npm run dev                 # Start dev server with hot reload
npm run build              # Build for production
npm run preview            # Preview production build

# Testing
npm test                   # Run unit tests
npm run test:ui            # Run tests with UI
npm run test:e2e           # Run E2E tests

# Code Quality
npm run lint               # Run ESLint
npm run lint:fix           # Fix ESLint issues
npm run typecheck          # Run TypeScript type checking
npm run format             # Format code with Prettier
npm run format:check       # Check formatting

# Storybook
npm run storybook          # Start Storybook dev server
npm run build-storybook    # Build Storybook
```

## Code Style

- Use TypeScript for all new code
- Follow the existing ESLint and Prettier configuration
- Write tests for new functionality
- Add JSDoc comments for public APIs
- Use conventional commit messages

## Testing

### Unit Tests
Using Vitest with React Testing Library:

```typescript
import { render, screen } from '@testing-library/react'
import { CausalGraph } from './CausalGraph'

test('renders causal graph', () => {
  const nodes = [{ id: 'A', label: 'Node A', position: { x: 0, y: 0 } }]
  const edges = []
  
  render(<CausalGraph nodes={nodes} edges={edges} />)
  expect(screen.getByText('Node A')).toBeInTheDocument()
})
```

### E2E Tests
Using Playwright for end-to-end testing:

```typescript
import { test, expect } from '@playwright/test'

test('intervention updates graph', async ({ page }) => {
  await page.goto('/experiments/supply-demand')
  await page.click('[data-testid="intervention-button"]')
  await expect(page.locator('[data-testid="graph"]')).toHaveClass(/updated/)
})
```

## Adding New Components

1. Create the component in `src/components/`
2. Add TypeScript interfaces in `src/types/`
3. Export from `src/components/index.ts`
4. Write unit tests
5. Add Storybook stories (when Storybook is configured)
6. Update documentation

## Working with Causal Models

The framework uses a standardized format for causal DAGs:

```typescript
interface CausalDAG {
  nodes: CausalNode[]
  edges: CausalEdge[]
}

interface CausalNode {
  id: string
  label: string
  position: { x: number; y: number }
}

interface CausalEdge {
  source: string
  target: string
  weight?: number
}
```

## Pre-commit Hooks

We use pre-commit hooks to maintain code quality:

- ESLint for JavaScript/TypeScript linting
- Prettier for code formatting
- Black and isort for Python formatting
- MyPy for Python type checking
- Security checks for secrets detection

## Troubleshooting

### Common Issues

**Node modules issues:**
```bash
rm -rf node_modules package-lock.json
npm install
```

**TypeScript errors:**
```bash
npm run typecheck
# Fix reported errors
```

**Pre-commit failures:**
```bash
pre-commit run --all-files
# Fix reported issues
```

### Getting Help

- Check existing issues on GitHub
- Join our Discord community
- Read the [Contributing Guide](../CONTRIBUTING.md)

## Release Process

1. Update version in `package.json`
2. Update `CHANGELOG.md`
3. Create release PR
4. Tag release after merge
5. Automated deployment handles the rest