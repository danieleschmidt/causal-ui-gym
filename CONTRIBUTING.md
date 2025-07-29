# Contributing to Causal UI Gym

We welcome contributions! This guide will help you get started.

## 🎯 Ways to Contribute

- **Bug reports**: Found a bug? Let us know!
- **Feature requests**: Have an idea? Share it with us!
- **Code contributions**: Fix bugs, add features, improve documentation
- **Documentation**: Help improve our docs and examples
- **Testing**: Add test cases, improve test coverage

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ and npm
- Python 3.9+ and pip
- Git

### Setup Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/causal-ui-gym.git
cd causal-ui-gym

# Install frontend dependencies
npm install

# Install backend dependencies
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
npm test
python -m pytest
```

## 📋 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Follow existing code style and conventions
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Commit Changes

We use [Conventional Commits](https://conventionalcommits.org/):

```bash
git commit -m "feat: add causal graph visualization component"
git commit -m "fix: resolve intervention tracking bug"
git commit -m "docs: update API documentation"
```

### 4. Submit Pull Request

- Push your branch to your fork
- Create a pull request with clear description
- Link any related issues
- Ensure CI passes

## 🏗️ Project Structure

```
causal-ui-gym/
├── src/                    # Frontend React components
├── backend/                # Python JAX backend
├── tests/                  # Test files
├── docs/                   # Documentation
├── examples/               # Example experiments
└── figma-plugin/          # Figma integration
```

## 🧪 Testing

### Frontend Tests

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run with coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e
```

### Backend Tests

```bash
# Run Python tests
python -m pytest

# Run with coverage
python -m pytest --cov=causal_ui_gym

# Run specific test file
python -m pytest tests/test_causal_engine.py
```

## 📝 Code Style

### Frontend (TypeScript/React)

- Use TypeScript for all new code
- Follow existing component patterns
- Use functional components with hooks
- Add prop types and documentation

```typescript
interface CausalGraphProps {
  nodes: CausalNode[]
  edges: CausalEdge[]
  onIntervene?: (node: string, value: number) => void
}

export function CausalGraph({ nodes, edges, onIntervene }: CausalGraphProps) {
  // Component implementation
}
```

### Backend (Python)

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Add docstrings for public APIs
- Use JAX for numerical computations

```python
def compute_intervention(
    dag: CausalDAG, 
    intervention: Dict[str, float],
    evidence: Dict[str, float]
) -> Dict[str, float]:
    """Compute intervention effects using do-calculus.
    
    Args:
        dag: Causal directed acyclic graph
        intervention: Variables to intervene on
        evidence: Observed evidence
        
    Returns:
        Posterior distribution after intervention
    """
    # Implementation
```

## 🐛 Bug Reports

When reporting bugs, please include:

- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Node.js version, etc.)
- Screenshots if applicable

Use our [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).

## 💡 Feature Requests

For feature requests, please provide:

- Clear description of the feature
- Use case and motivation
- Possible implementation approach
- Any related examples or references

## 📚 Documentation

- Update README.md for significant changes
- Add JSDoc comments for TypeScript functions
- Add docstrings for Python functions
- Update API documentation
- Add examples for new features

## 🔄 Release Process

1. Version bump following [Semantic Versioning](https://semver.org/)
2. Update CHANGELOG.md
3. Create release PR
4. Tag release after merge
5. Automated deployment to npm and PyPI

## 🤝 Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Provide constructive feedback
- Follow our [Code of Conduct](CODE_OF_CONDUCT.md)

## 📞 Getting Help

- **Discord**: Join our [community server](https://discord.gg/causal-ui)
- **GitHub Discussions**: For questions and ideas
- **Issues**: For bug reports and feature requests
- **Email**: causal-ui@yourdomain.com

## 🏆 Recognition

Contributors are recognized in:
- README.md contributors section
- Release notes
- Annual contributor spotlight

Thank you for contributing to Causal UI Gym! 🎉