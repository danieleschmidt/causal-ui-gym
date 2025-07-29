# Contributing to Causal UI Gym

We welcome contributions! This guide will help you get started with contributing to the Causal UI Gym project.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/causal-ui-gym.git
   cd causal-ui-gym
   ```
3. **Set up the development environment**:
   ```bash
   # Frontend dependencies
   npm install
   
   # Backend dependencies
   pip install -r requirements.txt
   pip install -e ".[dev]"
   
   # Install pre-commit hooks
   pre-commit install
   ```

## 🛠️ Development Workflow

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Run tests and linting**:
   ```bash
   # Frontend
   npm run lint
   npm run typecheck
   npm test
   
   # Backend
   black .
   isort .
   flake8
   pytest
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create a Pull Request**

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for formatting changes
- `refactor:` for code refactoring
- `test:` for adding tests
- `chore:` for maintenance tasks

## 🎯 Areas for Contribution

### High Priority
- **New experiment templates** - Create reusable causal reasoning experiments
- **Additional causal metrics** - Implement new metrics for LLM evaluation
- **Performance optimizations** - Improve JAX backend efficiency
- **Documentation improvements** - Enhance guides and API docs

### Medium Priority
- **Figma plugin enhancements** - Extend design-to-experiment workflow
- **LLM agent implementations** - Add support for new models
- **Testing coverage** - Expand test suite
- **Accessibility improvements** - Enhance UI accessibility

### Getting Started Ideas
- Fix typos in documentation
- Add type annotations
- Improve error messages
- Add example experiments

## 📋 Pull Request Guidelines

### Before Submitting
- [ ] Code follows project style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated if needed
- [ ] Commit messages follow conventional format
- [ ] No merge conflicts with main branch

### PR Template
When creating a PR, please include:

1. **Description** - What does this PR do?
2. **Motivation** - Why is this change needed?
3. **Testing** - How was this tested?
4. **Screenshots** - For UI changes
5. **Breaking Changes** - Any backwards compatibility issues?

## 🧪 Testing

### Frontend Testing
```bash
# Unit tests
npm test

# E2E tests
npm run test:e2e

# Visual regression tests
npm run test:visual
```

### Backend Testing
```bash
# Unit tests
pytest

# Coverage
pytest --cov=causal_ui_gym

# Integration tests
pytest tests/integration/
```

## 📝 Documentation

### Code Documentation
- Use TypeScript types for all frontend code
- Add docstrings to Python functions
- Include inline comments for complex logic
- Update README when adding features

### API Documentation
- Update OpenAPI specs for backend changes
- Add Storybook stories for new components
- Include usage examples in docstrings

## 🔒 Security

### Reporting Vulnerabilities
- **DO NOT** create public issues for security vulnerabilities
- Email security concerns to: security@causal-ui-gym.dev
- Include detailed reproduction steps

### Security Guidelines
- Never commit secrets or API keys
- Use environment variables for configuration
- Validate all user inputs
- Follow OWASP guidelines for web security

## 📞 Getting Help

### Community Channels
- **GitHub Discussions** - For questions and ideas
- **Discord** - Real-time chat ([invite link](https://discord.gg/causal-ui))
- **Issues** - Bug reports and feature requests

### Maintainer Contact
- **GitHub**: [@danieleschmidt](https://github.com/danieleschmidt)
- **Email**: daniel@causal-ui-gym.dev

## 🎉 Recognition

Contributors are recognized in:
- README contributors section
- Release notes
- Annual contributor spotlight
- Conference presentation acknowledgments

### Contributor Levels
- **First-time contributor** - Welcome package
- **Regular contributor** - Project stickers
- **Core contributor** - Direct maintainer access
- **Major contributor** - Conference speaking opportunities

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## ✨ Thank You!

Every contribution, no matter how small, helps make Causal UI Gym better for everyone. We appreciate your time and effort! 🙏