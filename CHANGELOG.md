# Changelog

All notable changes to the Causal UI Gym project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- SDLC checkpoint implementation strategy
- Comprehensive project documentation structure
- Architecture Decision Records (ADR) framework
- Project charter and roadmap documentation

### Changed
- Enhanced README with comprehensive framework overview
- Improved project structure documentation

### Deprecated
- Nothing

### Removed
- Nothing

### Fixed
- Nothing

### Security
- Nothing

## [0.1.0] - 2025-01-XX

### Added
- Initial project structure with React + TypeScript frontend
- JAX backend foundation for causal computations
- Basic causal graph visualization components
- Testing infrastructure with Vitest and Playwright
- Development tooling (ESLint, Prettier, TypeScript)
- Docker containerization setup
- Initial documentation framework
- Security and monitoring configurations

### Dependencies
- React 18.2+ for UI components
- JAX 0.4.28+ for numerical computations
- TypeScript 5.0+ for type safety
- Material-UI 5.15+ for design system
- D3.js 7.9+ for graph visualizations

---

## Release Notes Template

For future releases, use this template:

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New features and capabilities

### Changed
- Changes to existing functionality

### Deprecated
- Features that will be removed in future versions

### Removed
- Features that have been removed

### Fixed
- Bug fixes and corrections

### Security
- Security-related changes and fixes
```

---

## Maintenance Guidelines

1. **Update Frequency**: Update changelog with every release
2. **Entry Format**: Use present tense, imperative mood ("Add feature" not "Added feature")
3. **Version Links**: Link version numbers to release tags
4. **Categories**: Always include all categories, mark as "Nothing" if empty
5. **Detail Level**: Include enough detail for users to understand impact

## Contributing

When contributing changes:
1. Add your changes to the "Unreleased" section
2. Use the appropriate category (Added, Changed, Fixed, etc.)
3. Include breaking changes in the description
4. Reference issue numbers where applicable

Example:
```markdown
### Added
- New CausalGraph component with intervention controls (#123)
```