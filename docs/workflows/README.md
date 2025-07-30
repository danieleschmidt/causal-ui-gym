# GitHub Actions Workflows

This directory contains CI/CD workflow templates and documentation for the Causal UI Gym project.

## Overview

Due to security considerations, actual GitHub Actions workflows are documented here rather than implemented directly. Repository maintainers should manually create these workflows in `.github/workflows/` directory.

## Required Workflows

### 1. Continuous Integration (`ci.yml`)

**Purpose**: Run tests, linting, and security checks on all pull requests and pushes.

**Trigger Events**:
- Push to `main` branch
- Pull request creation/updates
- Manual workflow dispatch

**Jobs**:
- **Lint and Format**: ESLint, Prettier, Black, isort
- **Type Check**: TypeScript and mypy validation
- **Unit Tests**: Frontend (Vitest) and backend (pytest)
- **Security Scan**: Bandit, Safety, detect-secrets
- **Build Verification**: Ensure code builds successfully

**Required Secrets**:
- None (uses public tools only)

**Example trigger configuration**:
```yaml
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  workflow_dispatch:
```

### 2. End-to-End Testing (`e2e.yml`)

**Purpose**: Run comprehensive end-to-end tests using Playwright.

**Trigger Events**:
- Pull request to `main` branch
- Nightly schedule
- Manual dispatch

**Jobs**:
- **E2E Tests**: Multi-browser testing
- **Visual Regression**: Screenshot comparisons
- **Performance Tests**: Load testing with k6
- **Accessibility Tests**: WCAG compliance

**Required Secrets**:
- Test environment credentials

### 3. Security Scanning (`security.yml`)

**Purpose**: Comprehensive security analysis and vulnerability assessment.

**Trigger Events**:
- Push to `main` branch
- Schedule (daily)
- Manual dispatch

**Jobs**:
- **Dependency Scan**: Snyk, Safety, npm audit
- **Container Scan**: Trivy security scanning
- **SAST**: Static Application Security Testing
- **License Check**: License compliance verification

**Required Secrets**:
- `SNYK_TOKEN`: Snyk vulnerability scanning
- Container registry credentials

### 4. Build and Deploy (`deploy.yml`)

**Purpose**: Build and deploy application to various environments.

**Trigger Events**:
- Tag creation (production)
- Push to `main` (staging)
- Manual dispatch

**Jobs**:
- **Build**: Multi-stage Docker build
- **Deploy Staging**: Automated staging deployment
- **Deploy Production**: Manual approval required
- **Smoke Tests**: Post-deployment verification

**Required Secrets**:
- `DOCKER_REGISTRY_TOKEN`: Container registry access
- `STAGING_DEPLOY_KEY`: Staging environment access
- `PROD_DEPLOY_KEY`: Production environment access
- `OPENAI_API_KEY`: LLM service access
- `ANTHROPIC_API_KEY`: LLM service access

### 5. Release Management (`release.yml`)

**Purpose**: Automated release creation and changelog generation.

**Trigger Events**:
- Tag creation matching `v*`
- Manual dispatch

**Jobs**:
- **Create Release**: GitHub release creation
- **Generate Changelog**: Automated changelog
- **Publish Packages**: NPM and PyPI publishing
- **Update Documentation**: Version documentation

**Required Secrets**:
- `NPM_TOKEN`: NPM package publishing
- `PYPI_TOKEN`: PyPI package publishing
- `GITHUB_TOKEN`: GitHub release creation

## Implementation Guide

### Step 1: Create Workflow Files

Create the following files in `.github/workflows/`:

```bash
mkdir -p .github/workflows
```

### Step 2: Security Configuration

Set up required secrets in repository settings:

1. Go to Repository Settings → Secrets and Variables → Actions
2. Add all required secrets listed above
3. Configure environment protection rules

### Step 3: Branch Protection

Configure branch protection rules:

1. Go to Repository Settings → Branches
2. Add rule for `main` branch
3. Require status checks before merging
4. Require reviews from code owners
5. Restrict pushes to administrators

### Step 4: Environment Configuration

Set up deployment environments:

1. Go to Repository Settings → Environments
2. Create `staging` and `production` environments
3. Configure protection rules and secrets
4. Set up required reviewers for production

## Workflow Templates

### Basic CI Template

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: npm ci
      - run: pip install -r requirements.txt
      - run: npm run lint
      - run: npm run typecheck
      - run: npm run test
      - run: pytest
```

### Security Scan Template

```yaml
name: Security
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 6 * * *'

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Snyk
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
      - name: Run Bandit
        run: |
          pip install bandit
          bandit -r . -f json -o bandit-report.json
```

## Monitoring and Alerts

### Workflow Monitoring

- Set up notifications for workflow failures
- Monitor workflow performance and duration
- Track security scan results
- Alert on deployment failures

### Metrics Collection

- Workflow success/failure rates
- Test coverage trends
- Security vulnerability counts
- Deployment frequency and lead time

## Best Practices

### Security
- Use least privilege access
- Store secrets securely
- Audit workflow permissions
- Review third-party actions

### Performance
- Use caching effectively
- Parallelize jobs when possible
- Optimize Docker builds
- Monitor resource usage

### Reliability
- Handle failures gracefully
- Implement retry mechanisms
- Use stable action versions
- Test workflow changes

## Troubleshooting

### Common Issues

1. **Failed Tests**: Check test environment setup
2. **Security Scan Failures**: Review vulnerability reports
3. **Deployment Issues**: Verify secrets and permissions
4. **Build Failures**: Check dependency versions

### Debug Workflows

Enable debug logging:
```yaml
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Security Best Practices](https://docs.github.com/en/actions/security-guides)
- [Workflow Templates](https://github.com/actions/starter-workflows)

---

*Last Updated: January 2025*  
*Review: Required before implementing workflows*