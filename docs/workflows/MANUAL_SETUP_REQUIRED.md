# Manual Setup Required for GitHub Workflows

## Overview

Due to GitHub App permission limitations, the following GitHub Actions workflows must be manually created by repository maintainers. This document provides all necessary templates and setup instructions.

## Required Actions

### 1. Create Workflow Files

Copy the following template files from `docs/workflows/examples/` to `.github/workflows/`:

```bash
mkdir -p .github/workflows
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/deploy.yml .github/workflows/
```

### 2. Configure Repository Secrets

Add the following secrets in Repository Settings → Secrets and Variables → Actions:

#### Required Secrets
- `OPENAI_API_KEY` - OpenAI API access for LLM integration
- `ANTHROPIC_API_KEY` - Anthropic API access for Claude integration
- `DOCKER_REGISTRY_TOKEN` - GitHub Container Registry access
- `SLACK_WEBHOOK_URL` - Slack notifications (optional)
- `GRAFANA_API_KEY` - Monitoring dashboard updates (optional)

#### Production Deployment Secrets
- `STAGING_DEPLOY_KEY` - Staging environment SSH key
- `PROD_DEPLOY_KEY` - Production environment SSH key
- `KUBECONFIG` - Kubernetes cluster configuration (if using K8s)

### 3. Setup Branch Protection Rules

Configure branch protection for `main`:

1. Go to Repository Settings → Branches
2. Add rule for `main` branch
3. Configure the following:
   - ✅ Require a pull request before merging
   - ✅ Require approvals (1 reviewer minimum)
   - ✅ Dismiss stale PR approvals when new commits are pushed
   - ✅ Require review from code owners
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Require conversation resolution before merging
   - ✅ Restrict pushes that create files that could be executable

#### Required Status Checks
- `Lint and Format`
- `Frontend Tests`
- `Backend Tests`
- `Integration Tests`
- `Security Scan`
- `Merge Requirement`

### 4. Configure Deployment Environments

Create deployment environments in Repository Settings → Environments:

#### Staging Environment
- **Name**: `staging`
- **Deployment URL**: `https://staging.causal-ui-gym.com`
- **Protection Rules**:
  - ✅ Required reviewers: 1 maintainer
  - ✅ Wait timer: 0 minutes
  - ✅ Deployment branches: Selected branches → `main`

#### Production Environment
- **Name**: `production`
- **Deployment URL**: `https://causal-ui-gym.com`
- **Protection Rules**:
  - ✅ Required reviewers: 2 maintainers
  - ✅ Wait timer: 5 minutes
  - ✅ Deployment branches: Selected tags → `v*`

### 5. Setup Code Owners

Create `.github/CODEOWNERS` file:

```
# Global owners
* @danieleschmidt

# Frontend code
/src/ @frontend-team
/package.json @frontend-team
/vite.config.ts @frontend-team

# Backend code
/causal_ui_gym/ @backend-team
/requirements.txt @backend-team
/pyproject.toml @backend-team

# Infrastructure
/docker-compose*.yml @devops-team
/Dockerfile @devops-team
/nginx/ @devops-team
/monitoring/ @devops-team

# Documentation
/docs/ @documentation-team
*.md @documentation-team

# Security-sensitive files
/.github/ @security-team @danieleschmidt
/scripts/ @security-team @devops-team
```

### 6. Additional Workflow Templates

The repository includes templates for additional workflows that can be implemented:

#### Security Scanning (`security.yml`)
- Daily vulnerability scans
- Dependency security checks
- Container image scanning
- SAST/DAST analysis

#### End-to-End Testing (`e2e.yml`)
- Playwright browser testing
- Visual regression testing
- Performance testing
- Accessibility compliance

#### Release Management (`release.yml`)
- Automated semantic versioning
- Changelog generation
- Package publishing
- GitHub release creation

### 7. Notification Setup

#### Slack Integration
1. Create Slack app in your workspace
2. Add incoming webhooks
3. Configure webhook URL in repository secrets
4. Test notifications

#### Email Notifications
Configure in Repository Settings → Notifications:
- ✅ Actions - Send notifications for workflow runs on this repository
- ✅ Dependabot alerts
- ✅ Security alerts

## Workflow Permissions

Ensure the repository has the following permissions configured:

### Actions Permissions
- Repository Settings → Actions → General
- ✅ Allow all actions and reusable workflows
- ✅ Allow actions created by GitHub
- ✅ Allow actions by Marketplace verified creators

### Token Permissions
Update workflow files to include necessary permissions:

```yaml
permissions:
  contents: read
  security-events: write
  packages: write
  deployments: write
  pull-requests: write
```

## Validation Checklist

After setup, verify the following:

- [ ] All workflow files are in `.github/workflows/`
- [ ] Required secrets are configured
- [ ] Branch protection rules are active
- [ ] Deployment environments are configured
- [ ] Code owners file is present
- [ ] First CI run completes successfully
- [ ] Staging deployment works
- [ ] Production deployment requires proper approvals

## Support

For issues with workflow setup:

1. Check GitHub Actions documentation
2. Review workflow run logs
3. Validate secret configurations
4. Test with minimal workflow first
5. Contact repository maintainers

## Security Considerations

⚠️ **Important Security Notes**:

- Never commit secrets to the repository
- Use least-privilege access for service accounts
- Regularly rotate API keys and tokens
- Monitor workflow runs for suspicious activity
- Review third-party actions before use
- Enable dependency scanning and security alerts

---

*This setup is required due to GitHub App permission limitations. Repository maintainers must complete these steps manually.*