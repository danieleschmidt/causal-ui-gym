# Advanced CI/CD Workflows

This document provides comprehensive GitHub Actions workflows for advanced CI/CD automation, security, and deployment strategies.

## Workflow Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Code Push     │───▶│   CI Pipeline   │───▶│   CD Pipeline   │
│                 │    │   - Test        │    │   - Build       │
│                 │    │   - Lint        │    │   - Deploy      │
│                 │    │   - Security    │    │   - Monitor     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core CI Pipeline

### Main CI Workflow

```yaml
# .github/workflows/ci.yml
name: Continuous Integration

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  workflow_dispatch:
    inputs:
      run_performance_tests:
        description: 'Run performance tests'
        required: false
        default: false
        type: boolean

env:
  NODE_VERSION: '20'
  PYTHON_VERSION: '3.11'
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  # ==========================================
  # Code Quality and Static Analysis
  # ==========================================
  quality:
    name: Code Quality
    runs-on: ubuntu-latest
    outputs:
      cache-key: ${{ steps.cache-key.outputs.key }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for better analysis

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Generate cache key
        id: cache-key
        run: |
          echo "key=deps-${{ runner.os }}-node${{ env.NODE_VERSION }}-python${{ env.PYTHON_VERSION }}-${{ hashFiles('**/package-lock.json', '**/requirements.txt') }}" >> $GITHUB_OUTPUT

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: |
            ~/.npm
            ~/.cache/pip
            node_modules
          key: ${{ steps.cache-key.outputs.key }}
          restore-keys: |
            deps-${{ runner.os }}-node${{ env.NODE_VERSION }}-python${{ env.PYTHON_VERSION }}-

      - name: Install dependencies
        run: |
          npm ci
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: TypeScript type checking
        run: npm run typecheck

      - name: ESLint
        run: npm run lint -- --format=@microsoft/eslint-formatter-sarif --output-file=eslint-results.sarif

      - name: Prettier format check
        run: npm run format:check

      - name: Python linting (flake8)
        run: flake8 backend/ --format=sarif --output-file=flake8-results.sarif

      - name: Python type checking (mypy)
        run: mypy backend/ --junit-xml=mypy-results.xml

      - name: Upload lint results to GitHub Security
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: |
            eslint-results.sarif
            flake8-results.sarif

      - name: SonarCloud Scan
        uses: SonarSource/sonarcloud-github-action@master
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}

  # ==========================================
  # Security Scanning
  # ==========================================
  security:
    name: Security Scan
    runs-on: ubuntu-latest
    needs: quality
    permissions:
      security-events: write
      contents: read
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup dependencies
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: |
            ~/.npm
            ~/.cache/pip
            node_modules
          key: ${{ needs.quality.outputs.cache-key }}

      - name: Install dependencies
        run: npm ci

      - name: npm audit
        run: npm audit --audit-level=moderate --json > npm-audit.json
        continue-on-error: true

      - name: Snyk Security Scan
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          args: --severity-threshold=high --sarif-file-output=snyk.sarif

      - name: Python Security Scan (Bandit)
        run: |
          pip install bandit[toml]
          bandit -r backend/ -f sarif -o bandit-results.sarif

      - name: Python Dependency Scan (Safety)
        run: |
          pip install safety
          safety check --json > safety-results.json
        continue-on-error: true

      - name: Secrets Scanning (TruffleHog)
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: main
          head: HEAD
          extra_args: --debug --only-verified

      - name: CodeQL Analysis
        uses: github/codeql-action/init@v3
        with:
          languages: javascript, python
          queries: security-extended,security-and-quality

      - name: CodeQL Autobuild
        uses: github/codeql-action/autobuild@v3

      - name: CodeQL Analysis
        uses: github/codeql-action/analyze@v3

      - name: Upload security scan results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: |
            snyk.sarif
            bandit-results.sarif

  # ==========================================
  # Unit and Integration Tests
  # ==========================================
  test:
    name: Test Suite
    runs-on: ubuntu-latest
    needs: quality
    strategy:
      matrix:
        test-group: [frontend, backend, integration]
      fail-fast: false
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        if: matrix.test-group == 'frontend' || matrix.test-group == 'integration'
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'

      - name: Setup Python
        if: matrix.test-group == 'backend' || matrix.test-group == 'integration'
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: |
            ~/.npm
            ~/.cache/pip
            node_modules
          key: ${{ needs.quality.outputs.cache-key }}

      - name: Install dependencies
        run: |
          if [[ "${{ matrix.test-group }}" == "frontend" || "${{ matrix.test-group }}" == "integration" ]]; then
            npm ci
          fi
          if [[ "${{ matrix.test-group }}" == "backend" || "${{ matrix.test-group }}" == "integration" ]]; then
            pip install -r requirements.txt
            pip install -r requirements-dev.txt
          fi

      - name: Frontend Tests
        if: matrix.test-group == 'frontend'
        run: |
          npm run test -- --coverage --reporter=json --outputFile=frontend-test-results.json
          npm run test:coverage

      - name: Backend Tests
        if: matrix.test-group == 'backend'
        run: |
          pytest backend/ --cov=backend --cov-report=xml --cov-report=json --junit-xml=backend-test-results.xml

      - name: Integration Tests
        if: matrix.test-group == 'integration'
        run: |
          # Start services for integration testing
          docker-compose -f docker-compose.test.yml up -d
          sleep 30
          
          # Run integration tests
          npm run test:integration
          pytest tests/integration/ --junit-xml=integration-test-results.xml

      - name: Upload test results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-results-${{ matrix.test-group }}
          path: |
            *-test-results.*
            coverage/
            htmlcov/

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        if: always()
        with:
          files: coverage/coverage-final.json,coverage.xml
          flags: ${{ matrix.test-group }}
          name: ${{ matrix.test-group }}-coverage

  # ==========================================
  # End-to-End Tests
  # ==========================================
  e2e:
    name: E2E Tests
    runs-on: ubuntu-latest
    needs: [quality, test]
    if: github.event_name == 'push' || github.event.inputs.run_performance_tests == 'true'
    strategy:
      matrix:
        browser: [chromium, firefox, webkit]
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: |
            ~/.npm
            node_modules
          key: ${{ needs.quality.outputs.cache-key }}

      - name: Install dependencies
        run: npm ci

      - name: Install Playwright
        run: npx playwright install --with-deps ${{ matrix.browser }}

      - name: Build application
        run: npm run build

      - name: Start application
        run: |
          npm run preview &
          sleep 10

      - name: Run E2E tests
        run: npx playwright test --project=${{ matrix.browser }}

      - name: Upload E2E results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: e2e-results-${{ matrix.browser }}
          path: |
            playwright-report/
            test-results/

  # ==========================================
  # Performance Tests
  # ==========================================
  performance:
    name: Performance Tests
    runs-on: ubuntu-latest
    needs: [quality, test]
    if: github.event.inputs.run_performance_tests == 'true' || github.ref == 'refs/heads/main'
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Build application
        run: npm run build

      - name: Install k6
        run: |
          sudo gpg -k
          sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
          echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
          sudo apt-get update
          sudo apt-get install k6

      - name: Start application
        run: |
          npm run preview &
          sleep 10

      - name: Run performance tests
        run: k6 run tests/performance/k6-load-test.js --out json=performance-results.json

      - name: Performance regression check
        run: |
          # Compare with baseline (implementation depends on your needs)
          node scripts/check-performance-regression.js performance-results.json

      - name: Upload performance results
        uses: actions/upload-artifact@v4
        with:
          name: performance-results
          path: performance-results.json

  # ==========================================
  # Build and Package
  # ==========================================
  build:
    name: Build Application
    runs-on: ubuntu-latest
    needs: [quality, security, test]
    outputs:
      image-digest: ${{ steps.build.outputs.digest }}
      image-tags: ${{ steps.meta.outputs.tags }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha,prefix={{branch}}-

      - name: Build and push Docker image
        id: build
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./Dockerfile.secure
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64,linux/arm64
          provenance: true
          sbom: true

      - name: Generate SBOM
        uses: anchore/sbom-action@v0
        with:
          image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}

      - name: Scan image with Trivy
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'trivy-results.sarif'

  # ==========================================
  # Deploy to Staging
  # ==========================================
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: [build, e2e]
    if: github.ref == 'refs/heads/main'
    environment:
      name: staging
      url: https://staging.causal-ui-gym.dev
    steps:
      - name: Deploy to staging
        run: |
          # Implementation depends on your infrastructure
          echo "Deploying ${{ needs.build.outputs.image-tags }} to staging"
          # kubectl set image deployment/causal-ui-gym causal-ui-gym=${{ needs.build.outputs.image-tags }}

      - name: Run smoke tests
        run: |
          # Basic smoke tests after deployment
          curl -f https://staging.causal-ui-gym.dev/health || exit 1

      - name: Notify deployment
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          channel: '#deployments'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}

  # ==========================================
  # Release Gate
  # ==========================================
  release-gate:
    name: Release Gate
    runs-on: ubuntu-latest
    needs: [deploy-staging, performance]
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Check deployment health
        run: |
          # Automated health checks
          curl -f https://staging.causal-ui-gym.dev/health || exit 1
          
      - name: Performance gate
        run: |
          # Check if performance meets thresholds
          node scripts/performance-gate.js

      - name: Security gate
        run: |
          # Check for critical security issues
          echo "Checking security scan results..."

      - name: Approve for production
        run: echo "All gates passed - ready for production deployment"
```

### Production Deployment Workflow

```yaml
# .github/workflows/deploy-production.yml
name: Production Deployment

on:
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      image_tag:
        description: 'Image tag to deploy'
        required: true
        type: string

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://causal-ui-gym.dev
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup kubectl
        uses: azure/setup-kubectl@v3
        with:
          version: 'v1.28.0'

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-west-2

      - name: Update kubeconfig
        run: aws eks update-kubeconfig --name production-cluster

      - name: Blue-Green Deployment
        run: |
          # Blue-Green deployment script
          IMAGE_TAG=${{ github.event.inputs.image_tag || github.event.release.tag_name }}
          IMAGE="${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${IMAGE_TAG}"
          
          # Deploy to green environment
          kubectl set image deployment/causal-ui-gym-green causal-ui-gym=${IMAGE}
          kubectl rollout status deployment/causal-ui-gym-green --timeout=600s
          
          # Health check on green
          kubectl run smoke-test --image=curlimages/curl --rm -i --restart=Never -- \
            curl -f http://causal-ui-gym-green-service/health
          
          # Switch traffic to green
          kubectl patch service causal-ui-gym-service -p '{"spec":{"selector":{"version":"green"}}}'
          
          # Wait and verify
          sleep 60
          kubectl run final-check --image=curlimages/curl --rm -i --restart=Never -- \
            curl -f https://causal-ui-gym.dev/health

      - name: Rollback on failure
        if: failure()
        run: |
          echo "Deployment failed, rolling back..."
          kubectl patch service causal-ui-gym-service -p '{"spec":{"selector":{"version":"blue"}}}'

      - name: Update monitoring
        run: |
          # Update monitoring dashboards with new version
          curl -X POST "${{ secrets.GRAFANA_API_URL }}/api/annotations" \
            -H "Authorization: Bearer ${{ secrets.GRAFANA_API_KEY }}" \
            -H "Content-Type: application/json" \
            -d '{
              "text": "Production deployment: ${{ github.event.release.tag_name }}",
              "tags": ["deployment", "production"]
            }'

      - name: Notify success
        uses: 8398a7/action-slack@v3
        with:
          status: success
          channel: '#production'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
          text: |
            🚀 Production deployment successful!
            Version: ${{ github.event.release.tag_name }}
            URL: https://causal-ui-gym.dev
```

### Security and Compliance Workflow

```yaml
# .github/workflows/security-compliance.yml
name: Security and Compliance

on:
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM UTC
  workflow_dispatch:

jobs:
  dependency-scan:
    name: Dependency Security Scan
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          npm ci
          pip install -r requirements.txt

      - name: OWASP Dependency Check
        uses: dependency-check/Dependency-Check_Action@main
        with:
          project: 'causal-ui-gym'
          path: '.'
          format: 'JSON'
          args: >
            --enableRetired
            --enableExperimental
            --suppression suppression.xml

      - name: Upload OWASP results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: reports/dependency-check-report.sarif

  container-scan:
    name: Container Security Scan
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Build image
        run: docker build -f Dockerfile.secure -t causal-ui-gym:scan .

      - name: Run Trivy scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'causal-ui-gym:scan'
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Run Snyk Container scan
        uses: snyk/actions/docker@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
        with:
          image: causal-ui-gym:scan

  compliance-check:
    name: Compliance Check
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: SLSA Provenance
        uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.9.0
        with:
          base64-subjects: ${{ steps.hash.outputs.hashes }}

      - name: Generate compliance report
        run: |
          # Generate compliance documentation
          node scripts/generate-compliance-report.js > compliance-report.md

      - name: Upload compliance artifacts
        uses: actions/upload-artifact@v4
        with:
          name: compliance-report
          path: compliance-report.md
```

## Workflow Utilities

### Reusable Workflows

```yaml
# .github/workflows/reusable-security-scan.yml
name: Reusable Security Scan

on:
  workflow_call:
    inputs:
      image-name:
        required: true
        type: string
      severity-threshold:
        required: false
        type: string
        default: 'HIGH'
    outputs:
      scan-results:
        description: 'Security scan results'
        value: ${{ jobs.security-scan.outputs.results }}

jobs:
  security-scan:
    runs-on: ubuntu-latest
    outputs:
      results: ${{ steps.scan.outputs.results }}
    steps:
      - name: Run Trivy scan
        id: scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ inputs.image-name }}
          severity: ${{ inputs.severity-threshold }}
          format: 'json'
```

### Custom Actions

```yaml
# .github/actions/setup-environment/action.yml
name: 'Setup Environment'
description: 'Setup Node.js, Python, and dependencies'
inputs:
  node-version:
    description: 'Node.js version'
    required: false
    default: '20'
  python-version:
    description: 'Python version'
    required: false
    default: '3.11'
runs:
  using: 'composite'
  steps:
    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: ${{ inputs.node-version }}
        cache: 'npm'
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ inputs.python-version }}
        cache: 'pip'
    
    - name: Install dependencies
      shell: bash
      run: |
        npm ci
        pip install -r requirements.txt
```

## Deployment Strategies

### Canary Deployment

```yaml
# .github/workflows/canary-deployment.yml
name: Canary Deployment

on:
  workflow_dispatch:
    inputs:
      percentage:
        description: 'Canary traffic percentage'
        required: true
        default: '10'
        type: choice
        options: ['10', '25', '50', '100']

jobs:
  canary-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy Canary
        run: |
          # Implement canary deployment logic
          kubectl patch deployment causal-ui-gym-canary -p \
            '{"spec":{"template":{"metadata":{"labels":{"version":"canary"}}}}}'
          
          # Update traffic split
          kubectl patch virtualservice causal-ui-gym -p \
            '{"spec":{"http":[{"match":[{"headers":{"canary":{"exact":"true"}}}],"route":[{"destination":{"host":"causal-ui-gym","subset":"canary"}}]},{"route":[{"destination":{"host":"causal-ui-gym","subset":"stable"},"weight":$((100-${{ github.event.inputs.percentage }}))},{"destination":{"host":"causal-ui-gym","subset":"canary"},"weight":${{ github.event.inputs.percentage }}}]}]}}'

      - name: Monitor Canary
        run: |
          # Monitor metrics for 10 minutes
          sleep 600
          
          # Check error rates
          ERROR_RATE=$(kubectl exec -n monitoring deployment/prometheus -- \
            promtool query instant 'rate(http_requests_total{status=~"5.."}[5m])')
          
          if (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
            echo "High error rate detected, rolling back"
            exit 1
          fi

      - name: Promote or Rollback
        run: |
          if [ "${{ github.event.inputs.percentage }}" == "100" ]; then
            # Full promotion
            kubectl patch service causal-ui-gym -p \
              '{"spec":{"selector":{"version":"canary"}}}'
          fi
```

## Monitoring and Observability

### Deployment Monitoring

```yaml
# .github/workflows/deployment-monitoring.yml
name: Deployment Health Check

on:
  deployment_status:

jobs:
  health-check:
    if: github.event.deployment_status.state == 'success'
    runs-on: ubuntu-latest
    steps:
      - name: Wait for deployment stabilization
        run: sleep 60

      - name: Health check
        run: |
          for i in {1..10}; do
            if curl -f ${{ github.event.deployment.payload.web_url }}/health; then
              echo "Health check passed"
              break
            fi
            echo "Health check failed, attempt $i/10"
            sleep 30
          done

      - name: Performance check
        run: |
          # Run performance tests against new deployment
          k6 run --env URL=${{ github.event.deployment.payload.web_url }} \
            tests/performance/post-deployment.js

      - name: Update monitoring
        run: |
          curl -X POST "${{ secrets.DATADOG_API_URL }}/api/v1/events" \
            -H "DD-API-KEY: ${{ secrets.DATADOG_API_KEY }}" \
            -d '{
              "title": "Deployment Complete",
              "text": "Successfully deployed to ${{ github.event.deployment.environment }}",
              "tags": ["deployment", "${{ github.event.deployment.environment }}"]
            }'
```

## Best Practices Summary

### Security
- Use OIDC for authentication instead of long-lived tokens
- Scan all dependencies and container images
- Implement least privilege access
- Use signed commits and verified builds

### Performance
- Use caching extensively
- Parallelize jobs where possible
- Use matrix builds for different environments
- Implement proper artifact management

### Reliability
- Use timeouts and retries
- Implement proper error handling
- Use environment protection rules
- Monitor workflow performance

### Maintainability
- Use reusable workflows and actions
- Document all custom workflows
- Implement proper versioning
- Use semantic commit messages

---

*These advanced workflows provide comprehensive CI/CD automation with security, monitoring, and deployment best practices integrated throughout the development lifecycle.*