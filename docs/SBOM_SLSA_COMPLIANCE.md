# SBOM Generation and SLSA Compliance

This document outlines the Software Bill of Materials (SBOM) generation and Supply-chain Levels for Software Artifacts (SLSA) compliance setup for Causal UI Gym.

## Overview

### SBOM (Software Bill of Materials)
A complete inventory of all software components, dependencies, and metadata used in building the application.

### SLSA (Supply-chain Levels for Software Artifacts)
A security framework that provides standards for securing the software supply chain.

## SBOM Generation

### Frontend (npm/Node.js)

#### Using CycloneDX
```bash
# Install CycloneDX generator
npm install -g @cyclonedx/cyclonedx-npm

# Generate SBOM in multiple formats
cyclonedx-npm --output-format json --output-file sbom-frontend.json
cyclonedx-npm --output-format xml --output-file sbom-frontend.xml
cyclonedx-npm --output-format csv --output-file sbom-frontend.csv
```

#### Using Syft (Alternative)
```bash
# Install Syft
curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin

# Generate SBOM for package.json
syft package.json -o spdx-json=sbom-frontend-syft.json
syft package.json -o cyclonedx-json=sbom-frontend-cyclonedx.json
```

### Backend (Python)

#### Using CycloneDX for Python
```bash
# Install CycloneDX Python generator
pip install cyclonedx-python-lib cyclonedx-bom

# Generate SBOM from requirements.txt
cyclonedx-py requirements -r requirements.txt -o sbom-backend.json --format json
cyclonedx-py requirements -r requirements.txt -o sbom-backend.xml --format xml
```

#### Using pip-audit
```bash
# Install pip-audit
pip install pip-audit

# Generate SBOM with vulnerability data
pip-audit --format=cyclonedx --output=sbom-backend-audit.json
```

### Container Images

#### Using Syft for Docker Images
```bash
# Generate SBOM for built Docker image
syft packages docker:causal-ui-gym:latest -o spdx-json=sbom-container.json
syft packages docker:causal-ui-gym:latest -o cyclonedx-json=sbom-container-cyclonedx.json
```

#### Using Docker SBOM (Experimental)
```bash
# Docker buildkit SBOM generation
docker buildx build --sbom=true -t causal-ui-gym:latest .
```

## SLSA Compliance Implementation

### Level 1: Documentation
- ✅ Document build process in DEVELOPMENT.md
- ✅ Version control all source code
- ✅ Generate provenance for all artifacts

### Level 2: Hosted Build Service
- ✅ Use GitHub Actions for all builds
- ✅ Authenticate all contributors
- ✅ Generate signed provenance

### Level 3: Hardened Builds
- 🔄 Use hardened build platform
- 🔄 Prevent secret extraction
- 🔄 Isolate build processes

### Level 4: Reproducible Builds
- 🔄 Hermetic builds
- 🔄 Reproducible outputs
- 🔄 Two-party review

## GitHub Actions Integration

### SBOM Generation Workflow

Create `.github/workflows/sbom-generation.yml`:

```yaml
name: SBOM Generation
on:
  push:
    branches: [main]
  release:
    types: [published]
  workflow_dispatch:

jobs:
  generate-sbom:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write  # For SLSA provenance
      
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: |
          npm ci
          pip install -r requirements.txt
          
      - name: Install SBOM tools
        run: |
          npm install -g @cyclonedx/cyclonedx-npm
          pip install cyclonedx-python-lib cyclonedx-bom pip-audit
          curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
          
      - name: Generate Frontend SBOM
        run: |
          cyclonedx-npm --output-format json --output-file sbom-frontend.json
          syft package.json -o spdx-json=sbom-frontend-spdx.json
          
      - name: Generate Backend SBOM
        run: |
          cyclonedx-py requirements -r requirements.txt -o sbom-backend.json --format json
          pip-audit --format=cyclonedx --output=sbom-backend-audit.json
          
      - name: Build Docker image
        run: |
          docker build -t causal-ui-gym:${{ github.sha }} .
          
      - name: Generate Container SBOM
        run: |
          syft packages docker:causal-ui-gym:${{ github.sha }} -o spdx-json=sbom-container.json
          
      - name: Combine SBOMs
        run: |
          # Create combined SBOM manifest
          cat > sbom-manifest.json << EOF
          {
            "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
            "commit": "${{ github.sha }}",
            "ref": "${{ github.ref }}",
            "components": {
              "frontend": "sbom-frontend.json",
              "backend": "sbom-backend.json",
              "container": "sbom-container.json"
            }
          }
          EOF
          
      - name: Upload SBOMs
        uses: actions/upload-artifact@v4
        with:
          name: sboms-${{ github.sha }}
          path: |
            sbom-*.json
            sbom-manifest.json
          retention-days: 90
          
      - name: Generate SLSA Provenance
        uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.9.0
        with:
          base64-subjects: ${{ steps.hash.outputs.hashes }}
          provenance-name: provenance-${{ github.sha }}.intoto.jsonl
```

### Container Signing and Attestation

```yaml
      - name: Install Cosign
        uses: sigstore/cosign-installer@v3
        
      - name: Sign container image
        run: |
          cosign sign --yes causal-ui-gym:${{ github.sha }}
          
      - name: Generate and attach SBOM attestation
        run: |
          cosign attest --yes --predicate sbom-container.json --type cyclonedx causal-ui-gym:${{ github.sha }}
          
      - name: Generate and attach SLSA provenance
        run: |
          cosign attest --yes --predicate provenance-${{ github.sha }}.intoto.jsonl --type slsaprovenance causal-ui-gym:${{ github.sha }}
```

## Verification and Validation

### SBOM Validation
```bash
# Validate SBOM format
cyclonedx validate --input-format json --input-file sbom-frontend.json

# Check for known vulnerabilities
grype sbom:sbom-frontend.json -o table
```

### SLSA Provenance Verification
```bash
# Install slsa-verifier
go install github.com/slsa-framework/slsa-verifier/v2/cli/slsa-verifier@latest

# Verify provenance
slsa-verifier verify-artifact --provenance-path provenance.intoto.jsonl --source-uri github.com/yourusername/causal-ui-gym causal-ui-gym:latest
```

### Container Image Verification
```bash
# Verify signature
cosign verify causal-ui-gym:latest --certificate-identity-regexp=".*" --certificate-oidc-issuer-regexp=".*"

# Verify SBOM attestation
cosign verify-attestation causal-ui-gym:latest --type cyclonedx --certificate-identity-regexp=".*" --certificate-oidc-issuer-regexp=".*"
```

## Automation Scripts

### SBOM Generation Script
```bash
#!/bin/bash
# scripts/generate-sbom.sh

set -e

echo "🔍 Generating Software Bill of Materials..."

# Create output directory
mkdir -p ./sbom-output

# Frontend SBOM
echo "📦 Generating Frontend SBOM..."
cyclonedx-npm --output-format json --output-file ./sbom-output/frontend.json
syft package.json -o spdx-json=./sbom-output/frontend-spdx.json

# Backend SBOM
echo "🐍 Generating Backend SBOM..."
cyclonedx-py requirements -r requirements.txt -o ./sbom-output/backend.json --format json
pip-audit --format=cyclonedx --output=./sbom-output/backend-audit.json

# Container SBOM (if image exists)
if docker image inspect causal-ui-gym:latest >/dev/null 2>&1; then
    echo "🐳 Generating Container SBOM..."
    syft packages docker:causal-ui-gym:latest -o spdx-json=./sbom-output/container.json
fi

echo "✅ SBOM generation complete. Files saved to ./sbom-output/"
```

### Vulnerability Scanning Script
```bash
#!/bin/bash
# scripts/scan-vulnerabilities.sh

set -e

echo "🔍 Scanning for vulnerabilities..."

# Scan SBOMs with Grype
grype sbom:./sbom-output/frontend.json -o table --fail-on medium
grype sbom:./sbom-output/backend.json -o table --fail-on medium

# Scan with Trivy
trivy fs . --format table --severity HIGH,CRITICAL

# Scan container (if available)
if docker image inspect causal-ui-gym:latest >/dev/null 2>&1; then
    trivy image causal-ui-gym:latest --severity HIGH,CRITICAL
fi

echo "✅ Vulnerability scanning complete."
```

## Compliance Checklist

- [ ] **SBOM Generation**: Automated SBOM generation for all components
- [ ] **Vulnerability Tracking**: Regular scanning and monitoring
- [ ] **Provenance Generation**: SLSA provenance for all builds
- [ ] **Digital Signatures**: Sign all artifacts
- [ ] **Attestations**: Attach SBOMs and provenance as attestations
- [ ] **Verification Tools**: Scripts to verify signatures and provenance
- [ ] **Documentation**: Complete documentation of supply chain practices
- [ ] **Monitoring**: Continuous monitoring for supply chain threats

## Tools and Dependencies

### Required Tools
- **CycloneDX**: SBOM generation (npm, Python)
- **Syft**: Multi-format SBOM generation
- **Cosign**: Container signing and attestation
- **SLSA GitHub Generator**: Provenance generation
- **Grype**: Vulnerability scanning
- **Trivy**: Security scanner

### Installation
```bash
# Install all required tools
./scripts/install-sbom-tools.sh
```

## References

- [SLSA Framework](https://slsa.dev/)
- [CycloneDX Standard](https://cyclonedx.org/)
- [SPDX Standard](https://spdx.dev/)
- [Sigstore Cosign](https://docs.sigstore.dev/cosign/overview/)
- [NIST SSDF](https://csrc.nist.gov/Projects/ssdf)

---

*This document should be reviewed and updated quarterly to ensure compliance with evolving standards.*