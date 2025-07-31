# SLSA Provenance Configuration

Supply-chain Levels for Software Artifacts (SLSA) provenance generation for enhanced security and compliance.

## SLSA Level 3 Configuration

### Build Requirements
- **Hermetic builds**: All builds run in isolated environments
- **Reproducible builds**: Same inputs produce identical outputs  
- **Signed provenance**: All artifacts include cryptographic attestation
- **Verified dependencies**: All dependencies verified before use

### Implementation Status

#### ✅ Current Capabilities
- Dependency pinning via package-lock.json and poetry.lock
- Multi-stage container builds for isolation
- Pre-commit hooks for supply chain security
- Automated dependency vulnerability scanning

#### 🔄 Planned Enhancements
- **GitHub Actions SLSA Generator**: Use slsa-framework/slsa-github-generator
- **Cosign integration**: Sign container images and artifacts
- **SBOM generation**: Automated Software Bill of Materials
- **Attestation storage**: Store provenance in Sigstore transparency log

### Provenance Generation Workflow

```yaml
# In .github/workflows/release.yml
- name: Generate SLSA Provenance
  uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.4.0
  with:
    base64-subjects: ${{ steps.hash.outputs.hash }}
    provenance-name: causal-ui-gym-provenance.intoto.jsonl
```

### Verification Commands

```bash
# Verify SLSA provenance
slsa-verifier verify-artifact \
  --provenance-path causal-ui-gym-provenance.intoto.jsonl \
  --source-uri github.com/yourusername/causal-ui-gym \
  causal-ui-gym-v1.0.0.tar.gz

# Verify container image
cosign verify \
  --certificate-identity-regexp "^https://github.com/yourusername/causal-ui-gym" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com \
  ghcr.io/yourusername/causal-ui-gym:latest
```

### Supply Chain Security Checklist

- [ ] **MANUAL ACTION**: Enable GitHub Actions SLSA provenance generation
- [ ] **MANUAL ACTION**: Configure Cosign for artifact signing
- [ ] **MANUAL ACTION**: Set up SBOM generation in CI/CD
- [ ] **MANUAL ACTION**: Implement dependency verification gates
- [ ] **MANUAL ACTION**: Configure Sigstore integration

### Compliance Mapping

| Requirement | Implementation | Status |
|-------------|----------------|---------|
| Build platform | GitHub Actions | ⏳ Pending |
| Source control | GitHub with 2FA | ✅ Active |
| Dependency management | Locked versions | ✅ Active |
| Vulnerability scanning | Multiple tools | ✅ Active |
| Artifact signing | Cosign | ⏳ Pending |
| Provenance generation | SLSA Generator | ⏳ Pending |

### Security Contact

For security issues related to SLSA compliance:
- Create issue using SECURITY.md template
- Tag with `security` and `slsa` labels
- Provide provenance verification logs if applicable

## Resources

- [SLSA Framework](https://slsa.dev/)
- [GitHub SLSA Generator](https://github.com/slsa-framework/slsa-github-generator)
- [Cosign Documentation](https://docs.sigstore.dev/cosign/overview/)
- [SBOM Tools](https://github.com/anchore/syft)