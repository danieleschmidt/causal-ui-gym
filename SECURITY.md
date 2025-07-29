# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of Causal UI Gym seriously. If you believe you have found a security vulnerability, please report it to us as described below.

**Please do not report security vulnerabilities through public GitHub issues.**

### How to Report

Send an email to **security@causal-ui-gym.dev** with:

- Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### What to Expect

After you submit a report, we will:

1. **Acknowledge receipt** within 24 hours
2. **Confirm the vulnerability** and determine its severity within 72 hours
3. **Work on a fix** and keep you updated on progress
4. **Release a patched version** with credit to you (if desired)
5. **Publish a security advisory** detailing the vulnerability

### Security Best Practices

When using Causal UI Gym:

- Always use the latest version
- Keep your dependencies up to date
- Use HTTPS for all network communications
- Validate all user inputs in your applications
- Follow secure coding practices for React and JAX
- Don't commit secrets or API keys to version control

### Dependencies

We regularly audit our dependencies for security vulnerabilities using:

- GitHub Dependabot
- npm audit
- Snyk security scanning

If you discover a vulnerability in one of our dependencies, please report it to us as well as the upstream maintainer.

### Disclosure Policy

We follow responsible disclosure principles:

- We will acknowledge valid reports within 24 hours
- We will work to fix vulnerabilities in a timely manner
- We will coordinate public disclosure with the reporter
- We will credit reporters (unless they prefer to remain anonymous)

### Comments on this Policy

If you have suggestions on how this policy could be improved, please submit a pull request or open an issue.

Thank you for helping keep Causal UI Gym and our users safe!