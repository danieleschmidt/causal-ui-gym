#!/usr/bin/env python3
"""
Advanced security scanning automation for Causal UI Gym.
Integrates multiple security tools and generates comprehensive reports.
"""

import subprocess
import json
import os
import sys
from typing import Dict, List, Any
from pathlib import Path

class SecurityScanner:
    """Comprehensive security scanning suite."""
    
    def __init__(self):
        self.results = {}
        self.project_root = Path.cwd()
    
    def run_bandit_scan(self) -> Dict[str, Any]:
        """Run Bandit security scan on Python code."""
        try:
            result = subprocess.run([
                'bandit', '-r', 'src/', 'tests/', '-f', 'json', '-o', 'bandit-report.json'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if os.path.exists('bandit-report.json'):
                with open('bandit-report.json', 'r') as f:
                    report = json.load(f)
                return {
                    'status': 'success',
                    'high_severity': len([i for i in report.get('results', []) if i.get('issue_severity') == 'HIGH']),
                    'medium_severity': len([i for i in report.get('results', []) if i.get('issue_severity') == 'MEDIUM']),
                    'total_issues': len(report.get('results', [])),
                    'report_file': 'bandit-report.json'
                }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def run_safety_scan(self) -> Dict[str, Any]:
        """Run Safety scan for Python dependencies."""
        try:
            result = subprocess.run([
                'safety', 'check', '--json', '--output', 'safety-report.json'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if os.path.exists('safety-report.json'):
                with open('safety-report.json', 'r') as f:
                    report = json.load(f)
                return {
                    'status': 'success',
                    'vulnerabilities': len(report),
                    'report_file': 'safety-report.json'
                }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def run_npm_audit(self) -> Dict[str, Any]:
        """Run npm audit for JavaScript dependencies."""
        try:
            result = subprocess.run([
                'npm', 'audit', '--json'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.stdout:
                audit_data = json.loads(result.stdout)
                return {
                    'status': 'success',
                    'vulnerabilities': audit_data.get('metadata', {}).get('vulnerabilities', {}),
                    'total_dependencies': audit_data.get('metadata', {}).get('totalDependencies', 0)
                }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def run_semgrep_scan(self) -> Dict[str, Any]:
        """Run Semgrep SAST scan."""
        try:
            result = subprocess.run([
                'semgrep', '--config=auto', '--json', '--output=semgrep-report.json', '.'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if os.path.exists('semgrep-report.json'):
                with open('semgrep-report.json', 'r') as f:
                    report = json.load(f)
                return {
                    'status': 'success',
                    'findings': len(report.get('results', [])),
                    'errors': len(report.get('errors', [])),
                    'report_file': 'semgrep-report.json'
                }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def check_secrets_baseline(self) -> Dict[str, Any]:
        """Verify secrets detection baseline."""
        try:
            result = subprocess.run([
                'detect-secrets', 'scan', '--baseline', '.secrets.baseline'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            return {
                'status': 'success' if result.returncode == 0 else 'warnings',
                'message': 'Secrets baseline validation completed',
                'new_secrets_found': result.returncode != 0
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        print("Running comprehensive security scan...")
        
        # Run all security scans
        self.results['bandit'] = self.run_bandit_scan()
        self.results['safety'] = self.run_safety_scan()
        self.results['npm_audit'] = self.run_npm_audit()
        self.results['semgrep'] = self.run_semgrep_scan()
        self.results['secrets'] = self.check_secrets_baseline()
        
        # Calculate security score
        score = self._calculate_security_score()
        
        # Generate recommendations
        recommendations = self._generate_security_recommendations()
        
        report = {
            'timestamp': subprocess.check_output(['date', '-u']).decode().strip(),
            'security_score': score,
            'grade': self._get_security_grade(score),
            'scan_results': self.results,
            'recommendations': recommendations,
            'compliance_status': self._check_compliance_status()
        }
        
        # Save comprehensive report
        with open('security-report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _calculate_security_score(self) -> int:
        """Calculate overall security score (0-100)."""
        score = 100
        
        # Deduct points for security issues
        bandit_result = self.results.get('bandit', {})
        if bandit_result.get('status') == 'success':
            score -= bandit_result.get('high_severity', 0) * 20
            score -= bandit_result.get('medium_severity', 0) * 10
        
        # Deduct for dependency vulnerabilities
        safety_result = self.results.get('safety', {})
        if safety_result.get('status') == 'success':
            score -= safety_result.get('vulnerabilities', 0) * 15
        
        npm_result = self.results.get('npm_audit', {})
        if npm_result.get('status') == 'success':
            vulns = npm_result.get('vulnerabilities', {})
            score -= vulns.get('critical', 0) * 25
            score -= vulns.get('high', 0) * 15
            score -= vulns.get('moderate', 0) * 10
        
        return max(0, score)
    
    def _get_security_grade(self, score: int) -> str:
        """Convert security score to letter grade."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security improvement recommendations."""
        recommendations = []
        
        # Bandit recommendations
        if self.results.get('bandit', {}).get('high_severity', 0) > 0:
            recommendations.append("Address high-severity Bandit findings immediately")
        
        # Dependency recommendations
        if self.results.get('safety', {}).get('vulnerabilities', 0) > 0:
            recommendations.append("Update Python dependencies with known vulnerabilities")
        
        npm_vulns = self.results.get('npm_audit', {}).get('vulnerabilities', {})
        if npm_vulns.get('critical', 0) > 0 or npm_vulns.get('high', 0) > 0:
            recommendations.append("Update npm dependencies with critical/high vulnerabilities")
        
        # General recommendations
        recommendations.extend([
            "Run security scans in CI/CD pipeline",
            "Implement dependency update automation",
            "Enable automated security monitoring",
            "Regular security training for developers"
        ])
        
        return recommendations
    
    def _check_compliance_status(self) -> Dict[str, bool]:
        """Check compliance with security standards."""
        return {
            'secrets_detection': self.results.get('secrets', {}).get('status') == 'success',
            'dependency_scanning': any([
                self.results.get('safety', {}).get('status') == 'success',
                self.results.get('npm_audit', {}).get('status') == 'success'
            ]),
            'static_analysis': any([
                self.results.get('bandit', {}).get('status') == 'success',
                self.results.get('semgrep', {}).get('status') == 'success'
            ]),
            'container_scanning': False,  # Would be True if Trivy/Docker scan implemented
        }

def main():
    """Run security scanning suite."""
    scanner = SecurityScanner()
    report = scanner.generate_security_report()
    
    print(f"\n=== Security Scan Results ===")
    print(f"Security Grade: {report['grade']}")
    print(f"Security Score: {report['security_score']}/100")
    
    print(f"\n=== Scan Summary ===")
    for tool, result in report['scan_results'].items():
        status = result.get('status', 'unknown')
        print(f"{tool.capitalize()}: {status}")
    
    if report['recommendations']:
        print(f"\n=== Recommendations ===")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
    
    print(f"\nDetailed report saved to: security-report.json")
    
    # Exit with error code if security score is too low
    if report['security_score'] < 70:
        print(f"\nWARNING: Security score below threshold (70)")
        sys.exit(1)

if __name__ == "__main__":
    main()