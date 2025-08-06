#!/usr/bin/env python3
"""
Comprehensive Quality Gates Validator for Causal UI Gym

This script runs all quality checks including security scanning,
performance validation, code quality analysis, and health checks.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QualityGateValidator:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results: Dict[str, Any] = {}
        self.start_time = datetime.now()
        
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all quality gate checks."""
        logger.info("🚀 Starting comprehensive quality gate validation...")
        
        checks = [
            ("security", self.check_security),
            ("performance", self.check_performance),
            ("code_quality", self.check_code_quality),
            ("dependency_health", self.check_dependencies),
            ("configuration", self.check_configuration),
            ("documentation", self.check_documentation),
            ("build_integrity", self.check_build_integrity)
        ]
        
        for check_name, check_function in checks:
            try:
                logger.info(f"🔍 Running {check_name} checks...")
                result = check_function()
                self.results[check_name] = result
                
                if result.get('status') == 'pass':
                    logger.info(f"✅ {check_name} checks passed")
                else:
                    logger.warning(f"⚠️  {check_name} checks had issues")
                    
            except Exception as e:
                logger.error(f"❌ Error running {check_name} checks: {e}")
                self.results[check_name] = {
                    'status': 'error',
                    'error': str(e),
                    'issues': [f"Check failed with error: {e}"]
                }
        
        # Calculate overall status
        self.results['overall'] = self.calculate_overall_status()
        self.results['execution_time'] = (datetime.now() - self.start_time).total_seconds()
        
        return self.results
    
    def check_security(self) -> Dict[str, Any]:
        """Run security vulnerability checks."""
        issues = []
        score = 100
        
        # Check for common security issues in code
        security_patterns = [
            (r'eval\s*\(', 'Use of eval() function'),
            (r'dangerouslySetInnerHTML', 'Use of dangerouslySetInnerHTML'),
            (r'document\.write', 'Use of document.write'),
            (r'localStorage\.setItem.*password', 'Storing passwords in localStorage'),
            (r'sessionStorage\.setItem.*password', 'Storing passwords in sessionStorage'),
            (r'console\.log.*password', 'Logging passwords to console'),
            (r'process\.env\.[A-Z_]*KEY', 'Potential API key in environment variables'),
            (r'["\'][a-f0-9]{32,}["\']', 'Potential hardcoded API keys or secrets'),
            (r'window\.location\.hash', 'Potential URL fragment security issue'),
            (r'innerHTML\s*=\s*.*\+', 'Potential XSS via innerHTML concatenation')
        ]
        
        # Scan source files
        source_dirs = ['src', 'backend', 'examples']
        for source_dir in source_dirs:
            source_path = self.project_root / source_dir
            if source_path.exists():
                issues.extend(self.scan_directory_for_patterns(source_path, security_patterns))
        
        # Check for insecure dependencies
        issues.extend(self.check_insecure_dependencies())
        
        # Check file permissions
        issues.extend(self.check_file_permissions())
        
        # Check for exposed sensitive files
        issues.extend(self.check_exposed_files())
        
        score -= len(issues) * 10
        status = 'pass' if score >= 80 else 'fail' if score < 60 else 'warning'
        
        return {
            'status': status,
            'score': max(0, score),
            'issues': issues,
            'checks_performed': len(security_patterns) + 3,
            'recommendations': self.get_security_recommendations(issues)
        }
    
    def check_performance(self) -> Dict[str, Any]:
        """Run performance and scalability checks."""
        issues = []
        metrics = {}
        
        # Check bundle size (estimate)
        bundle_info = self.estimate_bundle_size()
        if bundle_info['size_mb'] > 5:
            issues.append(f"Large estimated bundle size: {bundle_info['size_mb']:.1f}MB")
        
        # Check for performance anti-patterns
        perf_patterns = [
            (r'setInterval\s*\([^,]*,\s*[0-9]{1,2}[^0-9]', 'Frequent setInterval (< 100ms)'),
            (r'setTimeout\s*\([^,]*,\s*[0-9]{1}[^0-9]', 'Very short setTimeout (< 10ms)'),
            (r'document\.querySelector(?!All)', 'Inefficient DOM queries'),
            (r'\.map\s*\([^)]*\)\.map\s*\(', 'Chained array methods'),
            (r'useState\s*\([^)]*Object\.assign', 'Inefficient state updates'),
            (r'useEffect\s*\(\s*\(\s*\)\s*=>\s*\{[\s\S]*fetch', 'Unoptimized API calls in useEffect')
        ]
        
        source_path = self.project_root / 'src'
        if source_path.exists():
            issues.extend(self.scan_directory_for_patterns(source_path, perf_patterns))
        
        # Check memory usage patterns
        memory_issues = self.check_memory_patterns()
        issues.extend(memory_issues)
        
        # Check caching implementation
        caching_score = self.check_caching_implementation()
        metrics['caching_score'] = caching_score
        
        # Check concurrent processing
        concurrency_score = self.check_concurrency_patterns()
        metrics['concurrency_score'] = concurrency_score
        
        score = 100 - len(issues) * 5
        if caching_score < 70:
            score -= 10
        if concurrency_score < 70:
            score -= 10
        
        status = 'pass' if score >= 80 else 'fail' if score < 60 else 'warning'
        
        return {
            'status': status,
            'score': max(0, score),
            'issues': issues,
            'metrics': metrics,
            'bundle_info': bundle_info,
            'recommendations': self.get_performance_recommendations(issues, metrics)
        }
    
    def check_code_quality(self) -> Dict[str, Any]:
        """Check code quality metrics."""
        issues = []
        metrics = {}
        
        # Count lines of code
        loc_info = self.count_lines_of_code()
        metrics.update(loc_info)
        
        # Check code complexity
        complexity_issues = self.check_code_complexity()
        issues.extend(complexity_issues)
        
        # Check for code smells
        code_smell_patterns = [
            (r'function\s+\w+\s*\([^)]{50,}', 'Function with too many parameters'),
            (r'if\s*\([^{]{100,}', 'Complex conditional statements'),
            (r'\.then\s*\([^}]*\.then\s*\([^}]*\.then', 'Deeply nested promises'),
            (r'catch\s*\([^}]*console\.log', 'Poor error handling'),
            (r'TODO|FIXME|HACK', 'Technical debt markers'),
            (r'var\s+', 'Use of var instead of let/const'),
            (r'==\s*[^=]', 'Use of loose equality'),
            (r'new\s+Date\s*\(\s*\)', 'Non-deterministic date creation')
        ]
        
        source_dirs = ['src', 'backend']
        for source_dir in source_dirs:
            source_path = self.project_root / source_dir
            if source_path.exists():
                issues.extend(self.scan_directory_for_patterns(source_path, code_smell_patterns))
        
        # Check test coverage (estimated)
        test_coverage = self.estimate_test_coverage()
        metrics['test_coverage'] = test_coverage
        
        if test_coverage < 80:
            issues.append(f"Low test coverage: {test_coverage}%")
        
        score = 100 - len(issues) * 5
        if test_coverage < 60:
            score -= 20
        
        status = 'pass' if score >= 80 else 'fail' if score < 60 else 'warning'
        
        return {
            'status': status,
            'score': max(0, score),
            'issues': issues,
            'metrics': metrics,
            'recommendations': self.get_code_quality_recommendations(issues, metrics)
        }
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check dependency health and security."""
        issues = []
        
        # Check package.json
        package_json = self.project_root / 'package.json'
        if package_json.exists():
            with open(package_json, 'r') as f:
                package_data = json.load(f)
            
            dependencies = package_data.get('dependencies', {})
            dev_dependencies = package_data.get('devDependencies', {})
            
            # Check for known vulnerable packages
            vulnerable_packages = [
                'lodash', 'moment', 'request', 'axios<0.21.0'
            ]
            
            for pkg in vulnerable_packages:
                if pkg.split('<')[0] in dependencies or pkg.split('<')[0] in dev_dependencies:
                    issues.append(f"Potentially vulnerable dependency: {pkg}")
            
            # Check for excessive dependencies
            total_deps = len(dependencies) + len(dev_dependencies)
            if total_deps > 100:
                issues.append(f"High number of dependencies: {total_deps}")
        
        # Check requirements.txt
        requirements_txt = self.project_root / 'requirements.txt'
        if requirements_txt.exists():
            python_deps = self.check_python_dependencies(requirements_txt)
            issues.extend(python_deps)
        
        score = 100 - len(issues) * 10
        status = 'pass' if score >= 80 else 'fail' if score < 60 else 'warning'
        
        return {
            'status': status,
            'score': max(0, score),
            'issues': issues,
            'total_dependencies': len(dependencies) + len(dev_dependencies) if 'dependencies' in locals() else 0
        }
    
    def check_configuration(self) -> Dict[str, Any]:
        """Check configuration and deployment readiness."""
        issues = []
        
        # Check essential config files
        config_files = [
            'tsconfig.json',
            'package.json',
            'vite.config.ts',
            'docker-compose.yml',
            'Dockerfile'
        ]
        
        missing_configs = []
        for config_file in config_files:
            if not (self.project_root / config_file).exists():
                missing_configs.append(config_file)
        
        if missing_configs:
            issues.append(f"Missing configuration files: {', '.join(missing_configs)}")
        
        # Check environment handling
        env_files = ['.env.example', '.env.production', '.env.development']
        if not any((self.project_root / env_file).exists() for env_file in env_files):
            issues.append("No environment configuration files found")
        
        # Check Docker configuration
        if (self.project_root / 'Dockerfile').exists():
            docker_issues = self.check_docker_config()
            issues.extend(docker_issues)
        
        score = 100 - len(issues) * 15
        status = 'pass' if score >= 80 else 'fail' if score < 60 else 'warning'
        
        return {
            'status': status,
            'score': max(0, score),
            'issues': issues,
            'missing_configs': missing_configs
        }
    
    def check_documentation(self) -> Dict[str, Any]:
        """Check documentation completeness."""
        issues = []
        
        # Check essential documentation files
        doc_files = [
            'README.md',
            'CONTRIBUTING.md',
            'LICENSE',
            'docs/API_DOCUMENTATION.md',
            'docs/DEPLOYMENT_GUIDE.md'
        ]
        
        missing_docs = []
        for doc_file in doc_files:
            doc_path = self.project_root / doc_file
            if not doc_path.exists():
                missing_docs.append(doc_file)
            elif doc_path.stat().st_size < 500:  # Less than 500 bytes
                issues.append(f"Documentation file too short: {doc_file}")
        
        if missing_docs:
            issues.append(f"Missing documentation: {', '.join(missing_docs)}")
        
        # Check code documentation
        source_files = list((self.project_root / 'src').glob('**/*.ts')) + \
                       list((self.project_root / 'src').glob('**/*.tsx'))
        
        documented_files = 0
        for file_path in source_files:
            content = file_path.read_text()
            if '/**' in content or '/*' in content:
                documented_files += 1
        
        if source_files:
            doc_ratio = documented_files / len(source_files)
            if doc_ratio < 0.5:
                issues.append(f"Low code documentation ratio: {doc_ratio:.1%}")
        
        score = 100 - len(issues) * 12
        status = 'pass' if score >= 80 else 'fail' if score < 60 else 'warning'
        
        return {
            'status': status,
            'score': max(0, score),
            'issues': issues,
            'missing_docs': missing_docs,
            'code_doc_ratio': doc_ratio if 'doc_ratio' in locals() else 0
        }
    
    def check_build_integrity(self) -> Dict[str, Any]:
        """Check build system integrity."""
        issues = []
        
        # Check if builds are possible
        if (self.project_root / 'package.json').exists():
            package_json = self.project_root / 'package.json'
            with open(package_json, 'r') as f:
                package_data = json.load(f)
            
            scripts = package_data.get('scripts', {})
            essential_scripts = ['build', 'test', 'dev']
            
            for script in essential_scripts:
                if script not in scripts:
                    issues.append(f"Missing build script: {script}")
        
        # Check TypeScript configuration
        tsconfig = self.project_root / 'tsconfig.json'
        if tsconfig.exists():
            try:
                with open(tsconfig, 'r') as f:
                    ts_config = json.load(f)
                
                compiler_options = ts_config.get('compilerOptions', {})
                if compiler_options.get('strict') is not True:
                    issues.append("TypeScript strict mode not enabled")
                
                if 'target' not in compiler_options:
                    issues.append("TypeScript target not specified")
                    
            except json.JSONDecodeError:
                issues.append("Invalid tsconfig.json")
        
        score = 100 - len(issues) * 15
        status = 'pass' if score >= 80 else 'fail' if score < 60 else 'warning'
        
        return {
            'status': status,
            'score': max(0, score),
            'issues': issues
        }
    
    def calculate_overall_status(self) -> Dict[str, Any]:
        """Calculate overall quality gate status."""
        check_results = {k: v for k, v in self.results.items() if k != 'overall'}
        
        total_score = 0
        max_possible_score = 0
        failed_checks = []
        warning_checks = []
        
        for check_name, result in check_results.items():
            if isinstance(result, dict) and 'status' in result:
                score = result.get('score', 0)
                total_score += score
                max_possible_score += 100
                
                if result['status'] == 'fail':
                    failed_checks.append(check_name)
                elif result['status'] == 'warning':
                    warning_checks.append(check_name)
        
        overall_score = (total_score / max_possible_score * 100) if max_possible_score > 0 else 0
        
        if len(failed_checks) > 2 or overall_score < 60:
            overall_status = 'fail'
        elif len(failed_checks) > 0 or len(warning_checks) > 3 or overall_score < 80:
            overall_status = 'warning'
        else:
            overall_status = 'pass'
        
        return {
            'status': overall_status,
            'score': round(overall_score, 1),
            'failed_checks': failed_checks,
            'warning_checks': warning_checks,
            'total_issues': sum(len(result.get('issues', [])) for result in check_results.values() if isinstance(result, dict))
        }
    
    # Helper methods
    def scan_directory_for_patterns(self, directory: Path, patterns: List[tuple]) -> List[str]:
        """Scan directory for pattern matches."""
        import re
        issues = []
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix in ['.ts', '.tsx', '.js', '.jsx', '.py']:
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    for pattern, description in patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            rel_path = file_path.relative_to(self.project_root)
                            issues.append(f"{description} in {rel_path}")
                except Exception as e:
                    continue  # Skip files that can't be read
        
        return issues
    
    def check_insecure_dependencies(self) -> List[str]:
        """Check for known insecure dependencies."""
        issues = []
        # This is a simplified check - in production, use tools like npm audit
        return issues
    
    def check_file_permissions(self) -> List[str]:
        """Check for insecure file permissions."""
        issues = []
        sensitive_files = ['.env', '.env.production', 'config/secrets.json']
        
        for file_path in sensitive_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                stat = full_path.stat()
                # Check if file is world-readable (simplified check)
                if stat.st_mode & 0o044:
                    issues.append(f"Sensitive file has permissive permissions: {file_path}")
        
        return issues
    
    def check_exposed_files(self) -> List[str]:
        """Check for accidentally exposed sensitive files."""
        issues = []
        sensitive_patterns = [
            '.env*',
            '*.key',
            '*.pem',
            '*.p12',
            'id_rsa*',
            'config/database.yml',
            'config/secrets.yml'
        ]
        
        for pattern in sensitive_patterns:
            matches = list(self.project_root.glob(pattern))
            for match in matches:
                if match.is_file():
                    issues.append(f"Sensitive file found: {match.relative_to(self.project_root)}")
        
        return issues
    
    def estimate_bundle_size(self) -> Dict[str, Any]:
        """Estimate bundle size based on source code."""
        total_size = 0
        file_count = 0
        
        source_dirs = ['src', 'node_modules']
        for source_dir in source_dirs:
            source_path = self.project_root / source_dir
            if source_path.exists():
                for file_path in source_path.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                        file_count += 1
        
        return {
            'size_mb': total_size / (1024 * 1024),
            'file_count': file_count,
            'estimated_gzipped_mb': total_size / (1024 * 1024) * 0.3  # Rough gzip estimate
        }
    
    def check_memory_patterns(self) -> List[str]:
        """Check for memory leak patterns."""
        issues = []
        # This would scan for common memory leak patterns
        return issues
    
    def check_caching_implementation(self) -> int:
        """Check quality of caching implementation."""
        cache_files = list(self.project_root.glob('**/cache.ts')) + \
                     list(self.project_root.glob('**/caching.ts'))
        
        if not cache_files:
            return 30  # Low score for no caching
        
        # Check for advanced caching features
        score = 70  # Base score for having caching
        
        for cache_file in cache_files:
            content = cache_file.read_text()
            if 'TTL' in content or 'expire' in content:
                score += 10
            if 'LRU' in content or 'evict' in content:
                score += 10
            if 'priority' in content:
                score += 10
        
        return min(score, 100)
    
    def check_concurrency_patterns(self) -> int:
        """Check concurrency implementation quality."""
        score = 50  # Base score
        
        source_path = self.project_root / 'src'
        if source_path.exists():
            for file_path in source_path.rglob('*.ts'):
                content = file_path.read_text()
                if 'Promise.all' in content:
                    score += 10
                if 'Promise.allSettled' in content:
                    score += 15
                if 'worker' in content.lower():
                    score += 20
                if 'concurrent' in content.lower():
                    score += 10
        
        return min(score, 100)
    
    def count_lines_of_code(self) -> Dict[str, int]:
        """Count lines of code by type."""
        loc_stats = {'typescript': 0, 'python': 0, 'total': 0}
        
        extensions = {
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.py': 'python'
        }
        
        for ext, lang in extensions.items():
            for file_path in self.project_root.rglob(f'*{ext}'):
                if 'node_modules' not in str(file_path):
                    try:
                        lines = len(file_path.read_text().splitlines())
                        loc_stats[lang] += lines
                        loc_stats['total'] += lines
                    except Exception:
                        continue
        
        return loc_stats
    
    def check_code_complexity(self) -> List[str]:
        """Check for overly complex code patterns."""
        issues = []
        # Simplified complexity check
        return issues
    
    def estimate_test_coverage(self) -> float:
        """Estimate test coverage based on test files vs source files."""
        source_files = len(list(self.project_root.glob('src/**/*.ts'))) + \
                      len(list(self.project_root.glob('src/**/*.tsx'))) + \
                      len(list(self.project_root.glob('backend/**/*.py')))
        
        test_files = len(list(self.project_root.glob('**/*test*.ts'))) + \
                    len(list(self.project_root.glob('**/*test*.py'))) + \
                    len(list(self.project_root.glob('**/*.spec.ts')))
        
        if source_files == 0:
            return 0
        
        # Rough estimation: each test file covers ~3 source files
        estimated_coverage = min((test_files * 3) / source_files * 100, 100)
        return round(estimated_coverage, 1)
    
    def check_python_dependencies(self, requirements_file: Path) -> List[str]:
        """Check Python dependencies for issues."""
        issues = []
        content = requirements_file.read_text()
        
        # Check for unpinned versions
        lines = content.strip().split('\\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                if '==' not in line and '>=' not in line and '<=' not in line:
                    issues.append(f"Unpinned Python dependency: {line}")
        
        return issues
    
    def check_docker_config(self) -> List[str]:
        """Check Docker configuration."""
        issues = []
        dockerfile = self.project_root / 'Dockerfile'
        
        if dockerfile.exists():
            content = dockerfile.read_text()
            
            if 'USER root' in content:
                issues.append("Docker container runs as root user")
            
            if 'ADD' in content and 'http' in content:
                issues.append("Docker uses ADD with URLs (security risk)")
            
            if '--no-cache' not in content and 'apt-get update' in content:
                issues.append("Docker apt-get without --no-cache")
        
        return issues
    
    def get_security_recommendations(self, issues: List[str]) -> List[str]:
        """Get security recommendations based on issues found."""
        recommendations = []
        
        if any('eval' in issue for issue in issues):
            recommendations.append("Replace eval() with safer alternatives like JSON.parse()")
        
        if any('innerHTML' in issue for issue in issues):
            recommendations.append("Use textContent or createElement instead of innerHTML")
        
        if any('API key' in issue for issue in issues):
            recommendations.append("Move API keys to environment variables")
        
        if any('password' in issue for issue in issues):
            recommendations.append("Never store passwords in client-side storage")
        
        return recommendations
    
    def get_performance_recommendations(self, issues: List[str], metrics: Dict[str, Any]) -> List[str]:
        """Get performance recommendations."""
        recommendations = []
        
        if metrics.get('caching_score', 0) < 70:
            recommendations.append("Implement intelligent caching for expensive computations")
        
        if metrics.get('concurrency_score', 0) < 70:
            recommendations.append("Add concurrent processing for batch operations")
        
        if any('setTimeout' in issue for issue in issues):
            recommendations.append("Review timer usage and consider requestAnimationFrame for UI updates")
        
        return recommendations
    
    def get_code_quality_recommendations(self, issues: List[str], metrics: Dict[str, Any]) -> List[str]:
        """Get code quality recommendations."""
        recommendations = []
        
        if metrics.get('test_coverage', 0) < 80:
            recommendations.append("Increase test coverage with unit and integration tests")
        
        if any('TODO' in issue for issue in issues):
            recommendations.append("Address technical debt marked with TODO/FIXME")
        
        if any('var' in issue for issue in issues):
            recommendations.append("Replace var declarations with let/const")
        
        return recommendations

def main():
    """Main function to run quality gates."""
    project_root = Path(__file__).parent.parent
    validator = QualityGateValidator(project_root)
    
    results = validator.run_all_checks()
    
    # Output results
    print("\\n" + "="*80)
    print("🎯 CAUSAL UI GYM - QUALITY GATES REPORT")
    print("="*80)
    print(f"📊 Overall Status: {results['overall']['status'].upper()}")
    print(f"📈 Overall Score: {results['overall']['score']}/100")
    print(f"⏱️  Execution Time: {results['execution_time']:.2f}s")
    print(f"🐛 Total Issues: {results['overall']['total_issues']}")
    
    if results['overall']['failed_checks']:
        print(f"❌ Failed Checks: {', '.join(results['overall']['failed_checks'])}")
    
    if results['overall']['warning_checks']:
        print(f"⚠️  Warning Checks: {', '.join(results['overall']['warning_checks'])}")
    
    print("\\n" + "-"*80)
    print("📋 DETAILED RESULTS")
    print("-"*80)
    
    for check_name, result in results.items():
        if check_name in ['overall', 'execution_time']:
            continue
        
        if isinstance(result, dict):
            status_icon = "✅" if result['status'] == 'pass' else "⚠️" if result['status'] == 'warning' else "❌"
            print(f"\\n{status_icon} {check_name.replace('_', ' ').title()}: {result['score']}/100")
            
            if result.get('issues'):
                print("   Issues:")
                for issue in result['issues'][:5]:  # Show first 5 issues
                    print(f"   • {issue}")
                if len(result['issues']) > 5:
                    print(f"   ... and {len(result['issues']) - 5} more")
            
            if result.get('recommendations'):
                print("   Recommendations:")
                for rec in result['recommendations'][:3]:  # Show first 3 recommendations
                    print(f"   → {rec}")
    
    # Save detailed report
    report_file = project_root / 'quality-gates-report.json'
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\\n📄 Detailed report saved to: {report_file}")
    
    # Exit with appropriate code
    if results['overall']['status'] == 'fail':
        print("\\n❌ Quality gates FAILED")
        sys.exit(1)
    elif results['overall']['status'] == 'warning':
        print("\\n⚠️  Quality gates passed with warnings")
        sys.exit(0)
    else:
        print("\\n✅ All quality gates PASSED")
        sys.exit(0)

if __name__ == '__main__':
    main()