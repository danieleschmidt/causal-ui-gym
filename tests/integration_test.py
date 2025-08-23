#!/usr/bin/env python3
"""
Comprehensive integration testing for the complete Causal UI Gym system.
"""

import requests
import time
import json
import subprocess
import os
import sys
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass 
class TestResult:
    name: str
    passed: bool
    message: str
    duration: float
    details: Dict[str, Any] = None

class IntegrationTester:
    """End-to-end integration testing suite"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.backend_url = "http://localhost:8002"
        self.frontend_url = "http://localhost:5173"
    
    def run_all_tests(self) -> List[TestResult]:
        """Execute complete test suite"""
        print("🧪 Running Comprehensive Integration Tests")
        print("=" * 45)
        
        test_methods = [
            self.test_backend_health,
            self.test_api_endpoints,
            self.test_experiment_workflow,
            self.test_intervention_system,
            self.test_performance_metrics,
            self.test_error_handling,
            self.test_data_persistence,
            self.test_concurrent_operations,
            self.test_frontend_build,
            self.test_security_headers
        ]
        
        for test_method in test_methods:
            try:
                start_time = time.time()
                result = test_method()
                duration = time.time() - start_time
                result.duration = duration
                self.results.append(result)
                
                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(f"{status} {result.name} ({duration:.2f}s)")
                if not result.passed:
                    print(f"     Error: {result.message}")
                    
            except Exception as e:
                duration = time.time() - start_time if 'start_time' in locals() else 0
                result = TestResult(test_method.__name__, False, str(e), duration)
                self.results.append(result)
                print(f"❌ FAIL {test_method.__name__} - Exception: {e}")
        
        return self.results
    
    def test_backend_health(self) -> TestResult:
        """Test backend health and readiness"""
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            data = response.json()
            
            checks = [
                response.status_code == 200,
                data.get("status") == "healthy",
                "performance" in data,
                data["performance"]["requests_per_second"] >= 0
            ]
            
            if all(checks):
                return TestResult(
                    "Backend Health Check",
                    True,
                    "All health indicators green",
                    0,
                    data
                )
            else:
                return TestResult(
                    "Backend Health Check", 
                    False,
                    f"Health check failed: {data}",
                    0
                )
                
        except Exception as e:
            return TestResult("Backend Health Check", False, str(e), 0)
    
    def test_api_endpoints(self) -> TestResult:
        """Test all critical API endpoints"""
        endpoints_to_test = [
            ("/", 200),
            ("/health", 200),
            ("/api/status", 200),
            ("/api/metrics", 200),
            ("/api/experiments", 200),
            ("/nonexistent", 404)
        ]
        
        failed_endpoints = []
        
        for endpoint, expected_status in endpoints_to_test:
            try:
                response = requests.get(f"{self.backend_url}{endpoint}", timeout=5)
                if response.status_code != expected_status:
                    failed_endpoints.append(f"{endpoint} returned {response.status_code}, expected {expected_status}")
            except Exception as e:
                failed_endpoints.append(f"{endpoint} failed: {str(e)}")
        
        if not failed_endpoints:
            return TestResult(
                "API Endpoints Test",
                True,
                f"All {len(endpoints_to_test)} endpoints responding correctly",
                0
            )
        else:
            return TestResult(
                "API Endpoints Test",
                False,
                f"Failed endpoints: {'; '.join(failed_endpoints)}",
                0
            )
    
    def test_experiment_workflow(self) -> TestResult:
        """Test complete experiment creation and management workflow"""
        try:
            # Create experiment
            exp_data = {
                "name": "Integration Test Experiment",
                "description": "Automated test experiment for validation"
            }
            
            create_response = requests.post(
                f"{self.backend_url}/api/experiments",
                json=exp_data,
                timeout=10
            )
            
            if create_response.status_code != 201:
                return TestResult(
                    "Experiment Workflow",
                    False,
                    f"Failed to create experiment: {create_response.status_code}",
                    0
                )
            
            created_exp = create_response.json()
            exp_id = created_exp.get("id")
            
            # Verify experiment was created
            list_response = requests.get(f"{self.backend_url}/api/experiments", timeout=5)
            experiments = list_response.json().get("experiments", [])
            
            found_experiment = any(exp.get("id") == exp_id for exp in experiments)
            
            if found_experiment:
                return TestResult(
                    "Experiment Workflow",
                    True,
                    f"Successfully created and verified experiment {exp_id}",
                    0,
                    created_exp
                )
            else:
                return TestResult(
                    "Experiment Workflow",
                    False,
                    "Created experiment not found in list",
                    0
                )
                
        except Exception as e:
            return TestResult("Experiment Workflow", False, str(e), 0)
    
    def test_intervention_system(self) -> TestResult:
        """Test causal intervention processing"""
        try:
            # Test multiple intervention scenarios
            interventions = [
                {"variable": "price", "value": 100},
                {"variable": "demand", "value": 50},
                {"variable": "temperature", "value": 25.5}
            ]
            
            results = []
            
            for intervention in interventions:
                response = requests.post(
                    f"{self.backend_url}/api/interventions",
                    json=intervention,
                    timeout=10
                )
                
                if response.status_code != 200:
                    return TestResult(
                        "Intervention System",
                        False,
                        f"Intervention failed for {intervention['variable']}: {response.status_code}",
                        0
                    )
                
                result = response.json()
                results.append(result)
                
                # Validate intervention result structure
                required_fields = ["intervention_id", "variable", "value", "result"]
                if not all(field in result for field in required_fields):
                    return TestResult(
                        "Intervention System",
                        False,
                        f"Invalid intervention result structure: missing fields",
                        0
                    )
            
            return TestResult(
                "Intervention System",
                True,
                f"Successfully processed {len(interventions)} interventions",
                0,
                {"interventions_processed": len(results)}
            )
            
        except Exception as e:
            return TestResult("Intervention System", False, str(e), 0)
    
    def test_performance_metrics(self) -> TestResult:
        """Test real-time performance metrics collection"""
        try:
            # Make some requests to generate metrics
            for _ in range(10):
                requests.get(f"{self.backend_url}/health", timeout=2)
            
            time.sleep(1)  # Allow metrics to update
            
            metrics_response = requests.get(f"{self.backend_url}/api/metrics", timeout=5)
            
            if metrics_response.status_code != 200:
                return TestResult(
                    "Performance Metrics",
                    False,
                    f"Failed to fetch metrics: {metrics_response.status_code}",
                    0
                )
            
            metrics = metrics_response.json()
            
            # Validate metrics structure
            required_sections = ["current_metrics", "cache_stats", "system_info"]
            if not all(section in metrics for section in required_sections):
                return TestResult(
                    "Performance Metrics",
                    False,
                    "Invalid metrics structure",
                    0
                )
            
            # Check for reasonable metric values
            current_metrics = metrics["current_metrics"]
            if (current_metrics.get("requests_per_second", -1) < 0 or
                current_metrics.get("average_response_time", -1) < 0):
                return TestResult(
                    "Performance Metrics",
                    False,
                    "Invalid metric values",
                    0
                )
            
            return TestResult(
                "Performance Metrics",
                True,
                "Metrics collection and reporting working correctly",
                0,
                metrics
            )
            
        except Exception as e:
            return TestResult("Performance Metrics", False, str(e), 0)
    
    def test_error_handling(self) -> TestResult:
        """Test error handling and recovery"""
        error_scenarios = [
            # Invalid JSON
            {
                "method": "POST",
                "endpoint": "/api/experiments",
                "data": "invalid json",
                "headers": {"Content-Type": "application/json"},
                "expected_status": 400
            },
            # Non-existent endpoint
            {
                "method": "GET", 
                "endpoint": "/api/nonexistent",
                "expected_status": 404
            }
        ]
        
        try:
            for scenario in error_scenarios:
                if scenario["method"] == "GET":
                    response = requests.get(
                        f"{self.backend_url}{scenario['endpoint']}", 
                        timeout=5
                    )
                elif scenario["method"] == "POST":
                    response = requests.post(
                        f"{self.backend_url}{scenario['endpoint']}",
                        data=scenario.get("data"),
                        headers=scenario.get("headers", {}),
                        timeout=5
                    )
                
                if response.status_code != scenario["expected_status"]:
                    return TestResult(
                        "Error Handling",
                        False,
                        f"Expected {scenario['expected_status']}, got {response.status_code} for {scenario['endpoint']}",
                        0
                    )
            
            return TestResult(
                "Error Handling",
                True,
                f"All {len(error_scenarios)} error scenarios handled correctly",
                0
            )
            
        except Exception as e:
            return TestResult("Error Handling", False, str(e), 0)
    
    def test_data_persistence(self) -> TestResult:
        """Test data persistence and retrieval"""
        try:
            # Create unique experiment
            unique_id = f"persist_test_{int(time.time() * 1000)}"
            exp_data = {
                "name": f"Persistence Test {unique_id}",
                "description": "Testing data persistence"
            }
            
            # Create experiment
            create_response = requests.post(
                f"{self.backend_url}/api/experiments",
                json=exp_data,
                timeout=10
            )
            
            if create_response.status_code != 201:
                return TestResult(
                    "Data Persistence",
                    False,
                    "Failed to create test experiment",
                    0
                )
            
            created_exp = create_response.json()
            
            # Verify persistence by retrieving
            time.sleep(0.1)  # Brief delay
            
            list_response = requests.get(f"{self.backend_url}/api/experiments", timeout=5)
            experiments = list_response.json().get("experiments", [])
            
            # Find our experiment
            found = any(
                exp.get("name") == exp_data["name"] and 
                exp.get("description") == exp_data["description"]
                for exp in experiments
            )
            
            if found:
                return TestResult(
                    "Data Persistence",
                    True,
                    "Data successfully persisted and retrieved",
                    0
                )
            else:
                return TestResult(
                    "Data Persistence",
                    False,
                    "Created data not found after retrieval",
                    0
                )
                
        except Exception as e:
            return TestResult("Data Persistence", False, str(e), 0)
    
    def test_concurrent_operations(self) -> TestResult:
        """Test system behavior under concurrent operations"""
        try:
            import concurrent.futures
            
            def create_experiment(index):
                data = {
                    "name": f"Concurrent Test {index}",
                    "description": f"Concurrent test experiment #{index}"
                }
                response = requests.post(
                    f"{self.backend_url}/api/experiments",
                    json=data,
                    timeout=10
                )
                return response.status_code == 201
            
            # Run 20 concurrent experiment creations
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(create_experiment, i) for i in range(20)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            success_count = sum(results)
            
            if success_count >= 18:  # Allow for a few failures due to timing
                return TestResult(
                    "Concurrent Operations",
                    True,
                    f"Successfully handled {success_count}/20 concurrent requests",
                    0
                )
            else:
                return TestResult(
                    "Concurrent Operations",
                    False,
                    f"Only {success_count}/20 concurrent requests succeeded",
                    0
                )
                
        except Exception as e:
            return TestResult("Concurrent Operations", False, str(e), 0)
    
    def test_frontend_build(self) -> TestResult:
        """Test frontend build process"""
        try:
            # Check if build directory exists and has content
            dist_path = "/root/repo/dist"
            if os.path.exists(dist_path):
                files = os.listdir(dist_path)
                if len(files) > 0:
                    return TestResult(
                        "Frontend Build",
                        True,
                        f"Build artifacts present: {len(files)} files",
                        0,
                        {"files": files}
                    )
            
            return TestResult(
                "Frontend Build",
                False,
                "Build artifacts not found",
                0
            )
            
        except Exception as e:
            return TestResult("Frontend Build", False, str(e), 0)
    
    def test_security_headers(self) -> TestResult:
        """Test security headers and CORS"""
        try:
            response = requests.get(f"{self.backend_url}/health", timeout=5)
            headers = response.headers
            
            # Check CORS headers
            cors_present = 'Access-Control-Allow-Origin' in headers
            
            # Check for basic security considerations
            content_type_correct = headers.get('Content-Type') == 'application/json'
            
            if cors_present and content_type_correct:
                return TestResult(
                    "Security Headers",
                    True,
                    "Security headers properly configured",
                    0,
                    {"cors_enabled": cors_present}
                )
            else:
                return TestResult(
                    "Security Headers",
                    False,
                    "Missing required security headers",
                    0
                )
                
        except Exception as e:
            return TestResult("Security Headers", False, str(e), 0)

def generate_test_report(results: List[TestResult]) -> str:
    """Generate comprehensive test report"""
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.passed)
    failed_tests = total_tests - passed_tests
    total_duration = sum(r.duration for r in results)
    
    report = []
    report.append("=" * 60)
    report.append("🧪 CAUSAL UI GYM INTEGRATION TEST REPORT")
    report.append("=" * 60)
    report.append("")
    
    report.append(f"📊 Test Summary:")
    report.append(f"   Total Tests: {total_tests}")
    report.append(f"   Passed: {passed_tests}")
    report.append(f"   Failed: {failed_tests}")
    report.append(f"   Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    report.append(f"   Total Duration: {total_duration:.2f}s")
    report.append("")
    
    # Detailed results
    report.append("📋 Detailed Results:")
    report.append("-" * 40)
    
    for result in results:
        status = "✅ PASS" if result.passed else "❌ FAIL"
        report.append(f"{status} {result.name}")
        report.append(f"   Duration: {result.duration:.3f}s")
        if result.message:
            report.append(f"   Message: {result.message}")
        if not result.passed and result.details:
            report.append(f"   Details: {result.details}")
        report.append("")
    
    # Quality assessment
    report.append("🏆 Quality Assessment:")
    report.append("-" * 25)
    
    if passed_tests == total_tests:
        report.append("   Status: 🎉 EXCELLENT - All tests passing!")
        report.append("   Quality Gate: ✅ PASSED")
    elif passed_tests >= total_tests * 0.9:
        report.append("   Status: 👍 GOOD - Most tests passing")
        report.append("   Quality Gate: ⚠️  CONDITIONAL PASS")
    else:
        report.append("   Status: ⚠️  NEEDS IMPROVEMENT - Multiple test failures")
        report.append("   Quality Gate: ❌ FAILED")
    
    return "\n".join(report)

def main():
    """Run integration tests and generate report"""
    tester = IntegrationTester()
    results = tester.run_all_tests()
    
    print("\n" + generate_test_report(results))
    
    # Exit with appropriate code
    passed_count = sum(1 for r in results if r.passed)
    if passed_count == len(results):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()