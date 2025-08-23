#!/usr/bin/env python3
"""
Performance testing suite for Generation 3 scalability validation.
"""

import requests
import time
import statistics
import concurrent.futures
import json
from typing import List, Dict, Tuple
import threading

class PerformanceTester:
    """Comprehensive performance testing suite"""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.results = []
        self.lock = threading.Lock()
    
    def single_request(self, endpoint: str = "/health") -> Tuple[bool, float]:
        """Execute single request and measure response time"""
        start_time = time.time()
        try:
            response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
            end_time = time.time()
            return response.status_code == 200, (end_time - start_time) * 1000
        except Exception:
            end_time = time.time()
            return False, (end_time - start_time) * 1000
    
    def concurrent_load_test(self, concurrency: int, total_requests: int, endpoint: str = "/health") -> Dict:
        """Run concurrent load test"""
        print(f"🚀 Running load test: {total_requests} requests with {concurrency} concurrent users")
        
        def worker():
            success, response_time = self.single_request(endpoint)
            with self.lock:
                self.results.append((success, response_time))
        
        start_time = time.time()
        self.results.clear()
        
        # Execute concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker) for _ in range(total_requests)]
            concurrent.futures.wait(futures)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Analyze results
        successful_requests = sum(1 for success, _ in self.results if success)
        response_times = [rt for success, rt in self.results if success]
        
        results = {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": total_requests - successful_requests,
            "success_rate": (successful_requests / total_requests) * 100,
            "total_duration": total_duration,
            "requests_per_second": total_requests / total_duration,
            "average_response_time": statistics.mean(response_times) if response_times else 0,
            "median_response_time": statistics.median(response_times) if response_times else 0,
            "p95_response_time": self._percentile(response_times, 95) if response_times else 0,
            "p99_response_time": self._percentile(response_times, 99) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0
        }
        
        return results
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile from data"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def test_caching_performance(self) -> Dict:
        """Test caching effectiveness"""
        print("🔄 Testing caching performance...")
        
        endpoint = "/api/experiments"
        
        # First request (cache miss)
        start_time = time.time()
        response1 = requests.get(f"{self.base_url}{endpoint}")
        first_request_time = (time.time() - start_time) * 1000
        
        # Second request (should be cached)
        start_time = time.time()
        response2 = requests.get(f"{self.base_url}{endpoint}")
        second_request_time = (time.time() - start_time) * 1000
        
        return {
            "first_request_time": first_request_time,
            "second_request_time": second_request_time,
            "cache_improvement": ((first_request_time - second_request_time) / first_request_time) * 100,
            "cache_working": second_request_time < first_request_time
        }
    
    def test_compression_benefit(self) -> Dict:
        """Test compression effectiveness"""
        print("📦 Testing compression benefits...")
        
        endpoint = "/api/metrics"
        
        # Request without compression
        start_time = time.time()
        response_no_comp = requests.get(f"{self.base_url}{endpoint}")
        no_comp_time = (time.time() - start_time) * 1000
        no_comp_size = len(response_no_comp.content)
        
        # Request with compression
        headers = {"Accept-Encoding": "gzip"}
        start_time = time.time()
        response_comp = requests.get(f"{self.base_url}{endpoint}", headers=headers)
        comp_time = (time.time() - start_time) * 1000
        comp_size = len(response_comp.content)
        
        return {
            "uncompressed_size": no_comp_size,
            "compressed_size": comp_size,
            "compression_ratio": ((no_comp_size - comp_size) / no_comp_size) * 100 if no_comp_size > 0 else 0,
            "uncompressed_time": no_comp_time,
            "compressed_time": comp_time,
            "compression_enabled": response_comp.headers.get('Content-Encoding') == 'gzip'
        }
    
    def test_scalability_curve(self) -> List[Dict]:
        """Test performance across different concurrency levels"""
        print("📈 Testing scalability curve...")
        
        concurrency_levels = [1, 5, 10, 20, 50]
        total_requests = 100
        results = []
        
        for concurrency in concurrency_levels:
            print(f"   Testing {concurrency} concurrent users...")
            result = self.concurrent_load_test(concurrency, total_requests)
            result['concurrency'] = concurrency
            results.append(result)
            time.sleep(2)  # Brief pause between tests
        
        return results
    
    def stress_test(self, duration_seconds: int = 30) -> Dict:
        """Run sustained load for specified duration"""
        print(f"💪 Running {duration_seconds}s stress test...")
        
        end_time = time.time() + duration_seconds
        request_count = 0
        success_count = 0
        response_times = []
        
        def worker():
            nonlocal request_count, success_count
            while time.time() < end_time:
                success, response_time = self.single_request("/health")
                with self.lock:
                    request_count += 1
                    if success:
                        success_count += 1
                    response_times.append(response_time)
                time.sleep(0.01)  # Small delay to prevent overwhelming
        
        # Run with moderate concurrency
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(worker) for _ in range(20)]
            concurrent.futures.wait(futures, timeout=duration_seconds + 5)
        
        return {
            "duration_seconds": duration_seconds,
            "total_requests": request_count,
            "successful_requests": success_count,
            "requests_per_second": request_count / duration_seconds,
            "success_rate": (success_count / request_count) * 100 if request_count > 0 else 0,
            "average_response_time": statistics.mean(response_times) if response_times else 0,
            "p95_response_time": self._percentile(response_times, 95) if response_times else 0
        }

def print_results(title: str, results: Dict):
    """Pretty print test results"""
    print(f"\n📊 {title}")
    print("=" * (len(title) + 4))
    
    if isinstance(results, dict):
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value}")
    else:
        print(json.dumps(results, indent=2))

def main():
    """Run complete performance test suite"""
    print("🎯 Causal UI Gym Performance Test Suite")
    print("=" * 45)
    
    tester = PerformanceTester()
    
    # Basic connectivity test
    print("🔍 Testing server connectivity...")
    success, response_time = tester.single_request("/health")
    if not success:
        print("❌ Server not responding. Please ensure scalable server is running on port 8002.")
        return
    
    print(f"✅ Server responding in {response_time:.2f}ms")
    
    # Run test suite
    tests = [
        ("Basic Load Test (100 requests, 10 concurrent)", 
         lambda: tester.concurrent_load_test(10, 100)),
        
        ("Caching Performance Test", 
         lambda: tester.test_caching_performance()),
        
        ("Compression Benefits Test", 
         lambda: tester.test_compression_benefit()),
        
        ("High Concurrency Test (1000 requests, 50 concurrent)", 
         lambda: tester.concurrent_load_test(50, 1000)),
        
        ("Stress Test (30 seconds sustained load)", 
         lambda: tester.stress_test(30)),
    ]
    
    results_summary = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}")
            result = test_func()
            results_summary[test_name] = result
            print_results(test_name, result)
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results_summary[test_name] = {"error": str(e)}
    
    # Scalability curve (separate due to duration)
    print(f"\n🧪 Running: Scalability Curve Analysis")
    try:
        scalability_results = tester.test_scalability_curve()
        results_summary["Scalability Curve"] = scalability_results
        
        print("\n📈 Scalability Analysis")
        print("=" * 25)
        print("Concurrency | RPS    | Avg Response | Success Rate")
        print("-" * 50)
        
        for result in scalability_results:
            print(f"{result['concurrency']:10} | "
                  f"{result['requests_per_second']:6.1f} | "
                  f"{result['average_response_time']:11.1f}ms | "
                  f"{result['success_rate']:10.1f}%")
        
    except Exception as e:
        print(f"❌ Scalability test failed: {e}")
    
    # Performance summary
    print("\n🏆 Performance Summary")
    print("=" * 23)
    
    basic_load = results_summary.get("Basic Load Test (100 requests, 10 concurrent)", {})
    high_load = results_summary.get("High Concurrency Test (1000 requests, 50 concurrent)", {})
    stress = results_summary.get("Stress Test (30 seconds sustained load)", {})
    caching = results_summary.get("Caching Performance Test", {})
    compression = results_summary.get("Compression Benefits Test", {})
    
    print(f"✅ Basic Load RPS: {basic_load.get('requests_per_second', 0):.1f}")
    print(f"✅ High Concurrency RPS: {high_load.get('requests_per_second', 0):.1f}")
    print(f"✅ Sustained Load RPS: {stress.get('requests_per_second', 0):.1f}")
    print(f"✅ Cache Improvement: {caching.get('cache_improvement', 0):.1f}%")
    print(f"✅ Compression Ratio: {compression.get('compression_ratio', 0):.1f}%")
    
    # Overall assessment
    overall_rps = high_load.get('requests_per_second', 0)
    overall_success = high_load.get('success_rate', 0)
    
    if overall_rps > 500 and overall_success > 95:
        print("\n🎉 EXCELLENT: System demonstrates high scalability!")
    elif overall_rps > 200 and overall_success > 90:
        print("\n👍 GOOD: System shows solid performance characteristics")
    elif overall_rps > 50 and overall_success > 80:
        print("\n⚠️  MODERATE: System functional but could benefit from optimization")
    else:
        print("\n⚠️  NEEDS IMPROVEMENT: Performance below expectations")

if __name__ == "__main__":
    main()