#!/usr/bin/env python3
"""
Performance benchmarking for JAX backend and causal computations.
Monitors memory usage, computation time, and scaling characteristics.
"""

import time
import psutil
import json
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager

@dataclass
class BenchmarkResult:
    """Performance benchmark result."""
    test_name: str
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    success: bool
    error: str = ""

class PerformanceBenchmarker:
    """JAX backend performance monitoring."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    @contextmanager
    def measure_performance(self, test_name: str):
        """Context manager for performance measurement."""
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.perf_counter()
        
        try:
            yield
            success = True
            error = ""
        except Exception as e:
            success = False
            error = str(e)
        finally:
            end_time = time.perf_counter()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            cpu_percent = process.cpu_percent()
            
            result = BenchmarkResult(
                test_name=test_name,
                duration_ms=(end_time - start_time) * 1000,
                memory_mb=end_memory - start_memory,
                cpu_percent=cpu_percent,
                success=success,
                error=error
            )
            self.results.append(result)
    
    def benchmark_causal_computation(self):
        """Benchmark causal graph computations."""
        with self.measure_performance("causal_dag_creation"):
            # Simulate DAG creation (replace with actual JAX code)
            import numpy as np
            dag = np.random.randn(100, 100)
            
        with self.measure_performance("intervention_computation"):
            # Simulate do-calculus computation
            intervention = np.random.randn(50, 50)
            result = np.matmul(dag[:50, :50], intervention)
    
    def benchmark_ui_data_processing(self):
        """Benchmark UI data processing pipeline."""
        with self.measure_performance("data_preprocessing"):
            # Simulate data preprocessing for React components
            data = [{"node": i, "value": i**2} for i in range(1000)]
            
        with self.measure_performance("metrics_calculation"):
            # Simulate metrics calculation (ATE, TE)
            metrics = {"ate": 0.15, "te": 0.23, "confidence": 0.95}
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        successful_results = [r for r in self.results if r.success]
        
        report = {
            "timestamp": time.time(),
            "total_tests": len(self.results),
            "successful_tests": len(successful_results),
            "average_duration_ms": sum(r.duration_ms for r in successful_results) / len(successful_results) if successful_results else 0,
            "max_memory_mb": max(r.memory_mb for r in successful_results) if successful_results else 0,
            "performance_grade": self._calculate_grade(successful_results),
            "recommendations": self._generate_recommendations(successful_results),
            "detailed_results": [asdict(r) for r in self.results]
        }
        
        return report
    
    def _calculate_grade(self, results: List[BenchmarkResult]) -> str:
        """Calculate performance grade based on benchmarks."""
        if not results:
            return "F"
        
        avg_duration = sum(r.duration_ms for r in results) / len(results)
        max_memory = max(r.memory_mb for r in results)
        
        if avg_duration < 100 and max_memory < 50:
            return "A"
        elif avg_duration < 500 and max_memory < 100:
            return "B"
        elif avg_duration < 1000 and max_memory < 200:
            return "C"
        else:
            return "D"
    
    def _generate_recommendations(self, results: List[BenchmarkResult]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        avg_duration = sum(r.duration_ms for r in results) / len(results) if results else 0
        max_memory = max(r.memory_mb for r in results) if results else 0
        
        if avg_duration > 500:
            recommendations.append("Consider optimizing computational algorithms")
            recommendations.append("Implement caching for repeated calculations")
        
        if max_memory > 100:
            recommendations.append("Optimize memory usage in JAX computations")
            recommendations.append("Consider using JAX memory management features")
        
        return recommendations

def main():
    """Run performance benchmarks."""
    benchmarker = PerformanceBenchmarker()
    
    print("Running performance benchmarks...")
    benchmarker.benchmark_causal_computation()
    benchmarker.benchmark_ui_data_processing()
    
    report = benchmarker.generate_report()
    
    # Save report
    with open("performance-report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Performance grade: {report['performance_grade']}")
    print(f"Average duration: {report['average_duration_ms']:.2f}ms")
    print(f"Max memory usage: {report['max_memory_mb']:.2f}MB")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")

if __name__ == "__main__":
    main()