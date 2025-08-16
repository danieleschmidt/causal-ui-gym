"""
Benchmarking framework for causal inference methods.
"""

from .causal_benchmarks import (
    BenchmarkDataset,
    BenchmarkResult,
    BenchmarkSuite,
    SyntheticDataGenerator,
    CausalMethodBenchmarker,
    linear_regression_method,
    jax_causal_engine_method
)

__all__ = [
    "BenchmarkDataset",
    "BenchmarkResult", 
    "BenchmarkSuite",
    "SyntheticDataGenerator",
    "CausalMethodBenchmarker",
    "linear_regression_method",
    "jax_causal_engine_method"
]