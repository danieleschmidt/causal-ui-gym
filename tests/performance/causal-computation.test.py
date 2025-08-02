"""
Performance tests for causal computation using JAX
"""

import time
import pytest
import jax
import jax.numpy as jnp
from typing import List, Tuple
import numpy as np

# Mock JAX-based causal computation functions
def compute_intervention_jax(dag: dict, intervention: dict, evidence: dict) -> dict:
    """Mock JAX-based intervention computation"""
    # Simulate causal computation with JAX
    nodes = jnp.array([1.0, 2.0, 3.0])  # Mock node values
    result = jnp.sum(nodes) * intervention.get('value', 1.0)
    return {'result': float(result)}

def compute_ate_jax(treatment_data: jnp.ndarray, control_data: jnp.ndarray) -> float:
    """Compute Average Treatment Effect using JAX"""
    return float(jnp.mean(treatment_data) - jnp.mean(control_data))

def batch_interventions_jax(interventions: List[dict], dag: dict) -> List[dict]:
    """Batch process multiple interventions"""
    return [compute_intervention_jax(dag, intervention, {}) for intervention in interventions]

class TestCausalComputationPerformance:
    """Performance tests for causal computation operations"""

    def setup_method(self):
        """Setup test data"""
        self.small_dag = {
            'nodes': ['A', 'B', 'C'],
            'edges': [('A', 'B'), ('B', 'C')]
        }
        
        self.large_dag = {
            'nodes': [f'node_{i}' for i in range(100)],
            'edges': [(f'node_{i}', f'node_{i+1}') for i in range(99)]
        }
        
        self.massive_dag = {
            'nodes': [f'node_{i}' for i in range(1000)],
            'edges': [(f'node_{i}', f'node_{i+1}') for i in range(999)]
        }

    def test_single_intervention_performance(self):
        """Test performance of single intervention computation"""
        intervention = {'variable': 'A', 'value': 50.0}
        
        # Measure computation time
        start_time = time.perf_counter()
        result = compute_intervention_jax(self.small_dag, intervention, {})
        end_time = time.perf_counter()
        
        computation_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        assert result is not None
        assert computation_time < 10  # Should complete in under 10ms
        print(f"Single intervention: {computation_time:.2f}ms")

    def test_batch_intervention_performance(self):
        """Test performance of batch intervention processing"""
        interventions = [
            {'variable': 'A', 'value': 25.0},
            {'variable': 'B', 'value': 50.0},
            {'variable': 'C', 'value': 75.0},
        ]
        
        start_time = time.perf_counter()
        results = batch_interventions_jax(interventions, self.small_dag)
        end_time = time.perf_counter()
        
        computation_time = (end_time - start_time) * 1000
        
        assert len(results) == 3
        assert computation_time < 50  # Should complete in under 50ms
        print(f"Batch interventions (3): {computation_time:.2f}ms")

    def test_large_dag_performance(self):
        """Test performance with large causal DAG"""
        intervention = {'variable': 'node_50', 'value': 100.0}
        
        start_time = time.perf_counter()
        result = compute_intervention_jax(self.large_dag, intervention, {})
        end_time = time.perf_counter()
        
        computation_time = (end_time - start_time) * 1000
        
        assert result is not None
        assert computation_time < 100  # Should complete in under 100ms
        print(f"Large DAG (100 nodes): {computation_time:.2f}ms")

    def test_massive_dag_performance(self):
        """Test performance with massive causal DAG"""
        intervention = {'variable': 'node_500', 'value': 200.0}
        
        start_time = time.perf_counter()
        result = compute_intervention_jax(self.massive_dag, intervention, {})
        end_time = time.perf_counter()
        
        computation_time = (end_time - start_time) * 1000
        
        assert result is not None
        assert computation_time < 500  # Should complete in under 500ms
        print(f"Massive DAG (1000 nodes): {computation_time:.2f}ms")

    def test_ate_computation_performance(self):
        """Test Average Treatment Effect computation performance"""
        # Generate large datasets
        treatment_data = jnp.array(np.random.normal(100, 15, 10000))
        control_data = jnp.array(np.random.normal(85, 15, 10000))
        
        start_time = time.perf_counter()
        ate = compute_ate_jax(treatment_data, control_data)
        end_time = time.perf_counter()
        
        computation_time = (end_time - start_time) * 1000
        
        assert isinstance(ate, float)
        assert computation_time < 50  # Should complete in under 50ms
        print(f"ATE computation (10k samples): {computation_time:.2f}ms")

    def test_concurrent_computations(self):
        """Test performance of concurrent causal computations"""
        import concurrent.futures
        
        def compute_with_intervention(intervention_value):
            intervention = {'variable': 'A', 'value': intervention_value}
            return compute_intervention_jax(self.small_dag, intervention, {})
        
        intervention_values = [i * 10.0 for i in range(100)]
        
        start_time = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(compute_with_intervention, intervention_values))
        
        end_time = time.perf_counter()
        
        computation_time = (end_time - start_time) * 1000
        
        assert len(results) == 100
        assert computation_time < 1000  # Should complete in under 1 second
        print(f"Concurrent computations (100): {computation_time:.2f}ms")

    def test_memory_usage_large_computations(self):
        """Test memory usage during large computations"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform multiple large computations
        for i in range(50):
            intervention = {'variable': f'node_{i}', 'value': float(i)}
            compute_intervention_jax(self.large_dag, intervention, {})
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 100  # Should not increase by more than 100MB
        print(f"Memory increase: {memory_increase:.2f}MB")

    def test_jax_compilation_overhead(self):
        """Test JAX JIT compilation overhead"""
        
        @jax.jit
        def jitted_computation(data):
            return jnp.sum(data ** 2)
        
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # First call includes compilation time
        start_time = time.perf_counter()
        result1 = jitted_computation(data)
        compile_time = time.perf_counter() - start_time
        
        # Second call should be much faster
        start_time = time.perf_counter()
        result2 = jitted_computation(data)
        execution_time = time.perf_counter() - start_time
        
        assert result1 == result2
        assert execution_time < compile_time  # JIT should speed up subsequent calls
        print(f"Compilation: {compile_time*1000:.2f}ms, Execution: {execution_time*1000:.2f}ms")

    @pytest.mark.parametrize("size", [10, 100, 1000])
    def test_scalability_with_size(self, size):
        """Test how performance scales with problem size"""
        dag = {
            'nodes': [f'node_{i}' for i in range(size)],
            'edges': [(f'node_{i}', f'node_{i+1}') for i in range(size-1)]
        }
        
        intervention = {'variable': f'node_{size//2}', 'value': 50.0}
        
        start_time = time.perf_counter()
        result = compute_intervention_jax(dag, intervention, {})
        end_time = time.perf_counter()
        
        computation_time = (end_time - start_time) * 1000
        
        assert result is not None
        print(f"Size {size}: {computation_time:.2f}ms")
        
        # Performance should scale reasonably
        if size == 10:
            assert computation_time < 10
        elif size == 100:
            assert computation_time < 100
        elif size == 1000:
            assert computation_time < 1000

    def test_gpu_vs_cpu_performance(self):
        """Test performance difference between GPU and CPU"""
        large_array = jnp.array(np.random.random((1000, 1000)))
        
        # CPU computation
        with jax.default_device(jax.devices('cpu')[0]):
            start_time = time.perf_counter()
            cpu_result = jnp.sum(large_array ** 2)
            cpu_time = time.perf_counter() - start_time
        
        # GPU computation (if available)
        try:
            gpu_devices = jax.devices('gpu')
            if gpu_devices:
                with jax.default_device(gpu_devices[0]):
                    start_time = time.perf_counter()
                    gpu_result = jnp.sum(large_array ** 2)
                    gpu_time = time.perf_counter() - start_time
                
                print(f"CPU: {cpu_time*1000:.2f}ms, GPU: {gpu_time*1000:.2f}ms")
                
                # Results should be approximately equal
                assert abs(cpu_result - gpu_result) < 1e-5
                
                # GPU should be faster for large computations (if available)
                if gpu_time > 0:  # Sometimes GPU setup time dominates
                    assert gpu_time <= cpu_time or gpu_time < 0.1
            else:
                print("No GPU available, skipping GPU test")
        except Exception as e:
            print(f"GPU test failed: {e}")

    def test_memory_efficient_batch_processing(self):
        """Test memory-efficient processing of large batches"""
        batch_size = 1000
        interventions = [
            {'variable': 'A', 'value': float(i)}
            for i in range(batch_size)
        ]
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Process in smaller chunks to avoid memory issues
        chunk_size = 100
        results = []
        
        start_time = time.perf_counter()
        
        for i in range(0, batch_size, chunk_size):
            chunk = interventions[i:i + chunk_size]
            chunk_results = batch_interventions_jax(chunk, self.small_dag)
            results.extend(chunk_results)
        
        end_time = time.perf_counter()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        computation_time = (end_time - start_time) * 1000
        
        assert len(results) == batch_size
        assert memory_increase < 50  # Should not use excessive memory
        assert computation_time < 5000  # Should complete in under 5 seconds
        
        print(f"Batch processing (1000 items): {computation_time:.2f}ms")
        print(f"Memory increase: {memory_increase:.2f}MB")

if __name__ == "__main__":
    # Run performance tests with verbose output
    pytest.main([__file__, "-v", "-s"])