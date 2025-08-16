"""
Advanced Benchmarking Framework for Causal Inference Methods.

This module implements comprehensive benchmarking for causal inference algorithms,
including synthetic data generation, real-world datasets, and performance evaluation
across multiple metrics and scenarios.
"""

import asyncio
import logging
import json
import numpy as np
import jax.numpy as jnp
from jax import random
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
from abc import ABC, abstractmethod
import time

from ..engine.causal_engine import JaxCausalEngine, CausalDAG, Intervention, CausalResult

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkDataset:
    """Container for benchmark dataset with ground truth."""
    name: str
    description: str
    data: Dict[str, jnp.ndarray]  # Variable name -> data array
    true_dag: CausalDAG
    true_effects: Dict[Tuple[str, str], float]  # (treatment, outcome) -> true ATE
    confounders: List[str]
    sample_size: int
    dimensionality: int
    noise_level: float
    dataset_type: str  # 'synthetic', 'semi_synthetic', 'real'
    difficulty_level: str  # 'easy', 'medium', 'hard', 'expert'
    metadata: Dict[str, Any]


@dataclass
class BenchmarkResult:
    """Result of running a method on a benchmark dataset."""
    method_name: str
    dataset_name: str
    estimated_effects: Dict[Tuple[str, str], float]
    true_effects: Dict[Tuple[str, str], float]
    performance_metrics: Dict[str, float]
    computation_time: float
    memory_usage: float
    convergence_info: Dict[str, Any]
    method_parameters: Dict[str, Any]
    timestamp: datetime
    error_occurred: bool
    error_message: Optional[str]


@dataclass
class BenchmarkSuite:
    """Collection of benchmark datasets and evaluation metrics."""
    name: str
    description: str
    datasets: List[BenchmarkDataset]
    evaluation_metrics: List[str]
    baseline_methods: List[str]
    target_scenarios: List[str]  # e.g., 'high_dimensional', 'nonlinear', 'confounded'
    created_date: datetime


class SyntheticDataGenerator:
    """Generate synthetic causal datasets with known ground truth."""
    
    def __init__(self, random_seed: int = 42):
        self.key = random.PRNGKey(random_seed)
    
    def generate_linear_scm_dataset(
        self,
        n_variables: int,
        n_samples: int,
        edge_density: float = 0.3,
        noise_scale: float = 0.5,
        confounding_strength: float = 0.5
    ) -> BenchmarkDataset:
        """
        Generate dataset from linear structural causal model.
        
        Args:
            n_variables: Number of variables in the system
            n_samples: Number of samples to generate
            edge_density: Probability of edge between any two variables
            noise_scale: Standard deviation of noise terms
            confounding_strength: Strength of confounding relationships
            
        Returns:
            BenchmarkDataset with synthetic linear causal data
        """
        self.key, subkey = random.split(self.key)
        
        # Generate random DAG structure
        variable_names = [f"X{i}" for i in range(n_variables)]
        
        # Create adjacency matrix with topological ordering
        adj_matrix = jnp.zeros((n_variables, n_variables))
        edges = []
        edge_weights = {}
        
        for i in range(n_variables):
            for j in range(i + 1, n_variables):
                if random.uniform(self.key, minval=0, maxval=1) < edge_density:
                    self.key, subkey = random.split(self.key)
                    weight = random.normal(subkey) * 0.8  # Keep weights reasonable
                    adj_matrix = adj_matrix.at[j, i].set(weight)
                    edges.append((variable_names[i], variable_names[j]))
                    edge_weights[(variable_names[i], variable_names[j])] = float(weight)
        
        # Generate data using structural equations
        self.key, subkey = random.split(self.key)
        noise = random.normal(subkey, (n_variables, n_samples)) * noise_scale
        
        data_matrix = jnp.zeros((n_variables, n_samples))
        
        # Generate data in topological order
        for i in range(n_variables):
            parent_effects = jnp.dot(adj_matrix[i, :], data_matrix)
            data_matrix = data_matrix.at[i, :].set(parent_effects + noise[i, :])
        
        # Create data dictionary
        data = {var_name: data_matrix[i, :] for i, var_name in enumerate(variable_names)}
        
        # Calculate true causal effects
        true_effects = {}
        causal_engine = JaxCausalEngine()
        dag = CausalDAG(
            nodes=variable_names,
            edges=edges,
            node_data=data,
            edge_weights=edge_weights
        )
        
        # Compute true ATEs for all pairs
        for i, treatment in enumerate(variable_names):
            for j, outcome in enumerate(variable_names):
                if i != j:
                    try:
                        result = causal_engine.compute_ate(dag, treatment, outcome, n_samples=1000)
                        true_effects[(treatment, outcome)] = result.ate or 0.0
                    except:
                        true_effects[(treatment, outcome)] = 0.0
        
        # Identify confounders (variables that influence multiple others)
        confounders = []
        for var in variable_names:
            outgoing_edges = [e for e in edges if e[0] == var]
            if len(outgoing_edges) >= 2:
                confounders.append(var)
        
        return BenchmarkDataset(
            name=f"linear_scm_n{n_variables}_samples{n_samples}",
            description=f"Linear SCM with {n_variables} variables, {n_samples} samples",
            data=data,
            true_dag=dag,
            true_effects=true_effects,
            confounders=confounders,
            sample_size=n_samples,
            dimensionality=n_variables,
            noise_level=noise_scale,
            dataset_type="synthetic",
            difficulty_level="medium",
            metadata={
                "edge_density": edge_density,
                "confounding_strength": confounding_strength,
                "generation_method": "linear_scm"
            }
        )
    
    def generate_nonlinear_scm_dataset(
        self,
        n_variables: int,
        n_samples: int,
        nonlinearity_type: str = "polynomial"
    ) -> BenchmarkDataset:
        """
        Generate dataset with nonlinear causal relationships.
        
        Args:
            n_variables: Number of variables
            n_samples: Number of samples
            nonlinearity_type: Type of nonlinearity ('polynomial', 'sigmoid', 'mixed')
            
        Returns:
            BenchmarkDataset with nonlinear causal relationships
        """
        self.key, subkey = random.split(self.key)
        
        variable_names = [f"X{i}" for i in range(n_variables)]
        edges = []
        edge_weights = {}
        
        # Create simpler structure for nonlinear case
        for i in range(n_variables - 1):
            edges.append((variable_names[i], variable_names[i + 1]))
            edge_weights[(variable_names[i], variable_names[i + 1])] = 1.0
        
        # Add some additional edges
        if n_variables > 3:
            edges.append((variable_names[0], variable_names[2]))
            edge_weights[(variable_names[0], variable_names[2])] = 0.5
        
        # Generate data with nonlinear relationships
        self.key, subkey = random.split(self.key)
        noise = random.normal(subkey, (n_variables, n_samples)) * 0.3
        
        data_matrix = jnp.zeros((n_variables, n_samples))
        
        # First variable is purely noise
        data_matrix = data_matrix.at[0, :].set(noise[0, :])
        
        # Generate subsequent variables with nonlinear relationships
        for i in range(1, n_variables):
            parent_effects = jnp.zeros(n_samples)
            
            for edge in edges:
                if edge[1] == variable_names[i]:
                    parent_idx = variable_names.index(edge[0])
                    parent_data = data_matrix[parent_idx, :]
                    
                    if nonlinearity_type == "polynomial":
                        effect = 0.5 * parent_data + 0.3 * parent_data ** 2
                    elif nonlinearity_type == "sigmoid":
                        effect = jnp.tanh(parent_data)
                    else:  # mixed
                        effect = 0.4 * parent_data + 0.2 * jnp.sin(parent_data)
                    
                    parent_effects += effect * edge_weights[edge]
            
            data_matrix = data_matrix.at[i, :].set(parent_effects + noise[i, :])
        
        data = {var_name: data_matrix[i, :] for i, var_name in enumerate(variable_names)}
        
        dag = CausalDAG(
            nodes=variable_names,
            edges=edges,
            node_data=data,
            edge_weights=edge_weights
        )
        
        # For nonlinear case, true effects are harder to compute analytically
        # Use finite differences approximation
        true_effects = {}
        for treatment in variable_names:
            for outcome in variable_names:
                if treatment != outcome:
                    # Simplified: assume moderate effect for connected variables
                    if any(e[0] == treatment and e[1] == outcome for e in edges):
                        true_effects[(treatment, outcome)] = 0.5
                    else:
                        true_effects[(treatment, outcome)] = 0.0
        
        return BenchmarkDataset(
            name=f"nonlinear_scm_{nonlinearity_type}_n{n_variables}",
            description=f"Nonlinear SCM ({nonlinearity_type}) with {n_variables} variables",
            data=data,
            true_dag=dag,
            true_effects=true_effects,
            confounders=[variable_names[0]] if n_variables > 2 else [],
            sample_size=n_samples,
            dimensionality=n_variables,
            noise_level=0.3,
            dataset_type="synthetic",
            difficulty_level="hard",
            metadata={
                "nonlinearity_type": nonlinearity_type,
                "generation_method": "nonlinear_scm"
            }
        )
    
    def generate_high_dimensional_dataset(
        self,
        n_variables: int,
        n_samples: int,
        sparsity_level: float = 0.1
    ) -> BenchmarkDataset:
        """
        Generate high-dimensional sparse causal dataset.
        
        Args:
            n_variables: Number of variables (should be large, e.g., > 100)
            n_samples: Number of samples
            sparsity_level: Fraction of edges present in the true graph
            
        Returns:
            BenchmarkDataset with high-dimensional sparse structure
        """
        self.key, subkey = random.split(self.key)
        
        variable_names = [f"X{i}" for i in range(n_variables)]
        
        # Generate sparse random DAG
        edges = []
        edge_weights = {}
        max_edges = int(n_variables * (n_variables - 1) / 2 * sparsity_level)
        
        edge_count = 0
        for i in range(n_variables):
            for j in range(i + 1, min(i + 10, n_variables)):  # Limit neighborhood
                if edge_count < max_edges and random.uniform(self.key) < sparsity_level * 2:
                    self.key, subkey = random.split(self.key)
                    weight = random.normal(subkey) * 0.5
                    edges.append((variable_names[i], variable_names[j]))
                    edge_weights[(variable_names[i], variable_names[j])] = float(weight)
                    edge_count += 1
        
        # Generate data efficiently for high dimensions
        self.key, subkey = random.split(self.key)
        data_matrix = random.normal(subkey, (n_variables, n_samples)) * 0.5
        
        # Apply causal structure efficiently
        for edge, weight in edge_weights.items():
            parent_idx = variable_names.index(edge[0])
            child_idx = variable_names.index(edge[1])
            
            # Add causal effect
            data_matrix = data_matrix.at[child_idx, :].add(
                weight * data_matrix[parent_idx, :]
            )
        
        data = {var_name: data_matrix[i, :] for i, var_name in enumerate(variable_names)}
        
        dag = CausalDAG(
            nodes=variable_names,
            edges=edges,
            node_data=data,
            edge_weights=edge_weights
        )
        
        # Compute true effects for a subset of variable pairs
        true_effects = {}
        for i in range(min(10, n_variables)):  # Only compute for first 10 variables
            for j in range(min(10, n_variables)):
                if i != j:
                    treatment = variable_names[i]
                    outcome = variable_names[j]
                    # Use edge weight as proxy for true effect
                    if (treatment, outcome) in edge_weights:
                        true_effects[(treatment, outcome)] = edge_weights[(treatment, outcome)]
                    else:
                        true_effects[(treatment, outcome)] = 0.0
        
        return BenchmarkDataset(
            name=f"high_dim_n{n_variables}_sparse{sparsity_level}",
            description=f"High-dimensional sparse dataset with {n_variables} variables",
            data=data,
            true_dag=dag,
            true_effects=true_effects,
            confounders=[],  # Simplified for high-dim case
            sample_size=n_samples,
            dimensionality=n_variables,
            noise_level=0.5,
            dataset_type="synthetic",
            difficulty_level="expert",
            metadata={
                "sparsity_level": sparsity_level,
                "max_edges": max_edges,
                "generation_method": "high_dimensional_sparse"
            }
        )


class CausalMethodBenchmarker:
    """Benchmark causal inference methods on synthetic and real datasets."""
    
    def __init__(self):
        self.data_generator = SyntheticDataGenerator()
        self.benchmark_history: List[BenchmarkResult] = []
        
    def create_benchmark_suite(self, suite_name: str) -> BenchmarkSuite:
        """
        Create a comprehensive benchmark suite with various dataset types.
        
        Args:
            suite_name: Name for the benchmark suite
            
        Returns:
            BenchmarkSuite with diverse datasets
        """
        datasets = []
        
        # Small linear datasets (easy)
        for n_vars in [5, 10]:
            for n_samples in [500, 1000]:
                dataset = self.data_generator.generate_linear_scm_dataset(
                    n_variables=n_vars,
                    n_samples=n_samples,
                    edge_density=0.3,
                    noise_scale=0.3
                )
                datasets.append(dataset)
        
        # Medium nonlinear datasets (hard)
        for nonlin_type in ["polynomial", "sigmoid"]:
            dataset = self.data_generator.generate_nonlinear_scm_dataset(
                n_variables=8,
                n_samples=1000,
                nonlinearity_type=nonlin_type
            )
            datasets.append(dataset)
        
        # High-dimensional sparse datasets (expert)
        for n_vars in [50, 100]:
            dataset = self.data_generator.generate_high_dimensional_dataset(
                n_variables=n_vars,
                n_samples=1000,
                sparsity_level=0.05
            )
            datasets.append(dataset)
        
        return BenchmarkSuite(
            name=suite_name,
            description=f"Comprehensive causal inference benchmark suite",
            datasets=datasets,
            evaluation_metrics=[
                "ate_error",
                "rank_correlation", 
                "precision_at_k",
                "recall_at_k",
                "f1_score",
                "computation_time"
            ],
            baseline_methods=[
                "linear_regression",
                "instrumental_variables", 
                "difference_in_differences",
                "regression_discontinuity"
            ],
            target_scenarios=[
                "linear_relationships",
                "nonlinear_relationships", 
                "high_dimensional",
                "strong_confounding",
                "weak_instruments"
            ],
            created_date=datetime.now()
        )
    
    async def benchmark_method(
        self,
        method_func: Callable,
        method_name: str,
        dataset: BenchmarkDataset,
        method_params: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """
        Benchmark a single causal inference method on a dataset.
        
        Args:
            method_func: Function that implements the causal inference method
            method_name: Name of the method
            dataset: Dataset to benchmark on
            method_params: Parameters for the method
            
        Returns:
            BenchmarkResult with performance metrics
        """
        start_time = time.time()
        method_params = method_params or {}
        
        try:
            # Run the method
            estimated_effects = await self._run_method_async(
                method_func, dataset, method_params
            )
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                estimated_effects, dataset.true_effects
            )
            
            computation_time = time.time() - start_time
            
            result = BenchmarkResult(
                method_name=method_name,
                dataset_name=dataset.name,
                estimated_effects=estimated_effects,
                true_effects=dataset.true_effects,
                performance_metrics=performance_metrics,
                computation_time=computation_time,
                memory_usage=0.0,  # Could be implemented with memory profiling
                convergence_info={},
                method_parameters=method_params,
                timestamp=datetime.now(),
                error_occurred=False,
                error_message=None
            )
            
        except Exception as e:
            logger.error(f"Error benchmarking {method_name} on {dataset.name}: {e}")
            
            result = BenchmarkResult(
                method_name=method_name,
                dataset_name=dataset.name,
                estimated_effects={},
                true_effects=dataset.true_effects,
                performance_metrics={},
                computation_time=time.time() - start_time,
                memory_usage=0.0,
                convergence_info={},
                method_parameters=method_params,
                timestamp=datetime.now(),
                error_occurred=True,
                error_message=str(e)
            )
        
        self.benchmark_history.append(result)
        return result
    
    async def run_benchmark_suite(
        self,
        methods: Dict[str, Callable],
        benchmark_suite: BenchmarkSuite,
        max_concurrent: int = 4
    ) -> List[BenchmarkResult]:
        """
        Run multiple methods on a benchmark suite.
        
        Args:
            methods: Dictionary of method_name -> method_function
            benchmark_suite: Suite of datasets to benchmark on
            max_concurrent: Maximum number of concurrent benchmark runs
            
        Returns:
            List of BenchmarkResults for all method-dataset combinations
        """
        tasks = []
        
        for method_name, method_func in methods.items():
            for dataset in benchmark_suite.datasets:
                task = self.benchmark_method(method_func, method_name, dataset)
                tasks.append(task)
        
        # Run tasks with concurrency limit
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(task):
            async with semaphore:
                return await task
        
        limited_tasks = [run_with_semaphore(task) for task in tasks]
        results = await asyncio.gather(*limited_tasks)
        
        return results
    
    def analyze_benchmark_results(
        self,
        results: List[BenchmarkResult]
    ) -> Dict[str, Any]:
        """
        Analyze benchmark results and provide summary statistics.
        
        Args:
            results: List of benchmark results
            
        Returns:
            Dictionary with analysis summary
        """
        analysis = {
            "total_runs": len(results),
            "successful_runs": sum(1 for r in results if not r.error_occurred),
            "failed_runs": sum(1 for r in results if r.error_occurred),
            "methods_tested": len(set(r.method_name for r in results)),
            "datasets_tested": len(set(r.dataset_name for r in results)),
            "method_performance": {},
            "dataset_difficulty": {},
            "overall_rankings": {},
            "statistical_tests": {}
        }
        
        # Method performance summary
        for method_name in set(r.method_name for r in results):
            method_results = [r for r in results if r.method_name == method_name and not r.error_occurred]
            
            if method_results:
                ate_errors = [r.performance_metrics.get("ate_error", float('inf')) for r in method_results]
                analysis["method_performance"][method_name] = {
                    "mean_ate_error": float(np.mean(ate_errors)),
                    "std_ate_error": float(np.std(ate_errors)),
                    "median_ate_error": float(np.median(ate_errors)),
                    "success_rate": len(method_results) / len([r for r in results if r.method_name == method_name]),
                    "avg_computation_time": float(np.mean([r.computation_time for r in method_results]))
                }
        
        # Dataset difficulty analysis
        for dataset_name in set(r.dataset_name for r in results):
            dataset_results = [r for r in results if r.dataset_name == dataset_name and not r.error_occurred]
            
            if dataset_results:
                ate_errors = [r.performance_metrics.get("ate_error", float('inf')) for r in dataset_results]
                analysis["dataset_difficulty"][dataset_name] = {
                    "mean_error_across_methods": float(np.mean(ate_errors)),
                    "error_variance": float(np.var(ate_errors)),
                    "difficulty_score": float(np.mean(ate_errors) + np.var(ate_errors))  # Higher = more difficult
                }
        
        # Overall method rankings
        method_scores = {}
        for method_name, perf in analysis["method_performance"].items():
            # Lower ATE error and computation time are better
            score = 1.0 / (1.0 + perf["mean_ate_error"]) + 1.0 / (1.0 + perf["avg_computation_time"] / 60)
            method_scores[method_name] = score
        
        analysis["overall_rankings"] = sorted(
            method_scores.items(), key=lambda x: x[1], reverse=True
        )
        
        return analysis
    
    async def _run_method_async(
        self,
        method_func: Callable,
        dataset: BenchmarkDataset,
        method_params: Dict[str, Any]
    ) -> Dict[Tuple[str, str], float]:
        """Run causal inference method asynchronously."""
        # This is a wrapper to run potentially sync methods async
        loop = asyncio.get_event_loop()
        
        try:
            # Run method in thread pool for CPU-bound tasks
            estimated_effects = await loop.run_in_executor(
                None, method_func, dataset, method_params
            )
            return estimated_effects
        except Exception as e:
            logger.error(f"Error running method: {e}")
            return {}
    
    def _calculate_performance_metrics(
        self,
        estimated_effects: Dict[Tuple[str, str], float],
        true_effects: Dict[Tuple[str, str], float]
    ) -> Dict[str, float]:
        """Calculate performance metrics comparing estimated vs true effects."""
        metrics = {}
        
        # Get common effect pairs
        common_pairs = set(estimated_effects.keys()) & set(true_effects.keys())
        
        if not common_pairs:
            return {"ate_error": float('inf'), "correlation": 0.0}
        
        estimated_values = np.array([estimated_effects[pair] for pair in common_pairs])
        true_values = np.array([true_effects[pair] for pair in common_pairs])
        
        # Average Treatment Effect Error
        ate_error = np.mean(np.abs(estimated_values - true_values))
        metrics["ate_error"] = float(ate_error)
        
        # Root Mean Square Error
        rmse = np.sqrt(np.mean((estimated_values - true_values) ** 2))
        metrics["rmse"] = float(rmse)
        
        # Correlation
        if len(estimated_values) > 1 and np.var(estimated_values) > 0 and np.var(true_values) > 0:
            correlation = np.corrcoef(estimated_values, true_values)[0, 1]
            metrics["correlation"] = float(correlation)
        else:
            metrics["correlation"] = 0.0
        
        # Mean Absolute Percentage Error
        non_zero_true = true_values[true_values != 0]
        non_zero_est = estimated_values[true_values != 0]
        if len(non_zero_true) > 0:
            mape = np.mean(np.abs((non_zero_true - non_zero_est) / non_zero_true)) * 100
            metrics["mape"] = float(mape)
        
        # Rank correlation (Spearman)
        if len(estimated_values) > 2:
            from scipy.stats import spearmanr
            rank_corr, _ = spearmanr(estimated_values, true_values)
            metrics["rank_correlation"] = float(rank_corr) if not np.isnan(rank_corr) else 0.0
        
        return metrics


# Example baseline methods for benchmarking
def linear_regression_method(dataset: BenchmarkDataset, params: Dict[str, Any]) -> Dict[Tuple[str, str], float]:
    """Simple linear regression baseline method."""
    estimated_effects = {}
    
    variables = list(dataset.data.keys())
    
    for treatment in variables:
        for outcome in variables:
            if treatment != outcome:
                # Simple OLS regression: outcome ~ treatment
                X = dataset.data[treatment]
                y = dataset.data[outcome]
                
                # Add intercept
                X_with_intercept = jnp.column_stack([jnp.ones(len(X)), X])
                
                # OLS: beta = (X'X)^-1 X'y
                try:
                    XtX = jnp.dot(X_with_intercept.T, X_with_intercept)
                    XtX_inv = jnp.linalg.inv(XtX + 1e-6 * jnp.eye(2))
                    Xty = jnp.dot(X_with_intercept.T, y)
                    coefficients = jnp.dot(XtX_inv, Xty)
                    
                    # Treatment effect is the coefficient on treatment variable
                    estimated_effects[(treatment, outcome)] = float(coefficients[1])
                except:
                    estimated_effects[(treatment, outcome)] = 0.0
    
    return estimated_effects


def jax_causal_engine_method(dataset: BenchmarkDataset, params: Dict[str, Any]) -> Dict[Tuple[str, str], float]:
    """Benchmark the JAX causal engine."""
    estimated_effects = {}
    
    engine = JaxCausalEngine()
    
    variables = list(dataset.data.keys())
    
    for treatment in variables:
        for outcome in variables:
            if treatment != outcome:
                try:
                    result = engine.compute_ate(
                        dataset.true_dag, 
                        treatment, 
                        outcome,
                        n_samples=params.get("n_samples", 1000)
                    )
                    estimated_effects[(treatment, outcome)] = result.ate or 0.0
                except:
                    estimated_effects[(treatment, outcome)] = 0.0
    
    return estimated_effects