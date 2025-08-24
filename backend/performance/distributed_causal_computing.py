"""
Distributed Causal Computing System for Large-Scale Operations

This module implements distributed causal inference computation using 
JAX's advanced parallelization capabilities for research-grade scalability.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass
import asyncio
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import logging
from functools import partial

from ..error_handling.advanced_error_system import (
    resilient_computation, resilient_causal_inference,
    validate_causal_data, causal_computation_context
)

logger = logging.getLogger(__name__)


@dataclass
class ComputationTask:
    """Represents a computational task for distributed execution"""
    task_id: str
    task_type: str
    input_data: Dict[str, Any]
    priority: int = 1
    estimated_runtime: float = 1.0
    memory_requirement: int = 1024  # MB
    gpu_required: bool = False
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class ComputationResult:
    """Result from distributed computation"""
    task_id: str
    result: Any
    execution_time: float
    memory_used: int
    worker_id: str
    success: bool
    error_message: Optional[str] = None


class DistributedCausalCompute:
    """High-performance distributed causal inference system"""
    
    def __init__(
        self,
        n_workers: Optional[int] = None,
        use_gpu: bool = True,
        memory_limit_gb: float = 16.0
    ):
        self.n_workers = n_workers or mp.cpu_count()
        self.use_gpu = use_gpu and len(jax.devices()) > 1
        self.memory_limit = memory_limit_gb * 1024  # MB
        
        # Initialize JAX for distributed computing
        if self.use_gpu:
            self._setup_gpu_distribution()
        else:
            self._setup_cpu_distribution()
        
        # Task management
        self.active_tasks: Dict[str, ComputationTask] = {}
        self.completed_tasks: Dict[str, ComputationResult] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        
        # Performance metrics
        self.performance_metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_execution_time': 0.0,
            'throughput': 0.0,  # tasks per second
            'memory_efficiency': 0.0,
            'gpu_utilization': 0.0
        }
        
        logger.info(f"Initialized distributed compute with {self.n_workers} workers, GPU: {self.use_gpu}")
    
    def _setup_gpu_distribution(self):
        """Setup JAX for multi-GPU distributed computing"""
        devices = jax.devices()
        self.devices = devices
        self.n_devices = len(devices)
        
        logger.info(f"Available devices: {[d.device_kind for d in devices]}")
        
        # Set up device mesh for parallel computation
        if self.n_devices > 1:
            self.device_mesh = jax.sharding.Mesh(devices, ('batch',))
        else:
            self.device_mesh = None
    
    def _setup_cpu_distribution(self):
        """Setup for CPU-based distributed computing"""
        self.devices = [jax.devices('cpu')[0]]
        self.n_devices = 1
        self.device_mesh = None
        
        # Set up process pool for CPU parallelization
        self.process_pool = ProcessPoolExecutor(max_workers=self.n_workers)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.n_workers * 2)
    
    @resilient_causal_inference
    async def submit_causal_discovery_task(
        self,
        data: jnp.ndarray,
        method: str = 'distributed_notears',
        hyperparameters: Optional[Dict[str, Any]] = None,
        priority: int = 1
    ) -> str:
        """Submit a causal discovery task for distributed execution"""
        
        task_id = f"causal_discovery_{int(time.time() * 1000)}_{hash(str(data.shape)) % 10000}"
        
        # Estimate resource requirements
        n_samples, n_vars = data.shape
        estimated_memory = n_vars**2 * n_samples * 8 / (1024**2)  # MB estimate
        estimated_runtime = n_vars**2 * 0.1  # seconds estimate
        
        task = ComputationTask(
            task_id=task_id,
            task_type='causal_discovery',
            input_data={
                'data': data,
                'method': method,
                'hyperparameters': hyperparameters or {}
            },
            priority=priority,
            estimated_runtime=estimated_runtime,
            memory_requirement=int(estimated_memory),
            gpu_required=n_vars > 20  # Use GPU for larger problems
        )
        
        self.active_tasks[task_id] = task
        await self.task_queue.put(task)
        self.performance_metrics['total_tasks'] += 1
        
        logger.info(f"Submitted causal discovery task {task_id} for {n_vars} variables")
        return task_id
    
    @resilient_causal_inference
    async def submit_intervention_task(
        self,
        dag: jnp.ndarray,
        intervention: Dict[str, float],
        data: jnp.ndarray,
        n_samples: int = 1000,
        priority: int = 2
    ) -> str:
        """Submit an intervention computation task"""
        
        task_id = f"intervention_{int(time.time() * 1000)}_{hash(str(intervention)) % 10000}"
        
        # Estimate resource requirements
        n_vars = dag.shape[0]
        estimated_memory = n_vars * n_samples * 8 / (1024**2)
        estimated_runtime = n_samples * 0.001
        
        task = ComputationTask(
            task_id=task_id,
            task_type='intervention',
            input_data={
                'dag': dag,
                'intervention': intervention,
                'data': data,
                'n_samples': n_samples
            },
            priority=priority,
            estimated_runtime=estimated_runtime,
            memory_requirement=int(estimated_memory),
            gpu_required=n_samples > 10000
        )
        
        self.active_tasks[task_id] = task
        await self.task_queue.put(task)
        self.performance_metrics['total_tasks'] += 1
        
        return task_id
    
    @resilient_causal_inference
    async def submit_batch_computation(
        self,
        tasks: List[ComputationTask],
        batch_size: Optional[int] = None
    ) -> List[str]:
        """Submit multiple tasks as a batch for optimized execution"""
        
        batch_size = batch_size or min(len(tasks), self.n_workers)
        task_ids = []
        
        # Sort tasks by priority and resource requirements
        sorted_tasks = sorted(tasks, key=lambda t: (-t.priority, -t.estimated_runtime))
        
        # Submit in batches
        for i in range(0, len(sorted_tasks), batch_size):
            batch = sorted_tasks[i:i + batch_size]
            
            for task in batch:
                task_id = task.task_id or f"batch_task_{int(time.time() * 1000)}_{i}"
                task.task_id = task_id
                
                self.active_tasks[task_id] = task
                await self.task_queue.put(task)
                task_ids.append(task_id)
        
        self.performance_metrics['total_tasks'] += len(tasks)
        logger.info(f"Submitted batch of {len(tasks)} tasks")
        
        return task_ids
    
    async def start_workers(self):
        """Start distributed workers for task execution"""
        
        workers = []
        
        if self.use_gpu and self.device_mesh:
            # GPU-based distributed workers
            for i in range(self.n_devices):
                worker = asyncio.create_task(
                    self._gpu_worker(i, self.devices[i])
                )
                workers.append(worker)
        else:
            # CPU-based workers
            for i in range(self.n_workers):
                worker = asyncio.create_task(
                    self._cpu_worker(i)
                )
                workers.append(worker)
        
        logger.info(f"Started {len(workers)} distributed workers")
        return workers
    
    async def _gpu_worker(self, worker_id: int, device: jax.Device):
        """GPU worker for distributed computation"""
        
        worker_name = f"gpu_worker_{worker_id}"
        logger.info(f"Starting {worker_name} on device {device}")
        
        while True:
            try:
                # Get next task
                task = await self.task_queue.get()
                
                if task is None:  # Shutdown signal
                    break
                
                start_time = time.time()
                
                # Execute task on specific device
                with jax.default_device(device):
                    result = await self._execute_task_gpu(task, worker_name)
                
                execution_time = time.time() - start_time
                
                # Record result
                self.completed_tasks[task.task_id] = ComputationResult(
                    task_id=task.task_id,
                    result=result,
                    execution_time=execution_time,
                    memory_used=task.memory_requirement,
                    worker_id=worker_name,
                    success=True
                )
                
                self.active_tasks.pop(task.task_id, None)
                self.performance_metrics['completed_tasks'] += 1
                
                # Update metrics
                self._update_performance_metrics(execution_time)
                
                logger.debug(f"{worker_name} completed task {task.task_id} in {execution_time:.2f}s")
                
            except Exception as e:
                logger.error(f"{worker_name} failed on task {task.task_id}: {e}")
                
                if task.task_id in self.active_tasks:
                    self.completed_tasks[task.task_id] = ComputationResult(
                        task_id=task.task_id,
                        result=None,
                        execution_time=time.time() - start_time,
                        memory_used=0,
                        worker_id=worker_name,
                        success=False,
                        error_message=str(e)
                    )
                    
                    self.active_tasks.pop(task.task_id, None)
                    self.performance_metrics['failed_tasks'] += 1
    
    async def _cpu_worker(self, worker_id: int):
        """CPU worker for distributed computation"""
        
        worker_name = f"cpu_worker_{worker_id}"
        logger.info(f"Starting {worker_name}")
        
        while True:
            try:
                # Get next task
                task = await self.task_queue.get()
                
                if task is None:  # Shutdown signal
                    break
                
                start_time = time.time()
                
                # Execute task in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.thread_pool,
                    self._execute_task_cpu,
                    task,
                    worker_name
                )
                
                execution_time = time.time() - start_time
                
                # Record result
                self.completed_tasks[task.task_id] = ComputationResult(
                    task_id=task.task_id,
                    result=result,
                    execution_time=execution_time,
                    memory_used=task.memory_requirement,
                    worker_id=worker_name,
                    success=True
                )
                
                self.active_tasks.pop(task.task_id, None)
                self.performance_metrics['completed_tasks'] += 1
                
                # Update metrics
                self._update_performance_metrics(execution_time)
                
                logger.debug(f"{worker_name} completed task {task.task_id} in {execution_time:.2f}s")
                
            except Exception as e:
                logger.error(f"{worker_name} failed on task {task.task_id}: {e}")
                
                if task.task_id in self.active_tasks:
                    self.completed_tasks[task.task_id] = ComputationResult(
                        task_id=task.task_id,
                        result=None,
                        execution_time=time.time() - start_time,
                        memory_used=0,
                        worker_id=worker_name,
                        success=False,
                        error_message=str(e)
                    )
                    
                    self.active_tasks.pop(task.task_id, None)
                    self.performance_metrics['failed_tasks'] += 1
    
    async def _execute_task_gpu(self, task: ComputationTask, worker_id: str) -> Any:
        """Execute a task on GPU with JAX acceleration"""
        
        if task.task_type == 'causal_discovery':
            return await self._gpu_causal_discovery(task.input_data, worker_id)
        elif task.task_type == 'intervention':
            return await self._gpu_intervention(task.input_data, worker_id)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
    
    def _execute_task_cpu(self, task: ComputationTask, worker_id: str) -> Any:
        """Execute a task on CPU"""
        
        if task.task_type == 'causal_discovery':
            return self._cpu_causal_discovery(task.input_data, worker_id)
        elif task.task_type == 'intervention':
            return self._cpu_intervention(task.input_data, worker_id)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
    
    async def _gpu_causal_discovery(self, input_data: Dict[str, Any], worker_id: str) -> Dict[str, Any]:
        """GPU-accelerated causal discovery"""
        
        data = input_data['data']
        method = input_data['method']
        hyperparameters = input_data['hyperparameters']
        
        n_samples, n_vars = data.shape
        
        if method == 'distributed_notears':
            result = await self._distributed_notears_gpu(data, hyperparameters)
        elif method == 'parallel_pc':
            result = await self._parallel_pc_gpu(data, hyperparameters)
        elif method == 'scalable_ges':
            result = await self._scalable_ges_gpu(data, hyperparameters)
        else:
            # Fallback to basic gradient-based method
            result = await self._basic_gradient_discovery_gpu(data, hyperparameters)
        
        return {
            'adjacency_matrix': result,
            'method': method,
            'worker_id': worker_id,
            'n_variables': n_vars,
            'n_samples': n_samples
        }
    
    @jax.jit
    def _distributed_notears_gpu(self, data: jnp.ndarray, hyperparameters: Dict[str, Any]) -> jnp.ndarray:
        """Distributed NOTEARS implementation with JAX parallelization"""
        
        n_samples, n_vars = data.shape
        learning_rate = hyperparameters.get('learning_rate', 0.01)
        max_iterations = hyperparameters.get('max_iterations', 1000)
        
        # Initialize adjacency matrix
        W = jax.random.normal(jax.random.PRNGKey(42), (n_vars, n_vars)) * 0.01
        W = W.at[jnp.diag_indices(n_vars)].set(0.0)
        
        @jax.jit
        def notears_loss(W: jnp.ndarray, data: jnp.ndarray) -> float:
            """NOTEARS loss function"""
            # Reconstruction loss
            residuals = data - data @ W.T
            reconstruction_loss = jnp.mean(residuals ** 2)
            
            # Sparsity penalty
            sparsity_loss = 0.1 * jnp.sum(jnp.abs(W))
            
            # Acyclicity constraint using matrix exponential
            acyclicity_loss = jnp.trace(jax.scipy.linalg.expm(W * W)) - n_vars
            
            return reconstruction_loss + sparsity_loss + acyclicity_loss
        
        @jax.jit
        def update_step(W: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
            """Single optimization step with automatic differentiation"""
            grad = jax.grad(notears_loss)(W, data)
            W_new = W - learning_rate * grad
            W_new = W_new.at[jnp.diag_indices(n_vars)].set(0.0)
            return W_new
        
        # Parallel optimization across devices
        if self.device_mesh and self.n_devices > 1:
            # Replicate computation across devices
            W = jax.device_put_replicated(W, self.devices)
            data = jax.device_put_replicated(data, self.devices)
            
            # Vectorize update step for parallel execution
            parallel_update = jax.pmap(update_step)
            
            for _ in range(max_iterations):
                W = parallel_update(W, data)
            
            # Average results from all devices
            W = jnp.mean(W, axis=0)
        else:
            # Single device optimization
            for _ in range(max_iterations):
                W = update_step(W, data)
        
        # Apply threshold for sparsity
        threshold = hyperparameters.get('threshold', 0.1)
        W = jnp.where(jnp.abs(W) > threshold, W, 0.0)
        
        return W
    
    @jax.jit
    def _parallel_pc_gpu(self, data: jnp.ndarray, hyperparameters: Dict[str, Any]) -> jnp.ndarray:
        """Parallel PC algorithm implementation"""
        
        n_samples, n_vars = data.shape
        alpha = hyperparameters.get('alpha', 0.05)
        
        # Initialize skeleton (fully connected)
        skeleton = jnp.ones((n_vars, n_vars)) - jnp.eye(n_vars)
        
        # Parallel independence testing
        @jax.jit
        def test_independence(i: int, j: int, data: jnp.ndarray) -> bool:
            """Test independence between variables i and j"""
            correlation = jnp.corrcoef(data[:, i], data[:, j])[0, 1]
            # Fisher's z-transform
            z_score = jnp.abs(0.5 * jnp.log((1 + correlation) / (1 - correlation)) * jnp.sqrt(n_samples - 3))
            critical_value = 1.96  # Approximate for alpha=0.05
            return z_score < critical_value
        
        # Vectorized independence testing
        i_indices, j_indices = jnp.meshgrid(jnp.arange(n_vars), jnp.arange(n_vars), indexing='ij')
        mask = i_indices < j_indices
        
        # Test all pairs in parallel
        independence_tests = jax.vmap(lambda i, j: test_independence(i, j, data))(
            i_indices[mask], j_indices[mask]
        )
        
        # Update skeleton based on independence tests
        pair_indices = jnp.column_stack([i_indices[mask], j_indices[mask]])
        for idx, is_independent in enumerate(independence_tests):
            if is_independent:
                i, j = pair_indices[idx]
                skeleton = skeleton.at[i, j].set(0.0)
                skeleton = skeleton.at[j, i].set(0.0)
        
        return skeleton
    
    @jax.jit 
    def _scalable_ges_gpu(self, data: jnp.ndarray, hyperparameters: Dict[str, Any]) -> jnp.ndarray:
        """Scalable GES (Greedy Equivalence Search) implementation"""
        
        n_samples, n_vars = data.shape
        max_iterations = hyperparameters.get('max_iterations', 100)
        
        # Initialize empty graph
        adjacency = jnp.zeros((n_vars, n_vars))
        
        @jax.jit
        def bic_score_parallel(adj_matrix: jnp.ndarray, data: jnp.ndarray) -> float:
            """Parallel BIC score computation"""
            total_score = 0.0
            
            for i in range(n_vars):
                parents = jnp.where(adj_matrix[:, i] > 0, size=n_vars, fill_value=-1)
                parents = parents[parents >= 0]
                n_parents = len(parents)
                
                if n_parents > 0:
                    X_parents = data[:, parents]
                    y = data[:, i]
                    
                    # Parallel least squares using JAX
                    coeffs = jnp.linalg.lstsq(X_parents, y, rcond=None)[0]
                    predictions = X_parents @ coeffs
                    residual_var = jnp.var(y - predictions)
                else:
                    residual_var = jnp.var(data[:, i])
                
                # BIC computation
                log_likelihood = -0.5 * n_samples * jnp.log(2 * jnp.pi * residual_var) - 0.5 * n_samples
                bic = -2 * log_likelihood + n_parents * jnp.log(n_samples)
                total_score += bic
            
            return total_score
        
        # Greedy search with parallel score evaluation
        current_score = bic_score_parallel(adjacency, data)
        
        for iteration in range(max_iterations):
            best_score = current_score
            best_adjacency = adjacency
            
            # Parallel evaluation of all possible edge additions
            edge_candidates = []
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j and adjacency[i, j] == 0:
                        edge_candidates.append((i, j))
            
            if not edge_candidates:
                break
            
            # Vectorized score evaluation
            candidate_scores = []
            for i, j in edge_candidates:
                test_adj = adjacency.at[i, j].set(1.0)
                score = bic_score_parallel(test_adj, data)
                candidate_scores.append((score, i, j))
            
            # Find best improvement
            best_candidate = min(candidate_scores, key=lambda x: x[0])
            if best_candidate[0] < best_score:
                best_score = best_candidate[0]
                best_adjacency = adjacency.at[best_candidate[1], best_candidate[2]].set(1.0)
            
            if best_score >= current_score:
                break
            
            adjacency = best_adjacency
            current_score = best_score
        
        return adjacency
    
    @jax.jit
    def _basic_gradient_discovery_gpu(self, data: jnp.ndarray, hyperparameters: Dict[str, Any]) -> jnp.ndarray:
        """Basic gradient-based causal discovery"""
        
        n_samples, n_vars = data.shape
        learning_rate = hyperparameters.get('learning_rate', 0.01)
        max_iterations = hyperparameters.get('max_iterations', 500)
        
        # Initialize adjacency matrix
        W = jax.random.normal(jax.random.PRNGKey(42), (n_vars, n_vars)) * 0.01
        W = W.at[jnp.diag_indices(n_vars)].set(0.0)
        
        @jax.jit
        def loss_fn(W: jnp.ndarray) -> float:
            residuals = data - data @ W.T
            return jnp.mean(residuals ** 2) + 0.1 * jnp.sum(jnp.abs(W))
        
        # Optimization loop
        for _ in range(max_iterations):
            grad = jax.grad(loss_fn)(W)
            W = W - learning_rate * grad
            W = W.at[jnp.diag_indices(n_vars)].set(0.0)
        
        # Apply threshold
        threshold = hyperparameters.get('threshold', 0.1)
        return jnp.where(jnp.abs(W) > threshold, W, 0.0)
    
    def _cpu_causal_discovery(self, input_data: Dict[str, Any], worker_id: str) -> Dict[str, Any]:
        """CPU-based causal discovery (fallback implementation)"""
        
        data = input_data['data']
        method = input_data['method']
        
        # Simple correlation-based discovery for CPU fallback
        n_vars = data.shape[1]
        correlations = jnp.corrcoef(data.T)
        
        # Apply threshold to create adjacency matrix
        threshold = 0.3
        adjacency = jnp.where(jnp.abs(correlations) > threshold, correlations, 0.0)
        adjacency = adjacency.at[jnp.diag_indices(n_vars)].set(0.0)
        
        return {
            'adjacency_matrix': adjacency,
            'method': f"{method}_cpu_fallback",
            'worker_id': worker_id
        }
    
    async def _gpu_intervention(self, input_data: Dict[str, Any], worker_id: str) -> Dict[str, Any]:
        """GPU-accelerated intervention computation"""
        
        dag = input_data['dag']
        intervention = input_data['intervention']
        data = input_data['data']
        n_samples = input_data['n_samples']
        
        result = await self._parallel_intervention_computation(dag, intervention, data, n_samples)
        
        return {
            'pre_intervention_mean': result['pre_mean'],
            'post_intervention_mean': result['post_mean'],
            'intervention_effect': result['effect'],
            'confidence_interval': result['ci'],
            'worker_id': worker_id
        }
    
    @jax.jit
    def _parallel_intervention_computation(
        self,
        dag: jnp.ndarray,
        intervention: Dict[str, float],
        data: jnp.ndarray,
        n_samples: int
    ) -> Dict[str, jnp.ndarray]:
        """Parallel computation of intervention effects"""
        
        n_vars = dag.shape[0]
        
        # Create intervention samples in parallel
        @jax.jit
        def generate_intervention_sample(key: jax.random.PRNGKey) -> jnp.ndarray:
            """Generate single intervention sample"""
            sample = jax.random.normal(key, (n_vars,))
            
            # Apply interventions
            for var_name, value in intervention.items():
                var_idx = int(var_name) if var_name.isdigit() else hash(var_name) % n_vars
                sample = sample.at[var_idx].set(value)
            
            # Propagate through causal structure
            for i in range(n_vars):
                parents = jnp.where(dag[:, i] > 0)[0]
                if len(parents) > 0:
                    effect = jnp.sum(dag[parents, i] * sample[parents])
                    if i not in [int(k) if k.isdigit() else hash(k) % n_vars for k in intervention.keys()]:
                        sample = sample.at[i].add(effect)
            
            return sample
        
        # Generate multiple samples in parallel
        keys = jax.random.split(jax.random.PRNGKey(42), n_samples)
        intervention_samples = jax.vmap(generate_intervention_sample)(keys)
        
        # Compute observational baseline
        observational_mean = jnp.mean(data, axis=0)
        intervention_mean = jnp.mean(intervention_samples, axis=0)
        
        # Calculate effects
        effect = intervention_mean - observational_mean
        
        # Bootstrap confidence intervals
        bootstrap_effects = []
        n_bootstrap = 100
        
        for i in range(n_bootstrap):
            key = jax.random.PRNGKey(i)
            bootstrap_indices = jax.random.choice(key, n_samples, shape=(n_samples,), replace=True)
            bootstrap_samples = intervention_samples[bootstrap_indices]
            bootstrap_effect = jnp.mean(bootstrap_samples, axis=0) - observational_mean
            bootstrap_effects.append(bootstrap_effect)
        
        bootstrap_effects = jnp.stack(bootstrap_effects)
        ci_lower = jnp.percentile(bootstrap_effects, 2.5, axis=0)
        ci_upper = jnp.percentile(bootstrap_effects, 97.5, axis=0)
        
        return {
            'pre_mean': observational_mean,
            'post_mean': intervention_mean,
            'effect': effect,
            'ci': jnp.stack([ci_lower, ci_upper])
        }
    
    def _cpu_intervention(self, input_data: Dict[str, Any], worker_id: str) -> Dict[str, Any]:
        """CPU-based intervention computation (fallback)"""
        
        dag = input_data['dag']
        intervention = input_data['intervention']
        data = input_data['data']
        
        # Simple mean difference computation
        baseline_mean = jnp.mean(data, axis=0)
        
        # Estimate intervention effect using linear approximation
        effect_estimate = jnp.zeros_like(baseline_mean)
        for var_name, value in intervention.items():
            var_idx = int(var_name) if var_name.isdigit() else 0
            if var_idx < len(baseline_mean):
                effect_estimate = effect_estimate.at[var_idx].add(value - baseline_mean[var_idx])
        
        return {
            'pre_intervention_mean': baseline_mean,
            'post_intervention_mean': baseline_mean + effect_estimate,
            'intervention_effect': effect_estimate,
            'confidence_interval': jnp.stack([-jnp.abs(effect_estimate), jnp.abs(effect_estimate)]),
            'worker_id': worker_id
        }
    
    def _update_performance_metrics(self, execution_time: float):
        """Update performance metrics"""
        
        # Update average execution time
        total_completed = self.performance_metrics['completed_tasks']
        current_avg = self.performance_metrics['avg_execution_time']
        new_avg = (current_avg * (total_completed - 1) + execution_time) / total_completed
        self.performance_metrics['avg_execution_time'] = new_avg
        
        # Update throughput (tasks per second)
        if execution_time > 0:
            self.performance_metrics['throughput'] = 1.0 / new_avg
    
    async def get_result(self, task_id: str, timeout: float = 300.0) -> Optional[ComputationResult]:
        """Get result for a specific task with timeout"""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            
            if task_id not in self.active_tasks:
                return None  # Task doesn't exist
            
            await asyncio.sleep(0.1)  # Check every 100ms
        
        # Timeout reached
        logger.warning(f"Task {task_id} timed out after {timeout}s")
        return None
    
    async def get_batch_results(
        self, 
        task_ids: List[str], 
        timeout: float = 300.0
    ) -> Dict[str, ComputationResult]:
        """Get results for multiple tasks"""
        
        results = {}
        start_time = time.time()
        
        while len(results) < len(task_ids) and time.time() - start_time < timeout:
            for task_id in task_ids:
                if task_id not in results and task_id in self.completed_tasks:
                    results[task_id] = self.completed_tasks[task_id]
            
            if len(results) < len(task_ids):
                await asyncio.sleep(0.1)
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        
        total_tasks = self.performance_metrics['total_tasks']
        completed_tasks = self.performance_metrics['completed_tasks']
        
        return {
            **self.performance_metrics,
            'success_rate': completed_tasks / max(1, total_tasks),
            'queue_size': self.task_queue.qsize(),
            'active_tasks': len(self.active_tasks),
            'workers': self.n_workers,
            'devices': len(self.devices),
            'memory_limit_mb': self.memory_limit
        }
    
    async def shutdown(self):
        """Shutdown the distributed compute system"""
        
        logger.info("Shutting down distributed compute system...")
        
        # Send shutdown signals to workers
        for _ in range(self.n_workers):
            await self.task_queue.put(None)
        
        # Close executor pools
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=True)
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        
        logger.info("Distributed compute system shutdown complete")


# Global distributed compute instance
distributed_compute = None

def get_distributed_compute(
    n_workers: Optional[int] = None,
    use_gpu: bool = True,
    memory_limit_gb: float = 16.0
) -> DistributedCausalCompute:
    """Get or create global distributed compute instance"""
    
    global distributed_compute
    
    if distributed_compute is None:
        distributed_compute = DistributedCausalCompute(
            n_workers=n_workers,
            use_gpu=use_gpu,
            memory_limit_gb=memory_limit_gb
        )
    
    return distributed_compute


# Convenience functions for common operations
async def distributed_causal_discovery(
    data: jnp.ndarray,
    method: str = 'distributed_notears',
    hyperparameters: Optional[Dict[str, Any]] = None,
    timeout: float = 300.0
) -> Optional[Dict[str, Any]]:
    """Convenience function for distributed causal discovery"""
    
    compute = get_distributed_compute()
    
    # Start workers if not already running
    workers = await compute.start_workers()
    
    try:
        # Submit task
        task_id = await compute.submit_causal_discovery_task(
            data=data,
            method=method,
            hyperparameters=hyperparameters
        )
        
        # Get result
        result = await compute.get_result(task_id, timeout=timeout)
        
        if result and result.success:
            return result.result
        else:
            logger.error(f"Task failed: {result.error_message if result else 'Timeout'}")
            return None
            
    finally:
        # Note: In practice, you might want to keep workers running
        # await compute.shutdown()
        pass


async def distributed_intervention(
    dag: jnp.ndarray,
    intervention: Dict[str, float],
    data: jnp.ndarray,
    n_samples: int = 1000,
    timeout: float = 300.0
) -> Optional[Dict[str, Any]]:
    """Convenience function for distributed intervention computation"""
    
    compute = get_distributed_compute()
    
    # Start workers if not already running
    workers = await compute.start_workers()
    
    try:
        # Submit task
        task_id = await compute.submit_intervention_task(
            dag=dag,
            intervention=intervention,
            data=data,
            n_samples=n_samples
        )
        
        # Get result
        result = await compute.get_result(task_id, timeout=timeout)
        
        if result and result.success:
            return result.result
        else:
            logger.error(f"Task failed: {result.error_message if result else 'Timeout'}")
            return None
            
    finally:
        # Note: Keep workers running for subsequent tasks
        pass


# Export main classes and functions
__all__ = [
    'DistributedCausalCompute',
    'ComputationTask',
    'ComputationResult',
    'get_distributed_compute',
    'distributed_causal_discovery',
    'distributed_intervention'
]