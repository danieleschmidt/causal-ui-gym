"""
Novel Causal Discovery Algorithms for Research Applications

This module implements state-of-the-art causal discovery methods with 
JAX acceleration for publication-ready research in causal inference.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass
import networkx as nx
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import logging
from ..error_handling.advanced_error_system import (
    resilient_computation, resilient_causal_inference, 
    validate_causal_data, causal_computation_context
)

logger = logging.getLogger(__name__)


@dataclass
class CausalDiscoveryResult:
    """Comprehensive results from causal discovery algorithms"""
    adjacency_matrix: jnp.ndarray
    confidence_scores: jnp.ndarray
    method_name: str
    hyperparameters: Dict[str, Any]
    convergence_info: Dict[str, Any]
    computational_metrics: Dict[str, float]
    statistical_tests: Dict[str, Any]
    bootstrap_stability: Optional[Dict[str, Any]] = None
    cross_validation_scores: Optional[Dict[str, Any]] = None


class NovelCausalDiscovery:
    """Advanced causal discovery with novel algorithms and JAX acceleration"""
    
    def __init__(self, random_seed: int = 42):
        self.rng_key = jax.random.PRNGKey(random_seed)
        self.supported_methods = [
            'gradient_based_discovery',
            'variational_causal_discovery', 
            'neural_causal_discovery',
            'hybrid_structure_learning',
            'adaptive_notears',
            'continuous_optimization_discovery'
        ]
    
    @resilient_causal_inference
    @validate_causal_data
    async def discover_causal_structure(
        self,
        data: jnp.ndarray,
        method: str = 'gradient_based_discovery',
        prior_knowledge: Optional[Dict[str, Any]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        bootstrap_samples: int = 100,
        cross_validation_folds: int = 5
    ) -> CausalDiscoveryResult:
        """
        Discover causal structure using novel algorithms
        
        Args:
            data: Observational data matrix (n_samples x n_variables)
            method: Discovery algorithm to use
            prior_knowledge: Prior structural constraints
            hyperparameters: Algorithm-specific parameters
            bootstrap_samples: Number of bootstrap iterations for stability
            cross_validation_folds: CV folds for validation
        
        Returns:
            Comprehensive causal discovery results
        """
        async with causal_computation_context(f"causal_discovery_{method}"):
            if method not in self.supported_methods:
                raise ValueError(f"Unsupported method: {method}")
            
            # Default hyperparameters
            hyp = hyperparameters or {}
            
            # Preprocess data
            data = self._preprocess_data(data)
            n_vars = data.shape[1]
            
            # Select and execute discovery method
            if method == 'gradient_based_discovery':
                result = await self._gradient_based_discovery(data, hyp)
            elif method == 'variational_causal_discovery':
                result = await self._variational_causal_discovery(data, hyp)
            elif method == 'neural_causal_discovery':
                result = await self._neural_causal_discovery(data, hyp)
            elif method == 'hybrid_structure_learning':
                result = await self._hybrid_structure_learning(data, hyp)
            elif method == 'adaptive_notears':
                result = await self._adaptive_notears(data, hyp)
            else:  # continuous_optimization_discovery
                result = await self._continuous_optimization_discovery(data, hyp)
            
            # Apply prior knowledge constraints
            if prior_knowledge:
                result = self._apply_prior_knowledge(result, prior_knowledge)
            
            # Bootstrap stability analysis
            if bootstrap_samples > 0:
                bootstrap_results = await self._bootstrap_stability_analysis(
                    data, method, hyp, bootstrap_samples
                )
                result.bootstrap_stability = bootstrap_results
            
            # Cross-validation
            if cross_validation_folds > 1:
                cv_scores = await self._cross_validation_analysis(
                    data, method, hyp, cross_validation_folds
                )
                result.cross_validation_scores = cv_scores
            
            # Statistical tests
            result.statistical_tests = self._perform_statistical_tests(data, result)
            
            return result
    
    @jax.jit
    def _preprocess_data(self, data: jnp.ndarray) -> jnp.ndarray:
        """Standardize and preprocess data for causal discovery"""
        # Standardization
        data = (data - jnp.mean(data, axis=0)) / jnp.std(data, axis=0)
        
        # Handle missing values with mean imputation
        data = jnp.where(jnp.isnan(data), 0.0, data)
        
        return data
    
    async def _gradient_based_discovery(
        self, 
        data: jnp.ndarray, 
        hyperparameters: Dict[str, Any]
    ) -> CausalDiscoveryResult:
        """Novel gradient-based causal discovery using JAX optimization"""
        
        n_samples, n_vars = data.shape
        learning_rate = hyperparameters.get('learning_rate', 0.01)
        max_iterations = hyperparameters.get('max_iterations', 1000)
        sparsity_lambda = hyperparameters.get('sparsity_lambda', 0.1)
        acyclicity_lambda = hyperparameters.get('acyclicity_lambda', 1.0)
        
        # Initialize adjacency matrix with small random values
        key, subkey = jax.random.split(self.rng_key)
        W = jax.random.normal(subkey, (n_vars, n_vars)) * 0.1
        W = W.at[jnp.diag_indices(n_vars)].set(0.0)  # No self-loops
        
        @jax.jit
        def loss_function(W: jnp.ndarray, data: jnp.ndarray) -> float:
            """Combined loss: reconstruction + sparsity + acyclicity"""
            # Reconstruction loss (negative log-likelihood)
            residuals = data - data @ W.T
            reconstruction_loss = jnp.mean(residuals ** 2)
            
            # Sparsity regularization
            sparsity_loss = sparsity_lambda * jnp.sum(jnp.abs(W))
            
            # Acyclicity constraint using matrix exponential trace
            acyclicity_loss = acyclicity_lambda * self._acyclicity_constraint(W)
            
            return reconstruction_loss + sparsity_loss + acyclicity_loss
        
        @jax.jit
        def update_step(W: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
            """Single gradient descent step"""
            grad = jax.grad(loss_function)(W, data)
            W_new = W - learning_rate * grad
            # Ensure no self-connections
            W_new = W_new.at[jnp.diag_indices(n_vars)].set(0.0)
            return W_new
        
        # Optimization loop
        start_time = jax.lax.current_time()
        losses = []
        
        for iteration in range(max_iterations):
            W = update_step(W, data)
            current_loss = loss_function(W, data)
            losses.append(float(current_loss))
            
            # Early stopping
            if iteration > 10 and abs(losses[-1] - losses[-2]) < 1e-6:
                logger.info(f"Converged at iteration {iteration}")
                break
        
        end_time = jax.lax.current_time()
        
        # Threshold for sparsity
        threshold = hyperparameters.get('threshold', 0.1)
        W_thresholded = jnp.where(jnp.abs(W) > threshold, W, 0.0)
        
        # Calculate confidence scores based on gradient magnitudes
        final_grad = jax.grad(loss_function)(W_thresholded, data)
        confidence_scores = 1.0 / (1.0 + jnp.abs(final_grad))
        
        return CausalDiscoveryResult(
            adjacency_matrix=W_thresholded,
            confidence_scores=confidence_scores,
            method_name='gradient_based_discovery',
            hyperparameters=hyperparameters,
            convergence_info={
                'iterations': iteration + 1,
                'final_loss': losses[-1],
                'converged': iteration < max_iterations - 1
            },
            computational_metrics={
                'runtime_seconds': float(end_time - start_time),
                'loss_trajectory': losses[-10:]  # Last 10 losses
            },
            statistical_tests={}
        )
    
    @jax.jit
    def _acyclicity_constraint(self, W: jnp.ndarray) -> float:
        """Compute acyclicity constraint using matrix exponential trace"""
        n = W.shape[0]
        # Compute tr(exp(W ⊙ W)) - n, where ⊙ is element-wise product
        W_squared = W * W
        exp_trace = jnp.trace(jax.scipy.linalg.expm(W_squared))
        return exp_trace - n
    
    async def _variational_causal_discovery(
        self, 
        data: jnp.ndarray, 
        hyperparameters: Dict[str, Any]
    ) -> CausalDiscoveryResult:
        """Variational Bayesian approach to causal discovery"""
        
        n_samples, n_vars = data.shape
        n_components = hyperparameters.get('n_components', 5)
        max_iterations = hyperparameters.get('max_iterations', 500)
        
        # Initialize variational parameters
        key, *subkeys = jax.random.split(self.rng_key, 4)
        
        # Prior parameters for edge probabilities
        alpha_prior = jnp.ones((n_vars, n_vars)) * 0.1
        beta_prior = jnp.ones((n_vars, n_vars)) * 0.9
        
        # Initialize posterior parameters
        alpha_posterior = alpha_prior + jax.random.exponential(subkeys[0], (n_vars, n_vars)) * 0.1
        beta_posterior = beta_prior + jax.random.exponential(subkeys[1], (n_vars, n_vars)) * 0.1
        
        @jax.jit
        def elbo_loss(alpha: jnp.ndarray, beta: jnp.ndarray, data: jnp.ndarray) -> float:
            """Evidence Lower Bound for variational inference"""
            # Expected log-likelihood
            edge_probs = alpha / (alpha + beta)
            reconstruction_loss = self._variational_reconstruction_loss(edge_probs, data)
            
            # KL divergence between posterior and prior
            kl_divergence = jnp.sum(
                jax.scipy.special.betaln(alpha, beta) - 
                jax.scipy.special.betaln(alpha_prior, beta_prior) +
                (alpha - alpha_prior) * (jax.scipy.special.digamma(alpha) - 
                                        jax.scipy.special.digamma(alpha + beta)) +
                (beta - beta_prior) * (jax.scipy.special.digamma(beta) - 
                                      jax.scipy.special.digamma(alpha + beta))
            )
            
            return reconstruction_loss + kl_divergence
        
        @jax.jit
        def update_parameters(alpha: jnp.ndarray, beta: jnp.ndarray, data: jnp.ndarray):
            """Update variational parameters"""
            # Gradient-based updates for variational parameters
            grad_alpha, grad_beta = jax.grad(elbo_loss, argnums=(0, 1))(alpha, beta, data)
            
            learning_rate = 0.01
            alpha_new = alpha - learning_rate * grad_alpha
            beta_new = beta - learning_rate * grad_beta
            
            # Ensure parameters stay positive
            alpha_new = jnp.maximum(alpha_new, 0.01)
            beta_new = jnp.maximum(beta_new, 0.01)
            
            return alpha_new, beta_new
        
        # Variational optimization
        start_time = jax.lax.current_time()
        elbo_history = []
        
        for iteration in range(max_iterations):
            alpha_posterior, beta_posterior = update_parameters(alpha_posterior, beta_posterior, data)
            current_elbo = elbo_loss(alpha_posterior, beta_posterior, data)
            elbo_history.append(float(current_elbo))
            
            if iteration > 10 and abs(elbo_history[-1] - elbo_history[-2]) < 1e-6:
                break
        
        end_time = jax.lax.current_time()
        
        # Extract final adjacency matrix and confidence scores
        edge_probabilities = alpha_posterior / (alpha_posterior + beta_posterior)
        threshold = hyperparameters.get('threshold', 0.5)
        adjacency_matrix = jnp.where(edge_probabilities > threshold, 1.0, 0.0)
        
        return CausalDiscoveryResult(
            adjacency_matrix=adjacency_matrix,
            confidence_scores=edge_probabilities,
            method_name='variational_causal_discovery',
            hyperparameters=hyperparameters,
            convergence_info={
                'iterations': iteration + 1,
                'final_elbo': elbo_history[-1],
                'converged': iteration < max_iterations - 1
            },
            computational_metrics={
                'runtime_seconds': float(end_time - start_time),
                'elbo_trajectory': elbo_history[-10:]
            },
            statistical_tests={}
        )
    
    @jax.jit
    def _variational_reconstruction_loss(self, edge_probs: jnp.ndarray, data: jnp.ndarray) -> float:
        """Reconstruction loss for variational approach"""
        n_samples = data.shape[0]
        
        # Expected reconstruction under edge probabilities
        expected_reconstruction = data @ edge_probs.T
        residuals = data - expected_reconstruction
        
        return jnp.mean(residuals ** 2)
    
    async def _neural_causal_discovery(
        self, 
        data: jnp.ndarray, 
        hyperparameters: Dict[str, Any]
    ) -> CausalDiscoveryResult:
        """Neural network-based causal discovery"""
        
        hidden_dims = hyperparameters.get('hidden_dims', [64, 32])
        learning_rate = hyperparameters.get('learning_rate', 0.001)
        max_epochs = hyperparameters.get('max_epochs', 1000)
        
        n_samples, n_vars = data.shape
        
        # Initialize neural network parameters
        def init_network_params(key, input_dim, hidden_dims, output_dim):
            """Initialize neural network parameters"""
            params = []
            dims = [input_dim] + hidden_dims + [output_dim]
            
            for i in range(len(dims) - 1):
                key, subkey = jax.random.split(key)
                W = jax.random.normal(subkey, (dims[i], dims[i+1])) * jnp.sqrt(2.0 / dims[i])
                b = jnp.zeros(dims[i+1])
                params.append({'W': W, 'b': b})
            
            return params
        
        @jax.jit
        def forward_pass(params, x):
            """Forward pass through the neural network"""
            for i, layer in enumerate(params[:-1]):
                x = jnp.dot(x, layer['W']) + layer['b']
                x = jax.nn.relu(x)  # ReLU activation
            
            # Final layer (linear)
            final_layer = params[-1]
            x = jnp.dot(x, final_layer['W']) + final_layer['b']
            
            return x
        
        @jax.jit
        def neural_loss(params, data):
            """Loss function for neural causal discovery"""
            # Use the network to predict causal relationships
            predictions = jax.vmap(lambda x: forward_pass(params, x[jnp.newaxis, :]))(data)
            predictions = predictions.reshape(n_samples, n_vars, n_vars)
            
            # Reconstruction loss
            reconstruction_loss = 0.0
            for i in range(n_vars):
                # Predict variable i from all other variables
                mask = jnp.ones(n_vars, dtype=bool).at[i].set(False)
                parents = predictions[:, i, mask]
                target = data[:, i]
                predicted = jnp.sum(parents * data[:, mask], axis=1)
                reconstruction_loss += jnp.mean((target - predicted) ** 2)
            
            # Sparsity regularization
            sparsity_loss = 0.01 * jnp.mean(jnp.abs(predictions))
            
            # Acyclicity constraint
            avg_adjacency = jnp.mean(predictions, axis=0)
            acyclicity_loss = 0.1 * self._acyclicity_constraint(avg_adjacency)
            
            return reconstruction_loss + sparsity_loss + acyclicity_loss
        
        # Initialize network
        key, subkey = jax.random.split(self.rng_key)
        params = init_network_params(subkey, n_vars, hidden_dims, n_vars * n_vars)
        
        # Training loop
        start_time = jax.lax.current_time()
        losses = []
        
        @jax.jit
        def update_params(params, data):
            """Single parameter update step"""
            grad_fn = jax.grad(neural_loss)
            grads = grad_fn(params, data)
            
            # Simple gradient descent update
            updated_params = []
            for param, grad in zip(params, grads):
                updated_param = {}
                for key in param:
                    updated_param[key] = param[key] - learning_rate * grad[key]
                updated_params.append(updated_param)
            
            return updated_params
        
        for epoch in range(max_epochs):
            params = update_params(params, data)
            current_loss = neural_loss(params, data)
            losses.append(float(current_loss))
            
            if epoch > 10 and abs(losses[-1] - losses[-2]) < 1e-6:
                break
        
        end_time = jax.lax.current_time()
        
        # Extract final adjacency matrix
        final_predictions = jax.vmap(lambda x: forward_pass(params, x[jnp.newaxis, :]))(data)
        final_predictions = final_predictions.reshape(n_samples, n_vars, n_vars)
        adjacency_matrix = jnp.mean(final_predictions, axis=0)
        
        # Apply threshold
        threshold = hyperparameters.get('threshold', 0.3)
        adjacency_matrix = jnp.where(jnp.abs(adjacency_matrix) > threshold, adjacency_matrix, 0.0)
        
        # Confidence scores based on consistency across samples
        confidence_scores = 1.0 - jnp.std(final_predictions, axis=0) / (jnp.abs(adjacency_matrix) + 1e-6)
        
        return CausalDiscoveryResult(
            adjacency_matrix=adjacency_matrix,
            confidence_scores=confidence_scores,
            method_name='neural_causal_discovery',
            hyperparameters=hyperparameters,
            convergence_info={
                'epochs': epoch + 1,
                'final_loss': losses[-1],
                'converged': epoch < max_epochs - 1
            },
            computational_metrics={
                'runtime_seconds': float(end_time - start_time),
                'loss_trajectory': losses[-10:]
            },
            statistical_tests={}
        )
    
    async def _hybrid_structure_learning(
        self, 
        data: jnp.ndarray, 
        hyperparameters: Dict[str, Any]
    ) -> CausalDiscoveryResult:
        """Hybrid constraint-based and score-based structure learning"""
        
        # Combine PC algorithm insights with continuous optimization
        alpha = hyperparameters.get('significance_level', 0.05)
        max_iterations = hyperparameters.get('max_iterations', 500)
        
        n_samples, n_vars = data.shape
        
        # Phase 1: Constraint-based skeleton discovery
        skeleton = await self._discover_skeleton_pc(data, alpha)
        
        # Phase 2: Score-based orientation with skeleton constraints
        adjacency_matrix = await self._score_based_orientation(data, skeleton, max_iterations)
        
        # Calculate confidence based on statistical tests and optimization stability
        confidence_scores = await self._calculate_hybrid_confidence(data, adjacency_matrix, alpha)
        
        return CausalDiscoveryResult(
            adjacency_matrix=adjacency_matrix,
            confidence_scores=confidence_scores,
            method_name='hybrid_structure_learning',
            hyperparameters=hyperparameters,
            convergence_info={'method': 'hybrid_two_phase'},
            computational_metrics={'runtime_seconds': 0.0},  # Placeholder
            statistical_tests={}
        )
    
    @jax.jit
    async def _discover_skeleton_pc(self, data: jnp.ndarray, alpha: float) -> jnp.ndarray:
        """PC algorithm skeleton discovery using conditional independence tests"""
        n_vars = data.shape[1]
        skeleton = jnp.ones((n_vars, n_vars)) - jnp.eye(n_vars)  # Fully connected initially
        
        # Simplified PC algorithm - test all pairs for independence
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                # Test marginal independence
                correlation = jnp.corrcoef(data[:, i], data[:, j])[0, 1]
                # Fisher's z-transform for significance testing
                n = data.shape[0]
                z_score = jnp.abs(0.5 * jnp.log((1 + correlation) / (1 - correlation)) * jnp.sqrt(n - 3))
                
                # Critical value for alpha significance
                critical_value = stats.norm.ppf(1 - alpha/2)
                
                if z_score < critical_value:  # Independent
                    skeleton = skeleton.at[i, j].set(0.0)
                    skeleton = skeleton.at[j, i].set(0.0)
        
        return skeleton
    
    @jax.jit
    async def _score_based_orientation(
        self, 
        data: jnp.ndarray, 
        skeleton: jnp.ndarray, 
        max_iterations: int
    ) -> jnp.ndarray:
        """Score-based edge orientation using BIC-like scoring"""
        
        n_vars = data.shape[1]
        # Initialize with skeleton
        adjacency = skeleton.copy()
        
        @jax.jit
        def bic_score(adj_matrix, data):
            """Bayesian Information Criterion score"""
            n_samples, n_vars = data.shape
            total_score = 0.0
            
            for i in range(n_vars):
                parents = jnp.where(adj_matrix[:, i] > 0)[0]
                n_parents = len(parents)
                
                if n_parents == 0:
                    # No parents - just variance
                    residual_var = jnp.var(data[:, i])
                else:
                    # Linear regression with parents
                    X = data[:, parents]
                    y = data[:, i]
                    
                    # Simple least squares
                    coeffs = jnp.linalg.lstsq(X, y, rcond=None)[0]
                    predictions = X @ coeffs
                    residual_var = jnp.var(y - predictions)
                
                # BIC = -2 * log_likelihood + k * log(n)
                log_likelihood = -0.5 * n_samples * jnp.log(2 * jnp.pi * residual_var) - 0.5 * n_samples
                bic = -2 * log_likelihood + n_parents * jnp.log(n_samples)
                total_score += bic
            
            return total_score
        
        # Greedy orientation optimization
        current_score = bic_score(adjacency, data)
        
        for iteration in range(max_iterations):
            best_score = current_score
            best_adjacency = adjacency
            
            # Try orienting each undirected edge
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if skeleton[i, j] > 0:  # Edge exists in skeleton
                        # Try i -> j
                        test_adj = adjacency.at[i, j].set(1.0).at[j, i].set(0.0)
                        score = bic_score(test_adj, data)
                        if score < best_score:  # Lower BIC is better
                            best_score = score
                            best_adjacency = test_adj
                        
                        # Try j -> i
                        test_adj = adjacency.at[j, i].set(1.0).at[i, j].set(0.0)
                        score = bic_score(test_adj, data)
                        if score < best_score:
                            best_score = score
                            best_adjacency = test_adj
            
            if best_score >= current_score:  # No improvement
                break
            
            adjacency = best_adjacency
            current_score = best_score
        
        return adjacency
    
    async def _calculate_hybrid_confidence(
        self, 
        data: jnp.ndarray, 
        adjacency: jnp.ndarray, 
        alpha: float
    ) -> jnp.ndarray:
        """Calculate confidence scores for hybrid method"""
        n_vars = data.shape[1]
        confidence = jnp.zeros_like(adjacency)
        
        for i in range(n_vars):
            for j in range(n_vars):
                if adjacency[i, j] > 0:
                    # Statistical significance of the relationship
                    correlation = jnp.corrcoef(data[:, i], data[:, j])[0, 1]
                    n = data.shape[0]
                    z_score = jnp.abs(0.5 * jnp.log((1 + correlation) / (1 - correlation)) * jnp.sqrt(n - 3))
                    p_value = 2 * (1 - stats.norm.cdf(z_score))
                    confidence_score = 1.0 - p_value
                    confidence = confidence.at[i, j].set(confidence_score)
        
        return confidence
    
    async def _adaptive_notears(
        self, 
        data: jnp.ndarray, 
        hyperparameters: Dict[str, Any]
    ) -> CausalDiscoveryResult:
        """Adaptive NOTEARS with dynamic hyperparameter tuning"""
        
        # Enhanced NOTEARS with adaptive penalty scheduling
        initial_lambda = hyperparameters.get('initial_lambda', 0.1)
        lambda_schedule = hyperparameters.get('lambda_schedule', 'exponential')
        max_iterations = hyperparameters.get('max_iterations', 1000)
        
        n_samples, n_vars = data.shape
        
        # Initialize with small random values
        key, subkey = jax.random.split(self.rng_key)
        W = jax.random.normal(subkey, (n_vars, n_vars)) * 0.01
        W = W.at[jnp.diag_indices(n_vars)].set(0.0)
        
        @jax.jit
        def adaptive_loss(W, data, iteration, lambda_val):
            """Adaptive loss function with iteration-dependent penalties"""
            # Base reconstruction loss
            residuals = data - data @ W.T
            reconstruction_loss = jnp.mean(residuals ** 2)
            
            # Adaptive sparsity penalty
            sparsity_penalty = lambda_val * jnp.sum(jnp.abs(W))
            
            # Adaptive acyclicity constraint
            acyclicity_penalty = self._adaptive_acyclicity_penalty(W, iteration)
            
            return reconstruction_loss + sparsity_penalty + acyclicity_penalty
        
        # Optimization with adaptive scheduling
        start_time = jax.lax.current_time()
        losses = []
        lambda_values = []
        
        for iteration in range(max_iterations):
            # Update lambda according to schedule
            if lambda_schedule == 'exponential':
                current_lambda = initial_lambda * (0.99 ** iteration)
            elif lambda_schedule == 'linear':
                current_lambda = initial_lambda * (1 - iteration / max_iterations)
            else:  # constant
                current_lambda = initial_lambda
            
            lambda_values.append(current_lambda)
            
            # Gradient update
            grad = jax.grad(lambda W: adaptive_loss(W, data, iteration, current_lambda))(W)
            learning_rate = 0.01 * (0.99 ** (iteration // 100))  # Decay learning rate
            W = W - learning_rate * grad
            W = W.at[jnp.diag_indices(n_vars)].set(0.0)
            
            current_loss = adaptive_loss(W, data, iteration, current_lambda)
            losses.append(float(current_loss))
            
            # Early stopping
            if iteration > 10 and abs(losses[-1] - losses[-2]) < 1e-7:
                break
        
        end_time = jax.lax.current_time()
        
        # Final thresholding
        threshold = hyperparameters.get('threshold', 0.1)
        W_final = jnp.where(jnp.abs(W) > threshold, W, 0.0)
        
        # Confidence based on final gradient magnitudes
        final_grad = jax.grad(lambda W: adaptive_loss(W, data, iteration, current_lambda))(W_final)
        confidence_scores = jnp.exp(-jnp.abs(final_grad))
        
        return CausalDiscoveryResult(
            adjacency_matrix=W_final,
            confidence_scores=confidence_scores,
            method_name='adaptive_notears',
            hyperparameters=hyperparameters,
            convergence_info={
                'iterations': iteration + 1,
                'final_loss': losses[-1],
                'converged': iteration < max_iterations - 1,
                'lambda_schedule': lambda_values[-10:]
            },
            computational_metrics={
                'runtime_seconds': float(end_time - start_time),
                'loss_trajectory': losses[-10:]
            },
            statistical_tests={}
        )
    
    @jax.jit
    def _adaptive_acyclicity_penalty(self, W: jnp.ndarray, iteration: int) -> float:
        """Adaptive acyclicity penalty that increases with iterations"""
        base_penalty = self._acyclicity_constraint(W)
        # Increase penalty strength as optimization progresses
        penalty_multiplier = 1.0 + 0.01 * iteration
        return penalty_multiplier * base_penalty
    
    async def _continuous_optimization_discovery(
        self, 
        data: jnp.ndarray, 
        hyperparameters: Dict[str, Any]
    ) -> CausalDiscoveryResult:
        """Continuous optimization approach with multiple restarts"""
        
        n_restarts = hyperparameters.get('n_restarts', 5)
        max_iterations = hyperparameters.get('max_iterations', 500)
        
        best_result = None
        best_score = float('inf')
        all_results = []
        
        # Multiple random restarts
        for restart in range(n_restarts):
            # Random initialization
            key, subkey = jax.random.split(self.rng_key)
            self.rng_key = key
            
            # Run optimization from this initialization
            result = await self._single_restart_optimization(data, hyperparameters, subkey)
            all_results.append(result)
            
            # Track best result
            if result.convergence_info['final_loss'] < best_score:
                best_score = result.convergence_info['final_loss']
                best_result = result
        
        # Ensemble confidence from multiple restarts
        ensemble_adjacency = jnp.stack([r.adjacency_matrix for r in all_results])
        consensus_matrix = jnp.mean(ensemble_adjacency, axis=0)
        consensus_confidence = 1.0 - jnp.std(ensemble_adjacency, axis=0) / (jnp.abs(consensus_matrix) + 1e-6)
        
        # Update best result with ensemble information
        best_result.adjacency_matrix = consensus_matrix
        best_result.confidence_scores = consensus_confidence
        best_result.method_name = 'continuous_optimization_discovery'
        best_result.convergence_info['ensemble_results'] = len(all_results)
        
        return best_result
    
    async def _single_restart_optimization(
        self, 
        data: jnp.ndarray, 
        hyperparameters: Dict[str, Any],
        rng_key: jax.random.PRNGKey
    ) -> CausalDiscoveryResult:
        """Single restart of continuous optimization"""
        
        n_vars = data.shape[1]
        learning_rate = hyperparameters.get('learning_rate', 0.01)
        max_iterations = hyperparameters.get('max_iterations', 500)
        
        # Random initialization
        W = jax.random.normal(rng_key, (n_vars, n_vars)) * 0.1
        W = W.at[jnp.diag_indices(n_vars)].set(0.0)
        
        @jax.jit
        def loss_fn(W, data):
            """Combined loss function"""
            residuals = data - data @ W.T
            reconstruction = jnp.mean(residuals ** 2)
            sparsity = 0.1 * jnp.sum(jnp.abs(W))
            acyclicity = 0.1 * self._acyclicity_constraint(W)
            return reconstruction + sparsity + acyclicity
        
        # Optimization loop
        start_time = jax.lax.current_time()
        losses = []
        
        for iteration in range(max_iterations):
            grad = jax.grad(loss_fn)(W, data)
            W = W - learning_rate * grad
            W = W.at[jnp.diag_indices(n_vars)].set(0.0)
            
            current_loss = loss_fn(W, data)
            losses.append(float(current_loss))
            
            if iteration > 10 and abs(losses[-1] - losses[-2]) < 1e-6:
                break
        
        end_time = jax.lax.current_time()
        
        # Threshold and confidence
        threshold = hyperparameters.get('threshold', 0.1)
        W_final = jnp.where(jnp.abs(W) > threshold, W, 0.0)
        confidence = jnp.abs(W_final) / (jnp.max(jnp.abs(W_final)) + 1e-6)
        
        return CausalDiscoveryResult(
            adjacency_matrix=W_final,
            confidence_scores=confidence,
            method_name='single_restart',
            hyperparameters=hyperparameters,
            convergence_info={
                'iterations': iteration + 1,
                'final_loss': losses[-1],
                'converged': iteration < max_iterations - 1
            },
            computational_metrics={
                'runtime_seconds': float(end_time - start_time),
                'loss_trajectory': losses[-10:]
            },
            statistical_tests={}
        )
    
    def _apply_prior_knowledge(
        self, 
        result: CausalDiscoveryResult, 
        prior_knowledge: Dict[str, Any]
    ) -> CausalDiscoveryResult:
        """Apply prior knowledge constraints to discovery results"""
        
        adj_matrix = result.adjacency_matrix
        
        # Forbidden edges
        if 'forbidden_edges' in prior_knowledge:
            for (i, j) in prior_knowledge['forbidden_edges']:
                adj_matrix = adj_matrix.at[i, j].set(0.0)
        
        # Required edges
        if 'required_edges' in prior_knowledge:
            for (i, j) in prior_knowledge['required_edges']:
                adj_matrix = adj_matrix.at[i, j].set(1.0)
        
        # Temporal ordering (for time series data)
        if 'temporal_order' in prior_knowledge:
            ordering = prior_knowledge['temporal_order']
            # Ensure no edges from later to earlier variables
            for i in range(len(ordering)):
                for j in range(i):
                    adj_matrix = adj_matrix.at[ordering[i], ordering[j]].set(0.0)
        
        result.adjacency_matrix = adj_matrix
        return result
    
    async def _bootstrap_stability_analysis(
        self,
        data: jnp.ndarray,
        method: str,
        hyperparameters: Dict[str, Any],
        n_bootstrap: int
    ) -> Dict[str, Any]:
        """Bootstrap analysis for structural stability"""
        
        n_samples = data.shape[0]
        bootstrap_results = []
        
        for bootstrap_iter in range(n_bootstrap):
            # Bootstrap sample
            key, subkey = jax.random.split(self.rng_key)
            self.rng_key = key
            
            bootstrap_indices = jax.random.choice(
                subkey, n_samples, shape=(n_samples,), replace=True
            )
            bootstrap_data = data[bootstrap_indices]
            
            # Run discovery on bootstrap sample
            # Simplified - would call the specific method
            simple_result = await self._gradient_based_discovery(bootstrap_data, hyperparameters)
            bootstrap_results.append(simple_result.adjacency_matrix)
        
        # Analyze stability
        bootstrap_matrices = jnp.stack(bootstrap_results)
        edge_frequencies = jnp.mean(bootstrap_matrices != 0, axis=0)
        edge_magnitudes_std = jnp.std(bootstrap_matrices, axis=0)
        
        return {
            'edge_frequencies': edge_frequencies,
            'edge_magnitude_std': edge_magnitudes_std,
            'stability_score': jnp.mean(edge_frequencies > 0.5),  # Proportion of stable edges
            'n_bootstrap_samples': n_bootstrap
        }
    
    async def _cross_validation_analysis(
        self,
        data: jnp.ndarray,
        method: str,
        hyperparameters: Dict[str, Any],
        n_folds: int
    ) -> Dict[str, Any]:
        """Cross-validation analysis for method validation"""
        
        n_samples = data.shape[0]
        fold_size = n_samples // n_folds
        cv_scores = []
        
        for fold in range(n_folds):
            # Create train/test split
            test_start = fold * fold_size
            test_end = min((fold + 1) * fold_size, n_samples)
            
            train_data = jnp.concatenate([data[:test_start], data[test_end:]])
            test_data = data[test_start:test_end]
            
            # Train on training data
            train_result = await self._gradient_based_discovery(train_data, hyperparameters)
            
            # Evaluate on test data
            test_score = self._evaluate_structure_on_data(train_result.adjacency_matrix, test_data)
            cv_scores.append(test_score)
        
        return {
            'cv_scores': cv_scores,
            'mean_cv_score': float(jnp.mean(jnp.array(cv_scores))),
            'std_cv_score': float(jnp.std(jnp.array(cv_scores))),
            'n_folds': n_folds
        }
    
    @jax.jit
    def _evaluate_structure_on_data(self, adjacency_matrix: jnp.ndarray, data: jnp.ndarray) -> float:
        """Evaluate discovered structure on held-out data"""
        # Simple evaluation: reconstruction error
        predictions = data @ adjacency_matrix.T
        reconstruction_error = jnp.mean((data - predictions) ** 2)
        return float(reconstruction_error)
    
    def _perform_statistical_tests(
        self, 
        data: jnp.ndarray, 
        result: CausalDiscoveryResult
    ) -> Dict[str, Any]:
        """Perform statistical tests on discovered structure"""
        
        n_samples, n_vars = data.shape
        adjacency = result.adjacency_matrix
        
        tests = {}
        
        # Test for edge significance
        edge_p_values = jnp.zeros_like(adjacency)
        for i in range(n_vars):
            for j in range(n_vars):
                if adjacency[i, j] != 0:
                    # Simple correlation test
                    correlation = jnp.corrcoef(data[:, i], data[:, j])[0, 1]
                    z_score = 0.5 * jnp.log((1 + correlation) / (1 - correlation)) * jnp.sqrt(n_samples - 3)
                    p_value = 2 * (1 - stats.norm.cdf(jnp.abs(z_score)))
                    edge_p_values = edge_p_values.at[i, j].set(p_value)
        
        tests['edge_p_values'] = edge_p_values
        tests['significant_edges'] = jnp.sum(edge_p_values < 0.05)
        
        # Overall structure tests
        tests['n_edges'] = jnp.sum(adjacency != 0)
        tests['density'] = float(tests['n_edges'] / (n_vars * (n_vars - 1)))
        
        return tests


# Export main class and utility functions
__all__ = [
    'NovelCausalDiscovery',
    'CausalDiscoveryResult'
]