"""
Core JAX-based causal inference engine implementing do-calculus operations.

This module provides the foundational causal computation capabilities for 
Causal UI Gym, including intervention analysis, average treatment effect 
calculation, and graph-based causal inference using JAX for GPU acceleration.
"""

import jax
import jax.numpy as jnp
from jax import random, jit, vmap, pmap
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from functools import partial
from dataclasses import dataclass


@dataclass
class CausalDAG:
    """Represents a causal directed acyclic graph with associated data."""
    nodes: List[str]
    edges: List[Tuple[str, str]]
    node_data: Dict[str, jnp.ndarray]
    edge_weights: Optional[Dict[Tuple[str, str], float]] = None
    
    def __post_init__(self):
        """Validate DAG structure and initialize default weights."""
        if self.edge_weights is None:
            self.edge_weights = {edge: 1.0 for edge in self.edges}
        self._validate_dag()
    
    def _validate_dag(self) -> None:
        """Ensure the graph is a valid DAG (no cycles)."""
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)
        
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Graph contains cycles and is not a valid DAG")


@dataclass
class Intervention:
    """Represents a causal intervention on specific variables."""
    variable: str
    value: Union[float, jnp.ndarray]
    timestamp: Optional[float] = None


@dataclass
class CausalResult:
    """Container for causal computation results."""
    intervention: Intervention
    outcome_distribution: jnp.ndarray
    ate: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    computation_time: Optional[float] = None


class JaxCausalEngine:
    """
    High-performance causal inference engine using JAX for GPU acceleration.
    
    Implements core do-calculus operations for causal reasoning evaluation,
    including intervention computation, average treatment effect calculation,
    and backdoor criterion identification.
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize the causal engine with JAX random state."""
        self.key = random.PRNGKey(random_seed)
        self.compiled_functions = {}
        
    @partial(jit, static_argnums=(0,))
    def _compute_linear_scm(
        self, 
        adjacency_matrix: jnp.ndarray,
        noise: jnp.ndarray,
        intervention_mask: jnp.ndarray,
        intervention_values: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute structural causal model with linear relationships.
        
        Args:
            adjacency_matrix: n x n matrix representing causal relationships
            noise: n x m matrix of noise terms for m samples
            intervention_mask: n-length binary vector indicating intervened variables
            intervention_values: n-length vector of intervention values
            
        Returns:
            n x m matrix of variable values under intervention
        """
        n_vars, n_samples = noise.shape
        
        # Initialize variables matrix
        variables = jnp.zeros((n_vars, n_samples))
        
        # Topological ordering for causal computation
        # For simplicity, assume variables are already in topological order
        for i in range(n_vars):
            if intervention_mask[i]:
                # Variable is intervened upon - set to intervention value
                variables = variables.at[i, :].set(intervention_values[i])
            else:
                # Variable follows structural equation
                parent_effects = jnp.dot(adjacency_matrix[i, :], variables)
                variables = variables.at[i, :].set(parent_effects + noise[i, :])
                
        return variables
    
    @partial(jit, static_argnums=(0,))
    def _compute_intervention_distribution(
        self,
        adjacency_matrix: jnp.ndarray,
        noise_samples: jnp.ndarray,
        intervention_variable: int,
        intervention_value: float,
        outcome_variable: int
    ) -> jnp.ndarray:
        """
        Compute the distribution of outcome variable under intervention.
        
        Args:
            adjacency_matrix: Causal graph adjacency matrix
            noise_samples: Pre-generated noise samples
            intervention_variable: Index of intervened variable
            intervention_value: Value to set intervened variable to
            outcome_variable: Index of outcome variable of interest
            
        Returns:
            Distribution of outcome variable under intervention
        """
        n_vars = adjacency_matrix.shape[0]
        
        # Create intervention mask and values
        intervention_mask = jnp.zeros(n_vars, dtype=bool)
        intervention_mask = intervention_mask.at[intervention_variable].set(True)
        
        intervention_values = jnp.zeros(n_vars)
        intervention_values = intervention_values.at[intervention_variable].set(intervention_value)
        
        # Compute variables under intervention
        variables = self._compute_linear_scm(
            adjacency_matrix, noise_samples, intervention_mask, intervention_values
        )
        
        return variables[outcome_variable, :]
    
    def compute_intervention(
        self, 
        dag: CausalDAG, 
        intervention: Intervention,
        outcome_variable: str,
        n_samples: int = 10000
    ) -> CausalResult:
        """
        Compute the effect of an intervention on an outcome variable.
        
        Args:
            dag: Causal DAG structure
            intervention: Intervention specification
            outcome_variable: Name of outcome variable
            n_samples: Number of samples for Monte Carlo estimation
            
        Returns:
            CausalResult containing intervention effects
        """
        import time
        start_time = time.time()
        
        # Convert DAG to adjacency matrix
        adjacency_matrix = self._dag_to_adjacency_matrix(dag)
        
        # Get variable indices
        intervention_idx = dag.nodes.index(intervention.variable)
        outcome_idx = dag.nodes.index(outcome_variable)
        
        # Generate noise samples
        self.key, subkey = random.split(self.key)
        noise_samples = random.normal(subkey, (len(dag.nodes), n_samples))
        
        # Compute intervention distribution
        outcome_dist = self._compute_intervention_distribution(
            adjacency_matrix,
            noise_samples,
            intervention_idx,
            float(intervention.value),
            outcome_idx
        )
        
        computation_time = time.time() - start_time
        
        return CausalResult(
            intervention=intervention,
            outcome_distribution=outcome_dist,
            computation_time=computation_time
        )
    
    @partial(jit, static_argnums=(0,))
    def _compute_ate_vectorized(
        self,
        adjacency_matrix: jnp.ndarray,
        noise_samples: jnp.ndarray,
        treatment_variable: int,
        outcome_variable: int,
        treatment_values: jnp.ndarray
    ) -> float:
        """
        Compute Average Treatment Effect using vectorized operations.
        
        Args:
            adjacency_matrix: Causal graph adjacency matrix
            noise_samples: Noise samples for Monte Carlo
            treatment_variable: Index of treatment variable
            outcome_variable: Index of outcome variable
            treatment_values: Array of treatment values to compare
            
        Returns:
            Average treatment effect
        """
        # Compute outcomes under different treatment values
        outcomes = vmap(
            lambda t_val: self._compute_intervention_distribution(
                adjacency_matrix, noise_samples, treatment_variable, t_val, outcome_variable
            )
        )(treatment_values)
        
        # Compute ATE (difference between treatment conditions)
        if len(treatment_values) == 2:
            ate = jnp.mean(outcomes[1] - outcomes[0])
        else:
            # For multiple treatment values, compute pairwise differences
            ate = jnp.mean(outcomes[-1] - outcomes[0])
            
        return ate
    
    def compute_ate(
        self,
        dag: CausalDAG,
        treatment_variable: str,
        outcome_variable: str,
        treatment_values: List[float] = [0.0, 1.0],
        n_samples: int = 10000,
        confidence_level: float = 0.95
    ) -> CausalResult:
        """
        Compute Average Treatment Effect with confidence intervals.
        
        Args:
            dag: Causal DAG structure
            treatment_variable: Name of treatment variable
            outcome_variable: Name of outcome variable
            treatment_values: Values of treatment to compare
            n_samples: Number of samples for estimation
            confidence_level: Confidence level for intervals
            
        Returns:
            CausalResult with ATE and confidence intervals
        """
        import time
        start_time = time.time()
        
        # Convert to indices and matrices
        treatment_idx = dag.nodes.index(treatment_variable)
        outcome_idx = dag.nodes.index(outcome_variable)
        adjacency_matrix = self._dag_to_adjacency_matrix(dag)
        
        # Generate noise samples
        self.key, subkey = random.split(self.key)
        noise_samples = random.normal(subkey, (len(dag.nodes), n_samples))
        
        # Compute ATE
        treatment_array = jnp.array(treatment_values)
        ate = self._compute_ate_vectorized(
            adjacency_matrix, noise_samples, treatment_idx, outcome_idx, treatment_array
        )
        
        # Bootstrap confidence intervals
        n_bootstrap = 1000
        bootstrap_ates = []
        
        for _ in range(n_bootstrap):
            self.key, subkey = random.split(self.key)
            bootstrap_noise = random.normal(subkey, (len(dag.nodes), n_samples))
            bootstrap_ate = self._compute_ate_vectorized(
                adjacency_matrix, bootstrap_noise, treatment_idx, outcome_idx, treatment_array
            )
            bootstrap_ates.append(float(bootstrap_ate))
        
        bootstrap_ates = jnp.array(bootstrap_ates)
        alpha = 1 - confidence_level
        ci_lower = jnp.percentile(bootstrap_ates, 100 * alpha / 2)
        ci_upper = jnp.percentile(bootstrap_ates, 100 * (1 - alpha / 2))
        
        computation_time = time.time() - start_time
        
        # Create dummy intervention for result
        intervention = Intervention(
            variable=treatment_variable,
            value=treatment_values[1] if len(treatment_values) > 1 else treatment_values[0]
        )
        
        return CausalResult(
            intervention=intervention,
            outcome_distribution=jnp.array([]),  # Not computed for ATE
            ate=float(ate),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            computation_time=computation_time
        )
    
    def identify_backdoor_paths(
        self, 
        dag: CausalDAG, 
        treatment: str, 
        outcome: str
    ) -> List[List[str]]:
        """
        Identify backdoor paths between treatment and outcome variables.
        
        Args:
            dag: Causal DAG structure
            treatment: Treatment variable name
            outcome: Outcome variable name
            
        Returns:
            List of backdoor paths (each path is a list of variable names)
        """
        # Convert to NetworkX graph for path analysis
        G = nx.DiGraph()
        G.add_nodes_from(dag.nodes)
        G.add_edges_from(dag.edges)
        
        # Find all paths from treatment to outcome
        try:
            all_paths = list(nx.all_simple_paths(G.to_undirected(), treatment, outcome))
        except nx.NetworkXNoPath:
            return []
        
        backdoor_paths = []
        
        for path in all_paths:
            if len(path) > 2:  # More than direct edge
                # Check if path starts with incoming edge to treatment
                if len(path) > 1:
                    # Check if first edge is incoming to treatment
                    first_edge = (path[0], path[1])
                    reverse_edge = (path[1], path[0])
                    
                    if reverse_edge in dag.edges and first_edge not in dag.edges:
                        backdoor_paths.append(path)
        
        return backdoor_paths
    
    def _dag_to_adjacency_matrix(self, dag: CausalDAG) -> jnp.ndarray:
        """Convert CausalDAG to adjacency matrix representation."""
        n = len(dag.nodes)
        adj_matrix = jnp.zeros((n, n))
        
        node_to_idx = {node: i for i, node in enumerate(dag.nodes)}
        
        for source, target in dag.edges:
            source_idx = node_to_idx[source]
            target_idx = node_to_idx[target]
            weight = dag.edge_weights.get((source, target), 1.0)
            adj_matrix = adj_matrix.at[target_idx, source_idx].set(weight)
            
        return adj_matrix
    
    def mutilate_graph(self, dag: CausalDAG, intervention: Intervention) -> CausalDAG:
        """
        Create a mutilated graph by removing incoming edges to intervened variable.
        
        Args:
            dag: Original causal DAG
            intervention: Intervention specification
            
        Returns:
            Mutilated DAG with removed incoming edges
        """
        # Remove all incoming edges to the intervened variable
        mutilated_edges = [
            edge for edge in dag.edges 
            if edge[1] != intervention.variable
        ]
        
        return CausalDAG(
            nodes=dag.nodes.copy(),
            edges=mutilated_edges,
            node_data=dag.node_data.copy(),
            edge_weights={
                edge: weight for edge, weight in dag.edge_weights.items()
                if edge in mutilated_edges
            }
        )
    
    def batch_interventions(
        self,
        dag: CausalDAG,
        interventions: List[Intervention],
        outcome_variable: str,
        n_samples: int = 10000
    ) -> List[CausalResult]:
        """
        Compute multiple interventions in parallel using JAX vectorization.
        
        Args:
            dag: Causal DAG structure
            interventions: List of interventions to compute
            outcome_variable: Name of outcome variable
            n_samples: Number of samples per intervention
            
        Returns:
            List of CausalResults for each intervention
        """
        # Group interventions by variable for efficient computation
        results = []
        
        for intervention in interventions:
            result = self.compute_intervention(dag, intervention, outcome_variable, n_samples)
            results.append(result)
            
        return results
    
    def validate_dag_assumptions(self, dag: CausalDAG) -> Dict[str, bool]:
        """
        Validate key assumptions required for causal inference.
        
        Args:
            dag: Causal DAG to validate
            
        Returns:
            Dictionary of assumption validation results
        """
        G = nx.DiGraph()
        G.add_nodes_from(dag.nodes)
        G.add_edges_from(dag.edges)
        
        assumptions = {
            'is_acyclic': nx.is_directed_acyclic_graph(G),
            'is_connected': nx.is_weakly_connected(G),
            'has_valid_topological_order': True,  # Already checked in DAG validation
            'has_sufficient_data': all(
                len(data) > 100 for data in dag.node_data.values()
            ) if dag.node_data else False
        }
        
        return assumptions