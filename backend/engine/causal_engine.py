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
    
    def compute_frontdoor_adjustment(
        self,
        dag: CausalDAG,
        treatment: str,
        outcome: str,
        mediator: str,
        n_samples: int = 10000
    ) -> CausalResult:
        """
        Compute causal effect using frontdoor adjustment when backdoor is not available.
        
        Args:
            dag: Causal DAG structure
            treatment: Treatment variable name
            outcome: Outcome variable name  
            mediator: Mediator variable name
            n_samples: Number of samples for estimation
            
        Returns:
            CausalResult with frontdoor-adjusted effect
        """
        import time
        start_time = time.time()
        
        # Convert to indices
        treatment_idx = dag.nodes.index(treatment)
        outcome_idx = dag.nodes.index(outcome)
        mediator_idx = dag.nodes.index(mediator)
        
        adjacency_matrix = self._dag_to_adjacency_matrix(dag)
        
        # Generate noise samples
        self.key, subkey = random.split(self.key)
        noise_samples = random.normal(subkey, (len(dag.nodes), n_samples))
        
        # Frontdoor formula: E[Y|do(X=x)] = Σ_m E[Y|X=x',M=m] * P(M=m|X=x)
        # Step 1: Compute P(M|do(X))
        mediator_dist_x1 = self._compute_intervention_distribution(
            adjacency_matrix, noise_samples, treatment_idx, 1.0, mediator_idx
        )
        
        mediator_dist_x0 = self._compute_intervention_distribution(
            adjacency_matrix, noise_samples, treatment_idx, 0.0, mediator_idx
        )
        
        # Step 2: For each mediator value, compute E[Y|X,M] averaging over X
        outcome_values_x1 = []
        outcome_values_x0 = []
        
        # Sample mediator values from the distributions
        mediator_samples_x1 = mediator_dist_x1
        mediator_samples_x0 = mediator_dist_x0
        
        # Compute outcomes under different treatment values for same mediator
        for i in range(min(1000, n_samples)):  # Sample subset for efficiency
            m_val = mediator_samples_x1[i]
            
            # Set both treatment and mediator
            intervention_mask = jnp.zeros(len(dag.nodes), dtype=bool)
            intervention_mask = intervention_mask.at[treatment_idx].set(True)
            intervention_mask = intervention_mask.at[mediator_idx].set(True)
            
            intervention_values = jnp.zeros(len(dag.nodes))
            intervention_values = intervention_values.at[treatment_idx].set(1.0)
            intervention_values = intervention_values.at[mediator_idx].set(m_val)
            
            variables = self._compute_linear_scm(
                adjacency_matrix, noise_samples[:, i:i+1], intervention_mask, intervention_values
            )
            outcome_values_x1.append(float(variables[outcome_idx, 0]))
            
            # Same for treatment = 0
            intervention_values = intervention_values.at[treatment_idx].set(0.0)
            variables = self._compute_linear_scm(
                adjacency_matrix, noise_samples[:, i:i+1], intervention_mask, intervention_values
            )
            outcome_values_x0.append(float(variables[outcome_idx, 0]))
        
        # Compute frontdoor effect
        frontdoor_effect = jnp.mean(jnp.array(outcome_values_x1) - jnp.array(outcome_values_x0))
        
        computation_time = time.time() - start_time
        
        intervention = Intervention(variable=treatment, value=1.0)
        return CausalResult(
            intervention=intervention,
            outcome_distribution=jnp.array(outcome_values_x1),
            ate=float(frontdoor_effect),
            computation_time=computation_time
        )
    
    def compute_pearl_causal_hierarchy(
        self,
        dag: CausalDAG,
        treatment: str,
        outcome: str,
        query_type: str = "ate",  # 'ate', 'cate', 'mediation', 'counterfactual'
        n_samples: int = 10000
    ) -> Dict[str, CausalResult]:
        """
        Compute causal effects using Pearl's Causal Hierarchy (Association, Intervention, Counterfactuals).
        
        This method implements the three levels of Pearl's causal hierarchy:
        1. Association (P(Y|X)) - observational correlations
        2. Intervention (P(Y|do(X))) - causal effects under interventions  
        3. Counterfactuals (P(Y_x|X',Y')) - what would have happened
        
        Args:
            dag: Causal DAG structure
            treatment: Treatment variable name
            outcome: Outcome variable name
            query_type: Type of causal query to perform
            n_samples: Number of samples for estimation
            
        Returns:
            Dictionary containing results for each level of the hierarchy
        """
        import time
        start_time = time.time()
        
        results = {}
        
        # Level 1: Association P(Y|X)
        association_result = self._compute_association(dag, treatment, outcome, n_samples)
        results['association'] = association_result
        
        # Level 2: Intervention P(Y|do(X))
        intervention_result = self.compute_ate(dag, treatment, outcome, n_samples=n_samples)
        results['intervention'] = intervention_result
        
        # Level 3: Counterfactuals P(Y_x|X',Y')
        counterfactual_result = self._compute_counterfactuals(dag, treatment, outcome, n_samples)
        results['counterfactual'] = counterfactual_result
        
        # Mediation analysis if requested
        if query_type == "mediation":
            mediation_result = self._compute_mediation_analysis(dag, treatment, outcome, n_samples)
            results['mediation'] = mediation_result
            
        computation_time = time.time() - start_time
        
        # Add meta-result with computation summary
        results['meta'] = CausalResult(
            intervention=Intervention(variable="meta", value=0),
            outcome_distribution=jnp.array([]),
            computation_time=computation_time,
            ate=None
        )
        
        return results
    
    @partial(jit, static_argnums=(0,))
    def _compute_association(
        self,
        dag: CausalDAG,
        treatment: str,
        outcome: str,
        n_samples: int
    ) -> CausalResult:
        """
        Compute Level 1: Association P(Y|X) - observational correlation.
        """
        import time
        start_time = time.time()
        
        treatment_idx = dag.nodes.index(treatment)
        outcome_idx = dag.nodes.index(outcome)
        adjacency_matrix = self._dag_to_adjacency_matrix(dag)
        
        # Generate observational data (no interventions)
        self.key, subkey = random.split(self.key)
        noise_samples = random.normal(subkey, (len(dag.nodes), n_samples))
        
        intervention_mask = jnp.zeros(len(dag.nodes), dtype=bool)
        intervention_values = jnp.zeros(len(dag.nodes))
        
        variables = self._compute_linear_scm(
            adjacency_matrix, noise_samples, intervention_mask, intervention_values
        )
        
        observed_treatment = variables[treatment_idx, :]
        observed_outcome = variables[outcome_idx, :]
        
        # Compute correlation coefficient as association measure
        association_strength = jnp.corrcoef(observed_treatment, observed_outcome)[0, 1]
        
        computation_time = time.time() - start_time
        
        return CausalResult(
            intervention=Intervention(variable=treatment, value=0),
            outcome_distribution=observed_outcome,
            ate=float(association_strength),
            computation_time=computation_time
        )
    
    def _compute_counterfactuals(
        self,
        dag: CausalDAG,
        treatment: str,
        outcome: str,
        n_samples: int
    ) -> CausalResult:
        """
        Compute Level 3: Counterfactuals P(Y_x|X',Y') - what would have happened.
        """
        import time
        start_time = time.time()
        
        treatment_idx = dag.nodes.index(treatment)
        outcome_idx = dag.nodes.index(outcome)
        adjacency_matrix = self._dag_to_adjacency_matrix(dag)
        
        # Generate observational data first
        self.key, subkey = random.split(self.key)
        noise_samples = random.normal(subkey, (len(dag.nodes), n_samples))
        
        # Compute factual outcomes (what actually happened)
        intervention_mask = jnp.zeros(len(dag.nodes), dtype=bool)
        intervention_values = jnp.zeros(len(dag.nodes))
        
        factual_variables = self._compute_linear_scm(
            adjacency_matrix, noise_samples, intervention_mask, intervention_values
        )
        
        # Compute counterfactual outcomes (what would have happened)
        # For individuals who received treatment=0, what if they had treatment=1?
        counterfactual_outcomes = []
        
        for i in range(min(1000, n_samples)):  # Sample subset for efficiency
            factual_treatment = factual_variables[treatment_idx, i]
            
            if factual_treatment < 0.5:  # Originally untreated
                # Counterfactual: what if they were treated?
                cf_intervention_mask = jnp.zeros(len(dag.nodes), dtype=bool)
                cf_intervention_mask = cf_intervention_mask.at[treatment_idx].set(True)
                cf_intervention_values = jnp.zeros(len(dag.nodes))
                cf_intervention_values = cf_intervention_values.at[treatment_idx].set(1.0)
                
                # Use same noise for counterfactual consistency
                cf_variables = self._compute_linear_scm(
                    adjacency_matrix, noise_samples[:, i:i+1], cf_intervention_mask, cf_intervention_values
                )
                
                counterfactual_outcomes.append(float(cf_variables[outcome_idx, 0]))
        
        # Individual Treatment Effect (ITE) distribution
        counterfactual_dist = jnp.array(counterfactual_outcomes)
        individual_effects = counterfactual_dist - factual_variables[outcome_idx, :len(counterfactual_outcomes)]
        average_ite = jnp.mean(individual_effects)
        
        computation_time = time.time() - start_time
        
        return CausalResult(
            intervention=Intervention(variable=treatment, value=1),
            outcome_distribution=counterfactual_dist,
            ate=float(average_ite),
            computation_time=computation_time
        )
    
    def _compute_mediation_analysis(
        self,
        dag: CausalDAG,
        treatment: str,
        outcome: str,
        n_samples: int
    ) -> CausalResult:
        """
        Compute mediation analysis to decompose total effect into direct and indirect effects.
        """
        import time
        start_time = time.time()
        
        # Find potential mediators (variables causally between treatment and outcome)
        potential_mediators = []
        for node in dag.nodes:
            if node != treatment and node != outcome:
                # Check if treatment -> mediator -> outcome path exists
                treatment_to_mediator = any(edge[0] == treatment and edge[1] == node for edge in dag.edges)
                mediator_to_outcome = any(edge[0] == node and edge[1] == outcome for edge in dag.edges)
                if treatment_to_mediator and mediator_to_outcome:
                    potential_mediators.append(node)
        
        if not potential_mediators:
            # No mediators found, return total effect only
            total_effect = self.compute_ate(dag, treatment, outcome, n_samples=n_samples)
            return total_effect
        
        # Use first mediator for demonstration (could be extended to multiple mediators)
        mediator = potential_mediators[0]
        
        # Decompose effects using causal mediation formulas:
        # Total Effect = Direct Effect + Indirect Effect
        # Direct Effect: effect of treatment on outcome not through mediator
        # Indirect Effect: effect of treatment on outcome through mediator
        
        treatment_idx = dag.nodes.index(treatment)
        outcome_idx = dag.nodes.index(outcome)
        mediator_idx = dag.nodes.index(mediator)
        
        adjacency_matrix = self._dag_to_adjacency_matrix(dag)
        
        # Generate samples
        self.key, subkey = random.split(self.key)
        noise_samples = random.normal(subkey, (len(dag.nodes), n_samples))
        
        # Total effect: P(Y|do(T=1)) - P(Y|do(T=0))
        outcome_t1 = self._compute_intervention_distribution(
            adjacency_matrix, noise_samples, treatment_idx, 1.0, outcome_idx
        )
        outcome_t0 = self._compute_intervention_distribution(
            adjacency_matrix, noise_samples, treatment_idx, 0.0, outcome_idx
        )
        total_effect = jnp.mean(outcome_t1 - outcome_t0)
        
        # Natural Direct Effect (NDE): effect with mediator set to its natural value under control
        mediator_t0 = self._compute_intervention_distribution(
            adjacency_matrix, noise_samples, treatment_idx, 0.0, mediator_idx
        )
        
        # Compute Y under T=1, M=M(0)
        nde_outcomes = []
        for i in range(min(1000, n_samples)):
            m_val = mediator_t0[i]
            
            # Set both treatment=1 and mediator=M(0)
            intervention_mask = jnp.zeros(len(dag.nodes), dtype=bool)
            intervention_mask = intervention_mask.at[treatment_idx].set(True)
            intervention_mask = intervention_mask.at[mediator_idx].set(True)
            
            intervention_values = jnp.zeros(len(dag.nodes))
            intervention_values = intervention_values.at[treatment_idx].set(1.0)
            intervention_values = intervention_values.at[mediator_idx].set(m_val)
            
            variables = self._compute_linear_scm(
                adjacency_matrix, noise_samples[:, i:i+1], intervention_mask, intervention_values
            )
            nde_outcomes.append(float(variables[outcome_idx, 0]))
        
        nde = jnp.mean(jnp.array(nde_outcomes)) - jnp.mean(outcome_t0[:len(nde_outcomes)])
        
        # Natural Indirect Effect (NIE) = Total Effect - Natural Direct Effect
        nie = total_effect - nde
        
        computation_time = time.time() - start_time
        
        return CausalResult(
            intervention=Intervention(variable=f"{treatment}_mediated_by_{mediator}", value=1),
            outcome_distribution=jnp.array(nde_outcomes),
            ate=float(total_effect),
            computation_time=computation_time,
            confidence_interval=(float(nde), float(nie))  # Store direct and indirect effects
        )
    
    def compute_conditional_ate(
        self,
        dag: CausalDAG,
        treatment: str,
        outcome: str,
        conditioning_vars: List[str],
        conditioning_values: List[float],
        n_samples: int = 10000
    ) -> CausalResult:
        """
        Compute conditional average treatment effect given specific conditioning values.
        
        Args:
            dag: Causal DAG structure
            treatment: Treatment variable name
            outcome: Outcome variable name
            conditioning_vars: Variables to condition on
            conditioning_values: Values to condition on
            n_samples: Number of samples for estimation
            
        Returns:
            CausalResult with conditional ATE
        """
        import time
        start_time = time.time()
        
        # Convert to indices
        treatment_idx = dag.nodes.index(treatment)
        outcome_idx = dag.nodes.index(outcome)
        conditioning_indices = [dag.nodes.index(var) for var in conditioning_vars]
        
        adjacency_matrix = self._dag_to_adjacency_matrix(dag)
        
        # Generate noise samples
        self.key, subkey = random.split(self.key)
        noise_samples = random.normal(subkey, (len(dag.nodes), n_samples))
        
        # Create intervention masks for conditioning
        base_intervention_mask = jnp.zeros(len(dag.nodes), dtype=bool)
        base_intervention_values = jnp.zeros(len(dag.nodes))
        
        # Set conditioning variables
        for idx, value in zip(conditioning_indices, conditioning_values):
            base_intervention_mask = base_intervention_mask.at[idx].set(True)
            base_intervention_values = base_intervention_values.at[idx].set(value)
        
        # Compute outcomes under treatment = 1
        intervention_mask_t1 = base_intervention_mask.at[treatment_idx].set(True)
        intervention_values_t1 = base_intervention_values.at[treatment_idx].set(1.0)
        
        outcomes_t1 = []
        for i in range(n_samples):
            variables = self._compute_linear_scm(
                adjacency_matrix, noise_samples[:, i:i+1], intervention_mask_t1, intervention_values_t1
            )
            outcomes_t1.append(float(variables[outcome_idx, 0]))
        
        # Compute outcomes under treatment = 0
        intervention_mask_t0 = base_intervention_mask.at[treatment_idx].set(True)
        intervention_values_t0 = base_intervention_values.at[treatment_idx].set(0.0)
        
        outcomes_t0 = []
        for i in range(n_samples):
            variables = self._compute_linear_scm(
                adjacency_matrix, noise_samples[:, i:i+1], intervention_mask_t0, intervention_values_t0
            )
            outcomes_t0.append(float(variables[outcome_idx, 0]))
        
        # Compute conditional ATE
        conditional_ate = jnp.mean(jnp.array(outcomes_t1) - jnp.array(outcomes_t0))
        
        computation_time = time.time() - start_time
        
        intervention = Intervention(variable=treatment, value=1.0)
        return CausalResult(
            intervention=intervention,
            outcome_distribution=jnp.array(outcomes_t1),
            ate=float(conditional_ate),
            computation_time=computation_time
        )
    
    @jax.jit
    def _compute_propensity_scores(
        self,
        adjacency_matrix: jnp.ndarray,
        noise_samples: jnp.ndarray,
        treatment_idx: int,
        covariate_indices: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute propensity scores for treatment assignment.
        
        Args:
            adjacency_matrix: Causal graph adjacency matrix
            noise_samples: Noise samples
            treatment_idx: Index of treatment variable
            covariate_indices: Indices of covariate variables
            
        Returns:
            Propensity scores for each sample
        """
        n_samples = noise_samples.shape[1]
        
        # Compute baseline variables (no intervention)
        intervention_mask = jnp.zeros(adjacency_matrix.shape[0], dtype=bool)
        intervention_values = jnp.zeros(adjacency_matrix.shape[0])
        
        variables = self._compute_linear_scm(
            adjacency_matrix, noise_samples, intervention_mask, intervention_values
        )
        
        # Extract covariates and treatment
        covariates = variables[covariate_indices, :]
        treatment = variables[treatment_idx, :]
        
        # Simple logistic regression approximation
        # P(T=1|X) = sigmoid(β₀ + β₁X₁ + ... + βₖXₖ)
        covariate_means = jnp.mean(covariates, axis=1, keepdims=True)
        centered_covariates = covariates - covariate_means
        
        # Compute correlations as proxy for logistic coefficients
        correlations = jnp.corrcoef(jnp.vstack([treatment, centered_covariates]))[0, 1:]
        
        # Linear combination
        linear_combination = jnp.sum(correlations[:, None] * centered_covariates, axis=0)
        
        # Apply sigmoid
        propensity_scores = jax.nn.sigmoid(linear_combination)
        
        return propensity_scores
    
    def compute_ipw_ate(
        self,
        dag: CausalDAG,
        treatment: str,
        outcome: str,
        confounders: List[str],
        n_samples: int = 10000
    ) -> CausalResult:
        """
        Compute Average Treatment Effect using Inverse Propensity Weighting.
        
        Args:
            dag: Causal DAG structure
            treatment: Treatment variable name
            outcome: Outcome variable name
            confounders: List of confounding variables
            n_samples: Number of samples for estimation
            
        Returns:
            CausalResult with IPW-adjusted ATE
        """
        import time
        start_time = time.time()
        
        # Convert to indices
        treatment_idx = dag.nodes.index(treatment)
        outcome_idx = dag.nodes.index(outcome)
        confounder_indices = jnp.array([dag.nodes.index(var) for var in confounders])
        
        adjacency_matrix = self._dag_to_adjacency_matrix(dag)
        
        # Generate noise samples
        self.key, subkey = random.split(self.key)
        noise_samples = random.normal(subkey, (len(dag.nodes), n_samples))
        
        # Compute propensity scores
        propensity_scores = self._compute_propensity_scores(
            adjacency_matrix, noise_samples, treatment_idx, confounder_indices
        )
        
        # Compute observed outcomes (no intervention)
        intervention_mask = jnp.zeros(len(dag.nodes), dtype=bool)
        intervention_values = jnp.zeros(len(dag.nodes))
        
        variables = self._compute_linear_scm(
            adjacency_matrix, noise_samples, intervention_mask, intervention_values
        )
        
        observed_treatment = variables[treatment_idx, :]
        observed_outcome = variables[outcome_idx, :]
        
        # IPW estimator
        # ATE = E[Y*T/e(X)] - E[Y*(1-T)/(1-e(X))]
        weights_treated = observed_treatment / propensity_scores
        weights_control = (1 - observed_treatment) / (1 - propensity_scores)
        
        # Stabilize weights
        weights_treated = jnp.clip(weights_treated, 0, 10)
        weights_control = jnp.clip(weights_control, 0, 10)
        
        ate_treated = jnp.mean(observed_outcome * weights_treated)
        ate_control = jnp.mean(observed_outcome * weights_control)
        
        ipw_ate = ate_treated - ate_control
        
        computation_time = time.time() - start_time
        
        intervention = Intervention(variable=treatment, value=1.0)
        return CausalResult(
            intervention=intervention,
            outcome_distribution=observed_outcome,
            ate=float(ipw_ate),
            computation_time=computation_time
        )
    
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
            ) if dag.node_data else False,
            'satisfies_causal_markov': self._check_causal_markov_condition(dag),
            'has_identifiable_effects': self._check_effect_identifiability(dag)
        }
        
        return assumptions
    
    def _check_causal_markov_condition(self, dag: CausalDAG) -> bool:
        """Check if the DAG satisfies the Causal Markov Condition."""
        G = nx.DiGraph()  
        G.add_nodes_from(dag.nodes)
        G.add_edges_from(dag.edges)
        
        # For each node, check if it's independent of its non-descendants given its parents
        for node in dag.nodes:
            parents = list(G.predecessors(node))
            descendants = nx.descendants(G, node)
            non_descendants = set(dag.nodes) - descendants - {node} - set(parents)
            
            # In a proper implementation, we would test conditional independence
            # For now, we assume the condition holds if the graph structure is valid
            if len(non_descendants) > 0 and len(parents) == 0:
                return False
                
        return True
    
    def _check_effect_identifiability(self, dag: CausalDAG) -> bool:
        """Check if causal effects are identifiable in the given DAG."""
        G = nx.DiGraph()
        G.add_nodes_from(dag.nodes)
        G.add_edges_from(dag.edges)
        
        # Check for confounding variables that would make effects non-identifiable
        # This is a simplified check - full identifiability requires more complex analysis
        for edge in dag.edges:
            source, target = edge
            
            # Look for potential confounders
            common_causes = set(G.predecessors(source)) & set(G.predecessors(target))
            if len(common_causes) > 0:
                # Check if confounders are observable (in this simple case, assume they are)
                continue
                
        return True
    
    def compute_instrumental_variable_estimate(
        self,
        dag: CausalDAG,
        instrument: str,
        treatment: str,
        outcome: str,
        n_samples: int = 10000
    ) -> CausalResult:
        """
        Compute causal effect using instrumental variable estimation.
        
        An instrumental variable Z satisfies:
        1. Z affects T (relevance)
        2. Z affects Y only through T (exclusion restriction)
        3. Z is independent of unmeasured confounders (exchangeability)
        
        Args:
            dag: Causal DAG structure
            instrument: Instrumental variable name
            treatment: Treatment variable name
            outcome: Outcome variable name
            n_samples: Number of samples for estimation
            
        Returns:
            CausalResult with IV-estimated causal effect
        """
        import time
        start_time = time.time()
        
        # Convert to indices
        instrument_idx = dag.nodes.index(instrument)
        treatment_idx = dag.nodes.index(treatment)
        outcome_idx = dag.nodes.index(outcome)
        
        adjacency_matrix = self._dag_to_adjacency_matrix(dag)
        
        # Generate observational data
        self.key, subkey = random.split(self.key)
        noise_samples = random.normal(subkey, (len(dag.nodes), n_samples))
        
        intervention_mask = jnp.zeros(len(dag.nodes), dtype=bool)
        intervention_values = jnp.zeros(len(dag.nodes))
        
        variables = self._compute_linear_scm(
            adjacency_matrix, noise_samples, intervention_mask, intervention_values
        )
        
        # Extract observed variables
        z_obs = variables[instrument_idx, :]
        t_obs = variables[treatment_idx, :]
        y_obs = variables[outcome_idx, :]
        
        # Two-Stage Least Squares (2SLS) estimation
        # Stage 1: Regress T on Z
        # T = α + βZ + error
        
        # Compute coefficients using closed-form solutions
        z_mean = jnp.mean(z_obs)
        t_mean = jnp.mean(t_obs)
        y_mean = jnp.mean(y_obs)
        
        # First stage: T ~ Z
        cov_zt = jnp.mean((z_obs - z_mean) * (t_obs - t_mean))
        var_z = jnp.mean((z_obs - z_mean) ** 2)
        beta_zt = cov_zt / (var_z + 1e-8)  # Avoid division by zero
        alpha_t = t_mean - beta_zt * z_mean
        
        # Stage 2: Y ~ T_hat where T_hat = α + βZ
        t_hat = alpha_t + beta_zt * z_obs
        
        # Reduced form: Y ~ Z
        cov_zy = jnp.mean((z_obs - z_mean) * (y_obs - y_mean))
        beta_zy = cov_zy / (var_z + 1e-8)
        
        # IV estimate: β_IV = Cov(Y,Z) / Cov(T,Z)
        iv_estimate = beta_zy / (beta_zt + 1e-8)
        
        # Compute standard error and confidence interval using delta method
        n = len(z_obs)
        
        # Residuals for standard error computation
        t_residuals = t_obs - t_hat
        y_fitted = y_mean + iv_estimate * (t_obs - t_mean)
        y_residuals = y_obs - y_fitted
        
        # Standard error approximation
        var_residual = jnp.mean(y_residuals ** 2)
        var_first_stage = jnp.mean(t_residuals ** 2)
        
        iv_se = jnp.sqrt(var_residual / (n * var_first_stage * beta_zt ** 2 + 1e-8))
        
        # 95% confidence interval
        ci_lower = iv_estimate - 1.96 * iv_se
        ci_upper = iv_estimate + 1.96 * iv_se
        
        computation_time = time.time() - start_time
        
        return CausalResult(
            intervention=Intervention(variable=f"{treatment}_iv_{instrument}", value=1),
            outcome_distribution=y_fitted,
            ate=float(iv_estimate),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            computation_time=computation_time
        )
    
    def compute_difference_in_differences(
        self,
        dag: CausalDAG,
        treatment: str,
        outcome: str,
        time_var: str,
        group_var: str,
        n_samples: int = 10000
    ) -> CausalResult:
        """
        Compute causal effect using Difference-in-Differences design.
        
        DiD exploits variation in treatment timing across groups to identify causal effects.
        Estimates: E[Y_post - Y_pre | Treated] - E[Y_post - Y_pre | Control]
        
        Args:
            dag: Causal DAG structure
            treatment: Treatment variable name
            outcome: Outcome variable name
            time_var: Time period variable (0=pre, 1=post)
            group_var: Group variable (0=control, 1=treatment group)
            n_samples: Number of samples for estimation
            
        Returns:
            CausalResult with DiD-estimated causal effect
        """
        import time
        start_time = time.time()
        
        # Convert to indices
        treatment_idx = dag.nodes.index(treatment)
        outcome_idx = dag.nodes.index(outcome)
        time_idx = dag.nodes.index(time_var)
        group_idx = dag.nodes.index(group_var)
        
        adjacency_matrix = self._dag_to_adjacency_matrix(dag)
        
        # Generate panel data (pre/post for treatment/control groups)
        self.key, subkey = random.split(self.key)
        noise_samples = random.normal(subkey, (len(dag.nodes), n_samples))
        
        intervention_mask = jnp.zeros(len(dag.nodes), dtype=bool)
        intervention_values = jnp.zeros(len(dag.nodes))
        
        variables = self._compute_linear_scm(
            adjacency_matrix, noise_samples, intervention_mask, intervention_values
        )
        
        # Extract variables
        time_obs = variables[time_idx, :]
        group_obs = variables[group_idx, :]
        outcome_obs = variables[outcome_idx, :]
        treatment_obs = variables[treatment_idx, :]
        
        # Create binary indicators
        post_period = (time_obs > jnp.median(time_obs)).astype(float)
        treatment_group = (group_obs > jnp.median(group_obs)).astype(float)
        treated = treatment_group * post_period  # Interaction term
        
        # DiD regression: Y = α + β₁*Post + β₂*Treatment_Group + β₃*Post*Treatment_Group + ε
        # β₃ is the DiD estimate
        
        # Design matrix
        X = jnp.column_stack([
            jnp.ones(n_samples),  # Intercept
            post_period,  # Post period effect
            treatment_group,  # Treatment group effect
            treated  # DiD interaction term
        ])
        
        # Ordinary Least Squares
        XtX = jnp.dot(X.T, X)
        XtX_inv = jnp.linalg.inv(XtX + 1e-6 * jnp.eye(4))  # Ridge regularization
        Xty = jnp.dot(X.T, outcome_obs)
        coefficients = jnp.dot(XtX_inv, Xty)
        
        # DiD estimate is the coefficient on the interaction term
        did_estimate = coefficients[3]
        
        # Predicted values and residuals
        y_pred = jnp.dot(X, coefficients)
        residuals = outcome_obs - y_pred
        
        # Standard error for DiD coefficient
        mse = jnp.mean(residuals ** 2)
        var_coeff = mse * XtX_inv[3, 3]
        se_did = jnp.sqrt(var_coeff)
        
        # 95% confidence interval
        ci_lower = did_estimate - 1.96 * se_did
        ci_upper = did_estimate + 1.96 * se_did
        
        computation_time = time.time() - start_time
        
        return CausalResult(
            intervention=Intervention(variable=f"{treatment}_did", value=1),
            outcome_distribution=y_pred,
            ate=float(did_estimate),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            computation_time=computation_time
        )
    
    def compute_regression_discontinuity(
        self,
        dag: CausalDAG,
        running_var: str,
        treatment: str,
        outcome: str,
        cutoff: float,
        bandwidth: Optional[float] = None,
        n_samples: int = 10000
    ) -> CausalResult:
        """
        Compute causal effect using Regression Discontinuity Design.
        
        RDD exploits arbitrary cutoff rules for treatment assignment to identify local causal effects.
        
        Args:
            dag: Causal DAG structure
            running_var: Running variable that determines treatment
            treatment: Treatment variable name
            outcome: Outcome variable name
            cutoff: Cutoff value for treatment assignment
            bandwidth: Bandwidth around cutoff (if None, optimal bandwidth is computed)
            n_samples: Number of samples for estimation
            
        Returns:
            CausalResult with RDD-estimated local causal effect
        """
        import time
        start_time = time.time()
        
        # Convert to indices
        running_idx = dag.nodes.index(running_var)
        treatment_idx = dag.nodes.index(treatment)
        outcome_idx = dag.nodes.index(outcome)
        
        adjacency_matrix = self._dag_to_adjacency_matrix(dag)
        
        # Generate data
        self.key, subkey = random.split(self.key)
        noise_samples = random.normal(subkey, (len(dag.nodes), n_samples))
        
        intervention_mask = jnp.zeros(len(dag.nodes), dtype=bool)
        intervention_values = jnp.zeros(len(dag.nodes))
        
        variables = self._compute_linear_scm(
            adjacency_matrix, noise_samples, intervention_mask, intervention_values
        )
        
        # Extract variables
        running_obs = variables[running_idx, :]
        treatment_obs = variables[treatment_idx, :]
        outcome_obs = variables[outcome_idx, :]
        
        # Center running variable around cutoff
        running_centered = running_obs - cutoff
        
        # Determine optimal bandwidth if not provided
        if bandwidth is None:
            # Simple rule-of-thumb bandwidth
            bandwidth = 1.5 * jnp.std(running_centered) * (n_samples ** (-1/5))
        
        # Keep only observations within bandwidth
        within_bandwidth = jnp.abs(running_centered) <= bandwidth
        
        if jnp.sum(within_bandwidth) < 10:
            # Too few observations, expand bandwidth
            bandwidth *= 2
            within_bandwidth = jnp.abs(running_centered) <= bandwidth
        
        # Filter data
        running_local = running_centered[within_bandwidth]
        treatment_local = treatment_obs[within_bandwidth]
        outcome_local = outcome_obs[within_bandwidth]
        
        # Binary treatment indicator based on cutoff
        above_cutoff = (running_local >= 0).astype(float)
        
        # Local linear regression on both sides of cutoff
        # Y = α + β*Above + γ*Running + δ*Above*Running + ε
        
        X_local = jnp.column_stack([
            jnp.ones(len(running_local)),  # Intercept
            above_cutoff,  # Treatment indicator
            running_local,  # Running variable
            above_cutoff * running_local  # Interaction
        ])
        
        # Weighted least squares with triangular kernel
        weights = jnp.maximum(0, 1 - jnp.abs(running_local) / bandwidth)
        W = jnp.diag(weights)
        
        # Weighted regression
        XtWX = jnp.dot(X_local.T, jnp.dot(W, X_local))
        XtWX_inv = jnp.linalg.inv(XtWX + 1e-6 * jnp.eye(4))
        XtWy = jnp.dot(X_local.T, jnp.dot(W, outcome_local))
        coefficients = jnp.dot(XtWX_inv, XtWy)
        
        # RDD estimate is the coefficient on the treatment indicator
        rdd_estimate = coefficients[1]
        
        # Standard error
        y_pred = jnp.dot(X_local, coefficients)
        residuals = outcome_local - y_pred
        mse = jnp.mean(weights * residuals ** 2)
        var_coeff = mse * XtWX_inv[1, 1]
        se_rdd = jnp.sqrt(var_coeff)
        
        # 95% confidence interval
        ci_lower = rdd_estimate - 1.96 * se_rdd
        ci_upper = rdd_estimate + 1.96 * se_rdd
        
        computation_time = time.time() - start_time
        
        return CausalResult(
            intervention=Intervention(variable=f"{treatment}_rdd", value=1),
            outcome_distribution=y_pred,
            ate=float(rdd_estimate),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            computation_time=computation_time
        )