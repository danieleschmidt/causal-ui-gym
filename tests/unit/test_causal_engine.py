"""
Unit tests for the JAX causal inference engine.

Tests cover basic functionality, edge cases, error handling,
and performance characteristics of the causal computation engine.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from unittest.mock import Mock, patch
import time
from datetime import datetime

from backend.engine.causal_engine import (
    JaxCausalEngine,
    CausalDAG,
    Intervention,
    CausalResult
)


class TestCausalDAG:
    """Test CausalDAG data structure."""
    
    def test_dag_creation(self):
        """Test basic DAG creation."""
        nodes = ['X', 'Y', 'Z']
        edges = [('X', 'Y'), ('Y', 'Z')]
        node_data = {node: jnp.array([1, 2, 3]) for node in nodes}
        
        dag = CausalDAG(
            nodes=nodes,
            edges=edges,
            node_data=node_data
        )
        
        assert dag.nodes == nodes
        assert dag.edges == edges
        assert dag.node_data == node_data
        assert dag.edge_weights == {('X', 'Y'): 1.0, ('Y', 'Z'): 1.0}
    
    def test_dag_validation_valid(self):
        """Test DAG validation with valid structure."""
        nodes = ['A', 'B', 'C']
        edges = [('A', 'B'), ('B', 'C')]
        node_data = {node: jnp.array([1, 2, 3]) for node in nodes}
        
        # Should not raise exception
        dag = CausalDAG(nodes=nodes, edges=edges, node_data=node_data)
        assert dag is not None
    
    def test_dag_validation_cycle_detection(self):
        """Test DAG validation catches cycles."""
        nodes = ['A', 'B', 'C']
        edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]  # Cycle!
        node_data = {node: jnp.array([1, 2, 3]) for node in nodes}
        
        with pytest.raises(ValueError, match="cycles"):
            CausalDAG(nodes=nodes, edges=edges, node_data=node_data)
    
    def test_dag_custom_weights(self):
        """Test DAG with custom edge weights."""
        nodes = ['X', 'Y']
        edges = [('X', 'Y')]
        node_data = {node: jnp.array([1, 2, 3]) for node in nodes}
        edge_weights = {('X', 'Y'): 2.5}
        
        dag = CausalDAG(
            nodes=nodes,
            edges=edges,
            node_data=node_data,
            edge_weights=edge_weights
        )
        
        assert dag.edge_weights[('X', 'Y')] == 2.5


class TestIntervention:
    """Test Intervention data structure."""
    
    def test_intervention_creation(self):
        """Test basic intervention creation."""
        intervention = Intervention(
            variable='X',
            value=5.0,
            timestamp=datetime.now()
        )
        
        assert intervention.variable == 'X'
        assert intervention.value == 5.0
        assert intervention.timestamp is not None
    
    def test_intervention_types(self):
        """Test different intervention types."""
        interventions = [
            Intervention(variable='X', value=1, intervention_type='do'),
            Intervention(variable='Y', value=2, intervention_type='soft'),
            Intervention(variable='Z', value=3, intervention_type='conditional')
        ]
        
        for i, intervention in enumerate(interventions):
            assert intervention.value == i + 1


class TestJaxCausalEngine:
    """Test JAX causal inference engine."""
    
    @pytest.fixture
    def engine(self):
        """Create engine instance for testing."""
        return JaxCausalEngine(random_seed=42)
    
    @pytest.fixture
    def simple_dag(self):
        """Create simple DAG for testing."""
        nodes = ['X', 'Y', 'Z']
        edges = [('X', 'Y'), ('Y', 'Z')]
        node_data = {node: jnp.array(np.random.normal(0, 1, 1000)) for node in nodes}
        
        return CausalDAG(
            nodes=nodes,
            edges=edges,
            node_data=node_data
        )
    
    @pytest.fixture
    def complex_dag(self):
        """Create more complex DAG for testing."""
        nodes = ['T', 'X', 'Y', 'Z', 'W']
        edges = [
            ('T', 'X'), ('T', 'Y'),
            ('X', 'Z'), ('Y', 'Z'),
            ('Z', 'W')
        ]
        node_data = {node: jnp.array(np.random.normal(0, 1, 1000)) for node in nodes}
        
        return CausalDAG(
            nodes=nodes,
            edges=edges,
            node_data=node_data
        )
    
    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine is not None
        assert hasattr(engine, 'key')
        assert hasattr(engine, 'compiled_functions')
    
    def test_compute_intervention_basic(self, engine, simple_dag):
        """Test basic intervention computation."""
        intervention = Intervention(variable='X', value=1.0)
        
        result = engine.compute_intervention(
            dag=simple_dag,
            intervention=intervention,
            outcome_variable='Z',
            n_samples=100
        )
        
        assert isinstance(result, CausalResult)
        assert result.intervention == intervention
        assert result.outcome_distribution is not None
        assert len(result.outcome_distribution) == 100
        assert result.computation_time > 0
    
    def test_compute_intervention_different_values(self, engine, simple_dag):
        """Test intervention with different values."""
        values = [0.0, 1.0, 2.0, -1.0]
        results = []
        
        for value in values:
            intervention = Intervention(variable='X', value=value)
            result = engine.compute_intervention(
                dag=simple_dag,
                intervention=intervention,
                outcome_variable='Z',
                n_samples=100
            )
            results.append(result)
        
        # Results should be different for different intervention values
        outcomes = [r.outcome_distribution for r in results]
        for i in range(len(outcomes)):
            for j in range(i + 1, len(outcomes)):
                assert not jnp.allclose(outcomes[i], outcomes[j], atol=0.1)
    
    def test_compute_ate_basic(self, engine, simple_dag):
        """Test average treatment effect computation."""
        result = engine.compute_ate(
            dag=simple_dag,
            treatment_variable='X',
            outcome_variable='Z',
            treatment_values=[0.0, 1.0],
            n_samples=100
        )
        
        assert isinstance(result, CausalResult)
        assert result.ate is not None
        assert isinstance(result.ate, float)
        assert result.confidence_interval is not None
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] <= result.confidence_interval[1]
    
    def test_compute_ate_multiple_values(self, engine, simple_dag):
        """Test ATE with multiple treatment values."""
        result = engine.compute_ate(
            dag=simple_dag,
            treatment_variable='X',
            outcome_variable='Z',
            treatment_values=[-1.0, 0.0, 1.0, 2.0],
            n_samples=100
        )
        
        assert result.ate is not None
        # ATE should be computed as difference between extreme values
        assert isinstance(result.ate, float)
    
    def test_identify_backdoor_paths(self, engine, complex_dag):
        """Test backdoor path identification."""
        paths = engine.identify_backdoor_paths(
            dag=complex_dag,
            treatment='X',
            outcome='W'
        )
        
        assert isinstance(paths, list)
        # Each path should be a list of variable names
        for path in paths:
            assert isinstance(path, list)
            assert all(isinstance(var, str) for var in path)
    
    def test_validate_dag_assumptions(self, engine, simple_dag):
        """Test DAG assumption validation."""
        assumptions = engine.validate_dag_assumptions(simple_dag)
        
        assert isinstance(assumptions, dict)
        assert 'is_acyclic' in assumptions
        assert 'is_connected' in assumptions
        assert 'has_valid_topological_order' in assumptions
        
        # Simple DAG should pass all basic assumptions
        assert assumptions['is_acyclic'] is True
        assert assumptions['is_connected'] is True
    
    def test_mutilate_graph(self, engine, simple_dag):
        """Test graph mutilation for interventions."""
        intervention = Intervention(variable='X', value=1.0)
        
        mutilated_dag = engine.mutilate_graph(simple_dag, intervention)
        
        assert isinstance(mutilated_dag, CausalDAG)
        assert mutilated_dag.nodes == simple_dag.nodes
        
        # Should have removed incoming edges to intervened variable
        incoming_edges = [edge for edge in simple_dag.edges if edge[1] == 'X']
        mutilated_incoming = [edge for edge in mutilated_dag.edges if edge[1] == 'X']
        assert len(mutilated_incoming) == 0
    
    def test_batch_interventions(self, engine, simple_dag):
        """Test batch intervention processing."""
        interventions = [
            Intervention(variable='X', value=0.0),
            Intervention(variable='X', value=1.0),
            Intervention(variable='Y', value=0.5)
        ]
        
        results = engine.batch_interventions(
            dag=simple_dag,
            interventions=interventions,
            outcome_variable='Z',
            n_samples=50
        )
        
        assert len(results) == len(interventions)
        assert all(isinstance(r, CausalResult) for r in results)
        assert all(len(r.outcome_distribution) == 50 for r in results)
    
    def test_compute_frontdoor_adjustment(self, engine):
        """Test frontdoor adjustment computation."""
        # Create DAG suitable for frontdoor adjustment
        nodes = ['X', 'M', 'Y', 'U']
        edges = [('X', 'M'), ('M', 'Y'), ('U', 'X'), ('U', 'Y')]
        node_data = {node: jnp.array(np.random.normal(0, 1, 100)) for node in nodes}
        
        dag = CausalDAG(nodes=nodes, edges=edges, node_data=node_data)
        
        result = engine.compute_frontdoor_adjustment(
            dag=dag,
            treatment='X',
            outcome='Y',
            mediator='M',
            n_samples=100
        )
        
        assert isinstance(result, CausalResult)
        assert result.ate is not None
        assert result.computation_time > 0
    
    def test_compute_conditional_ate(self, engine, complex_dag):
        """Test conditional ATE computation."""
        result = engine.compute_conditional_ate(
            dag=complex_dag,
            treatment='X',
            outcome='W',
            conditioning_vars=['T'],
            conditioning_values=[1.0],
            n_samples=100
        )
        
        assert isinstance(result, CausalResult)
        assert result.ate is not None
        assert result.computation_time > 0
    
    def test_compute_ipw_ate(self, engine, complex_dag):
        """Test inverse propensity weighting ATE."""
        result = engine.compute_ipw_ate(
            dag=complex_dag,
            treatment='X',
            outcome='W',
            confounders=['T'],
            n_samples=100
        )
        
        assert isinstance(result, CausalResult)
        assert result.ate is not None
        assert result.computation_time > 0
    
    def test_error_handling_invalid_variable(self, engine, simple_dag):
        """Test error handling for invalid variable names."""
        intervention = Intervention(variable='INVALID', value=1.0)
        
        with pytest.raises(ValueError):
            engine.compute_intervention(
                dag=simple_dag,
                intervention=intervention,
                outcome_variable='Z',
                n_samples=100
            )
    
    def test_error_handling_invalid_outcome(self, engine, simple_dag):
        """Test error handling for invalid outcome variable."""
        intervention = Intervention(variable='X', value=1.0)
        
        with pytest.raises(ValueError):
            engine.compute_intervention(
                dag=simple_dag,
                intervention=intervention,
                outcome_variable='INVALID',
                n_samples=100
            )
    
    def test_performance_large_samples(self, engine, simple_dag):
        """Test performance with large sample sizes."""
        intervention = Intervention(variable='X', value=1.0)
        
        start_time = time.time()
        result = engine.compute_intervention(
            dag=simple_dag,
            intervention=intervention,
            outcome_variable='Z',
            n_samples=10000
        )
        end_time = time.time()
        
        assert isinstance(result, CausalResult)
        assert len(result.outcome_distribution) == 10000
        # Should complete in reasonable time (adjust threshold as needed)
        assert end_time - start_time < 10.0
    
    def test_repeatability_with_same_seed(self):
        """Test that results are repeatable with same seed."""
        nodes = ['X', 'Y']
        edges = [('X', 'Y')]
        node_data = {node: jnp.array([1, 2, 3]) for node in nodes}
        dag = CausalDAG(nodes=nodes, edges=edges, node_data=node_data)
        
        intervention = Intervention(variable='X', value=1.0)
        
        # Run with same seed twice
        engine1 = JaxCausalEngine(random_seed=42)
        result1 = engine1.compute_intervention(dag, intervention, 'Y', n_samples=100)
        
        engine2 = JaxCausalEngine(random_seed=42)
        result2 = engine2.compute_intervention(dag, intervention, 'Y', n_samples=100)
        
        # Results should be identical
        assert jnp.allclose(result1.outcome_distribution, result2.outcome_distribution)
    
    def test_different_seeds_give_different_results(self):
        """Test that different seeds give different results."""
        nodes = ['X', 'Y']
        edges = [('X', 'Y')]
        node_data = {node: jnp.array([1, 2, 3]) for node in nodes}
        dag = CausalDAG(nodes=nodes, edges=edges, node_data=node_data)
        
        intervention = Intervention(variable='X', value=1.0)
        
        # Run with different seeds
        engine1 = JaxCausalEngine(random_seed=42)
        result1 = engine1.compute_intervention(dag, intervention, 'Y', n_samples=100)
        
        engine2 = JaxCausalEngine(random_seed=123)
        result2 = engine2.compute_intervention(dag, intervention, 'Y', n_samples=100)
        
        # Results should be different
        assert not jnp.allclose(result1.outcome_distribution, result2.outcome_distribution)


class TestCausalEngineIntegration:
    """Integration tests for causal engine components."""
    
    def test_full_causal_analysis_pipeline(self):
        """Test complete causal analysis pipeline."""
        # Create realistic DAG
        nodes = ['Education', 'Income', 'Health', 'Age']
        edges = [
            ('Education', 'Income'),
            ('Education', 'Health'),
            ('Age', 'Income'),
            ('Age', 'Health'),
            ('Income', 'Health')
        ]
        node_data = {
            node: jnp.array(np.random.normal(50, 15, 1000)) 
            for node in nodes
        }
        
        dag = CausalDAG(nodes=nodes, edges=edges, node_data=node_data)
        engine = JaxCausalEngine(random_seed=42)
        
        # Validate DAG
        assumptions = engine.validate_dag_assumptions(dag)
        assert assumptions['is_acyclic']
        
        # Test intervention
        intervention = Intervention(variable='Education', value=16.0)  # College education
        result = engine.compute_intervention(
            dag=dag,
            intervention=intervention,
            outcome_variable='Health',
            n_samples=1000
        )
        
        assert isinstance(result, CausalResult)
        assert result.outcome_distribution is not None
        assert len(result.outcome_distribution) == 1000
        
        # Test ATE
        ate_result = engine.compute_ate(
            dag=dag,
            treatment_variable='Education',
            outcome_variable='Health',
            treatment_values=[12.0, 16.0],  # High school vs college
            n_samples=1000
        )
        
        assert isinstance(ate_result, CausalResult)
        assert ate_result.ate is not None
        assert ate_result.confidence_interval is not None
    
    def test_confounding_detection_and_adjustment(self):
        """Test confounding detection and adjustment methods."""
        # Create DAG with confounding
        nodes = ['Confounder', 'Treatment', 'Outcome']
        edges = [
            ('Confounder', 'Treatment'),
            ('Confounder', 'Outcome'),
            ('Treatment', 'Outcome')
        ]
        node_data = {
            node: jnp.array(np.random.normal(0, 1, 1000)) 
            for node in nodes
        }
        
        dag = CausalDAG(nodes=nodes, edges=edges, node_data=node_data)
        engine = JaxCausalEngine(random_seed=42)
        
        # Test backdoor path identification
        backdoor_paths = engine.identify_backdoor_paths(dag, 'Treatment', 'Outcome')
        assert len(backdoor_paths) > 0
        
        # Test IPW adjustment
        ipw_result = engine.compute_ipw_ate(
            dag=dag,
            treatment='Treatment',
            outcome='Outcome',
            confounders=['Confounder'],
            n_samples=1000
        )
        
        assert isinstance(ipw_result, CausalResult)
        assert ipw_result.ate is not None
    
    def test_edge_cases_and_boundary_conditions(self):
        """Test edge cases and boundary conditions."""
        engine = JaxCausalEngine(random_seed=42)
        
        # Single node DAG
        single_node_dag = CausalDAG(
            nodes=['X'],
            edges=[],
            node_data={'X': jnp.array([1, 2, 3])}
        )
        
        assumptions = engine.validate_dag_assumptions(single_node_dag)
        assert assumptions['is_acyclic']
        
        # Large DAG
        n_nodes = 20
        nodes = [f'X{i}' for i in range(n_nodes)]
        edges = [(f'X{i}', f'X{i+1}') for i in range(n_nodes-1)]
        node_data = {
            node: jnp.array(np.random.normal(0, 1, 100)) 
            for node in nodes
        }
        
        large_dag = CausalDAG(nodes=nodes, edges=edges, node_data=node_data)
        
        # Should handle large DAG without issues
        intervention = Intervention(variable='X0', value=1.0)
        result = engine.compute_intervention(
            dag=large_dag,
            intervention=intervention,
            outcome_variable=f'X{n_nodes-1}',
            n_samples=100
        )
        
        assert isinstance(result, CausalResult)
        assert result.outcome_distribution is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])