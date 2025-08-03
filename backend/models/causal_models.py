"""
Pydantic models for causal inference data structures.

This module defines the data models used throughout the Causal UI Gym
backend for representing causal graphs, experiments, interventions,
and analysis results with proper validation and serialization.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Tuple, Union, Any
from datetime import datetime
from enum import Enum
import uuid


class NodePosition(BaseModel):
    """2D position for graph visualization."""
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")


class CausalNodeModel(BaseModel):
    """Represents a node in a causal graph."""
    id: str = Field(..., description="Unique node identifier")
    label: str = Field(..., description="Human-readable node label")
    position: NodePosition = Field(..., description="Position for visualization")
    variable_type: str = Field(default="continuous", description="Type of variable (continuous, discrete, binary)")
    description: Optional[str] = Field(None, description="Detailed description of the variable")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "price",
                "label": "Product Price",
                "position": {"x": 100, "y": 50},
                "variable_type": "continuous",
                "description": "The selling price of the product in USD"
            }
        }


class CausalEdgeModel(BaseModel):
    """Represents a directed edge in a causal graph."""
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    weight: float = Field(default=1.0, description="Edge weight/strength")
    edge_type: str = Field(default="causal", description="Type of causal relationship")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence in causal relationship")
    
    @validator('weight')
    def validate_weight(cls, v):
        if v == 0:
            raise ValueError("Edge weight cannot be zero")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "source": "price",
                "target": "demand",
                "weight": -0.8,
                "edge_type": "causal",
                "confidence": 0.95
            }
        }


class CausalDAGModel(BaseModel):
    """Complete causal directed acyclic graph specification."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique DAG identifier")
    name: str = Field(..., description="Human-readable DAG name")
    description: Optional[str] = Field(None, description="Detailed description of the causal model")
    nodes: List[CausalNodeModel] = Field(..., description="List of nodes in the DAG")
    edges: List[CausalEdgeModel] = Field(..., description="List of directed edges")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('edges')
    def validate_no_cycles(cls, edges, values):
        """Ensure the graph has no cycles."""
        if 'nodes' not in values:
            return edges
            
        nodes = values['nodes']
        node_ids = {node.id for node in nodes}
        
        # Basic cycle detection using DFS
        from collections import defaultdict, deque
        
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        for edge in edges:
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValueError(f"Edge references non-existent node: {edge.source} -> {edge.target}")
            graph[edge.source].append(edge.target)
            in_degree[edge.target] += 1
        
        # Topological sort to detect cycles
        queue = deque([node for node in node_ids if in_degree[node] == 0])
        processed = 0
        
        while queue:
            node = queue.popleft()
            processed += 1
            
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if processed != len(node_ids):
            raise ValueError("Graph contains cycles and is not a valid DAG")
        
        return edges
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Supply and Demand Model",
                "description": "Basic economic model of supply and demand",
                "nodes": [
                    {
                        "id": "price",
                        "label": "Price",
                        "position": {"x": 100, "y": 100},
                        "variable_type": "continuous"
                    },
                    {
                        "id": "demand",
                        "label": "Demand",
                        "position": {"x": 200, "y": 100},
                        "variable_type": "continuous"
                    }
                ],
                "edges": [
                    {
                        "source": "price",
                        "target": "demand",
                        "weight": -0.5
                    }
                ]
            }
        }


class InterventionType(str, Enum):
    """Types of causal interventions."""
    DO = "do"  # Hard intervention (set variable to value)
    SOFT = "soft"  # Soft intervention (shift distribution)
    CONDITIONAL = "conditional"  # Conditional intervention


class InterventionModel(BaseModel):
    """Specification for a causal intervention."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    variable: str = Field(..., description="Variable to intervene on")
    value: Union[float, int, str] = Field(..., description="Intervention value")
    intervention_type: InterventionType = Field(default=InterventionType.DO)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    description: Optional[str] = Field(None, description="Description of the intervention")
    
    class Config:
        schema_extra = {
            "example": {
                "variable": "price",
                "value": 29.99,
                "intervention_type": "do",
                "description": "Set price to $29.99"
            }
        }


class MetricType(str, Enum):
    """Types of causal metrics."""
    ATE = "ate"  # Average Treatment Effect
    ITE = "ite"  # Individual Treatment Effect
    CATE = "cate"  # Conditional Average Treatment Effect
    BACKDOOR = "backdoor"  # Backdoor criterion satisfaction
    FRONTDOOR = "frontdoor"  # Frontdoor criterion satisfaction


class CausalMetricModel(BaseModel):
    """Represents a computed causal metric."""
    metric_type: MetricType = Field(..., description="Type of causal metric")
    value: float = Field(..., description="Computed metric value")
    confidence_interval: Optional[Tuple[float, float]] = Field(None, description="95% confidence interval")
    standard_error: Optional[float] = Field(None, description="Standard error of estimate")
    p_value: Optional[float] = Field(None, description="Statistical significance")
    sample_size: int = Field(..., description="Sample size used for computation")
    computation_time: float = Field(..., description="Time taken to compute metric (seconds)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metric metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "metric_type": "ate",
                "value": -2.34,
                "confidence_interval": [-3.1, -1.58],
                "standard_error": 0.38,
                "p_value": 0.002,
                "sample_size": 10000,
                "computation_time": 0.045
            }
        }


class CausalResultModel(BaseModel):
    """Results from causal inference computation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dag_id: str = Field(..., description="ID of the causal DAG used")
    intervention: InterventionModel = Field(..., description="Applied intervention")
    outcome_variable: str = Field(..., description="Outcome variable analyzed")
    metrics: List[CausalMetricModel] = Field(..., description="Computed causal metrics")
    outcome_distribution: Optional[List[float]] = Field(None, description="Sampled outcome distribution")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "dag_id": "abc123",
                "intervention": {
                    "variable": "price",
                    "value": 25.0,
                    "intervention_type": "do"
                },
                "outcome_variable": "demand",
                "metrics": [
                    {
                        "metric_type": "ate",
                        "value": 5.2,
                        "confidence_interval": [4.1, 6.3],
                        "sample_size": 10000,
                        "computation_time": 0.023
                    }
                ]
            }
        }


class BeliefState(BaseModel):
    """Represents an LLM agent's belief about causal relationships."""
    agent_id: str = Field(..., description="LLM agent identifier")
    variable_pair: Tuple[str, str] = Field(..., description="Source and target variables")
    belief_strength: float = Field(..., ge=0.0, le=1.0, description="Strength of causal belief")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in belief")
    reasoning: Optional[str] = Field(None, description="Agent's reasoning for the belief")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExperimentStatus(str, Enum):
    """Status of an experiment."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentConfigModel(BaseModel):
    """Configuration for a causal reasoning experiment."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(None, description="Experiment description")
    dag: CausalDAGModel = Field(..., description="Causal DAG for the experiment")
    interventions: List[InterventionModel] = Field(..., description="Planned interventions")
    outcome_variables: List[str] = Field(..., description="Variables to analyze as outcomes")
    sample_size: int = Field(default=10000, ge=100, description="Sample size for Monte Carlo estimation")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    status: ExperimentStatus = Field(default=ExperimentStatus.CREATED)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('outcome_variables')
    def validate_outcome_variables(cls, v, values):
        """Ensure outcome variables exist in the DAG."""
        if 'dag' in values:
            dag_nodes = {node.id for node in values['dag'].nodes}
            invalid_outcomes = set(v) - dag_nodes
            if invalid_outcomes:
                raise ValueError(f"Outcome variables not in DAG: {invalid_outcomes}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Price Elasticity Experiment",
                "description": "Test LLM understanding of price-demand relationships",
                "sample_size": 5000,
                "random_seed": 42,
                "outcome_variables": ["demand", "revenue"]
            }
        }


class ValidationResult(BaseModel):
    """Result of DAG or experiment validation."""
    is_valid: bool = Field(..., description="Whether validation passed")
    errors: List[str] = Field(default_factory=list, description="Validation error messages")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    assumptions: Dict[str, bool] = Field(default_factory=dict, description="Causal assumption checks")
    
    class Config:
        schema_extra = {
            "example": {
                "is_valid": True,
                "errors": [],
                "warnings": ["Small sample size may affect precision"],
                "assumptions": {
                    "is_acyclic": True,
                    "is_connected": True,
                    "sufficient_data": False
                }
            }
        }