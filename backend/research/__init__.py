"""
Advanced Research Module for Novel Causal Inference Algorithms.

This module contains cutting-edge research implementations including:
- Deep learning-based causal inference
- Quantum-inspired causal discovery
- Meta-learning for causal inference
- Automated research agent systems
- Autonomous research cycle execution
"""

from .novel_algorithms import (
    NovelAlgorithmResult,
    DeepCausalInference,
    QuantumInspiredCausalInference, 
    MetaCausalInference,
    run_novel_algorithm_suite
)

from .automated_research_system import (
    AutomatedResearchSystem,
    ResearchProject,
    ExperimentResult,
    ResearchPaper
)

__all__ = [
    "NovelAlgorithmResult",
    "DeepCausalInference",
    "QuantumInspiredCausalInference",
    "MetaCausalInference", 
    "run_novel_algorithm_suite",
    "AutomatedResearchSystem",
    "ResearchProject",
    "ExperimentResult", 
    "ResearchPaper"
]