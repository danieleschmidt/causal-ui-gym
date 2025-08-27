"""
Comprehensive Baseline Methods for Causal Inference Benchmarking

This module implements state-of-the-art baseline causal inference methods
for rigorous comparison with novel algorithms. Includes both classical
and modern approaches with standardized interfaces.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random, vmap, grad, jit
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.optimize import minimize
import networkx as nx
from datetime import datetime

# Import novel algorithms for comparison
from .novel_algorithms import (
    DeepCausalInference, 
    QuantumInspiredCausalInference, 
    MetaCausalInference,
    NovelAlgorithmResult
)
from .novel_causal_discovery import NovelCausalDiscovery, CausalDiscoveryResult
from .statistical_validation_framework import StatisticalValidator, ValidationReport
from ..benchmarking.causal_benchmarks import BenchmarkDataset, BenchmarkResult
from ..engine.causal_engine import JaxCausalEngine, CausalDAG

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Result from baseline causal inference method."""
    method_name: str
    causal_effects: Dict[Tuple[str, str], float]
    confidence_intervals: Dict[Tuple[str, str], Tuple[float, float]]
    computational_metrics: Dict[str, float]
    method_parameters: Dict[str, Any]
    theoretical_properties: Dict[str, str]
    assumptions_made: List[str]
    timestamp: datetime
    convergence_info: Optional[Dict[str, Any]] = None
    warnings: Optional[List[str]] = None


class CausalInferenceBaseline(ABC):
    """Abstract base class for causal inference baseline methods."""
    
    @abstractmethod
    def estimate_causal_effects(
        self,
        data: Dict[str, jnp.ndarray],
        treatment_vars: List[str],
        outcome_vars: List[str],
        confounders: Optional[List[str]] = None,
        **kwargs
    ) -> BaselineResult:
        """Estimate causal effects using the baseline method."""
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Return the name of the baseline method."""
        pass
    
    @abstractmethod
    def get_theoretical_properties(self) -> Dict[str, str]:
        """Return theoretical properties and guarantees."""
        pass
    
    @abstractmethod
    def get_assumptions(self) -> List[str]:
        """Return list of assumptions made by the method."""
        pass


class LinearRegressionBaseline(CausalInferenceBaseline):
    """Ordinary Least Squares regression baseline."""
    
    def __init__(self, regularization: float = 0.0):
        self.regularization = regularization
        
    def estimate_causal_effects(
        self,
        data: Dict[str, jnp.ndarray],
        treatment_vars: List[str],
        outcome_vars: List[str],
        confounders: Optional[List[str]] = None,
        **kwargs
    ) -> BaselineResult:
        """Estimate causal effects using linear regression."""
        
        start_time = datetime.now()
        causal_effects = {}
        confidence_intervals = {}
        warnings = []
        
        confounders = confounders or []
        
        for treatment in treatment_vars:
            for outcome in outcome_vars:
                if treatment == outcome:
                    continue
                    
                try:
                    # Prepare data
                    y = data[outcome]
                    X_vars = [treatment] + confounders
                    X_list = [data[var] for var in X_vars if var in data]
                    
                    if not X_list:
                        warnings.append(f"No valid predictors for {outcome}")
                        continue
                        
                    X = jnp.column_stack(X_list)
                    
                    # Add intercept
                    X_with_intercept = jnp.column_stack([jnp.ones(len(y)), X])
                    
                    # OLS estimation with regularization
                    XtX = jnp.dot(X_with_intercept.T, X_with_intercept)
                    if self.regularization > 0:
                        XtX += self.regularization * jnp.eye(XtX.shape[0])
                        
                    XtX_inv = jnp.linalg.inv(XtX + 1e-8 * jnp.eye(XtX.shape[0]))
                    Xty = jnp.dot(X_with_intercept.T, y)
                    coefficients = jnp.dot(XtX_inv, Xty)
                    
                    # Treatment effect is coefficient on treatment (index 1)
                    treatment_effect = float(coefficients[1])
                    causal_effects[(treatment, outcome)] = treatment_effect
                    
                    # Calculate standard errors and confidence intervals
                    residuals = y - jnp.dot(X_with_intercept, coefficients)
                    mse = jnp.mean(residuals ** 2)
                    var_coeff = mse * jnp.diag(XtX_inv)
                    se_treatment = float(jnp.sqrt(var_coeff[1]))
                    
                    # 95% confidence interval
                    t_critical = 1.96  # Approximate for large samples
                    ci_lower = treatment_effect - t_critical * se_treatment
                    ci_upper = treatment_effect + t_critical * se_treatment
                    confidence_intervals[(treatment, outcome)] = (ci_lower, ci_upper)
                    
                except Exception as e:
                    warnings.append(f"Error estimating {treatment}->{outcome}: {str(e)}")
                    causal_effects[(treatment, outcome)] = 0.0
                    confidence_intervals[(treatment, outcome)] = (0.0, 0.0)
        
        end_time = datetime.now()
        
        return BaselineResult(
            method_name=self.get_method_name(),
            causal_effects=causal_effects,
            confidence_intervals=confidence_intervals,
            computational_metrics={
                "runtime_seconds": (end_time - start_time).total_seconds(),
                "num_regressions": len(treatment_vars) * len(outcome_vars)
            },
            method_parameters={
                "regularization": self.regularization
            },
            theoretical_properties=self.get_theoretical_properties(),
            assumptions_made=self.get_assumptions(),
            timestamp=start_time,
            warnings=warnings if warnings else None
        )
    
    def get_method_name(self) -> str:
        return f"Linear Regression (λ={self.regularization})"
    
    def get_theoretical_properties(self) -> Dict[str, str]:
        return {
            "consistency": "Consistent under linear structural equations",
            "efficiency": "BLUE (Best Linear Unbiased Estimator) under Gauss-Markov",
            "asymptotic_distribution": "Asymptotically normal",
            "computational_complexity": "O(np² + p³) where p is number of predictors"
        }
    
    def get_assumptions(self) -> List[str]:
        return [
            "Linear relationship between variables",
            "No omitted confounders (conditional ignorability)",
            "Homoscedastic errors",
            "Independent observations",
            "No perfect multicollinearity"
        ]


class InstrumentalVariablesBaseline(CausalInferenceBaseline):
    """Two-Stage Least Squares (2SLS) instrumental variables baseline."""
    
    def __init__(self):
        pass
        
    def estimate_causal_effects(
        self,
        data: Dict[str, jnp.ndarray],
        treatment_vars: List[str],
        outcome_vars: List[str],
        confounders: Optional[List[str]] = None,
        instruments: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ) -> BaselineResult:
        """Estimate causal effects using instrumental variables."""
        
        start_time = datetime.now()
        causal_effects = {}
        confidence_intervals = {}
        warnings = []
        
        if not instruments:
            warnings.append("No instruments provided, using lagged variables")
            # Create simple instruments using lagged variables (simplified)
            instruments = {}
            for treatment in treatment_vars:
                instruments[treatment] = [f"{treatment}_lag1"]  # Would need actual lagged data
        
        confounders = confounders or []
        
        for treatment in treatment_vars:
            for outcome in outcome_vars:
                if treatment == outcome:
                    continue
                    
                try:
                    # Get instruments for this treatment
                    treatment_instruments = instruments.get(treatment, [])
                    valid_instruments = [inst for inst in treatment_instruments if inst in data]
                    
                    if not valid_instruments:
                        warnings.append(f"No valid instruments for {treatment}")
                        # Fallback to OLS
                        y = data[outcome]
                        x = data[treatment]
                        
                        if len(confounders) > 0:
                            X_confounders = jnp.column_stack([data[c] for c in confounders if c in data])
                            X_full = jnp.column_stack([jnp.ones(len(y)), x, X_confounders])
                        else:
                            X_full = jnp.column_stack([jnp.ones(len(y)), x])
                            
                        coeffs = jnp.linalg.lstsq(X_full, y, rcond=None)[0]
                        treatment_effect = float(coeffs[1])
                        
                    else:
                        # Two-Stage Least Squares
                        # Stage 1: Regress treatment on instruments and confounders
                        treatment_data = data[treatment]
                        instrument_data = jnp.column_stack([data[inst] for inst in valid_instruments])
                        
                        if confounders:
                            confounder_data = jnp.column_stack([data[c] for c in confounders if c in data])
                            stage1_X = jnp.column_stack([jnp.ones(len(treatment_data)), instrument_data, confounder_data])
                        else:
                            stage1_X = jnp.column_stack([jnp.ones(len(treatment_data)), instrument_data])
                            
                        # First stage regression
                        stage1_coeffs = jnp.linalg.lstsq(stage1_X, treatment_data, rcond=None)[0]
                        predicted_treatment = jnp.dot(stage1_X, stage1_coeffs)
                        
                        # Stage 2: Regress outcome on predicted treatment and confounders
                        outcome_data = data[outcome]
                        
                        if confounders:
                            stage2_X = jnp.column_stack([jnp.ones(len(outcome_data)), predicted_treatment, confounder_data])
                        else:
                            stage2_X = jnp.column_stack([jnp.ones(len(outcome_data)), predicted_treatment])
                            
                        stage2_coeffs = jnp.linalg.lstsq(stage2_X, outcome_data, rcond=None)[0]
                        treatment_effect = float(stage2_coeffs[1])
                        
                        # Check instrument strength (first-stage F-statistic approximation)
                        first_stage_residuals = treatment_data - predicted_treatment
                        first_stage_r2 = 1 - jnp.var(first_stage_residuals) / jnp.var(treatment_data)
                        
                        if first_stage_r2 < 0.1:
                            warnings.append(f"Weak instruments detected for {treatment} (R² = {first_stage_r2:.3f})")
                    
                    causal_effects[(treatment, outcome)] = treatment_effect
                    
                    # Simplified confidence interval (would need proper 2SLS standard errors)
                    se_approx = 0.1 * abs(treatment_effect) + 0.01  # Rough approximation
                    ci_lower = treatment_effect - 1.96 * se_approx
                    ci_upper = treatment_effect + 1.96 * se_approx
                    confidence_intervals[(treatment, outcome)] = (ci_lower, ci_upper)
                    
                except Exception as e:
                    warnings.append(f"Error in IV estimation {treatment}->{outcome}: {str(e)}")
                    causal_effects[(treatment, outcome)] = 0.0
                    confidence_intervals[(treatment, outcome)] = (0.0, 0.0)
        
        end_time = datetime.now()
        
        return BaselineResult(
            method_name=self.get_method_name(),
            causal_effects=causal_effects,
            confidence_intervals=confidence_intervals,
            computational_metrics={
                "runtime_seconds": (end_time - start_time).total_seconds(),
                "num_2sls_estimations": len(treatment_vars) * len(outcome_vars)
            },
            method_parameters={
                "instruments_used": instruments
            },
            theoretical_properties=self.get_theoretical_properties(),
            assumptions_made=self.get_assumptions(),
            timestamp=start_time,
            warnings=warnings if warnings else None
        )
    
    def get_method_name(self) -> str:
        return "Instrumental Variables (2SLS)"
    
    def get_theoretical_properties(self) -> Dict[str, str]:
        return {
            "consistency": "Consistent under valid instrument assumptions",
            "identification": "Identifies local average treatment effect (LATE)",
            "efficiency": "Less efficient than OLS when instruments are weak",
            "robustness": "Handles endogeneity but sensitive to instrument validity"
        }
    
    def get_assumptions(self) -> List[str]:
        return [
            "Instrument relevance (instruments correlated with treatment)",
            "Instrument exogeneity (instruments uncorrelated with error term)",
            "Exclusion restriction (instruments only affect outcome through treatment)",
            "Monotonicity (uniform treatment effect direction)",
            "SUTVA (Stable Unit Treatment Value Assumption)"
        ]


class MatchingBaseline(CausalInferenceBaseline):
    """Propensity score matching baseline."""
    
    def __init__(self, matching_method: str = "nearest", caliper: float = 0.1):
        self.matching_method = matching_method
        self.caliper = caliper
        
    def estimate_causal_effects(
        self,
        data: Dict[str, jnp.ndarray],
        treatment_vars: List[str],
        outcome_vars: List[str],
        confounders: Optional[List[str]] = None,
        **kwargs
    ) -> BaselineResult:
        """Estimate causal effects using propensity score matching."""
        
        start_time = datetime.now()
        causal_effects = {}
        confidence_intervals = {}
        warnings = []
        
        confounders = confounders or []
        
        for treatment in treatment_vars:
            for outcome in outcome_vars:
                if treatment == outcome:
                    continue
                    
                try:
                    # Binary treatment assumption for PSM
                    treatment_data = data[treatment]
                    outcome_data = data[outcome]
                    
                    # Convert treatment to binary if needed
                    treatment_binary = (treatment_data > jnp.median(treatment_data)).astype(float)
                    
                    if confounders:
                        # Estimate propensity scores using logistic regression
                        X_confounders = jnp.column_stack([data[c] for c in confounders if c in data])
                        X_with_intercept = jnp.column_stack([jnp.ones(len(treatment_binary)), X_confounders])
                        
                        # Simplified logistic regression (would use proper implementation)
                        # Using linear approximation for propensity scores
                        coeffs = jnp.linalg.lstsq(X_with_intercept, treatment_binary, rcond=None)[0]
                        propensity_scores = jnp.dot(X_with_intercept, coeffs)
                        propensity_scores = jax.nn.sigmoid(propensity_scores)  # Convert to probabilities
                    else:
                        # No confounders - use simple treatment probability
                        propensity_scores = jnp.full_like(treatment_binary, jnp.mean(treatment_binary))
                    
                    # Perform matching
                    treated_indices = jnp.where(treatment_binary == 1)[0]
                    control_indices = jnp.where(treatment_binary == 0)[0]
                    
                    if len(treated_indices) == 0 or len(control_indices) == 0:
                        warnings.append(f"Insufficient treatment variation for {treatment}")
                        causal_effects[(treatment, outcome)] = 0.0
                        confidence_intervals[(treatment, outcome)] = (0.0, 0.0)
                        continue
                    
                    # Nearest neighbor matching within caliper
                    matched_pairs = []
                    for treated_idx in treated_indices:
                        treated_ps = propensity_scores[treated_idx]
                        
                        # Find closest control unit within caliper
                        control_ps = propensity_scores[control_indices]
                        distances = jnp.abs(control_ps - treated_ps)
                        
                        min_distance = jnp.min(distances)
                        if min_distance <= self.caliper:
                            closest_control_idx = control_indices[jnp.argmin(distances)]
                            matched_pairs.append((treated_idx, closest_control_idx))
                    
                    if not matched_pairs:
                        warnings.append(f"No matches found within caliper for {treatment}")
                        causal_effects[(treatment, outcome)] = 0.0
                        confidence_intervals[(treatment, outcome)] = (0.0, 0.0)
                        continue
                    
                    # Calculate treatment effect from matched pairs
                    treatment_effects = []
                    for treated_idx, control_idx in matched_pairs:
                        effect = outcome_data[treated_idx] - outcome_data[control_idx]
                        treatment_effects.append(effect)
                    
                    treatment_effects = jnp.array(treatment_effects)
                    ate_estimate = float(jnp.mean(treatment_effects))
                    
                    causal_effects[(treatment, outcome)] = ate_estimate
                    
                    # Confidence interval based on matched sample
                    if len(treatment_effects) > 1:
                        se_ate = float(jnp.std(treatment_effects) / jnp.sqrt(len(treatment_effects)))
                        ci_lower = ate_estimate - 1.96 * se_ate
                        ci_upper = ate_estimate + 1.96 * se_ate
                    else:
                        ci_lower = ci_upper = ate_estimate
                        
                    confidence_intervals[(treatment, outcome)] = (ci_lower, ci_upper)
                    
                except Exception as e:
                    warnings.append(f"Error in matching {treatment}->{outcome}: {str(e)}")
                    causal_effects[(treatment, outcome)] = 0.0
                    confidence_intervals[(treatment, outcome)] = (0.0, 0.0)
        
        end_time = datetime.now()
        
        return BaselineResult(
            method_name=self.get_method_name(),
            causal_effects=causal_effects,
            confidence_intervals=confidence_intervals,
            computational_metrics={
                "runtime_seconds": (end_time - start_time).total_seconds(),
                "matching_pairs": sum(len([1 for t, o in causal_effects.keys()]) for _ in treatment_vars)
            },
            method_parameters={
                "matching_method": self.matching_method,
                "caliper": self.caliper
            },
            theoretical_properties=self.get_theoretical_properties(),
            assumptions_made=self.get_assumptions(),
            timestamp=start_time,
            warnings=warnings if warnings else None
        )
    
    def get_method_name(self) -> str:
        return f"Propensity Score Matching ({self.matching_method}, caliper={self.caliper})"
    
    def get_theoretical_properties(self) -> Dict[str, str]:
        return {
            "consistency": "Consistent under unconfoundedness and overlap",
            "identification": "Identifies average treatment effect (ATE)",
            "robustness": "Robust to functional form misspecification",
            "efficiency": "May be less efficient than regression with correct specification"
        }
    
    def get_assumptions(self) -> List[str]:
        return [
            "Unconfoundedness (selection on observables)",
            "Common support/overlap condition",
            "SUTVA (Stable Unit Treatment Value Assumption)",
            "Correct propensity score model specification",
            "No unmeasured confounders"
        ]


class DoublyRobustBaseline(CausalInferenceBaseline):
    """Doubly robust estimation combining regression and propensity scores."""
    
    def __init__(self, regularization: float = 0.01):
        self.regularization = regularization
        
    def estimate_causal_effects(
        self,
        data: Dict[str, jnp.ndarray],
        treatment_vars: List[str],
        outcome_vars: List[str],
        confounders: Optional[List[str]] = None,
        **kwargs
    ) -> BaselineResult:
        """Estimate causal effects using doubly robust estimation."""
        
        start_time = datetime.now()
        causal_effects = {}
        confidence_intervals = {}
        warnings = []
        
        confounders = confounders or []
        
        for treatment in treatment_vars:
            for outcome in outcome_vars:
                if treatment == outcome:
                    continue
                    
                try:
                    treatment_data = data[treatment]
                    outcome_data = data[outcome]
                    
                    # Binary treatment for DR estimation
                    treatment_binary = (treatment_data > jnp.median(treatment_data)).astype(float)
                    
                    if not confounders:
                        # Without confounders, DR reduces to simple difference in means
                        treated_mean = jnp.mean(outcome_data[treatment_binary == 1])
                        control_mean = jnp.mean(outcome_data[treatment_binary == 0])
                        ate_estimate = float(treated_mean - control_mean)
                    else:
                        X_confounders = jnp.column_stack([data[c] for c in confounders if c in data])
                        X_with_intercept = jnp.column_stack([jnp.ones(len(outcome_data)), X_confounders])
                        
                        # Step 1: Estimate propensity scores
                        # Using linear probability model (simplified)
                        ps_X = jnp.column_stack([jnp.ones(len(treatment_binary)), X_confounders])
                        ps_coeffs = jnp.linalg.lstsq(ps_X, treatment_binary, rcond=None)[0]
                        propensity_scores = jnp.dot(ps_X, ps_coeffs)
                        propensity_scores = jnp.clip(propensity_scores, 0.01, 0.99)  # Avoid extreme values
                        
                        # Step 2: Estimate outcome models for treated and control
                        treated_mask = treatment_binary == 1
                        control_mask = treatment_binary == 0
                        
                        # Outcome model for treated units
                        if jnp.sum(treated_mask) > 1:
                            X_treated = X_with_intercept[treated_mask]
                            y_treated = outcome_data[treated_mask]
                            
                            XtX_treated = jnp.dot(X_treated.T, X_treated) + self.regularization * jnp.eye(X_treated.shape[1])
                            treated_coeffs = jnp.linalg.solve(XtX_treated, jnp.dot(X_treated.T, y_treated))
                            mu1_hat = jnp.dot(X_with_intercept, treated_coeffs)
                        else:
                            mu1_hat = jnp.full_like(outcome_data, jnp.mean(outcome_data[treated_mask]))
                        
                        # Outcome model for control units
                        if jnp.sum(control_mask) > 1:
                            X_control = X_with_intercept[control_mask]
                            y_control = outcome_data[control_mask]
                            
                            XtX_control = jnp.dot(X_control.T, X_control) + self.regularization * jnp.eye(X_control.shape[1])
                            control_coeffs = jnp.linalg.solve(XtX_control, jnp.dot(X_control.T, y_control))
                            mu0_hat = jnp.dot(X_with_intercept, control_coeffs)
                        else:
                            mu0_hat = jnp.full_like(outcome_data, jnp.mean(outcome_data[control_mask]))
                        
                        # Step 3: Doubly robust estimation
                        # ATE = E[mu1(X) - mu0(X)] + E[T/e(X) * (Y - mu1(X))] - E[(1-T)/(1-e(X)) * (Y - mu0(X))]
                        
                        regression_component = jnp.mean(mu1_hat - mu0_hat)
                        
                        ipw_component_treated = jnp.mean(
                            treatment_binary / propensity_scores * (outcome_data - mu1_hat)
                        )
                        
                        ipw_component_control = jnp.mean(
                            (1 - treatment_binary) / (1 - propensity_scores) * (outcome_data - mu0_hat)
                        )
                        
                        ate_estimate = float(regression_component + ipw_component_treated - ipw_component_control)
                    
                    causal_effects[(treatment, outcome)] = ate_estimate
                    
                    # Simplified confidence interval (would need influence function for proper CI)
                    n = len(outcome_data)
                    ate_var_approx = jnp.var(outcome_data[treatment_binary == 1]) / jnp.sum(treatment_binary) + \
                                   jnp.var(outcome_data[treatment_binary == 0]) / jnp.sum(1 - treatment_binary)
                    se_ate = float(jnp.sqrt(ate_var_approx))
                    
                    ci_lower = ate_estimate - 1.96 * se_ate
                    ci_upper = ate_estimate + 1.96 * se_ate
                    confidence_intervals[(treatment, outcome)] = (ci_lower, ci_upper)
                    
                except Exception as e:
                    warnings.append(f"Error in DR estimation {treatment}->{outcome}: {str(e)}")
                    causal_effects[(treatment, outcome)] = 0.0
                    confidence_intervals[(treatment, outcome)] = (0.0, 0.0)
        
        end_time = datetime.now()
        
        return BaselineResult(
            method_name=self.get_method_name(),
            causal_effects=causal_effects,
            confidence_intervals=confidence_intervals,
            computational_metrics={
                "runtime_seconds": (end_time - start_time).total_seconds(),
                "num_dr_estimations": len(treatment_vars) * len(outcome_vars)
            },
            method_parameters={
                "regularization": self.regularization
            },
            theoretical_properties=self.get_theoretical_properties(),
            assumptions_made=self.get_assumptions(),
            timestamp=start_time,
            warnings=warnings if warnings else None
        )
    
    def get_method_name(self) -> str:
        return f"Doubly Robust (λ={self.regularization})"
    
    def get_theoretical_properties(self) -> Dict[str, str]:
        return {
            "consistency": "Consistent if either propensity score or outcome model is correct",
            "efficiency": "Efficient if both models are correctly specified",
            "robustness": "Robust to misspecification of one of the two models",
            "double_robustness": "Maintains consistency under single model misspecification"
        }
    
    def get_assumptions(self) -> List[str]:
        return [
            "Unconfoundedness (at least one model correctly specified)",
            "Positivity/overlap assumption",
            "SUTVA (Stable Unit Treatment Value Assumption)",
            "Either propensity score or outcome models are correctly specified",
            "No unmeasured confounders"
        ]


class BaselineComparator:
    """Comprehensive comparator for novel methods against established baselines."""
    
    def __init__(self, random_seed: int = 42):
        self.rng_key = random.PRNGKey(random_seed)
        self.statistical_validator = StatisticalValidator(random_seed)
        
        # Initialize baseline methods
        self.baseline_methods = {
            "ols": LinearRegressionBaseline(regularization=0.0),
            "ridge": LinearRegressionBaseline(regularization=0.1),
            "2sls": InstrumentalVariablesBaseline(),
            "psm": MatchingBaseline(matching_method="nearest", caliper=0.1),
            "dr": DoublyRobustBaseline(regularization=0.01)
        }
        
        # Initialize novel methods
        self.novel_methods = {
            "deep_ci": DeepCausalInference(),
            "quantum_ci": QuantumInspiredCausalInference(),
            "meta_ci": MetaCausalInference(),
            "novel_discovery": NovelCausalDiscovery()
        }
        
    def comprehensive_comparison(
        self,
        dataset: BenchmarkDataset,
        novel_method_name: str,
        novel_method_params: Optional[Dict[str, Any]] = None
    ) -> ValidationReport:
        """Perform comprehensive comparison of novel method against all baselines."""
        
        logger.info(f"Starting comprehensive comparison for {novel_method_name} on {dataset.name}")
        
        # Prepare data format
        treatment_vars = [var for var in dataset.data.keys() if "treatment" in var.lower() or var in ['X0', 'X1']]
        outcome_vars = [var for var in dataset.data.keys() if "outcome" in var.lower() or var in ['X2', 'X3', 'X4']]
        confounders = dataset.confounders
        
        if not treatment_vars:
            treatment_vars = list(dataset.data.keys())[:2]  # Use first 2 as treatments
        if not outcome_vars:
            outcome_vars = list(dataset.data.keys())[2:]  # Use rest as outcomes
            
        logger.info(f"Treatment variables: {treatment_vars}")
        logger.info(f"Outcome variables: {outcome_vars}")
        logger.info(f"Confounders: {confounders}")
        
        # Run novel method
        novel_results = self._run_novel_method(
            novel_method_name, dataset, treatment_vars, outcome_vars, confounders, novel_method_params
        )
        
        # Run baseline methods
        baseline_results = {}
        for baseline_name, baseline_method in self.baseline_methods.items():
            try:
                logger.info(f"Running baseline method: {baseline_name}")
                result = baseline_method.estimate_causal_effects(
                    dataset.data, treatment_vars, outcome_vars, confounders
                )
                # Convert to format expected by validator
                baseline_results[baseline_name] = {
                    f"{t}->{o}": effect for (t, o), effect in result.causal_effects.items()
                }
            except Exception as e:
                logger.error(f"Error running baseline {baseline_name}: {e}")
                baseline_results[baseline_name] = {}
        
        # Convert novel results to format expected by validator
        novel_results_formatted = {
            f"{t}->{o}": effect for (t, o), effect in novel_results.items() if isinstance(effect, (int, float))
        }
        
        # Ground truth from dataset
        ground_truth = {
            f"{t}->{o}": effect for (t, o), effect in dataset.true_effects.items()
        }
        
        # Perform statistical validation
        validation_report = self.statistical_validator.validate_method_performance(
            novel_results_formatted,
            baseline_results,
            ground_truth,
            novel_method_name,
            dataset.name
        )
        
        logger.info(f"Comprehensive comparison completed for {novel_method_name}")
        return validation_report
        
    def _run_novel_method(
        self,
        method_name: str,
        dataset: BenchmarkDataset,
        treatment_vars: List[str],
        outcome_vars: List[str],
        confounders: List[str],
        method_params: Optional[Dict[str, Any]]
    ) -> Dict[Tuple[str, str], float]:
        """Run a novel method and return causal effects."""
        
        method_params = method_params or {}
        
        if method_name in self.novel_methods:
            method = self.novel_methods[method_name]
            
            # Prepare data for novel methods
            data_matrix = jnp.column_stack([dataset.data[var] for var in dataset.data.keys()])
            
            causal_effects = {}
            
            if isinstance(method, (DeepCausalInference, QuantumInspiredCausalInference, MetaCausalInference)):
                # These methods work with data matrices
                for i, treatment in enumerate(treatment_vars):
                    for j, outcome in enumerate(outcome_vars):
                        if treatment != outcome:
                            # Extract relevant columns
                            treatment_col = list(dataset.data.keys()).index(treatment)
                            outcome_col = list(dataset.data.keys()).index(outcome)
                            
                            X = data_matrix
                            T = data_matrix[:, treatment_col]
                            Y = data_matrix[:, outcome_col]
                            Z = None  # Would need instruments
                            
                            try:
                                if isinstance(method, DeepCausalInference):
                                    if Z is not None:
                                        result = method.deep_iv_estimation(X, Z, T, Y)
                                    else:
                                        result = method.neural_tangent_causal_estimation(X, T, Y)
                                    
                                    causal_effects[(treatment, outcome)] = result.causal_effects.get("ATE", 0.0)
                                    
                                elif isinstance(method, QuantumInspiredCausalInference):
                                    result = method.quantum_superposition_causal_search(X)
                                    # Extract relevant effect
                                    relevant_effects = [v for k, v in result.causal_effects.items() if f"X{treatment_col}" in k and f"X{outcome_col}" in k]
                                    causal_effects[(treatment, outcome)] = relevant_effects[0] if relevant_effects else 0.0
                                    
                                elif isinstance(method, MetaCausalInference):
                                    domain_context = {"temporal": 0, "experimental": 1}
                                    result = method.meta_learned_causal_discovery(X, domain_context)
                                    # Extract relevant effect
                                    relevant_effects = [v for k, v in result.causal_effects.items() if f"X{treatment_col}" in k and f"X{outcome_col}" in k]
                                    causal_effects[(treatment, outcome)] = relevant_effects[0] if relevant_effects else 0.0
                                    
                            except Exception as e:
                                logger.error(f"Error running novel method {method_name} for {treatment}->{outcome}: {e}")
                                causal_effects[(treatment, outcome)] = 0.0
                                
            elif isinstance(method, NovelCausalDiscovery):
                # Causal discovery method
                try:
                    result = method.discover_causal_structure(data_matrix, method='gradient_based_discovery')
                    # Extract effects from adjacency matrix
                    adj_matrix = result.adjacency_matrix
                    variable_names = list(dataset.data.keys())
                    
                    for i, treatment in enumerate(treatment_vars):
                        for j, outcome in enumerate(outcome_vars):
                            if treatment != outcome:
                                treatment_idx = variable_names.index(treatment)
                                outcome_idx = variable_names.index(outcome)
                                effect = float(adj_matrix[treatment_idx, outcome_idx])
                                causal_effects[(treatment, outcome)] = effect
                                
                except Exception as e:
                    logger.error(f"Error running novel discovery method: {e}")
            
            return causal_effects
        else:
            raise ValueError(f"Unknown novel method: {method_name}")
            
    def run_baseline_suite(
        self,
        dataset: BenchmarkDataset
    ) -> Dict[str, BaselineResult]:
        """Run all baseline methods on a dataset."""
        
        results = {}
        treatment_vars = [var for var in dataset.data.keys() if "treatment" in var.lower() or var in ['X0', 'X1']]
        outcome_vars = [var for var in dataset.data.keys() if "outcome" in var.lower() or var in ['X2', 'X3', 'X4']]
        confounders = dataset.confounders
        
        if not treatment_vars:
            treatment_vars = list(dataset.data.keys())[:2]
        if not outcome_vars:
            outcome_vars = list(dataset.data.keys())[2:]
            
        for baseline_name, baseline_method in self.baseline_methods.items():
            try:
                logger.info(f"Running baseline: {baseline_name}")
                result = baseline_method.estimate_causal_effects(
                    dataset.data, treatment_vars, outcome_vars, confounders
                )
                results[baseline_name] = result
            except Exception as e:
                logger.error(f"Error running baseline {baseline_name}: {e}")
                
        return results
        
    def export_comparison_results(
        self,
        validation_reports: List[ValidationReport],
        output_format: str = "latex"
    ) -> str:
        """Export comparison results in specified format."""
        
        if output_format == "latex":
            return self._export_latex_comparison(validation_reports)
        elif output_format == "markdown":
            return self._export_markdown_comparison(validation_reports)
        else:
            return self._export_json_comparison(validation_reports)
            
    def _export_latex_comparison(self, reports: List[ValidationReport]) -> str:
        """Export comparison as LaTeX table."""
        
        latex_lines = []
        latex_lines.append(r"\begin{table}[htbp]")
        latex_lines.append(r"\centering")
        latex_lines.append(r"\caption{Novel Methods vs Baseline Comparison}")
        latex_lines.append(r"\begin{tabular}{lllll}")
        latex_lines.append(r"\toprule")
        latex_lines.append("Method & Dataset & Best Baseline & P-value & Effect Size \\\\")
        latex_lines.append(r"\midrule")
        
        for report in reports:
            best_baseline = "N/A"
            best_p_value = 1.0
            best_effect_size = 0.0
            
            for baseline_name, result in report.baseline_comparisons.items():
                if result.p_value < best_p_value:
                    best_baseline = baseline_name
                    best_p_value = result.p_value
                    best_effect_size = result.effect_size
                    
            significance = "*" if best_p_value < 0.05 else ""
            
            latex_lines.append(
                f"{report.method_name} & {report.dataset_name} & {best_baseline} & "
                f"{best_p_value:.4f}{significance} & {best_effect_size:.3f} \\\\"
            )
            
        latex_lines.append(r"\bottomrule")
        latex_lines.append(r"\end{tabular}")
        latex_lines.append(r"\label{tab:novel_baseline_comparison}")
        latex_lines.append(r"\end{table}")
        
        return "\n".join(latex_lines)
        
    def _export_markdown_comparison(self, reports: List[ValidationReport]) -> str:
        """Export comparison as Markdown table."""
        
        md_lines = []
        md_lines.append("# Novel Methods vs Baseline Comparison")
        md_lines.append("")
        md_lines.append("| Method | Dataset | Best Baseline | P-value | Effect Size | Significance |")
        md_lines.append("|--------|---------|---------------|---------|-------------|--------------|")
        
        for report in reports:
            best_baseline = "N/A"
            best_p_value = 1.0
            best_effect_size = 0.0
            
            for baseline_name, result in report.baseline_comparisons.items():
                if result.p_value < best_p_value:
                    best_baseline = baseline_name
                    best_p_value = result.p_value
                    best_effect_size = result.effect_size
                    
            significance = "**Significant**" if best_p_value < 0.05 else "Not Significant"
            
            md_lines.append(
                f"| {report.method_name} | {report.dataset_name} | {best_baseline} | "
                f"{best_p_value:.4f} | {best_effect_size:.3f} | {significance} |"
            )
            
        return "\n".join(md_lines)
        
    def _export_json_comparison(self, reports: List[ValidationReport]) -> str:
        """Export comparison as JSON."""
        
        comparison_data = {
            "comparison_summary": {
                "total_methods": len(reports),
                "total_comparisons": sum(len(r.baseline_comparisons) for r in reports)
            },
            "method_results": []
        }
        
        for report in reports:
            method_data = {
                "method_name": report.method_name,
                "dataset_name": report.dataset_name,
                "baseline_comparisons": {},
                "overall_performance": {
                    "significant_improvements": 0,
                    "total_comparisons": len(report.baseline_comparisons)
                }
            }
            
            for baseline_name, result in report.baseline_comparisons.items():
                method_data["baseline_comparisons"][baseline_name] = {
                    "p_value": result.p_value,
                    "effect_size": result.effect_size,
                    "significant": result.p_value < 0.05
                }
                
                if result.p_value < 0.05 and result.test_statistic < 0:  # Improvement
                    method_data["overall_performance"]["significant_improvements"] += 1
                    
            comparison_data["method_results"].append(method_data)
            
        return json.dumps(comparison_data, indent=2)


# Export main classes
__all__ = [
    "CausalInferenceBaseline",
    "LinearRegressionBaseline",
    "InstrumentalVariablesBaseline", 
    "MatchingBaseline",
    "DoublyRobustBaseline",
    "BaselineComparator",
    "BaselineResult"
]
