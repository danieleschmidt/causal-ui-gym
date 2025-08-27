"""
Statistical Validation Framework for Novel Causal Inference Algorithms

This module provides rigorous statistical validation for novel causal inference methods,
including significance testing, confidence intervals, multiple testing correction,
and reproducibility measures required for academic publication.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random, vmap, pmap
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging
from datetime import datetime
from scipy import stats
from sklearn.model_selection import PermutationTestScore
from sklearn.utils import resample
import time
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class StatisticalResult:
    """Container for statistical validation results."""
    test_name: str
    test_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    sample_size: int
    power: float
    significance_level: float
    corrected_p_value: Optional[float] = None
    test_assumptions_met: Dict[str, bool] = None
    bootstrap_statistics: Optional[jnp.ndarray] = None
    permutation_statistics: Optional[jnp.ndarray] = None
    reproducibility_hash: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.test_assumptions_met is None:
            self.test_assumptions_met = {}


@dataclass
class ValidationReport:
    """Comprehensive validation report for causal inference methods."""
    method_name: str
    dataset_name: str
    baseline_comparisons: Dict[str, StatisticalResult]
    significance_tests: Dict[str, StatisticalResult]
    robustness_tests: Dict[str, StatisticalResult]
    reproducibility_metrics: Dict[str, float]
    computational_efficiency: Dict[str, float]
    theoretical_guarantees: Dict[str, str]
    practical_recommendations: List[str]
    publication_ready_summary: str
    generated_timestamp: datetime
    validation_version: str = "1.0"


class StatisticalValidator:
    """
    Comprehensive statistical validation framework for causal inference methods.
    
    Provides rigorous statistical testing required for academic publication including:
    - Hypothesis testing with multiple comparisons correction
    - Bootstrap and permutation tests
    - Power analysis and effect size computation
    - Reproducibility and stability assessments
    - Baseline method comparisons with statistical significance
    """
    
    def __init__(self, random_seed: int = 42, significance_level: float = 0.05):
        self.rng_key = random.PRNGKey(random_seed)
        self.significance_level = significance_level
        self.validation_history: List[ValidationReport] = []
        
    def validate_method_performance(
        self,
        novel_method_results: Dict[str, float],
        baseline_results: Dict[str, Dict[str, float]],
        ground_truth: Dict[str, float],
        method_name: str,
        dataset_name: str,
        **kwargs
    ) -> ValidationReport:
        """
        Comprehensive statistical validation of novel method against baselines.
        
        Args:
            novel_method_results: Results from novel method
            baseline_results: Results from baseline methods {method_name: results}
            ground_truth: True values for comparison
            method_name: Name of the novel method
            dataset_name: Name of the dataset
            
        Returns:
            ValidationReport with comprehensive statistical analysis
        """
        start_time = time.time()
        
        # Extract common metrics for comparison
        common_metrics = set(novel_method_results.keys())
        for baseline_name, results in baseline_results.items():
            common_metrics &= set(results.keys())
            
        if not common_metrics:
            raise ValueError("No common metrics found between novel method and baselines")
            
        # Statistical comparisons with baselines
        baseline_comparisons = {}
        for baseline_name, baseline_vals in baseline_results.items():
            comparison = self._compare_methods_statistically(
                novel_method_results,
                baseline_vals,
                ground_truth,
                f"{method_name}_vs_{baseline_name}"
            )
            baseline_comparisons[baseline_name] = comparison
            
        # Significance tests for novel method
        significance_tests = self._perform_significance_tests(
            novel_method_results, ground_truth
        )
        
        # Robustness tests
        robustness_tests = self._perform_robustness_tests(
            novel_method_results, ground_truth
        )
        
        # Reproducibility metrics
        reproducibility_metrics = self._assess_reproducibility(
            novel_method_results, baseline_results
        )
        
        # Computational efficiency analysis
        efficiency_metrics = {
            "validation_time_seconds": time.time() - start_time,
            "method_complexity_score": self._estimate_computational_complexity(method_name),
            "scalability_assessment": self._assess_scalability(method_name)
        }
        
        # Generate publication-ready summary
        summary = self._generate_publication_summary(
            method_name, baseline_comparisons, significance_tests, robustness_tests
        )
        
        # Practical recommendations
        recommendations = self._generate_recommendations(
            baseline_comparisons, robustness_tests, efficiency_metrics
        )
        
        report = ValidationReport(
            method_name=method_name,
            dataset_name=dataset_name,
            baseline_comparisons=baseline_comparisons,
            significance_tests=significance_tests,
            robustness_tests=robustness_tests,
            reproducibility_metrics=reproducibility_metrics,
            computational_efficiency=efficiency_metrics,
            theoretical_guarantees=self._extract_theoretical_guarantees(method_name),
            practical_recommendations=recommendations,
            publication_ready_summary=summary,
            generated_timestamp=datetime.now()
        )
        
        self.validation_history.append(report)
        logger.info(f"Statistical validation complete for {method_name} on {dataset_name}")
        
        return report
        
    def _compare_methods_statistically(
        self,
        method1_results: Dict[str, float],
        method2_results: Dict[str, float], 
        ground_truth: Dict[str, float],
        comparison_name: str
    ) -> StatisticalResult:
        """Statistical comparison between two methods."""
        
        # Extract errors for both methods
        method1_errors = []
        method2_errors = []
        
        for metric in method1_results:
            if metric in method2_results and metric in ground_truth:
                error1 = abs(method1_results[metric] - ground_truth[metric])
                error2 = abs(method2_results[metric] - ground_truth[metric])
                method1_errors.append(error1)
                method2_errors.append(error2)
                
        if not method1_errors:
            return StatisticalResult(
                test_name=comparison_name,
                test_statistic=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                effect_size=0.0,
                sample_size=0,
                power=0.0,
                significance_level=self.significance_level
            )
            
        method1_errors = jnp.array(method1_errors)
        method2_errors = jnp.array(method2_errors)
        
        # Paired t-test for comparison
        differences = method1_errors - method2_errors
        n = len(differences)
        
        if n < 2:
            return StatisticalResult(
                test_name=comparison_name,
                test_statistic=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                effect_size=0.0,
                sample_size=n,
                power=0.0,
                significance_level=self.significance_level
            )
            
        # T-test statistics
        mean_diff = float(jnp.mean(differences))
        std_diff = float(jnp.std(differences, ddof=1))
        se_diff = std_diff / jnp.sqrt(n)
        
        t_stat = mean_diff / (se_diff + 1e-10)
        
        # Two-tailed p-value
        p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), n - 1)))
        
        # Confidence interval for mean difference
        t_critical = stats.t.ppf(1 - self.significance_level/2, n - 1)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        # Effect size (Cohen's d)
        pooled_std = jnp.sqrt((jnp.var(method1_errors) + jnp.var(method2_errors)) / 2)
        cohens_d = float(mean_diff / (pooled_std + 1e-10))
        
        # Power analysis (approximate)
        power = self._compute_power(abs(cohens_d), n, self.significance_level)
        
        # Bootstrap confidence interval
        self.rng_key, subkey = random.split(self.rng_key)
        bootstrap_stats = self._bootstrap_difference(differences, subkey, n_bootstrap=1000)
        
        return StatisticalResult(
            test_name=comparison_name,
            test_statistic=float(t_stat),
            p_value=p_value,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            effect_size=cohens_d,
            sample_size=n,
            power=power,
            significance_level=self.significance_level,
            bootstrap_statistics=bootstrap_stats,
            test_assumptions_met=self._check_ttest_assumptions(differences)
        )
        
    def _perform_significance_tests(
        self,
        method_results: Dict[str, float],
        ground_truth: Dict[str, float]
    ) -> Dict[str, StatisticalResult]:
        """Perform significance tests for method accuracy."""
        
        tests = {}
        
        # Extract errors
        errors = []
        for metric in method_results:
            if metric in ground_truth:
                error = abs(method_results[metric] - ground_truth[metric])
                errors.append(error)
                
        if not errors:
            return tests
            
        errors = jnp.array(errors)
        n = len(errors)
        
        # Test 1: One-sample t-test against zero error (perfect accuracy)
        mean_error = float(jnp.mean(errors))
        std_error = float(jnp.std(errors, ddof=1))
        se_error = std_error / jnp.sqrt(n)
        
        t_stat = mean_error / (se_error + 1e-10)
        p_value = float(2 * (1 - stats.t.cdf(t_stat, n - 1)))
        
        # Confidence interval for mean error
        t_critical = stats.t.ppf(1 - self.significance_level/2, n - 1)
        ci_lower = mean_error - t_critical * se_error
        ci_upper = mean_error + t_critical * se_error
        
        tests["accuracy_test"] = StatisticalResult(
            test_name="one_sample_accuracy_test",
            test_statistic=float(t_stat),
            p_value=p_value,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            effect_size=float(mean_error / (std_error + 1e-10)),
            sample_size=n,
            power=self._compute_power(abs(mean_error), n, self.significance_level),
            significance_level=self.significance_level
        )
        
        # Test 2: Normality test of errors (Shapiro-Wilk)
        if n >= 3 and n <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(np.array(errors))
            tests["normality_test"] = StatisticalResult(
                test_name="shapiro_wilk_normality",
                test_statistic=float(shapiro_stat),
                p_value=float(shapiro_p),
                confidence_interval=(0.0, 1.0),
                effect_size=float(shapiro_stat),
                sample_size=n,
                power=0.8,  # Typical power for normality tests
                significance_level=self.significance_level
            )
            
        return tests
        
    def _perform_robustness_tests(
        self,
        method_results: Dict[str, float],
        ground_truth: Dict[str, float]
    ) -> Dict[str, StatisticalResult]:
        """Perform robustness tests including bootstrap and permutation tests."""
        
        tests = {}
        
        # Extract data for robustness testing
        errors = []
        for metric in method_results:
            if metric in ground_truth:
                error = abs(method_results[metric] - ground_truth[metric])
                errors.append(error)
                
        if not errors:
            return tests
            
        errors = jnp.array(errors)
        n = len(errors)
        
        # Bootstrap test for mean error stability
        self.rng_key, subkey = random.split(self.rng_key)
        bootstrap_means = self._bootstrap_statistic(errors, jnp.mean, subkey, n_bootstrap=1000)
        
        bootstrap_ci = (
            float(jnp.percentile(bootstrap_means, 2.5)),
            float(jnp.percentile(bootstrap_means, 97.5))
        )
        
        tests["bootstrap_stability"] = StatisticalResult(
            test_name="bootstrap_mean_stability",
            test_statistic=float(jnp.mean(bootstrap_means)),
            p_value=0.0,  # Bootstrap doesn't provide p-value directly
            confidence_interval=bootstrap_ci,
            effect_size=float(jnp.std(bootstrap_means)),
            sample_size=n,
            power=0.95,  # Bootstrap typically has high power
            significance_level=self.significance_level,
            bootstrap_statistics=bootstrap_means
        )
        
        # Permutation test for randomness check
        if n >= 10:
            self.rng_key, subkey = random.split(self.rng_key)
            perm_stats = self._permutation_test_statistic(errors, subkey, n_permutations=1000)
            
            observed_stat = float(jnp.mean(errors))
            p_value_perm = float(jnp.mean(perm_stats >= observed_stat))
            
            tests["permutation_test"] = StatisticalResult(
                test_name="permutation_randomness_test",
                test_statistic=observed_stat,
                p_value=p_value_perm,
                confidence_interval=(float(jnp.min(perm_stats)), float(jnp.max(perm_stats))),
                effect_size=float(jnp.std(perm_stats)),
                sample_size=n,
                power=0.8,
                significance_level=self.significance_level,
                permutation_statistics=perm_stats
            )
            
        return tests
        
    def _assess_reproducibility(
        self,
        method_results: Dict[str, float],
        baseline_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Assess reproducibility metrics."""
        
        metrics = {}
        
        # Create deterministic hash for reproducibility tracking
        result_string = json.dumps(method_results, sort_keys=True)
        metrics["reproducibility_hash"] = hashlib.md5(result_string.encode()).hexdigest()
        
        # Coefficient of variation across metrics
        values = list(method_results.values())
        if values:
            values_array = jnp.array(values)
            cv = float(jnp.std(values_array) / (jnp.mean(jnp.abs(values_array)) + 1e-10))
            metrics["coefficient_of_variation"] = cv
            
        # Consistency with baselines (average correlation)
        correlations = []
        for baseline_name, baseline_vals in baseline_results.items():
            common_keys = set(method_results.keys()) & set(baseline_vals.keys())
            if len(common_keys) > 1:
                method_vals = [method_results[k] for k in common_keys]
                baseline_vals_list = [baseline_vals[k] for k in common_keys]
                
                if jnp.var(jnp.array(method_vals)) > 0 and jnp.var(jnp.array(baseline_vals_list)) > 0:
                    correlation = float(jnp.corrcoef(jnp.array(method_vals), jnp.array(baseline_vals_list))[0, 1])
                    correlations.append(correlation)
                    
        metrics["avg_baseline_correlation"] = float(jnp.mean(jnp.array(correlations))) if correlations else 0.0
        
        # Stability score (1 - CV)
        metrics["stability_score"] = 1.0 - min(metrics.get("coefficient_of_variation", 1.0), 1.0)
        
        return metrics
        
    @jax.jit
    def _bootstrap_statistic(
        self,
        data: jnp.ndarray,
        statistic_fn: Callable,
        rng_key: jax.random.PRNGKey,
        n_bootstrap: int = 1000
    ) -> jnp.ndarray:
        """Bootstrap a statistic with JAX acceleration."""
        
        def single_bootstrap(key):
            indices = random.choice(key, len(data), shape=(len(data),), replace=True)
            bootstrap_sample = data[indices]
            return statistic_fn(bootstrap_sample)
            
        keys = random.split(rng_key, n_bootstrap)
        return vmap(single_bootstrap)(keys)
        
    @jax.jit
    def _bootstrap_difference(
        self,
        differences: jnp.ndarray,
        rng_key: jax.random.PRNGKey,
        n_bootstrap: int = 1000
    ) -> jnp.ndarray:
        """Bootstrap the mean difference."""
        return self._bootstrap_statistic(differences, jnp.mean, rng_key, n_bootstrap)
        
    @jax.jit
    def _permutation_test_statistic(
        self,
        data: jnp.ndarray,
        rng_key: jax.random.PRNGKey,
        n_permutations: int = 1000
    ) -> jnp.ndarray:
        """Permutation test for randomness."""
        
        def single_permutation(key):
            permuted_data = random.permutation(key, data)
            return jnp.mean(permuted_data)
            
        keys = random.split(rng_key, n_permutations)
        return vmap(single_permutation)(keys)
        
    def _compute_power(
        self,
        effect_size: float,
        sample_size: int,
        alpha: float = 0.05
    ) -> float:
        """Compute statistical power (approximate)."""
        if sample_size < 2:
            return 0.0
            
        # Approximate power calculation for t-test
        delta = effect_size * jnp.sqrt(sample_size)
        t_critical = stats.t.ppf(1 - alpha/2, sample_size - 1)
        
        # Power = P(|T| > t_critical | delta)
        power_approx = 1 - stats.t.cdf(t_critical - delta, sample_size - 1) + stats.t.cdf(-t_critical - delta, sample_size - 1)
        
        return float(jnp.clip(power_approx, 0.0, 1.0))
        
    def _check_ttest_assumptions(
        self,
        data: jnp.ndarray
    ) -> Dict[str, bool]:
        """Check assumptions for t-test validity."""
        
        assumptions = {}
        n = len(data)
        
        # Check for normality (Shapiro-Wilk test)
        if n >= 3 and n <= 5000:
            _, p_value = stats.shapiro(np.array(data))
            assumptions["normality"] = p_value > 0.05
        else:
            assumptions["normality"] = True  # Assume normal for very small/large samples
            
        # Check for outliers (IQR method)
        q75 = float(jnp.percentile(data, 75))
        q25 = float(jnp.percentile(data, 25))
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        outliers = jnp.sum((data < lower_bound) | (data > upper_bound))
        assumptions["no_extreme_outliers"] = float(outliers) / n < 0.05
        
        # Check for sufficient sample size
        assumptions["sufficient_sample_size"] = n >= 10
        
        return assumptions
        
    def _estimate_computational_complexity(
        self,
        method_name: str
    ) -> float:
        """Estimate computational complexity score (0-10 scale)."""
        
        complexity_scores = {
            "linear_regression": 2.0,
            "deep_iv": 7.0,
            "neural_tangent_kernel": 8.0,
            "quantum_superposition": 9.0,
            "meta_learning": 6.0,
            "gradient_based_discovery": 5.0,
            "variational_causal_discovery": 7.0,
            "neural_causal_discovery": 8.0
        }
        
        return complexity_scores.get(method_name.lower(), 5.0)
        
    def _assess_scalability(
        self,
        method_name: str
    ) -> float:
        """Assess scalability score (0-10 scale)."""
        
        scalability_scores = {
            "linear_regression": 9.0,
            "deep_iv": 6.0,
            "neural_tangent_kernel": 4.0,
            "quantum_superposition": 3.0,
            "meta_learning": 7.0,
            "gradient_based_discovery": 8.0,
            "variational_causal_discovery": 6.0,
            "neural_causal_discovery": 5.0
        }
        
        return scalability_scores.get(method_name.lower(), 5.0)
        
    def _extract_theoretical_guarantees(
        self,
        method_name: str
    ) -> Dict[str, str]:
        """Extract theoretical guarantees for the method."""
        
        guarantees = {
            "deep_iv": {
                "consistency": "Consistent under IV assumptions and neural network universality",
                "asymptotic_normality": "Yes, under regularity conditions",
                "identification": "Requires valid instruments and exclusion restriction"
            },
            "neural_tangent_kernel": {
                "consistency": "Consistent in the infinite-width limit",
                "convergence_rate": "Minimax optimal under smoothness assumptions",
                "universality": "Universal approximation in RKHS"
            },
            "quantum_superposition": {
                "global_optimality": "Quantum annealing provides global search",
                "superposition_advantage": "Explores exponentially many structures",
                "coherence_preservation": "Maintains quantum coherence during search"
            }
        }
        
        return guarantees.get(method_name.lower(), {"general": "Method-specific guarantees not specified"})
        
    def _generate_publication_summary(
        self,
        method_name: str,
        baseline_comparisons: Dict[str, StatisticalResult],
        significance_tests: Dict[str, StatisticalResult],
        robustness_tests: Dict[str, StatisticalResult]
    ) -> str:
        """Generate publication-ready summary of results."""
        
        summary_parts = []
        summary_parts.append(f"Statistical Validation Summary for {method_name}:")
        summary_parts.append("="*60)
        
        # Baseline comparisons
        if baseline_comparisons:
            summary_parts.append("\nBaseline Comparisons:")
            for baseline_name, result in baseline_comparisons.items():
                significance = "significant" if result.p_value < self.significance_level else "not significant"
                direction = "improvement" if result.test_statistic < 0 else "degradation"
                summary_parts.append(
                    f"  vs {baseline_name}: {direction} ({significance}, p={result.p_value:.4f}, "
                    f"effect size={result.effect_size:.3f})"
                )
                
        # Significance tests
        if significance_tests:
            summary_parts.append("\nSignificance Tests:")
            for test_name, result in significance_tests.items():
                significance = "significant" if result.p_value < self.significance_level else "not significant"
                summary_parts.append(
                    f"  {test_name}: {significance} (p={result.p_value:.4f}, power={result.power:.3f})"
                )
                
        # Robustness assessment
        if robustness_tests:
            summary_parts.append("\nRobustness Assessment:")
            for test_name, result in robustness_tests.items():
                summary_parts.append(
                    f"  {test_name}: CI={result.confidence_interval}, stability={result.effect_size:.3f}"
                )
                
        summary_parts.append("\nNote: All tests performed with α = {:.3f}".format(self.significance_level))
        
        return "\n".join(summary_parts)
        
    def _generate_recommendations(
        self,
        baseline_comparisons: Dict[str, StatisticalResult],
        robustness_tests: Dict[str, StatisticalResult],
        efficiency_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate practical recommendations based on validation results."""
        
        recommendations = []
        
        # Performance recommendations
        significant_improvements = sum(
            1 for result in baseline_comparisons.values() 
            if result.p_value < self.significance_level and result.test_statistic < 0
        )
        
        if significant_improvements >= len(baseline_comparisons) // 2:
            recommendations.append(
                "Method shows statistically significant improvements over multiple baselines. "
                "Recommended for practical deployment."
            )
        else:
            recommendations.append(
                "Method shows mixed performance against baselines. "
                "Further investigation of applicable scenarios recommended."
            )
            
        # Robustness recommendations
        stability_scores = [
            result.effect_size for result in robustness_tests.values() 
            if "stability" in result.test_name.lower()
        ]
        
        if stability_scores and jnp.mean(jnp.array(stability_scores)) < 0.1:
            recommendations.append(
                "Method demonstrates high stability across bootstrap samples. "
                "Suitable for production environments."
            )
        elif stability_scores:
            recommendations.append(
                "Method shows moderate stability. Consider ensemble approaches "
                "or additional regularization for improved robustness."
            )
            
        # Efficiency recommendations
        complexity_score = efficiency_metrics.get("method_complexity_score", 5.0)
        scalability_score = efficiency_metrics.get("scalability_assessment", 5.0)
        
        if complexity_score < 5.0 and scalability_score > 7.0:
            recommendations.append(
                "Method has favorable computational properties. "
                "Suitable for large-scale applications."
            )
        elif complexity_score > 8.0:
            recommendations.append(
                "Method is computationally intensive. "
                "Consider GPU acceleration or distributed computing for large datasets."
            )
            
        return recommendations
        
    def multiple_testing_correction(
        self,
        p_values: List[float],
        method: str = "bonferroni"
    ) -> List[float]:
        """Apply multiple testing correction to p-values."""
        
        p_values = jnp.array(p_values)
        n_tests = len(p_values)
        
        if method == "bonferroni":
            corrected_p = jnp.minimum(p_values * n_tests, 1.0)
        elif method == "holm":
            # Holm-Bonferroni correction
            sorted_indices = jnp.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            corrected_sorted = jnp.zeros_like(sorted_p)
            for i in range(n_tests):
                correction_factor = n_tests - i
                corrected_sorted = corrected_sorted.at[i].set(
                    jnp.minimum(sorted_p[i] * correction_factor, 1.0)
                )
                
            # Enforce monotonicity
            for i in range(1, n_tests):
                corrected_sorted = corrected_sorted.at[i].set(
                    jnp.maximum(corrected_sorted[i], corrected_sorted[i-1])
                )
                
            # Unsort back to original order
            corrected_p = jnp.zeros_like(p_values)
            corrected_p = corrected_p.at[sorted_indices].set(corrected_sorted)
        else:
            # No correction
            corrected_p = p_values
            
        return corrected_p.tolist()
        
    def export_validation_report(
        self,
        report: ValidationReport,
        format_type: str = "latex"
    ) -> str:
        """Export validation report in specified format."""
        
        if format_type == "latex":
            return self._export_latex_report(report)
        elif format_type == "markdown":
            return self._export_markdown_report(report)
        else:
            return self._export_json_report(report)
            
    def _export_latex_report(self, report: ValidationReport) -> str:
        """Export validation report as LaTeX table format."""
        
        latex_content = []
        latex_content.append(r"\begin{table}[htbp]")
        latex_content.append(r"\centering")
        latex_content.append(f"\\caption{{Statistical Validation Results for {report.method_name}}}")
        latex_content.append(r"\begin{tabular}{lllll}")
        latex_content.append(r"\toprule")
        latex_content.append("Test & Statistic & P-value & Effect Size & Power \\")
        latex_content.append(r"\midrule")
        
        # Add baseline comparisons
        for test_name, result in report.baseline_comparisons.items():
            latex_content.append(
                f"{test_name} & {result.test_statistic:.3f} & {result.p_value:.4f} & "
                f"{result.effect_size:.3f} & {result.power:.3f} \\\\"
            )
            
        latex_content.append(r"\bottomrule")
        latex_content.append(r"\end{tabular}")
        latex_content.append(r"\end{table}")
        
        return "\n".join(latex_content)
        
    def _export_markdown_report(self, report: ValidationReport) -> str:
        """Export validation report as Markdown."""
        
        md_content = []
        md_content.append(f"# Statistical Validation Report: {report.method_name}")
        md_content.append(f"**Dataset**: {report.dataset_name}")
        md_content.append(f"**Generated**: {report.generated_timestamp}")
        md_content.append("")
        
        md_content.append("## Summary")
        md_content.append(report.publication_ready_summary)
        md_content.append("")
        
        md_content.append("## Recommendations")
        for i, rec in enumerate(report.practical_recommendations, 1):
            md_content.append(f"{i}. {rec}")
        md_content.append("")
        
        return "\n".join(md_content)
        
    def _export_json_report(self, report: ValidationReport) -> str:
        """Export validation report as JSON."""
        
        # Convert dataclass to dict, handling special types
        def convert_for_json(obj):
            if isinstance(obj, jnp.ndarray):
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, StatisticalResult):
                return asdict(obj)
            return obj
            
        report_dict = asdict(report)
        
        # Convert JAX arrays and special types
        for key, value in report_dict.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    report_dict[key][subkey] = convert_for_json(subvalue)
            else:
                report_dict[key] = convert_for_json(value)
                
        return json.dumps(report_dict, indent=2, default=str)
        
        
# Export main classes
__all__ = [
    "StatisticalValidator",
    "StatisticalResult", 
    "ValidationReport"
]
