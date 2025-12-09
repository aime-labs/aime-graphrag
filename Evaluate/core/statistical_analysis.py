"""
Statistical analysis utilities for evaluation results.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import warnings

# Statistical libraries are optional - fallback to basic stats if not available
SCIPY_AVAILABLE = False
PINGOUIN_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available for statistical analysis")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    warnings.warn("Pandas not available for data analysis")

try:
    import pingouin as pg
    PINGOUIN_AVAILABLE = True
except ImportError:
    PINGOUIN_AVAILABLE = False
    warnings.warn("Pingouin not available for advanced statistical tests")

from interfaces.core_interfaces import StatisticalAnalyzerInterface


@dataclass
class ConfidenceInterval:
    """Represents a confidence interval."""
    lower: float
    upper: float
    confidence: float
    mean: float


@dataclass
class SignificanceTestResult:
    """Results of a statistical significance test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float]
    significant: bool
    confidence_level: float
    interpretation: str


@dataclass
class BootstrapResult:
    """Results of bootstrap analysis."""
    original_statistic: float
    bootstrap_mean: float
    bootstrap_std: float
    confidence_interval: ConfidenceInterval
    distribution: List[float]


class StatisticalAnalyzer(StatisticalAnalyzerInterface):
    """Statistical analysis for evaluation metrics."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def compute_confidence_interval(self, scores: List[float], 
                                  confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval for scores using t-distribution."""
        if not scores:
            return (0.0, 0.0)
        
        scores_array = np.array(scores)
        # Remove NaN values before computing statistics
        scores_array = scores_array[~np.isnan(scores_array)]
        if len(scores_array) == 0:
            return (0.0, 0.0)
        
        n = len(scores_array)
        mean = np.mean(scores_array)
        std_err = stats.sem(scores_array)
        
        # Use t-distribution for small samples, normal for large samples
        if n < 30:
            t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
        else:
            t_val = stats.norm.ppf((1 + confidence) / 2)
        
        margin_error = t_val * std_err
        
        return (mean - margin_error, mean + margin_error)
    
    def compute_detailed_confidence_interval(self, scores: List[float], 
                                           confidence: float = 0.95) -> ConfidenceInterval:
        """Compute detailed confidence interval."""
        lower, upper = self.compute_confidence_interval(scores, confidence)
        # Use nanmean to exclude NaN values
        mean = np.nanmean(scores) if scores else 0.0
        
        return ConfidenceInterval(
            lower=float(lower),
            upper=float(upper),
            confidence=confidence,
            mean=float(mean)
        )
    
    def significance_test(self, scores1: List[float], scores2: List[float],
                         test_type: str = 'auto', confidence: float = 0.95) -> Dict[str, Any]:
        """Perform statistical significance test between two sets of scores."""
        if not scores1 or not scores2:
            return {
                'test_name': 'invalid',
                'p_value': 1.0,
                'significant': False,
                'error': 'Empty score lists'
            }
        
        # Convert to arrays and remove NaN values
        scores1_array = np.array(scores1)
        scores2_array = np.array(scores2)
        scores1_array = scores1_array[~np.isnan(scores1_array)]
        scores2_array = scores2_array[~np.isnan(scores2_array)]
        
        if len(scores1_array) == 0 or len(scores2_array) == 0:
            return {
                'test_name': 'invalid',
                'p_value': 1.0,
                'significant': False,
                'error': 'All values were NaN after filtering'
            }
        
        # Determine test type
        if test_type == 'auto':
            if len(scores1) < 30 or len(scores2) < 30:
                test_type = 'ttest'
            else:
                # Check for normality
                _, p1 = stats.shapiro(scores1_array)
                _, p2 = stats.shapiro(scores2_array)
                
                if p1 > 0.05 and p2 > 0.05:
                    test_type = 'ttest'
                else:
                    test_type = 'mannwhitney'
        
        return self._perform_test(scores1_array, scores2_array, test_type, confidence)
    
    def _perform_test(self, scores1: np.ndarray, scores2: np.ndarray, 
                     test_type: str, confidence: float) -> Dict[str, Any]:
        """Perform the specified statistical test."""
        alpha = 1 - confidence
        
        try:
            if test_type == 'ttest':
                # Independent t-test
                statistic, p_value = stats.ttest_ind(scores1, scores2)
                effect_size = self._compute_cohens_d(scores1, scores2)
                test_name = "Independent t-test"
                
            elif test_type == 'paired_ttest':
                # Paired t-test (assumes same length)
                if len(scores1) != len(scores2):
                    raise ValueError("Paired t-test requires equal length arrays")
                statistic, p_value = stats.ttest_rel(scores1, scores2)
                effect_size = self._compute_cohens_d(scores1, scores2)
                test_name = "Paired t-test"
                
            elif test_type == 'mannwhitney':
                # Mann-Whitney U test (non-parametric)
                statistic, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
                effect_size = self._compute_rank_biserial_correlation(scores1, scores2)
                test_name = "Mann-Whitney U test"
                
            elif test_type == 'wilcoxon':
                # Wilcoxon signed-rank test (paired, non-parametric)
                if len(scores1) != len(scores2):
                    raise ValueError("Wilcoxon test requires equal length arrays")
                statistic, p_value = stats.wilcoxon(scores1, scores2)
                effect_size = self._compute_rank_biserial_correlation(scores1, scores2)
                test_name = "Wilcoxon signed-rank test"
                
            else:
                raise ValueError(f"Unknown test type: {test_type}")
            
            significant = p_value < alpha
            interpretation = self._interpret_results(p_value, effect_size, test_name)
            
            return {
                'test_name': test_name,
                'statistic': float(statistic),
                'p_value': float(p_value),
                'effect_size': float(effect_size) if effect_size is not None else None,
                'significant': significant,
                'confidence_level': confidence,
                'interpretation': interpretation,
                'alpha': alpha
            }
            
        except Exception as e:
            self.logger.error(f"Statistical test failed: {str(e)}")
            return {
                'test_name': test_type,
                'p_value': 1.0,
                'significant': False,
                'error': str(e)
            }
    
    def _compute_cohens_d(self, scores1: np.ndarray, scores2: np.ndarray) -> float:
        """Compute Cohen's d effect size."""
        mean1, mean2 = np.mean(scores1), np.mean(scores2)
        std1, std2 = np.std(scores1, ddof=1), np.std(scores2, ddof=1)
        n1, n2 = len(scores1), len(scores2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        return (mean1 - mean2) / pooled_std
    
    def _compute_rank_biserial_correlation(self, scores1: np.ndarray, scores2: np.ndarray) -> float:
        """Compute rank-biserial correlation for Mann-Whitney U test."""
        n1, n2 = len(scores1), len(scores2)
        u_statistic, _ = stats.mannwhitneyu(scores1, scores2)
        
        # Convert to rank-biserial correlation
        return 1 - (2 * u_statistic) / (n1 * n2)
    
    def _interpret_results(self, p_value: float, effect_size: Optional[float], test_name: str) -> str:
        """Interpret statistical test results."""
        interpretation = []
        
        # P-value interpretation
        if p_value < 0.001:
            interpretation.append("Very strong evidence against null hypothesis (p < 0.001)")
        elif p_value < 0.01:
            interpretation.append("Strong evidence against null hypothesis (p < 0.01)")
        elif p_value < 0.05:
            interpretation.append("Moderate evidence against null hypothesis (p < 0.05)")
        elif p_value < 0.1:
            interpretation.append("Weak evidence against null hypothesis (p < 0.1)")
        else:
            interpretation.append("No significant difference found (p >= 0.1)")
        
        # Effect size interpretation (for Cohen's d)
        if effect_size is not None and "t-test" in test_name:
            abs_effect = abs(effect_size)
            if abs_effect < 0.2:
                interpretation.append("Negligible effect size")
            elif abs_effect < 0.5:
                interpretation.append("Small effect size")
            elif abs_effect < 0.8:
                interpretation.append("Medium effect size")
            else:
                interpretation.append("Large effect size")
        
        return "; ".join(interpretation)
    
    def bootstrap_analysis(self, scores: List[float], n_bootstrap: int = 1000, 
                          statistic_func: Optional[Callable[[np.ndarray], float]] = None) -> Dict[str, Any]:
        """Perform bootstrap analysis on scores."""
        if not scores:
            return {'error': 'Empty score list'}
        
        if statistic_func is None:
            statistic_func = np.mean
        
        scores_array = np.array(scores)
        original_statistic = statistic_func(scores_array)
        
        # Bootstrap sampling
        bootstrap_statistics = []
        rng = np.random.RandomState(42)  # For reproducibility
        
        for _ in range(n_bootstrap):
            bootstrap_sample = rng.choice(scores_array, size=len(scores_array), replace=True)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_statistics.append(bootstrap_stat)
        
        bootstrap_array = np.array(bootstrap_statistics)
        bootstrap_mean = np.mean(bootstrap_array)
        bootstrap_std = np.std(bootstrap_array)
        
        # Compute confidence interval using percentile method
        ci_lower = np.percentile(bootstrap_array, 2.5)
        ci_upper = np.percentile(bootstrap_array, 97.5)
        
        confidence_interval = ConfidenceInterval(
            lower=float(ci_lower),
            upper=float(ci_upper),
            confidence=0.95,
            mean=float(bootstrap_mean)
        )
        
        return {
            'original_statistic': float(original_statistic),
            'bootstrap_mean': float(bootstrap_mean),
            'bootstrap_std': float(bootstrap_std),
            'confidence_interval': confidence_interval,
            'bias': float(bootstrap_mean - original_statistic),
            'bootstrap_samples': bootstrap_statistics
        }
    
    def multiple_comparisons_correction(self, p_values: List[float], 
                                      method: str = 'bonferroni') -> List[float]:
        """Apply multiple comparisons correction."""
        p_array = np.array(p_values)
        
        if method == 'bonferroni':
            corrected = p_array * len(p_array)
            corrected = np.minimum(corrected, 1.0)  # Cap at 1.0
            
        elif method == 'bh' or method == 'benjamini_hochberg':
            # Benjamini-Hochberg (FDR control)
            sorted_indices = np.argsort(p_array)
            sorted_p = p_array[sorted_indices]
            n = len(p_array)
            
            corrected_sorted = sorted_p * n / np.arange(1, n + 1)
            
            # Ensure monotonicity
            for i in range(n - 2, -1, -1):
                corrected_sorted[i] = min(corrected_sorted[i], corrected_sorted[i + 1])
            
            corrected = np.empty_like(p_array)
            corrected[sorted_indices] = corrected_sorted
            corrected = np.minimum(corrected, 1.0)
            
        else:
            raise ValueError(f"Unknown correction method: {method}")
        
        # Convert to list of floats - handle numpy array conversion
        result: List[float] = []
        try:
            # Handle numpy arrays and other iterables
            corrected_array = np.asarray(corrected).flatten()
            result = [float(val) for val in corrected_array]  # type: ignore
        except Exception:
            # Fallback - return original p-values converted to float
            result = [float(p) for p in p_values]  # type: ignore
        
        return result
    
    def effect_size_analysis(self, scores1: List[float], scores2: List[float]) -> Dict[str, float]:
        """Compute various effect size measures."""
        if not scores1 or not scores2:
            return {}
        
        scores1_array = np.array(scores1)
        scores2_array = np.array(scores2)
        
        # Cohen's d
        cohens_d = self._compute_cohens_d(scores1_array, scores2_array)
        
        # Glass's delta (using control group SD)
        glass_delta = (np.mean(scores1_array) - np.mean(scores2_array)) / np.std(scores2_array, ddof=1)
        
        # Hedge's g (small sample correction for Cohen's d)
        n1, n2 = len(scores1_array), len(scores2_array)
        j = 1 - (3 / (4 * (n1 + n2) - 9))
        hedges_g = cohens_d * j
        
        # Common Language Effect Size (probability of superiority)
        combined = np.concatenate([scores1_array, scores2_array])
        ranks = stats.rankdata(combined)
        rank_sum1 = np.sum(ranks[:len(scores1_array)])
        cles = (rank_sum1 - len(scores1_array) * (len(scores1_array) + 1) / 2) / (len(scores1_array) * len(scores2_array))
        
        return {
            'cohens_d': float(cohens_d),
            'glass_delta': float(glass_delta),
            'hedges_g': float(hedges_g),
            'common_language_effect_size': float(cles)
        }
    
    def comparison_summary(self, results_dict: Dict[str, List[float]], 
                          baseline_method: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive comparison summary."""
        methods = list(results_dict.keys())
        summary = {}
        
        # Descriptive statistics for each method
        for method, scores in results_dict.items():
            if not scores:
                continue
                
            scores_array = np.array(scores)
            ci = self.compute_detailed_confidence_interval(scores)
            
            summary[method] = {
                'mean': float(np.mean(scores_array)),
                'std': float(np.std(scores_array, ddof=1)),
                'median': float(np.median(scores_array)),
                'min': float(np.min(scores_array)),
                'max': float(np.max(scores_array)),
                'n': len(scores_array),
                'confidence_interval': ci
            }
        
        # Pairwise comparisons
        if len(methods) > 1:
            comparisons = {}
            p_values = []
            
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods[i+1:], i+1):
                    if method1 in results_dict and method2 in results_dict:
                        comparison_key = f"{method1}_vs_{method2}"
                        test_result = self.significance_test(
                            results_dict[method1], 
                            results_dict[method2]
                        )
                        comparisons[comparison_key] = test_result
                        p_values.append(test_result.get('p_value', 1.0))
            
            # Apply multiple comparisons correction
            if p_values:
                corrected_p = self.multiple_comparisons_correction(p_values)
                comparison_keys = list(comparisons.keys())
                
                for i, key in enumerate(comparison_keys):
                    comparisons[key]['corrected_p_value'] = corrected_p[i]
                    comparisons[key]['significant_corrected'] = corrected_p[i] < 0.05
            
            summary['pairwise_comparisons'] = comparisons
        
        # Best performing method
        if baseline_method and baseline_method in summary:
            best_method = max(summary.keys(), 
                            key=lambda x: summary[x]['mean'] if isinstance(summary[x], dict) and 'mean' in summary[x] else -float('inf'))
            summary['best_method'] = best_method
            
            if best_method != baseline_method:
                improvement = summary[best_method]['mean'] - summary[baseline_method]['mean']
                relative_improvement = improvement / summary[baseline_method]['mean'] * 100
                summary['improvement_over_baseline'] = {
                    'absolute': float(improvement),
                    'relative_percent': float(relative_improvement)
                }
        
        return summary
