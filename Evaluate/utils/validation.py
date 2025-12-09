"""
Validation utilities to ensure code quality and prevent production issues.
"""

import warnings
import os
import logging
from typing import Any, Callable, List
from functools import wraps


class ProductionValidationError(Exception):
    """Raised when placeholder or incomplete functionality is used in production."""
    pass


def validate_production_ready(func_name: str, description: str = None):
    """
    Decorator to ensure methods are not placeholders in production.
    
    Args:
        func_name: Name of the function being validated
        description: Optional description of what's missing
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            _check_production_readiness(func_name, description)
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            _check_production_readiness(func_name, description)
            return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def _check_production_readiness(func_name: str, description: str = None):
    """Check if we're in production and raise error if so."""
    is_production = os.getenv('ENVIRONMENT', '').lower() in ['production', 'prod']
    
    if is_production:
        error_msg = f"Function '{func_name}' is not ready for production use."
        if description:
            error_msg += f" Issue: {description}"
        raise ProductionValidationError(error_msg)
    
    # Issue warning in all environments
    warning_msg = f"Function '{func_name}' may not be fully implemented."
    if description:
        warning_msg += f" Note: {description}"
    
    warnings.warn(warning_msg, UserWarning, stacklevel=4)


def warn_fallback_mode(component: str, reason: str = None):
    """
    Issue warning when falling back to less optimal implementation.
    
    Args:
        component: Name of the component using fallback mode
        reason: Optional reason for fallback
    """
    warning_msg = f"Component '{component}' is using fallback mode."
    if reason:
        warning_msg += f" Reason: {reason}"
    warning_msg += " This may impact performance or accuracy."
    
    warnings.warn(warning_msg, UserWarning, stacklevel=2)
    
    # Also log as warning
    logger = logging.getLogger(__name__)
    logger.warning(warning_msg)


def validate_not_nan_or_none(value: Any, metric_name: str) -> bool:
    """
    Validate that a metric value is not NaN or None.
    
    Args:
        value: The metric value to check
        metric_name: Name of the metric for error reporting
        
    Returns:
        True if value is valid, False otherwise
    """
    import numpy as np
    
    if value is None:
        warnings.warn(f"Metric '{metric_name}' returned None", UserWarning)
        return False
    
    if isinstance(value, (int, float)) and np.isnan(value):
        warnings.warn(f"Metric '{metric_name}' returned NaN", UserWarning)
        return False
    
    return True


def validate_metric_range(value: Any, metric_name: str, min_val: float = 0.0, max_val: float = 100.0) -> bool:
    """
    Validate that a metric value is within expected range.
    
    Args:
        value: The metric value to check
        metric_name: Name of the metric for error reporting
        min_val: Minimum expected value
        max_val: Maximum expected value
        
    Returns:
        True if value is in range, False otherwise
    """
    if not validate_not_nan_or_none(value, metric_name):
        return False
    
    try:
        num_value = float(value)
        if not (min_val <= num_value <= max_val):
            warnings.warn(
                f"Metric '{metric_name}' value {num_value} is outside expected range [{min_val}, {max_val}]",
                UserWarning
            )
            return False
        return True
    except (ValueError, TypeError):
        warnings.warn(f"Metric '{metric_name}' value is not numeric: {value}", UserWarning)
        return False


def check_implementation_completeness(methods_to_check: List[str], obj: Any) -> List[str]:
    """
    Check if methods in an object are properly implemented (not just returning placeholders).
    
    Args:
        methods_to_check: List of method names to check
        obj: Object to inspect
        
    Returns:
        List of method names that appear to be placeholders
    """
    placeholder_methods = []
    
    for method_name in methods_to_check:
        if hasattr(obj, method_name):
            method = getattr(obj, method_name)
            # Basic check for placeholder patterns in method source
            try:
                import inspect
                source = inspect.getsource(method)
                
                # Check for common placeholder patterns
                placeholder_patterns = [
                    'return 0.8',
                    'return 0.5',
                    '# Placeholder',
                    '# TODO',
                    'NotImplemented',
                    'pass  # Placeholder'
                ]
                
                if any(pattern in source for pattern in placeholder_patterns):
                    placeholder_methods.append(method_name)
                    
            except (OSError, TypeError):
                # Can't get source, skip this check
                pass
    
    return placeholder_methods


class QualityValidator:
    """Validator for ensuring code quality and completeness."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_metrics_results(self, results: dict, expected_metrics: List[str]) -> bool:
        """
        Validate that metrics results are complete and reasonable.
        
        Args:
            results: Dictionary of metric results
            expected_metrics: List of metrics that should be present
            
        Returns:
            True if validation passes, False otherwise
        """
        is_valid = True
        
        # Check for missing metrics
        missing_metrics = set(expected_metrics) - set(results.keys())
        if missing_metrics:
            error_msg = f"Missing metrics: {missing_metrics}"
            self.validation_errors.append(error_msg)
            self.logger.error(error_msg)
            is_valid = False
        
        # Check metric values
        for metric_name, value in results.items():
            if not validate_not_nan_or_none(value, metric_name):
                is_valid = False
            elif not validate_metric_range(value, metric_name):
                self.validation_warnings.append(f"Metric {metric_name} has unusual value: {value}")
        
        return is_valid
    
    def get_validation_summary(self) -> dict:
        """Get summary of validation results."""
        return {
            'errors': self.validation_errors,
            'warnings': self.validation_warnings,
            'has_errors': len(self.validation_errors) > 0,
            'has_warnings': len(self.validation_warnings) > 0
        }
    
    def clear_validation_results(self):
        """Clear stored validation results."""
        self.validation_errors.clear()
        self.validation_warnings.clear()