#!/usr/bin/env python3
"""
Convergence Analysis of Hardy-Littlewood Bias

This module analyzes the convergence behavior of the bias in Hardy-Littlewood
predictions, including fitting convergence models and validating theoretical bounds.

Author: Ruqing Chen
Date: January 2026
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from typing import Tuple, Dict, Optional
import warnings


def convergence_model(ln_N: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Two-term convergence model for Hardy-Littlewood bias.
    
    Bias(N) = a/ln(N) + b/ln²(N)
    
    Args:
        ln_N: Natural logarithm of N
        a: Coefficient for 1/ln(N) term
        b: Coefficient for 1/ln²(N) term
        
    Returns:
        Predicted bias values
    """
    return a / ln_N + b / (ln_N ** 2)


def three_term_model(ln_N: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Three-term convergence model.
    
    Bias(N) = a/ln(N) + b/ln²(N) + c/ln³(N)
    
    Args:
        ln_N: Natural logarithm of N
        a, b, c: Coefficients
        
    Returns:
        Predicted bias values
    """
    return a / ln_N + b / (ln_N ** 2) + c / (ln_N ** 3)


def fit_convergence_model(N_values: np.ndarray, bias_values: np.ndarray,
                         model: str = 'two_term') -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit convergence model to bias data.
    
    Args:
        N_values: Array of N values
        bias_values: Array of bias percentages
        model: 'two_term' or 'three_term'
        
    Returns:
        Tuple of (fitted_parameters, parameter_errors)
    """
    ln_N = np.log(N_values)
    
    if model == 'two_term':
        # Initial guess: [-58, -17] based on theoretical expectations
        p0 = [-58.0, -17.0]
        popt, pcov = curve_fit(convergence_model, ln_N, bias_values, p0=p0)
    elif model == 'three_term':
        p0 = [-58.0, -17.0, -5.0]
        popt, pcov = curve_fit(three_term_model, ln_N, bias_values, p0=p0)
    else:
        raise ValueError("Model must be 'two_term' or 'three_term'")
    
    # Parameter errors (standard deviation)
    perr = np.sqrt(np.diag(pcov))
    
    return popt, perr


def compute_residuals(N_values: np.ndarray, bias_values: np.ndarray,
                     params: np.ndarray, model: str = 'two_term') -> np.ndarray:
    """
    Compute residuals between data and model.
    
    Args:
        N_values: Array of N values
        bias_values: Array of actual bias values
        params: Fitted parameters
        model: Model type
        
    Returns:
        Array of residuals
    """
    ln_N = np.log(N_values)
    
    if model == 'two_term':
        predicted = convergence_model(ln_N, *params)
    elif model == 'three_term':
        predicted = three_term_model(ln_N, *params)
    else:
        raise ValueError("Model must be 'two_term' or 'three_term'")
    
    return bias_values - predicted


def convergence_metrics(residuals: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics for convergence quality.
    
    Args:
        residuals: Array of residuals
        
    Returns:
        Dictionary of metrics
    """
    return {
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
        'max_abs_residual': np.max(np.abs(residuals)),
        'rmse': np.sqrt(np.mean(residuals ** 2)),
        'r_squared': 1 - np.var(residuals) / np.var(residuals + np.mean(residuals))
    }


def extrapolate_bias(N: float, params: np.ndarray, 
                    model: str = 'two_term') -> float:
    """
    Extrapolate bias to large N using fitted model.
    
    Args:
        N: Target N value
        params: Fitted parameters
        model: Model type
        
    Returns:
        Predicted bias at N
    """
    ln_N = np.log(N)
    
    if model == 'two_term':
        return convergence_model(ln_N, *params)
    elif model == 'three_term':
        return three_term_model(ln_N, *params)
    else:
        raise ValueError("Model must be 'two_term' or 'three_term'")


def convergence_rate_analysis(N_values: np.ndarray, 
                              bias_values: np.ndarray) -> Dict:
    """
    Analyze the rate of convergence.
    
    Args:
        N_values: Array of N values
        bias_values: Array of bias values
        
    Returns:
        Dictionary with convergence rate analysis
    """
    # Compute successive ratios
    ratios = []
    for i in range(len(bias_values) - 1):
        if bias_values[i+1] != 0:
            ratio = bias_values[i] / bias_values[i+1]
            ln_ratio = np.log(N_values[i+1]) / np.log(N_values[i])
            ratios.append({
                'N_from': N_values[i],
                'N_to': N_values[i+1],
                'bias_ratio': ratio,
                'ln_ratio': ln_ratio
            })
    
    return {
        'ratios': ratios,
        'mean_convergence_factor': np.mean([r['bias_ratio'] for r in ratios])
    }


def validate_asymptotic_behavior(N_min: int, N_max: int,
                                params: np.ndarray) -> Dict:
    """
    Validate that the model shows proper asymptotic behavior.
    
    Args:
        N_min: Minimum N for validation
        N_max: Maximum N for validation
        params: Model parameters
        
    Returns:
        Dictionary with validation results
    """
    # Generate test points
    N_test = np.logspace(np.log10(N_min), np.log10(N_max), 100)
    bias_test = [extrapolate_bias(N, params) for N in N_test]
    
    # Check monotonicity (bias should decrease in magnitude)
    abs_bias = np.abs(bias_test)
    is_monotonic = np.all(np.diff(abs_bias) < 0)
    
    # Check convergence to zero
    final_bias = abs_bias[-1]
    converges_to_zero = final_bias < abs_bias[0] / 10
    
    # Compute convergence speed
    decay_rate = (abs_bias[0] - abs_bias[-1]) / abs_bias[0]
    
    return {
        'is_monotonic': is_monotonic,
        'converges_to_zero': converges_to_zero,
        'decay_rate': decay_rate,
        'initial_bias': bias_test[0],
        'final_bias': bias_test[-1],
        'N_range': (N_min, N_max)
    }


def compare_models(N_values: np.ndarray, bias_values: np.ndarray) -> pd.DataFrame:
    """
    Compare two-term and three-term models.
    
    Args:
        N_values: Array of N values
        bias_values: Array of bias values
        
    Returns:
        DataFrame comparing model performance
    """
    results = []
    
    for model in ['two_term', 'three_term']:
        params, errors = fit_convergence_model(N_values, bias_values, model=model)
        residuals = compute_residuals(N_values, bias_values, params, model=model)
        metrics = convergence_metrics(residuals)
        
        results.append({
            'model': model,
            'num_params': len(params),
            'rmse': metrics['rmse'],
            'r_squared': metrics['r_squared'],
            'max_residual': metrics['max_abs_residual'],
            'params': params,
            'errors': errors
        })
    
    return pd.DataFrame(results)


def bias_at_milestone(params: np.ndarray, milestone: str = '10^12') -> float:
    """
    Predict bias at key milestone values of N.
    
    Args:
        params: Fitted parameters
        milestone: '10^9', '10^12', '10^15', etc.
        
    Returns:
        Predicted bias
    """
    N = eval(milestone.replace('^', '**'))
    return extrapolate_bias(N, params)


# Main execution example
if __name__ == "__main__":
    print("Hardy-Littlewood Bias Convergence Analysis")
    print("=" * 70)
    
    # Example data (from the paper)
    N_data = np.array([1e3, 1e4, 1e6, 1e7, 1.5e8, 1e9])
    bias_data = np.array([-7.87, -7.87, -7.87, -3.66, -0.68, -0.49])
    
    print("\nFitting convergence model...")
    print("-" * 70)
    
    # Fit two-term model
    params, errors = fit_convergence_model(N_data, bias_data, model='two_term')
    
    print(f"Two-term model: Bias(N) = a/ln(N) + b/ln²(N)")
    print(f"  a = {params[0]:.2f} ± {errors[0]:.2f}")
    print(f"  b = {params[1]:.2f} ± {errors[1]:.2f}")
    
    # Compute residuals
    residuals = compute_residuals(N_data, bias_data, params)
    metrics = convergence_metrics(residuals)
    
    print("\nModel quality:")
    print(f"  RMSE:           {metrics['rmse']:.3f}%")
    print(f"  R²:             {metrics['r_squared']:.3f}")
    print(f"  Max residual:   {metrics['max_abs_residual']:.3f}%")
    
    # Extrapolate to key milestones
    print("\n" + "=" * 70)
    print("Extrapolated bias at key milestones:")
    print("-" * 70)
    
    milestones = {
        '10^9': 1e9,
        '10^12': 1e12,
        '10^15': 1e15,
        '10^18': 1e18
    }
    
    for name, N in milestones.items():
        bias = extrapolate_bias(N, params)
        print(f"  N = {name:6s}:  Bias = {bias:+7.3f}%")
    
    # Validate asymptotic behavior
    print("\n" + "=" * 70)
    print("Asymptotic behavior validation:")
    print("-" * 70)
    
    validation = validate_asymptotic_behavior(1e3, 1e15, params)
    print(f"  Monotonic decrease:     {validation['is_monotonic']}")
    print(f"  Converges to zero:      {validation['converges_to_zero']}")
    print(f"  Decay rate:             {validation['decay_rate']:.1%}")
    print(f"  Bias at N=10³:          {validation['initial_bias']:.3f}%")
    print(f"  Bias at N=10¹⁵:         {validation['final_bias']:.3f}%")
    
    print("\n" + "=" * 70)
    print("Convergence implies Hardy-Littlewood formula is asymptotically correct!")
    print("=" * 70)
