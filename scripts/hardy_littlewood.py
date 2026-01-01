#!/usr/bin/env python3
"""
Hardy-Littlewood Formula and Asymptotic Predictions

This module implements the Hardy-Littlewood formula for predicting
Goldbach representations with various asymptotic expansions.

Author: Ruqing Chen
Date: January 2026
"""

import numpy as np
from typing import Union, List


# Twin prime constant (Π₂)
C2 = 0.6601618158468695739278121100145557784326


def singular_series(N: int) -> float:
    """
    Compute the singular series S(N) for Hardy-Littlewood formula.
    
    S(N) = ∏_{p|N, p>2} (p-1)/(p-2)
    
    CRITICAL: Must remove all factors of 2 before factorization!
    
    Args:
        N: Even integer
        
    Returns:
        Value of S(N)
        
    Example:
        >>> singular_series(12)  # 12 = 2² × 3
        1.5
    """
    if N % 2 != 0:
        raise ValueError("N must be even")
    
    # Remove all factors of 2 first (CRITICAL)
    temp = N
    while temp % 2 == 0:
        temp //= 2
    
    # Find odd prime factors
    factors = set()
    d = 3
    while d * d <= temp:
        if temp % d == 0:
            factors.add(d)
            while temp % d == 0:
                temp //= d
        d += 2
    
    if temp > 1:
        factors.add(temp)
    
    # Compute product
    S = 1.0
    for p in factors:
        S *= (p - 1) / (p - 2)
    
    return S


def hardy_littlewood_basic(N: int) -> float:
    """
    Basic Hardy-Littlewood formula without asymptotic expansion.
    
    G(N) ≈ C₂ · S(N) · N/ln²(N)
    
    Args:
        N: Even integer > 2
        
    Returns:
        Predicted count (basic formula)
    """
    if N < 4 or N % 2 != 0:
        raise ValueError("N must be an even integer >= 4")
    
    S_N = singular_series(N)
    ln_N = np.log(N)
    
    return C2 * S_N * N / (ln_N ** 2)


def hardy_littlewood_expanded(N: int, order: int = 5) -> float:
    """
    Hardy-Littlewood formula with asymptotic expansion.
    
    G(N) ≈ C₂ · S(N) · (N/ln²N) · [1 + 2/lnN + 6/ln²N + 24/ln³N + ...]
    
    Args:
        N: Even integer > 2
        order: Order of expansion (0-5)
        
    Returns:
        Predicted count with expansion
        
    Example:
        >>> hardy_littlewood_expanded(1000, order=2)
        168.36...
    """
    if N < 4 or N % 2 != 0:
        raise ValueError("N must be an even integer >= 4")
    
    if order < 0 or order > 5:
        raise ValueError("Order must be between 0 and 5")
    
    S_N = singular_series(N)
    ln_N = np.log(N)
    
    # Factorial coefficients: [1, 2, 6, 24, 120, 720]
    coefficients = [1, 2, 6, 24, 120, 720]
    
    # Asymptotic expansion
    expansion = sum(coefficients[k] / (ln_N ** k) 
                   for k in range(order + 1))
    
    return C2 * S_N * (N / (ln_N ** 2)) * expansion


def compare_orders(N: int) -> dict:
    """
    Compare Hardy-Littlewood predictions at different expansion orders.
    
    Args:
        N: Even integer
        
    Returns:
        Dictionary with predictions at orders 0-5
    """
    return {
        f"order_{k}": hardy_littlewood_expanded(N, order=k)
        for k in range(6)
    }


def compute_bias(N: int, actual: int, order: int = 5) -> float:
    """
    Compute percentage bias of Hardy-Littlewood prediction.
    
    Bias = (Actual - Predicted) / Predicted × 100%
    
    Args:
        N: Even integer
        actual: Actual Goldbach count
        order: Expansion order
        
    Returns:
        Percentage bias
    """
    predicted = hardy_littlewood_expanded(N, order=order)
    return (actual - predicted) / predicted * 100


def predict_for_range(N_min: int, N_max: int, num_points: int = 100,
                     order: int = 5) -> tuple:
    """
    Generate predictions over a range of N values.
    
    Args:
        N_min: Minimum N (will be rounded to even)
        N_max: Maximum N (will be rounded to even)
        num_points: Number of points to generate
        order: Expansion order
        
    Returns:
        Tuple of (N_values, predictions)
    """
    # Generate even N values
    N_values = np.logspace(np.log10(N_min), np.log10(N_max), num_points)
    N_values = (N_values // 2) * 2  # Round to even
    N_values = N_values.astype(int)
    
    # Compute predictions
    predictions = np.array([
        hardy_littlewood_expanded(N, order=order) 
        for N in N_values
    ])
    
    return N_values, predictions


def convergence_rate(N: int) -> dict:
    """
    Analyze the convergence rate of the asymptotic expansion.
    
    Args:
        N: Even integer
        
    Returns:
        Dictionary with convergence metrics
    """
    ln_N = np.log(N)
    
    # Leading term
    leading = C2 * singular_series(N) * N / (ln_N ** 2)
    
    # Corrections
    corrections = {
        '1/ln(N)': 2 / ln_N,
        '1/ln²(N)': 6 / (ln_N ** 2),
        '1/ln³(N)': 24 / (ln_N ** 3),
        '1/ln⁴(N)': 120 / (ln_N ** 4),
        '1/ln⁵(N)': 720 / (ln_N ** 5)
    }
    
    return {
        'leading_term': leading,
        'corrections': corrections,
        'total': hardy_littlewood_expanded(N, order=5)
    }


def error_bound(N: int, order: int = 5) -> float:
    """
    Estimate the error bound for truncated asymptotic expansion.
    
    The error is approximately O(1/ln^(order+1)(N)).
    
    Args:
        N: Even integer
        order: Expansion order
        
    Returns:
        Estimated relative error
    """
    ln_N = np.log(N)
    
    # Next term coefficient (factorial)
    next_coef = np.math.factorial(order + 1)
    
    # Error estimate
    error = next_coef / (ln_N ** (order + 1))
    
    return error


# Main execution example
if __name__ == "__main__":
    print("Hardy-Littlewood Formula Analysis")
    print("=" * 60)
    
    # Test case: N = 10000
    N = 10000
    print(f"\nAnalysis for N = {N:,}")
    print("-" * 60)
    
    # Basic vs Expanded
    basic = hardy_littlewood_basic(N)
    expanded = hardy_littlewood_expanded(N, order=5)
    
    print(f"Basic formula:    {basic:.2f}")
    print(f"Expanded (O=5):   {expanded:.2f}")
    print(f"Difference:       {expanded - basic:.2f} ({(expanded/basic - 1)*100:.2f}%)")
    
    # Compare orders
    print("\n" + "=" * 60)
    print("Predictions at different expansion orders:")
    print("-" * 60)
    
    orders = compare_orders(N)
    for order_name, pred in orders.items():
        order_num = int(order_name.split('_')[1])
        print(f"Order {order_num}:  {pred:10.2f}")
    
    # Convergence analysis
    print("\n" + "=" * 60)
    print("Convergence analysis:")
    print("-" * 60)
    
    conv = convergence_rate(N)
    print(f"Leading term: {conv['leading_term']:.2f}")
    print("\nCorrection terms:")
    for term, value in conv['corrections'].items():
        print(f"  {term:12s}: {value:8.4f}")
    
    # Error bound
    error = error_bound(N, order=5)
    print(f"\nEstimated error: {error:.6f} ({error*100:.4f}%)")
    
    print("\n" + "=" * 60)
