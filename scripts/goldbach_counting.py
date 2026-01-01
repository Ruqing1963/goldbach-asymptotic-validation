#!/usr/bin/env python3
"""
Goldbach Counting and Hardy-Littlewood Prediction

This script demonstrates the core algorithms used in the Hardy-Littlewood
Goldbach validation study.

Author: Ruqing Chen
Date: January 2026
"""

import numpy as np
from typing import List


def sieve_of_eratosthenes(n: int) -> List[int]:
    """
    Generate all primes up to n using the Sieve of Eratosthenes.
    
    Args:
        n: Upper limit for prime generation
        
    Returns:
        List of all primes <= n
        
    Example:
        >>> primes = sieve_of_eratosthenes(30)
        >>> primes
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    """
    if n < 2:
        return []
    
    # Create boolean array "is_prime[0..n]" and initialize all as true
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    # Sieve of Eratosthenes
    for i in range(2, int(np.sqrt(n)) + 1):
        if is_prime[i]:
            # Mark multiples of i as not prime
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    # Collect all primes
    primes = [i for i in range(n + 1) if is_prime[i]]
    return primes


def is_prime(n: int) -> bool:
    """
    Check if n is prime using trial division.
    
    Args:
        n: Number to test
        
    Returns:
        True if n is prime, False otherwise
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    # Test odd divisors up to sqrt(n)
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def goldbach_count(N: int) -> int:
    """
    Count the number of Goldbach representations of N.
    
    A Goldbach representation is N = p + q where p, q are primes.
    Counted with order: (p, q) and (q, p) are counted separately.
    
    Args:
        N: Even integer > 2
        
    Returns:
        Number of representations
        
    Example:
        >>> goldbach_count(10)
        4
        # 10 = 3+7 = 7+3 = 5+5 = 5+5 (counted twice except for 5+5)
    """
    if N < 4 or N % 2 != 0:
        raise ValueError("N must be an even integer >= 4")
    
    # Generate all primes up to N
    primes = sieve_of_eratosthenes(N)
    prime_set = set(primes)
    
    # Count representations
    count = 0
    for p in primes:
        if p > N // 2:
            break
        if (N - p) in prime_set:
            count += 1
    
    # Double count for ordered pairs, except when p = N/2
    count *= 2
    if N // 2 in prime_set and is_prime(N // 2):
        count -= 1  # Correct for the middle prime counted twice
    
    return count


def singular_series(N: int) -> float:
    """
    Compute the singular series S(N) for the Hardy-Littlewood formula.
    
    S(N) = ∏_{p|N, p>2} (p-1)/(p-2)
    
    CRITICAL: Must remove all factors of 2 before factorization!
    
    Args:
        N: Even integer
        
    Returns:
        Value of S(N)
    """
    if N % 2 != 0:
        raise ValueError("N must be even")
    
    # Remove all factors of 2 first (CRITICAL BUG FIX)
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


def hardy_littlewood_predict(N: int, order: int = 5) -> float:
    """
    Predict Goldbach count using Hardy-Littlewood formula with asymptotic expansion.
    
    G(N) ≈ C₂ · S(N) · (N/ln²N) · [1 + 2/lnN + 6/ln²N + ...]
    
    Args:
        N: Even integer > 2
        order: Order of asymptotic expansion (default: 5)
        
    Returns:
        Predicted Goldbach count
    """
    if N < 4 or N % 2 != 0:
        raise ValueError("N must be an even integer >= 4")
    
    # Twin prime constant
    C2 = 0.6601618158468695739278121100145557784326  
    
    # Singular series
    S_N = singular_series(N)
    
    # Asymptotic expansion
    ln_N = np.log(N)
    
    # Coefficients for expansion: [1, 2, 6, 24, 120, 720]
    # These are factorials: k! for k=0,1,2,3,4,5
    coefficients = [1, 2, 6, 24, 120, 720]
    
    expansion = sum(coefficients[k] / (ln_N ** k) 
                   for k in range(min(order + 1, len(coefficients))))
    
    # Final prediction
    prediction = C2 * S_N * (N / (ln_N ** 2)) * expansion
    
    return prediction


def compute_bias(N: int) -> float:
    """
    Compute the percentage bias of Hardy-Littlewood prediction.
    
    Bias = (Actual - Predicted) / Predicted * 100%
    
    Args:
        N: Even integer > 2
        
    Returns:
        Percentage bias
    """
    actual = goldbach_count(N)
    predicted = hardy_littlewood_predict(N)
    bias = (actual - predicted) / predicted * 100
    return bias


# Main execution example
if __name__ == "__main__":
    print("Hardy-Littlewood Goldbach Validation")
    print("=" * 50)
    
    # Test values
    test_values = [10, 100, 1000, 10000]
    
    print("\nTest Results:")
    print(f"{'N':<10} {'Actual':<10} {'Predicted':<12} {'Bias (%)':<10}")
    print("-" * 50)
    
    for N in test_values:
        actual = goldbach_count(N)
        predicted = hardy_littlewood_predict(N)
        bias = (actual - predicted) / predicted * 100
        
        print(f"{N:<10} {actual:<10} {predicted:<12.2f} {bias:<10.2f}")
    
    print("\n" + "=" * 50)
    print("Note: Negative bias means Hardy-Littlewood overestimates.")
    print("As N → ∞, bias → 0 (convergence).")
