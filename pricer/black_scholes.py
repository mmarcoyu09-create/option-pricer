"""
Black-Scholes pricer for European options, with analytic Greeks.

Model assumptions:
    dS_t = r S_t dt + sigma S_t dW_t   (under the risk-neutral measure)
    constant r, sigma, continuous dividend yield q, no frictions.

Notation:
    S     spot price
    K     strike
    T     time to maturity in years
    r     continuously compounded risk-free rate
    sigma annualised volatility
    q     continuous dividend yield (default 0)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def _validate(S, K, T, sigma):
    if np.any(np.asarray(S) <= 0) or np.any(np.asarray(K) <= 0):
        raise ValueError("S and K must be positive")
    if np.any(np.asarray(T) <= 0):
        raise ValueError("T must be positive (use the payoff for T = 0)")
    if np.any(np.asarray(sigma) <= 0):
        raise ValueError("sigma must be positive")


def d1_d2(S, K, T, r, sigma, q=0.0):
    """Return d1, d2 of the Black-Scholes formula."""
    _validate(S, K, T, sigma)
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def price(S, K, T, r, sigma, option="call", q=0.0):
    """Black-Scholes price of a European call or put."""
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    disc_S = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)
    if option == "call":
        return disc_S * norm.cdf(d1) - disc_K * norm.cdf(d2)
    if option == "put":
        return disc_K * norm.cdf(-d2) - disc_S * norm.cdf(-d1)
    raise ValueError("option must be 'call' or 'put'")


@dataclass
class Greeks:
    delta: float  # dV/dS
    gamma: float  # d2V/dS2
    vega: float   # dV/dsigma, per 1.00 change in vol (divide by 100 for per vol point)
    theta: float  # dV/dt, per year (divide by 365 for per calendar day)
    rho: float    # dV/dr, per 1.00 change in rate


def greeks(S, K, T, r, sigma, option="call", q=0.0) -> Greeks:
    """Analytic Black-Scholes Greeks."""
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = norm.pdf(d1)
    eq = np.exp(-q * T)
    er = np.exp(-r * T)
    sqrt_t = np.sqrt(T)

    gamma = eq * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * eq * pdf_d1 * sqrt_t
    common_theta = -S * eq * pdf_d1 * sigma / (2 * sqrt_t)

    if option == "call":
        delta = eq * norm.cdf(d1)
        theta = common_theta - r * K * er * norm.cdf(d2) + q * S * eq * norm.cdf(d1)
        rho = K * T * er * norm.cdf(d2)
    elif option == "put":
        delta = -eq * norm.cdf(-d1)
        theta = common_theta + r * K * er * norm.cdf(-d2) - q * S * eq * norm.cdf(-d1)
        rho = -K * T * er * norm.cdf(-d2)
    else:
        raise ValueError("option must be 'call' or 'put'")

    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)
