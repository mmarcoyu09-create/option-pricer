"""
Sanity checks for the Black-Scholes pricer.

Run with:  python -m pytest -q
"""

import numpy as np
import pytest

from pricer import greeks, price

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2


def test_textbook_value():
    # Hull, standard ATM example: call ~ 10.4506, put ~ 5.5735
    assert price(S, K, T, r, sigma, "call") == pytest.approx(10.4506, abs=1e-4)
    assert price(S, K, T, r, sigma, "put") == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("K_", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(K_, q):
    # C - P = S e^{-qT} - K e^{-rT}
    c = price(S, K_, T, r, sigma, "call", q)
    p = price(S, K_, T, r, sigma, "put", q)
    assert c - p == pytest.approx(S * np.exp(-q * T) - K_ * np.exp(-r * T), abs=1e-10)


def test_no_arbitrage_bounds():
    c = price(S, K, T, r, sigma, "call")
    assert max(S - K * np.exp(-r * T), 0) <= c <= S


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_greeks_match_finite_differences(option, q):
    """Analytic Greeks must agree with bump-and-reprice."""
    g = greeks(S, K, T, r, sigma, option, q)
    h = 1e-4
    f = lambda **kw: price(**{"S": S, "K": K, "T": T, "r": r, "sigma": sigma,
                              "option": option, "q": q, **kw})

    delta_fd = (f(S=S + h) - f(S=S - h)) / (2 * h)
    gamma_fd = (f(S=S + h) - 2 * f() + f(S=S - h)) / h**2
    vega_fd = (f(sigma=sigma + h) - f(sigma=sigma - h)) / (2 * h)
    theta_fd = -(f(T=T + h) - f(T=T - h)) / (2 * h)  # theta = -dV/dT
    rho_fd = (f(r=r + h) - f(r=r - h)) / (2 * h)

    assert g.delta == pytest.approx(delta_fd, rel=1e-5)
    assert g.gamma == pytest.approx(gamma_fd, rel=1e-3)
    assert g.vega == pytest.approx(vega_fd, rel=1e-5)
    assert g.theta == pytest.approx(theta_fd, rel=1e-5)
    assert g.rho == pytest.approx(rho_fd, rel=1e-5)


def test_vectorised():
    strikes = np.array([90.0, 100.0, 110.0])
    prices = price(S, strikes, T, r, sigma, "call")
    assert prices.shape == (3,)
    assert np.all(np.diff(prices) < 0)  # call price decreases in strike


def test_invalid_inputs():
    with pytest.raises(ValueError):
        price(S, K, 0.0, r, sigma)
    with pytest.raises(ValueError):
        price(S, K, T, r, sigma, "straddle")
