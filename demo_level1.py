"""
Level 1 demo: Black-Scholes prices and Greeks.

Run with:  python demo_level1.py
Produces greeks_vs_spot.png
"""

import matplotlib.pyplot as plt
import numpy as np

from pricer import greeks, price

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

print(f"Inputs: S={S}, K={K}, T={T}y, r={r:.0%}, sigma={sigma:.0%}\n")
for opt in ("call", "put"):
    g = greeks(S, K, T, r, sigma, opt)
    print(f"{opt.upper():4s} price = {price(S, K, T, r, sigma, opt):8.4f}")
    print(f"     delta = {g.delta:8.4f}   gamma = {g.gamma:.4f}")
    print(f"     vega  = {g.vega / 100:8.4f} per vol point")
    print(f"     theta = {g.theta / 365:8.4f} per day")
    print(f"     rho   = {g.rho / 100:8.4f} per 1% rate\n")

# Greeks as a function of spot, for three maturities
spots = np.linspace(60, 140, 400)
maturities = [0.1, 0.5, 1.0]
panels = [("Call price", lambda s, t: price(s, K, t, r, sigma, "call")),
          ("Delta", lambda s, t: greeks(s, K, t, r, sigma, "call").delta),
          ("Gamma", lambda s, t: greeks(s, K, t, r, sigma, "call").gamma),
          ("Vega (per vol point)", lambda s, t: greeks(s, K, t, r, sigma, "call").vega / 100)]

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, (title, fn) in zip(axes.flat, panels):
    for t in maturities:
        ax.plot(spots, fn(spots, t), label=f"T = {t}y")
    ax.axvline(K, color="grey", lw=0.8, ls="--")
    ax.set_title(title)
    ax.set_xlabel("Spot")
    ax.grid(alpha=0.3)
axes[0, 0].plot(spots, np.maximum(spots - K, 0), color="black", lw=1, ls=":", label="Payoff")
axes[0, 0].legend()
fig.suptitle(f"European call, K = {K:.0f}, r = {r:.0%}, sigma = {sigma:.0%}")
fig.tight_layout()
fig.savefig("greeks_vs_spot.png", dpi=150)
print("Saved greeks_vs_spot.png")
