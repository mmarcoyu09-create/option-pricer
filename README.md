# Option Pricing Engine

European option pricing in Python: closed-form Black-Scholes, Monte Carlo
simulation and (coming next) variance reduction, path-dependent options and
implied volatility from market data.

## Roadmap

- [x] **Level 1** – Black-Scholes pricer and analytic Greeks, verified against put-call parity, textbook values and finite differences
- [ ] **Level 2** – Monte Carlo pricer (GBM), convergence to the closed form, O(1/√N) error
- [ ] **Level 3** – Variance reduction: antithetic and control variates
- [ ] **Level 4** – Asian options (no closed form)
- [ ] **Level 5** – Implied volatility from real option chains, volatility smile

## Usage

```python
from pricer import price, greeks

price(S=100, K=100, T=1, r=0.05, sigma=0.2, option="call")   # 10.4506
greeks(S=100, K=100, T=1, r=0.05, sigma=0.2, option="call").delta  # 0.6368
```

```bash
pip install numpy scipy matplotlib pytest
python -m pytest -q        # run the checks
python demo_level1.py      # prices, Greeks and plot
```

## Structure

```
pricer/black_scholes.py   closed-form prices and Greeks (vectorised)
tests/                    parity, bounds, finite-difference Greek checks
demo_level1.py            example output and Greek profiles plot
```
