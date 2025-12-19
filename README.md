# Volatility Estimator (SPY)

Tool that computes and compares volatility measures:
- Historical volatility (baseline)
- Rolling volatility (20/60/120 days)
- EWMA volatility (λ=0.94)
- Event study around macro events (CPI, FOMC)

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
