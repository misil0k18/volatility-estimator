import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TICKER = "SPY"
RETURNS_PATH = f"data/{TICKER}_returns.csv"

TRADING_DAYS = 252
LAMBDA = 0.94 # RiskMetrics standard for daily data

def load_returns(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df["log_return"]

def ewma_volatility(log_returns: pd.Series, lambda_:float) -> pd.Series:
    """
    EWMA volatility (annualized)
    """
    returns_sq = log_returns.dropna() ** 2

    ewma_var = []
    var_t = returns_sq.iloc[0] # incicialización simple

    for r2 in returns_sq:
        var_t = lambda_ * var_t + (1 - lambda_) * r2
        ewma_var.append(var_t)

    ewma_vol = np.sqrt(pd.Series(ewma_var, index=returns_sq.index))
    return ewma_vol * np.sqrt(TRADING_DAYS)

def rolling_volatility(log_returns:pd.Series, window:int) -> pd.Series:
    return log_returns.rolling(window).std() * np.sqrt(TRADING_DAYS)

if __name__ == "__main__":
    log_returns = load_returns(RETURNS_PATH)

    ewma_vol = ewma_volatility(log_returns, LAMBDA)
    roll_20 = rolling_volatility(log_returns, 20)

    # Plot comparación 
    plt.figure(figsize=(12, 6))
    plt.plot(ewma_vol.index, ewma_vol, label = "EWMA (λ=0.94)")
    plt.plot(roll_20.index, roll_20, label = "Rolling 20d", alpha=0.7)

    plt.title(f"{TICKER} Volatility: EWMA vs Rolling 20d")
    plt.ylabel("Volatility")
    plt.legend()
    plt.tight_layout()
    plt.show()