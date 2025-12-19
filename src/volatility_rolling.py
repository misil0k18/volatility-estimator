import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TICKER = "SPY"
RETURNS_PATH = f"data/{TICKER}_returns.csv"

TRADING_DAYS = 252
WINDOWS = [20, 60, 120]  # en días

def load_returns(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df["log_return"]

def rolling_volatility(log_returns: pd.Series, window: int) -> pd.Series:
    return log_returns.rolling(window).std() * np.sqrt(TRADING_DAYS)

if __name__ == "__main__":
    log_returns = load_returns(RETURNS_PATH)

    vol_df = pd.DataFrame(index=log_returns.index)

    for w in WINDOWS:
        vol_df[f"vol_{w}d"] = rolling_volatility(log_returns, w)

    # Mostrar últimas filas
    print(vol_df.tail())

    # Plot
    plt.figure(figsize=(12, 6))
    for col in vol_df.columns:
        plt.plot(vol_df.index, vol_df[col], label=col)

    plt.title(f"{TICKER} Rolling Volatility (Annualized)")
    plt.ylabel("Volatility")
    plt.legend()
    plt.tight_layout()
    plt.show()