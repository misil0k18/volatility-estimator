import pandas as pd
import numpy as np

TICKER = "SPY"
RETURNS_PATH = f"data/{TICKER}_returns.csv"

TRADING_DAYS = 252

def load_returns(path:str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df["log_return"]

def historical_volatility(log_returns: pd.Series) -> float:
    """
    Volatilidad histórica anualizada
    """
    daily_vol = log_returns.dropna().std()
    annual_vol = daily_vol * np.sqrt(TRADING_DAYS)
    return annual_vol

if __name__ == "__main__":
    log_returns = load_returns(RETURNS_PATH)

    vol = historical_volatility(log_returns)

    print("Volatilidad histórica simple (anualizada)")
    print(f"{TICKER}: {vol:.2%}")