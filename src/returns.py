import pandas as pd
import numpy as np

TICKER = "SPY"

PRICES_PATH = f"data/{TICKER}_prices.csv"
RETURNS_PATH = f"data/{TICKER}_returns.csv"

def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df

def compute_log_returns(adj_close: pd.Series) -> pd.Series:
    # r_t = ln(P_t / P_{t-1})
    return np.log(adj_close / adj_close.shift(1))

if __name__ == "__main__":
    prices = load_prices(PRICES_PATH)

    if "Adj Close" not in prices.columns:
        raise ValueError("No encuentro la columna 'Adj Close' en el CSV de precios.")

    adj_close = prices["Adj Close"].copy()
    log_ret = compute_log_returns(adj_close).rename("log_return")

    # Validación rápida
    print("Primeras 5 filas de Adj Close:")
    print(adj_close.head())
    print("\nPrimeras 5 filas de log returns:")
    print(log_ret.head())

    print("\nStats log returns (sin NaNs):")
    print(log_ret.dropna().describe())

    # Guardar
    out = pd.DataFrame({"Adj Close": adj_close, "log_return": log_ret})
    out.to_csv(RETURNS_PATH)

    print(f"\nReturns guardados en: {RETURNS_PATH}")
    print(f"Filas totales: {len(out)} | NaNs en log_return: {out['log_return'].isna().sum()}")
