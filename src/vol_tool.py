import pandas as pd
import numpy as np

TRADING_DAYS = 252

def load_prices_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df

def compute_log_returns(price_series: pd.Series) -> pd.Series:
    return np.log(price_series / price_series.shift(1)).rename("log_return")

def vol_simple_annualized(log_returns: pd.Series) -> float:
    daily_vol = log_returns.dropna().std()
    return float(daily_vol * np.sqrt(TRADING_DAYS))

def vol_rolling_annualized(log_returns: pd.Series, windows=(20, 60, 120)) -> pd.DataFrame:
    out = {}
    for w in windows:
        out[f"vol_{w}d"] = log_returns.rolling(w).std() * np.sqrt(TRADING_DAYS)
    return pd.DataFrame(out)

def vol_ewma_annualized(log_returns: pd.Series, lambda_: float = 0.94) -> pd.Series:
    r2 = (log_returns.dropna() ** 2)

    ewma_var = []
    var_t = r2.iloc[0]  # inicialización simple

    for x in r2:
        var_t = lambda_ * var_t + (1 - lambda_) * x
        ewma_var.append(var_t)

    ewma_vol = np.sqrt(pd.Series(ewma_var, index=r2.index))
    return (ewma_vol * np.sqrt(TRADING_DAYS)).rename(f"vol_ewma_l{lambda_}")

def build_volatility_table(
    prices_df: pd.DataFrame,
    price_col: str = "Adj Close",
    windows=(20, 60, 120),
    lambda_: float = 0.94
) -> pd.DataFrame:
    if price_col not in prices_df.columns:
        raise ValueError(f"No encuentro la columna '{price_col}' en el DataFrame de precios.")

    prices = prices_df[price_col].copy().rename(price_col)
    log_ret = compute_log_returns(prices)

    roll = vol_rolling_annualized(log_ret, windows=windows)
    ewma = vol_ewma_annualized(log_ret, lambda_=lambda_)

    table = pd.concat([prices, log_ret, roll, ewma], axis=1)

    # baseline como referencia (constante)
    baseline = vol_simple_annualized(log_ret)
    table["vol_baseline"] = baseline

    return table

def compare_measures(vol_df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve una mini-tabla con correlaciones entre medidas de volatilidad (ignorando baseline constante).
    """
    vol_cols = [c for c in vol_df.columns if c.startswith("vol_") and c != "vol_baseline"]
    corr = vol_df[vol_cols].dropna().corr()
    return corr
