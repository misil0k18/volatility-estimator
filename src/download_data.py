import yfinance as yf
import pandas as pd

# Parámetros del proyecto (MVP)
TICKER = "SPY"
PERIOD = "10y"
INTERVAL = "1d"

def download_prices(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        group_by="column",   # <- esto suele evitar parte del lío
        progress=False
    )

    # --- Arreglo clave: si vienen columnas MultiIndex, las aplanamos ---
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance puede devolver algo tipo: ('Adj Close', 'SPY') o ('Price','Adj Close') etc.
        # Nos quedamos con la parte "real" del nombre de columna (la que contenga Open/Close/etc.)
        df.columns = [c[0] if c[0] in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] else c[1] for c in df.columns]

    # Asegurar índice de fechas limpio
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Validación mínima
    needed = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas esperadas: {missing}. Columnas recibidas: {df.columns.tolist()}")

    return df[needed]

if __name__ == "__main__":
    prices = download_prices(TICKER, PERIOD, INTERVAL)

    output_path = f"data/{TICKER}_prices.csv"
    prices.to_csv(output_path, index_label="Date")

    print(prices.head())
    print(f"\nDatos guardados en: {output_path}")
    print("Columnas:", prices.columns.tolist())
    print("Filas:", len(prices))
