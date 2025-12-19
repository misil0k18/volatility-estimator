import pandas as pd
from vol_tool import load_prices_csv, build_volatility_table, compare_measures

TICKER = "SPY"
PRICES_PATH = f"data/{TICKER}_prices.csv"
OUT_PATH = f"data/{TICKER}_volatility_measures.csv"

if __name__ == "__main__":
    # Cargar precios
    prices_df = load_prices_csv(PRICES_PATH)

    # Construir tabla de volatilidades
    vol_table = build_volatility_table(
        prices_df=prices_df,
        price_col="Adj Close", 
        windows=(20, 60, 120),
        lambda_=0.94
    )

    vol_table.to_csv(OUT_PATH)
    print(f"Guardado: {OUT_PATH}")
    print("\nÚltimas filas:")
    print(vol_table.tail())

    corr = compare_measures(vol_table)
    print("\nCorrelación entre medidas de volatilidad:")
    print(corr)