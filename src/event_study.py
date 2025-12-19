import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

TICKER = "SPY"
VOL_PATH = f"data/{TICKER}_volatility_measures.csv"
EVENTS_PATH = "data/events.csv"
OUT_SUMMARY = f"reports/{TICKER}_event_study_summary.csv"

# qué columna de volatilidad analizamos como principal
VOL_COL = "vol_ewma_l0.94"   # puedes cambiar a "vol_20d" si quieres

def load_vol_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    return df

def load_events(path: str) -> pd.DataFrame:
    ev = pd.read_csv(path, parse_dates=["date"])
    # defaults por si no están
    if "window_pre" not in ev.columns:
        ev["window_pre"] = 20
    if "window_post" not in ev.columns:
        ev["window_post"] = 20
    return ev

def slice_by_trading_days(index: pd.DatetimeIndex, event_date: pd.Timestamp, pre: int, post: int):
    """
    Encuentra el rango [event-pre, event+post] en términos de filas (días de trading).
    Si el event_date no cae en día de trading, lo alineamos al siguiente día de trading.
    """
    # posición del primer día de trading >= event_date
    pos = index.searchsorted(event_date)
    if pos >= len(index):
        return None

    start = max(pos - pre, 0)
    end = min(pos + post, len(index) - 1)
    return start, pos, end

def summarize_event(df: pd.DataFrame, event_date: pd.Timestamp, pre: int, post: int, vol_col: str):
    idx = df.index
    res = slice_by_trading_days(idx, event_date, pre, post)
    if res is None:
        return None

    start, pos, end = res
    event_trading_day = idx[pos]

    window = df.iloc[start:end+1].copy()
    pre_slice = df.iloc[start:pos].copy()
    post_slice = df.iloc[pos+1:end+1].copy()

    # stats robustos: mean y median
    pre_mean = pre_slice[vol_col].mean()
    post_mean = post_slice[vol_col].mean()
    pre_med = pre_slice[vol_col].median()
    post_med = post_slice[vol_col].median()

    delta_mean = post_mean - pre_mean
    pct_mean = (delta_mean / pre_mean) if pre_mean and pre_mean > 0 else np.nan

    return {
        "event_date_input": event_date.date(),
        "event_date_trading": event_trading_day.date(),
        "pre_days": pre,
        "post_days": post,
        "vol_col": vol_col,
        "pre_mean": pre_mean,
        "post_mean": post_mean,
        "delta_mean": delta_mean,
        "pct_change_mean": pct_mean,
        "pre_median": pre_med,
        "post_median": post_med,
        "delta_median": post_med - pre_med,
    }, window, event_trading_day

def plot_event(window: pd.DataFrame, event_trading_day: pd.Timestamp, vol_col: str, title: str, out_path: str):
    plt.figure(figsize=(12, 6))
    plt.plot(window.index, window[vol_col], label=vol_col)

    plt.axvline(event_trading_day, linestyle="--", label="event (trading day)")

    plt.title(title)
    plt.ylabel("Volatility (annualized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    vol = load_vol_table(VOL_PATH)

    if VOL_COL not in vol.columns:
        raise ValueError(f"No existe la columna {VOL_COL}. Columnas disponibles: {list(vol.columns)}")

    events = load_events(EVENTS_PATH)

    rows = []
    for _, e in events.iterrows():
        event_date = e["date"]
        pre = int(e["window_pre"])
        post = int(e["window_post"])
        label = e.get("label", "event")
        etype = e.get("event_type", "EVENT")

        result = summarize_event(vol, event_date, pre, post, VOL_COL)
        if result is None:
            print(f"[SKIP] Evento fuera de rango: {label} ({event_date.date()})")
            continue

        summary, window, event_trading_day = result
        summary["event_type"] = etype
        summary["label"] = label
        rows.append(summary)

        # plot por evento
        safe_label = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in label)
        out_plot = f"reports/{TICKER}_{etype}_{safe_label}_{event_date.date()}.png"
        plot_event(
            window=window,
            event_trading_day=event_trading_day,
            vol_col=VOL_COL,
            title=f"{TICKER} | {etype} | {label} | {VOL_COL}",
            out_path=out_plot
        )

        print(f"[OK] {etype} {label} ({event_date.date()}) -> plot: {out_plot}")

    summary_df = pd.DataFrame(rows).sort_values(["event_date_trading"])
    summary_df.to_csv(OUT_SUMMARY, index=False)
    print(f"\nResumen guardado en: {OUT_SUMMARY}")
    print("\nTop eventos por subida de vol (delta_mean):")
    print(summary_df.sort_values("delta_mean", ascending=False).head(10)[
        ["event_date_trading", "event_type", "label", "pre_mean", "post_mean", "delta_mean", "pct_change_mean"]
    ])
