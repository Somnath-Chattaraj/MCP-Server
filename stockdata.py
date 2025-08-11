import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def get_equity_summary(symbol):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5)

    ticker = yf.Ticker(symbol + ".NS")
    df = ticker.history(
        start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")
    )

    if df.empty:
        return {"error": f"No data found for {symbol}"}

    last_close = df["Close"].iloc[-1]
    last_open = df["Open"].iloc[-1]
    avg_volume = df["Volume"].mean()
    volatility = df["Close"].pct_change().std()

    ohlcv_summary = {
        "symbol_requested": symbol,
        "last_close": float(last_close),
        "last_open": float(last_open),
        "avg_volume": float(avg_volume),
        "volatility": round(volatility, 6) if not np.isnan(volatility) else None,
        "ohlcv_table": df.reset_index().to_dict(orient="records"),
    }

    return ohlcv_summary


if __name__ == "__main__":
    print(get_equity_summary("SBIN"))
