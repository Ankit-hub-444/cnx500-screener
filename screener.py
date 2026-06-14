"""
CNX 500 Stock Screener
Conditions:
  Market : Nifty 50 > 20 EMA
  Stock  : Price just crossed / near 150 EMA (within 3%)
           RS Alpha (55d) > 0
           Ratio Chart (vs Nifty 50) > 20 EMA of ratio
           RSI(14) > 50
           MACD Line > Signal Line
           Aroon Oscillator(25) > 0
           20 EMA > 50 EMA
           Price > Kaufman AMA (ER=14, Fast=2, Slow=30)
           A/D Line > 21 EMA of A/D
           Momentum(21) > 0
           Price > Donchian Channel (21-day) mid
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from io import StringIO
from datetime import datetime


# ── Technical indicators ──────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


def aroon_oscillator(high: pd.Series, low: pd.Series, period: int = 25) -> pd.Series:
    """
    Aroon Up   = (position of highest high in last period+1 bars from end) / period * 100
    Aroon Down = (position of lowest  low  in last period+1 bars from end) / period * 100
    Oscillator = Up - Down
    """
    # argmax/argmin returns 0-indexed position; position 'period' = most recent bar = 100%
    up = high.rolling(period + 1).apply(lambda x: float(np.argmax(x)) / period * 100, raw=True)
    dn = low.rolling(period + 1).apply(lambda x: float(np.argmin(x)) / period * 100, raw=True)
    return up - dn


def kaufman_ama(series: pd.Series, er_length: int = 14, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman Adaptive Moving Average."""
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    prices = series.values
    ama = np.full(len(prices), np.nan)

    # Seed the first valid AMA value
    start = er_length
    ama[start] = prices[start]

    for i in range(start + 1, len(prices)):
        direction = abs(prices[i] - prices[i - er_length])
        volatility = np.sum(np.abs(np.diff(prices[i - er_length: i + 1])))
        er = direction / volatility if volatility != 0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        ama[i] = ama[i - 1] + sc * (prices[i] - ama[i - 1])

    return pd.Series(ama, index=series.index)


def ad_line(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    hl_range = (high - low).replace(0, np.nan)
    clv = ((close - low) - (high - close)) / hl_range
    return (clv.fillna(0) * volume).cumsum()


def momentum(series: pd.Series, period: int = 21) -> pd.Series:
    return series - series.shift(period)


# ── CNX 500 symbol list ───────────────────────────────────────────────────

CNX500_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"

def get_cnx500_data() -> tuple[list[str], dict[str, str]]:
    """Returns (symbols_with_.NS, {SYMBOL: industry_string})"""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = requests.get(CNX500_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.columns = df.columns.str.strip()
        sym_col = next(c for c in df.columns if "symbol"   in c.lower())
        ind_col = next((c for c in df.columns if "industry" in c.lower()), None)
        df[sym_col] = df[sym_col].str.strip()
        symbols = df[sym_col].dropna().tolist()
        industry_map: dict[str, str] = {}
        if ind_col:
            df[ind_col] = df[ind_col].str.strip()
            industry_map = dict(zip(df[sym_col], df[ind_col]))
        print(f"      Loaded {len(symbols)} symbols from NSE")
        return [s + ".NS" for s in symbols], industry_map
    except Exception as exc:
        print(f"      [WARN] NSE download failed ({exc}). Exiting.")
        sys.exit(1)


def get_cnx500_symbols() -> list[str]:
    return get_cnx500_data()[0]


# ── Market condition ──────────────────────────────────────────────────────

def _extract_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Extract a named column as a float Series — handles flat and MultiIndex DataFrames."""
    data = df[col]
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    s = data.squeeze()
    # squeeze() on a single-element Series returns a scalar; guard against it
    if not isinstance(s, pd.Series):
        s = pd.Series([s], index=df.index[-1:])
    return s.astype(float)


def nifty_above_20ema(nifty_df: pd.DataFrame) -> tuple[bool, float, float]:
    if nifty_df is None or nifty_df.empty:
        return False, 0.0, 0.0
    try:
        close = _extract_col(nifty_df, "Close")
        if len(close) < 20:
            return False, 0.0, 0.0
        ema20 = ema(close, 20)
        last_close = float(close.iloc[-1])
        last_ema20 = float(ema20.iloc[-1])
        return last_close > last_ema20, round(last_close, 2), round(last_ema20, 2)
    except Exception:
        return False, 0.0, 0.0


# ── Per-stock screening ───────────────────────────────────────────────────

def screen(ticker: str, df: pd.DataFrame, nifty_df: pd.DataFrame,
           active_conds=None, bearish: bool = False) -> dict | None:
    try:
        # Align stock and Nifty on common trading dates
        common = df.index.intersection(nifty_df.index)
        if len(common) < 160:
            return None
        df = df.loc[common]
        nf = nifty_df.loc[common]

        # Extract OHLCV — _extract_col handles flat and MultiIndex column formats
        close  = _extract_col(df, "Close")
        high   = _extract_col(df, "High")
        low    = _extract_col(df, "Low")
        volume = _extract_col(df, "Volume")
        nifty  = _extract_col(nf, "Close")

        # Build the set of dates where ALL five series have non-NaN data.
        # Using explicit index intersection avoids any ambiguity with bool-series
        # alignment (different dtypes / NaN handling across pandas versions).
        idx = close.dropna().index
        for _s in (high, low, volume, nifty):
            idx = idx.intersection(_s.dropna().index)

        if len(idx) < 160:
            return None

        close  = close.reindex(idx)
        high   = high.reindex(idx)
        low    = low.reindex(idx)
        volume = volume.reindex(idx)
        nifty  = nifty.reindex(idx)

        r = {}

        # ── Compute raw indicator values (direction-agnostic) ─────────────

        # 150 EMA
        ema150      = ema(close, 150)
        last_price  = float(close.iloc[-1])
        last_ema150 = float(ema150.iloc[-1])
        crossed_above_150 = any(
            float(close.iloc[-(i+2)]) < float(ema150.iloc[-(i+2)])
            and float(close.iloc[-(i+1)]) >= float(ema150.iloc[-(i+1)])
            for i in range(5)
        )
        crossed_below_150 = any(
            float(close.iloc[-(i+2)]) > float(ema150.iloc[-(i+2)])
            and float(close.iloc[-(i+1)]) <= float(ema150.iloc[-(i+1)])
            for i in range(5)
        )
        near_above_150 = last_price >= last_ema150 and (last_price - last_ema150) / last_ema150 <= 0.03
        near_below_150 = last_price <= last_ema150 and (last_ema150 - last_price) / last_ema150 <= 0.03
        r["EMA150_dist%"] = round((last_price / last_ema150 - 1) * 100, 2)

        # RS Alpha
        rs_alpha = (close.iloc[-1] / close.iloc[-56] - 1) - (nifty.iloc[-1] / nifty.iloc[-56] - 1)
        r["RS_Alpha_55d"] = round(rs_alpha * 100, 2)

        # Ratio vs Nifty
        ratio      = close / nifty
        ratio_ema  = ema(ratio, 20)
        ratio_bull = float(ratio.iloc[-1]) > float(ratio_ema.iloc[-1])

        # RSI
        rsi_val = rsi(close, 14)
        r["RSI"] = round(float(rsi_val.iloc[-1]), 2)

        # MACD
        macd_line, sig_line = macd(close)
        macd_bull = float(macd_line.iloc[-1]) > float(sig_line.iloc[-1])

        # Aroon
        aroon_osc = aroon_oscillator(high, low, 25)
        r["Aroon"] = round(float(aroon_osc.iloc[-1]), 2)

        # EMA 20 vs 50 + 20 EMA cross detection
        ema20    = ema(close, 20)
        ema50    = ema(close, 50)
        ema_bull = float(ema20.iloc[-1]) > float(ema50.iloc[-1])
        crossed_above_ema20 = any(
            float(close.iloc[-(i+2)]) < float(ema20.iloc[-(i+2)])
            and float(close.iloc[-(i+1)]) >= float(ema20.iloc[-(i+1)])
            for i in range(5)
        )
        crossed_below_ema20 = any(
            float(close.iloc[-(i+2)]) > float(ema20.iloc[-(i+2)])
            and float(close.iloc[-(i+1)]) <= float(ema20.iloc[-(i+1)])
            for i in range(5)
        )

        # Kaufman AMA
        kama      = kaufman_ama(close, er_length=14, fast=2, slow=30)
        ama_bull  = float(close.iloc[-1]) > float(kama.iloc[-1])

        # A/D Line
        ad        = ad_line(high, low, close, volume)
        ad_bull   = float(ad.iloc[-1]) > float(ema(ad, 21).iloc[-1])

        # Momentum
        mom = momentum(close, 21)
        r["Mom_21"] = round(float(mom.iloc[-1]), 2)

        # Donchian mid
        dc_mid = (high.rolling(21).max() + low.rolling(21).min()) / 2
        r["Donchian_mid"] = round(float(dc_mid.iloc[-1]), 2)
        dc_bull = float(close.iloc[-1]) > float(dc_mid.iloc[-1])

        # ── Apply conditions (direction flips for bearish) ────────────────
        if not bearish:
            r["c0_150EMA"]    = crossed_above_150 or near_above_150
            r["c1_RS_Alpha"]  = rs_alpha > 0
            r["c2_Ratio"]     = ratio_bull
            r["c3_RSI"]       = r["RSI"] > 50
            r["c4_MACD"]      = macd_bull
            r["c5_Aroon"]     = r["Aroon"] > 0
            r["c6_EMA20_50"]   = ema_bull
            r["c7_AMA"]        = ama_bull
            r["c8_AD"]         = ad_bull
            r["c9_Mom"]        = r["Mom_21"] > 0
            r["c10_Donchian"]  = dc_bull
            r["c11_EMA20cross"]= crossed_above_ema20
        else:
            r["c0_150EMA"]     = crossed_below_150 or near_below_150
            r["c1_RS_Alpha"]   = rs_alpha < 0
            r["c2_Ratio"]      = not ratio_bull
            r["c3_RSI"]        = r["RSI"] < 50
            r["c4_MACD"]       = not macd_bull
            r["c5_Aroon"]      = r["Aroon"] < 0
            r["c6_EMA20_50"]   = not ema_bull
            r["c7_AMA"]        = not ama_bull
            r["c8_AD"]         = not ad_bull
            r["c9_Mom"]        = r["Mom_21"] < 0
            r["c10_Donchian"]  = not dc_bull
            r["c11_EMA20cross"]= crossed_below_ema20

        # ── Summary ───────────────────────────────────────────────────────
        all_cond_keys = sorted(k for k in r if k.startswith("c"))
        eval_keys = [k for k in all_cond_keys if active_conds is None or k in active_conds]
        passed = sum(1 for k in eval_keys if r[k])
        r["Passed"]   = f"{passed}/{len(eval_keys)}"
        r["All_Pass"] = len(eval_keys) > 0 and passed == len(eval_keys)
        r["Symbol"]   = ticker.replace(".NS", "")
        r["Price"]    = round(float(close.iloc[-1]), 2)
        r["50DMA"]    = round(float(close.rolling(50).mean().iloc[-1]), 2)

        return r

    except Exception:
        return None


# ── Main ──────────────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print("  CNX 500 STOCK SCREENER")
    print(f"  Run date : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)

    # Step 1 — symbol list
    print("\n[1/4] Fetching CNX 500 symbol list...")
    symbols = get_cnx500_symbols()

    # Step 2 — Nifty 50
    print("\n[2/4] Downloading Nifty 50 data...")
    nifty_df = yf.download("^NSEI", period="1y", interval="1d", progress=False, auto_adjust=True)
    if nifty_df.empty:
        print("ERROR: Could not fetch Nifty 50 data.")
        sys.exit(1)
    nifty_ok, nifty_price, nifty_ema20 = nifty_above_20ema(nifty_df)
    status = "YES" if nifty_ok else "NO"
    print(f"      Nifty 50 = {nifty_price}  |  20 EMA = {nifty_ema20}  |  Above 20 EMA: {status}")
    if not nifty_ok:
        print("      [WARN] Market condition not met — screener will still run.")

    # Step 3 — bulk download
    print(f"\n[3/4] Downloading 1-year daily data for {len(symbols)} stocks...")
    raw = yf.download(
        symbols, period="1y", interval="1d",
        progress=True, auto_adjust=True,
        group_by="ticker", threads=True,
    )

    # Step 4 — screen
    print("\n[4/4] Applying conditions...\n")
    results = []
    for sym in symbols:
        try:
            stock_df = raw[sym].dropna(how="all") if len(symbols) > 1 else raw
            if stock_df.empty:
                continue
            res = screen(sym, stock_df, nifty_df)
            if res:
                results.append(res)
        except Exception:
            continue

    passed = [r for r in results if r["All_Pass"]]

    # Output
    print("=" * 65)
    print(f"  {len(passed)} stock(s) passed ALL conditions  "
          f"(screened {len(results)} / {len(symbols)})")
    print("=" * 65)

    if passed:
        display_cols = [
            "Symbol", "Price", "50DMA", "Passed",
            "RSI", "Aroon", "RS_Alpha_55d", "Mom_21", "EMA150_dist%", "Donchian_mid",
            "c0_150EMA", "c1_RS_Alpha", "c2_Ratio", "c3_RSI", "c4_MACD",
            "c5_Aroon", "c6_EMA20_50", "c7_AMA", "c8_AD", "c9_Mom", "c10_Donchian",
        ]
        df_out = pd.DataFrame(passed)
        df_out = df_out[[c for c in display_cols if c in df_out.columns]]
        df_out = df_out.sort_values("RS_Alpha_55d", ascending=False)

        bool_cols = [c for c in df_out.columns if c.startswith("c")]
        df_out[bool_cols] = df_out[bool_cols].replace({True: "Y", False: "N"})

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 140)
        print(df_out.to_string(index=False))

        out_file = f"screener_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df_out.to_csv(out_file, index=False)
        print(f"\nSaved: {out_file}")
    else:
        print("No stocks passed all conditions today.")


if __name__ == "__main__":
    run()
