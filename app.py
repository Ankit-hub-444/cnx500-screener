import warnings
warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import threading
from datetime import datetime, timedelta

import numpy as np
import yfinance as yf
from flask import Flask, render_template, request, jsonify

from screener import (
    get_cnx500_symbols, nifty_above_20ema, screen,
)

app = Flask(__name__)

# ── In-memory cache (1 hour TTL) ─────────────────────────────────────────
_cache = {"symbols": None, "raw": None, "nifty": None, "fetched_at": None}
_CACHE_TTL = timedelta(hours=1)

# ── Single-job state ──────────────────────────────────────────────────────
_job = {
    "running": False, "status": "idle", "progress": 0,
    "message": "", "results": None, "market": None,
    "error": None, "screened": 0, "total": 0,
}
_lock = threading.Lock()


def _cache_valid():
    c = _cache
    return (c["raw"] is not None and c["fetched_at"] is not None
            and datetime.now() - c["fetched_at"] < _CACHE_TTL)


def _upd(**kw):
    with _lock:
        _job.update(kw)


def _to_python(v):
    if isinstance(v, (np.bool_,)):    return bool(v)
    if isinstance(v, (np.integer,)):  return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    return v


def _safe_row(row: dict) -> dict:
    return {k: _to_python(v) for k, v in row.items()}


# ── Background screening job ──────────────────────────────────────────────

def _run_job(active_conds, bearish: bool = False):
    try:
        _upd(running=True, status="loading", progress=5,
             message="Fetching CNX 500 symbol list...", error=None, results=None)

        if not _cache_valid():
            symbols = get_cnx500_symbols()
            _upd(progress=10, message=f"Loaded {len(symbols)} symbols — downloading Nifty 50...")

            nifty_df = yf.download("^NSEI", period="1y", interval="1d",
                                   progress=False, auto_adjust=True)
            _upd(progress=15, message=f"Downloading {len(symbols)} stocks — please wait ~1 min...")

            raw = yf.download(
                symbols, period="1y", interval="1d",
                progress=False, auto_adjust=True,
                group_by="ticker", threads=True,
            )
            with _lock:
                _cache.update({"symbols": symbols, "raw": raw,
                               "nifty": nifty_df, "fetched_at": datetime.now()})
        else:
            _upd(progress=50, message="Using cached data (downloaded < 1 hour ago)...")

        with _lock:
            symbols  = _cache["symbols"]
            raw      = _cache["raw"]
            nifty_df = _cache["nifty"]

        nifty_ok, nifty_price, nifty_ema20 = nifty_above_20ema(nifty_df)
        _upd(progress=55, message="Screening stocks...", total=len(symbols), screened=0)

        conds = set(active_conds) if active_conds is not None else None
        results = []
        n = len(symbols)

        for i, sym in enumerate(symbols):
            try:
                stock_df = raw[sym].dropna(how="all") if n > 1 else raw
                if stock_df.empty:
                    continue
                res = screen(sym, stock_df, nifty_df, active_conds=conds, bearish=bearish)
                if res and res["All_Pass"]:
                    results.append(_safe_row(res))
            except Exception:
                pass
            if i % 20 == 0:
                _upd(progress=55 + int((i + 1) / n * 40), screened=i + 1)

        # Fetch market cap only for the stocks that passed (typically < 20)
        if results:
            _upd(progress=96, message=f"Fetching market cap for {len(results)} stock(s)...")
            for res in results:
                try:
                    mc = yf.Ticker(res["Symbol"] + ".NS").fast_info.market_cap
                    res["Market_Cap_Cr"] = int(round(mc / 1e7)) if mc else None
                except Exception:
                    res["Market_Cap_Cr"] = None

        _upd(
            running=False, status="done", progress=100, screened=n,
            message=f"Done — {len(results)} stock(s) passed.",
            results=results,
            market={"ok": nifty_ok, "price": nifty_price, "ema20": nifty_ema20},
            bearish=bearish,
        )

    except Exception as exc:
        _upd(running=False, status="error", error=str(exc),
             message=f"Error: {exc}")


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    with _lock:
        if _job["running"]:
            return jsonify({"error": "Screener already running"}), 409
    data = request.json or {}
    active_conds = data.get("conditions")        # None → all conditions
    bearish      = data.get("mode", "bullish") == "bearish"
    threading.Thread(target=_run_job, args=(active_conds, bearish), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/status")
def status():
    with _lock:
        return jsonify(dict(_job))


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  CNX 500 Screener — Web UI")
    print(f"  Open  http://localhost:{port}  in your browser\n")
    app.run(debug=False, port=port, threaded=True)
