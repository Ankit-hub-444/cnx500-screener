"""
Runs the CNX 500 screener (bullish, all conditions) and sends any stocks
that pass every condition to a Telegram group.

Required environment variables:
  TELEGRAM_BOT_TOKEN  - bot token from @BotFather
  TELEGRAM_CHAT_ID    - target group chat id (negative number for groups)
"""

import os
import sys

import requests
import yfinance as yf

from screener import get_cnx500_data, nifty_above_20ema, screen

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


def send_telegram_message(text: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set.")
        sys.exit(1)
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    resp = requests.post(url, data={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }, timeout=20)
    resp.raise_for_status()


def format_message(results: list[dict], nifty_ok: bool, nifty_price: float, nifty_ema20: float) -> str:
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    market_line = f"Nifty 50: {nifty_price} | 20 EMA: {nifty_ema20} | Above 20 EMA: {'YES' if nifty_ok else 'NO'}"

    if not results:
        return f"<b>CNX 500 Screener — {ts}</b>\n{market_line}\n\nNo stocks passed all conditions."

    results = sorted(results, key=lambda r: r.get("RS_Alpha_55d", 0), reverse=True)
    lines = [f"<b>CNX 500 Screener — {ts}</b>", market_line, "",
             f"<b>{len(results)} stock(s) passed ALL conditions:</b>"]
    for r in results:
        mc = r.get("Market_Cap_Cr")
        mc_str = f"{mc:,} Cr" if mc else "—"
        lines.append(
            f"• <b>{r['Symbol']}</b> — ₹{r['Price']} | RSI {r['RSI']} | "
            f"RS α {r['RS_Alpha_55d']}% | MCap {mc_str} | {r.get('Sector', '—')}"
        )
    return "\n".join(lines)


def main() -> None:
    symbols, industry_map = get_cnx500_data()

    nifty_df = yf.download("^NSEI", period="1y", interval="1d",
                            progress=False, auto_adjust=True)
    nifty_ok, nifty_price, nifty_ema20 = nifty_above_20ema(nifty_df)

    raw = yf.download(
        symbols, period="1y", interval="1d",
        progress=False, auto_adjust=True,
        group_by="ticker", threads=True,
    )

    results = []
    for sym in symbols:
        try:
            stock_df = raw[sym].dropna(how="all") if len(symbols) > 1 else raw
            if stock_df.empty:
                continue
            res = screen(sym, stock_df, nifty_df, active_conds=None, bearish=False)
            if res and res["All_Pass"]:
                res["Sector"] = industry_map.get(res["Symbol"], "—")
                results.append(res)
        except Exception:
            continue

    for res in results:
        try:
            mc = yf.Ticker(res["Symbol"] + ".NS").fast_info.market_cap
            res["Market_Cap_Cr"] = int(round(mc / 1e7)) if mc else None
        except Exception:
            res["Market_Cap_Cr"] = None

    message = format_message(results, nifty_ok, nifty_price, nifty_ema20)
    print(message)
    send_telegram_message(message)


if __name__ == "__main__":
    main()
