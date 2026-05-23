"""
Polymarket Claude Copy Trading Bot v2
- Monitors RN1 for new positions
- Whale detection: any trade > $10K from any wallet
- Claude analyses all trades before alerting
- Telegram alerts to Polymarket bot only
"""

import os
import time
import json
import logging
import requests
from datetime import datetime, timezone
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
PK_PRIVATE_KEY    = os.environ["PK_PRIVATE_KEY"]
PK_API_KEY        = os.environ["PK_API_KEY"]
PK_API_SECRET     = os.environ["PK_API_SECRET"]
PK_PASSPHRASE     = os.environ["PK_PASSPHRASE"]
MY_PROXY_WALLET   = os.environ["MY_PROXY_WALLET"]

# Polymarket Telegram bot
TELEGRAM_TOKEN    = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID  = os.environ["TELEGRAM_CHAT_ID"]

TRADER_1          = os.environ.get("TRADER_1", "0x2005d16a84ceefa912d4e380cd32e7ff827875ea")
TRADER_2          = os.environ.get("TRADER_2", "disabled")

POLL_INTERVAL         = int(os.environ.get("POLL_INTERVAL", "30"))
TRADE_SIZE_USDC       = float(os.environ.get("TRADE_SIZE_USDC", "1.0"))
MIN_ODDS              = float(os.environ.get("MIN_ODDS", "0.05"))
MAX_ODDS              = float(os.environ.get("MAX_ODDS", "0.95"))
MAX_SPREAD            = float(os.environ.get("MAX_SPREAD", "0.10"))
MAX_TRADE_AGE_MINUTES = float(os.environ.get("MAX_TRADE_AGE_MINUTES", "30"))
WHALE_THRESHOLD       = float(os.environ.get("WHALE_THRESHOLD", "10000"))

DATA_API  = "https://data-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"

seen_trades: set = set()
seen_whale_trades: set = set()
open_positions: dict = {}


# TELEGRAM

def send_telegram(message: str):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10
        )
        if not r.ok:
            log.error(f"Telegram error: {r.text}")
    except Exception as e:
        log.error(f"Telegram failed: {e}")


# POLYMARKET DATA

def get_recent_trades(wallet: str, limit: int = 50) -> list:
    try:
        r = requests.get(f"{DATA_API}/trades", params={"user": wallet, "limit": limit}, timeout=10)
        r.raise_for_status()
        return r.json() if isinstance(r.json(), list) else []
    except Exception as e:
        log.error(f"Failed to fetch trades for {wallet[:10]}...: {e}")
        return []


def get_whale_trades(limit: int = 50) -> list:
    """Fetch recent large trades from Polymarket activity feed."""
    try:
        r = requests.get(
            f"{DATA_API}/trades",
            params={"limit": limit, "sortBy": "AMOUNT", "sortDirection": "DESC"},
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            return []
        # Filter trades above whale threshold
        whales = []
        for t in data:
            size = float(t.get("size", 0))
            price = float(t.get("price", 0))
            usd_value = size * price
            if usd_value >= WHALE_THRESHOLD:
                t["usd_value"] = usd_value
                whales.append(t)
        return whales
    except Exception as e:
        log.error(f"Failed to fetch whale trades: {e}")
        return []


def get_token_price(token_id: str) -> Optional[dict]:
    try:
        r_buy  = requests.get(f"{CLOB_API}/price", params={"token_id": token_id, "side": "BUY"}, timeout=10)
        r_sell = requests.get(f"{CLOB_API}/price", params={"token_id": token_id, "side": "SELL"}, timeout=10)
        if not r_buy.ok or not r_sell.ok:
            return None
        buy_price  = float(r_buy.json().get("price", 1.0))
        sell_price = float(r_sell.json().get("price", 0.0))
        spread     = buy_price - sell_price
        midpoint   = (buy_price + sell_price) / 2
        log.info(f"💰 Price — BUY: {buy_price:.3f} | SELL: {sell_price:.3f} | Spread: {spread:.3f} | Mid: {midpoint:.3f}")
        return {
            "buy_price": buy_price,
            "sell_price": sell_price,
            "spread": spread,
            "midpoint": midpoint,
            "has_liquidity": spread < MAX_SPREAD and buy_price > 0 and sell_price > 0
        }
    except Exception as e:
        log.error(f"Failed to fetch price: {e}")
        return None


def get_my_positions() -> list:
    try:
        r = requests.get(f"{DATA_API}/positions", params={"user": MY_PROXY_WALLET, "sizeThreshold": "0.01"}, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        log.error(f"Failed to fetch positions: {e}")
        return []


# CLAUDE ANALYSIS

def analyse_trade_with_claude(trade: dict, price_data: dict, is_whale: bool = False) -> dict:
    whale_context = f"\n⚠️ WHALE TRADE: ${trade.get('usd_value', 0):,.0f} position" if is_whale else ""

    prompt = f"""You are an expert Polymarket prediction market analyst filtering copy trades.

Analyse this trade and decide whether to copy it with $1.{whale_context}

TRADE DETAILS:
- Market: {trade.get('title', 'Unknown')}
- Outcome: {trade.get('outcome', 'Unknown')}
- Side: {trade.get('side', 'Unknown')}
- Price paid: {float(trade.get('price', 0)):.3f} ({float(trade.get('price', 0))*100:.1f}% implied probability)
- Position size: ${float(trade.get('size', 0)) * float(trade.get('price', 0)):.2f} USDC
{f"- Whale position: ${trade.get('usd_value', 0):,.0f} — high conviction signal" if is_whale else ""}

CURRENT MARKET PRICES:
- Best BUY: {price_data['buy_price']:.3f}
- Best SELL: {price_data['sell_price']:.3f}
- Spread: {price_data['spread']:.3f}
- Midpoint: {price_data['midpoint']:.3f}

REJECTION RULES:
- Spread > {MAX_SPREAD}: REJECT
- BUY price < 0.05 or > 0.95: REJECT
- Current price more than 15% above trader entry: REJECT
- Very niche or obscure market: REJECT

{"WHALE BONUS: If this is a legitimate whale trade with good liquidity, be slightly more lenient on the chasing rule (allow up to 20% movement)." if is_whale else ""}

Respond in JSON only:
{{"approve": true or false, "confidence": 1-10, "reason": "one sentence", "exit_target": 0.0}}"""

    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-haiku-4-5",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=20
        )
        if not r.ok:
            log.error(f"Claude API {r.status_code}: {r.text}")
            return {"approve": False, "confidence": 0, "reason": f"Claude error {r.status_code}", "exit_target": 0}
        raw = r.json()["content"][0]["text"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        log.error(f"Claude analysis failed: {e}")
        return {"approve": False, "confidence": 0, "reason": f"Error: {e}", "exit_target": 0}


# POSITION MONITORING

def monitor_open_positions():
    if not open_positions:
        return
    my_positions = get_my_positions()
    current_map = {p["asset"]: p for p in my_positions}
    for token_id, entry in list(open_positions.items()):
        if entry.get("dead"):
            continue
        pos = current_map.get(token_id)
        if not pos:
            log.info(f"Position {entry['title'][:30]}... resolved")
            del open_positions[token_id]
            continue
        entry_price   = entry["entry_price"]
        current_price = float(pos.get("curPrice", entry_price))
        pnl_pct       = ((current_price - entry_price) / entry_price) * 100
        if current_price >= 0.90:
            log.info(f"🎯 Price hit 0.90 target — close manually on Polymarket")
            send_telegram(
                f"🎯 <b>EXIT NOW!</b>\n{entry['title'][:50]}\n[{entry['outcome']}]\n"
                f"Price hit 0.90 target\nP&L: +{pnl_pct:.1f}%\n\n👉 Close on Polymarket manually"
            )
            entry["dead"] = True
        elif pnl_pct <= -70:
            log.info(f"💀 Down {pnl_pct:.1f}% — holding to resolution")
            entry["dead"] = True


# MAIN LOOP

def process_new_trade(trade: dict, trader_label: str, is_whale: bool = False):
    title    = trade.get("title", "Unknown market")
    outcome  = trade.get("outcome", "?")
    side     = trade.get("side", "BUY")
    price    = float(trade.get("price", 0))
    token_id = trade.get("asset", "")

    # Skip old trades
    trade_time = trade.get("timestamp", 0)
    if trade_time:
        try:
            age_minutes = (datetime.now(timezone.utc).timestamp() - float(trade_time)) / 60
            if age_minutes > MAX_TRADE_AGE_MINUTES:
                log.info(f"⏭️  Skipped — {age_minutes:.0f} mins old")
                return
        except:
            pass

    whale_tag = f" 🐋 ${trade.get('usd_value', 0):,.0f}" if is_whale else ""
    log.info(f"🔍 {trader_label}{whale_tag}: [{outcome}] '{title}' @ {price:.3f}")

    if price < MIN_ODDS or price > MAX_ODDS:
        log.info(f"⏭️  Skipped — price {price:.3f} outside [{MIN_ODDS}, {MAX_ODDS}]")
        return

    if not token_id:
        log.info("⏭️  Skipped — no token_id")
        return

    price_data = get_token_price(token_id)
    if not price_data:
        log.info("⏭️  Skipped — could not fetch price")
        return

    if not price_data["has_liquidity"]:
        log.info(f"⏭️  Skipped — spread too wide ({price_data['spread']:.3f})")
        return

    analysis = analyse_trade_with_claude(trade, price_data, is_whale)
    log.info(f"🤖 Claude: approve={analysis['approve']} | {analysis.get('confidence')}/10 | {analysis.get('reason')}")

    if not analysis["approve"]:
        log.info("❌ Rejected")
        return

    # Send Telegram alert for manual trading
    whale_header = f"🐋 <b>WHALE ALERT — ${trade.get('usd_value', 0):,.0f}</b>\n\n" if is_whale else ""
    send_telegram(
        f"{whale_header}✅ <b>COPY THIS TRADE</b>\n\n"
        f"📌 <b>Market:</b> {title[:50]}\n"
        f"🎯 <b>Bet:</b> {outcome}\n"
        f"💰 <b>Price:</b> {price_data['buy_price']:.3f}\n"
        f"📊 <b>Spread:</b> {price_data['spread']:.3f}\n"
        f"🤖 {analysis.get('reason')}\n"
        f"⭐ <b>Confidence:</b> {analysis.get('confidence')}/10\n"
        f"📍 <b>Source:</b> {trader_label}\n\n"
        f"👉 Place $1 on [{outcome}] on Polymarket now!"
    )
    log.info("📌 Manual trade alert sent")

    # Track for exit monitoring
    open_positions[token_id] = {
        "title": title,
        "outcome": outcome,
        "entry_price": price_data["buy_price"],
        "trader": trader_label,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "dead": False
    }


def run():
    traders = []
    if TRADER_1 and TRADER_1 != "disabled":
        traders.append((TRADER_1, "RN1"))
    if TRADER_2 and TRADER_2 != "disabled":
        traders.append((TRADER_2, "Trader2"))

    log.info("🚀 Polymarket Claude Bot starting...")
    log.info(f"   Monitoring: {', '.join([t[1] for t in traders])}")
    log.info(f"   Whale threshold: ${WHALE_THRESHOLD:,.0f}")
    send_telegram(
        f"🚀 <b>Polymarket Bot Started!</b>\n"
        f"Monitoring: {', '.join([t[1] for t in traders])}\n"
        f"Whale detection: ${WHALE_THRESHOLD:,.0f}+\n"
        f"Poll: every {POLL_INTERVAL}s"
    )

    while True:
        try:
            # Monitor followed traders
            for wallet, label in traders:
                trades = get_recent_trades(wallet, limit=50)
                for trade in trades:
                    tx = trade.get("transactionHash", "")
                    if tx and tx not in seen_trades:
                        seen_trades.add(tx)
                        process_new_trade(trade, label)

            # Whale detection
            whale_trades = get_whale_trades(limit=20)
            for trade in whale_trades:
                tx = trade.get("transactionHash", "")
                if tx and tx not in seen_whale_trades:
                    seen_whale_trades.add(tx)
                    # Skip if it's from a trader we already monitor
                    wallet = trade.get("proxyWallet", "")
                    if wallet.lower() not in [t[0].lower() for t in traders]:
                        process_new_trade(trade, "Whale", is_whale=True)

            monitor_open_positions()

        except KeyboardInterrupt:
            log.info("Bot stopped")
            break
        except Exception as e:
            log.error(f"Main loop error: {e}")
            send_telegram(f"⚠️ Bot error: {e}")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
