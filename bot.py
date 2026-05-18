"""
Polymarket Claude Copy Trading Bot
- Monitors 2 traders for new positions
- Sends each trade to Claude for analysis
- Places $1 copy trades if approved
- Uses /price endpoint (not /book which has a known Polymarket bug)
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

TRADER_1          = os.environ.get("TRADER_1", "0x2005d16a84ceefa912d4e380cd32e7ff827875ea")
TRADER_2          = os.environ.get("TRADER_2", "0x6e1d5040d0ac73709b0621f620d2a60b80d2d0fa")

POLL_INTERVAL     = int(os.environ.get("POLL_INTERVAL", "30"))
TRADE_SIZE_USDC   = float(os.environ.get("TRADE_SIZE_USDC", "1.0"))
MIN_ODDS          = float(os.environ.get("MIN_ODDS", "0.05"))
MAX_ODDS          = float(os.environ.get("MAX_ODDS", "0.95"))
EXIT_PROFIT_PCT   = float(os.environ.get("EXIT_PROFIT_PCT", "50"))
EXIT_LOSS_PCT     = float(os.environ.get("EXIT_LOSS_PCT", "70"))
MAX_SPREAD        = float(os.environ.get("MAX_SPREAD", "0.10"))

DATA_API  = "https://data-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"

seen_trades: set = set()
open_positions: dict = {}


# POLYMARKET DATA FETCHING

def get_recent_trades(wallet: str, limit: int = 20) -> list:
    try:
        r = requests.get(f"{DATA_API}/trades", params={"user": wallet, "limit": limit}, timeout=10)
        r.raise_for_status()
        return r.json() if isinstance(r.json(), list) else []
    except Exception as e:
        log.error(f"Failed to fetch trades for {wallet[:10]}...: {e}")
        return []


def get_token_price(token_id: str) -> Optional[dict]:
    """
    Use /price endpoint which returns live data.
    /book endpoint has a known bug returning 0.01/0.99 ghost spreads.
    Returns dict with buy_price, sell_price, spread, midpoint.
    """
    try:
        # Get buy price
        r_buy = requests.get(
            f"{CLOB_API}/price",
            params={"token_id": token_id, "side": "BUY"},
            timeout=10
        )
        r_sell = requests.get(
            f"{CLOB_API}/price",
            params={"token_id": token_id, "side": "SELL"},
            timeout=10
        )

        if not r_buy.ok or not r_sell.ok:
            log.warning(f"Price fetch failed: BUY={r_buy.status_code} SELL={r_sell.status_code}")
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
        log.error(f"Failed to fetch price for {token_id[:10]}...: {e}")
        return None


def get_spread(token_id: str) -> Optional[float]:
    """Use dedicated /spread endpoint as a cross-check."""
    try:
        r = requests.get(f"{CLOB_API}/spread", params={"token_id": token_id}, timeout=10)
        if r.ok:
            return float(r.json().get("spread", 1.0))
        return None
    except Exception as e:
        log.error(f"Spread fetch failed: {e}")
        return None


def get_my_positions() -> list:
    try:
        r = requests.get(
            f"{DATA_API}/positions",
            params={"user": MY_PROXY_WALLET, "sizeThreshold": "0.01"},
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        log.error(f"Failed to fetch my positions: {e}")
        return []


# CLAUDE ANALYSIS

def analyse_trade_with_claude(trade: dict, price_data: dict) -> dict:
    prompt = f"""You are an expert Polymarket prediction market analyst filtering copy trades.

Analyse this trade and decide whether to copy it with $1.

TRADE DETAILS:
- Market: {trade.get('title', 'Unknown')}
- Outcome: {trade.get('outcome', 'Unknown')}
- Side: {trade.get('side', 'Unknown')}
- Price paid by copied trader: {float(trade.get('price', 0)):.3f} ({float(trade.get('price', 0))*100:.1f}% implied probability)
- Trader position size: ${float(trade.get('size', 0)) * float(trade.get('price', 0)):.2f} USDC

CURRENT MARKET PRICES (live data):
- Best BUY price: {price_data['buy_price']:.3f}
- Best SELL price: {price_data['sell_price']:.3f}
- Spread: {price_data['spread']:.3f}
- Midpoint: {price_data['midpoint']:.3f}
- Adequate liquidity: {price_data['has_liquidity']}

REJECTION RULES (any one = reject):
- Spread > {MAX_SPREAD} (too wide, bad execution)
- BUY price < 0.05 or > 0.95 (near-certain, no value)
- Market is very niche, obscure, or unclear

APPROVAL CRITERIA:
- Spread under {MAX_SPREAD}
- Price between 0.05 and 0.95
- Clear sports or crypto market with reasonable volume
- Copied trader entering at a price close to current midpoint (not chasing)

Respond in this exact JSON only, no other text:
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
            return {"approve": False, "confidence": 0, "reason": f"Claude {r.status_code}", "exit_target": 0}
        raw = r.json()["content"][0]["text"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        log.error(f"Claude analysis failed: {e}")
        return {"approve": False, "confidence": 0, "reason": f"Error: {e}", "exit_target": 0}


# TRADE EXECUTION

def get_clob_client():
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds
        creds = ApiCreds(api_key=PK_API_KEY, api_secret=PK_API_SECRET, api_passphrase=PK_PASSPHRASE)
        return ClobClient(host=CLOB_API, key=PK_PRIVATE_KEY, chain_id=137, creds=creds)
    except Exception as e:
        log.error(f"Failed to init CLOB client: {e}")
        return None


def place_trade(token_id: str, side: str, price: float, size_usdc: float) -> Optional[str]:
    client = get_clob_client()
    if not client:
        return None
    try:
        from py_clob_client.clob_types import MarketOrderArgs, OrderType
        shares = round(size_usdc / price, 4)
        order_args = MarketOrderArgs(token_id=token_id, amount=shares)
        signed_order = client.create_market_order(order_args)
        resp = client.post_order(signed_order, OrderType.FOK)
        if resp and resp.get("success"):
            order_id = resp.get("orderID", "unknown")
            log.info(f"✅ Order placed: {order_id} | {side} {shares} shares @ {price}")
            return order_id
        else:
            log.warning(f"Order not filled: {resp}")
            return None
    except Exception as e:
        log.error(f"Trade execution failed: {e}")
        return None


def close_position(token_id: str, shares: float, current_price: float) -> bool:
    client = get_clob_client()
    if not client:
        return False
    try:
        from py_clob_client.clob_types import MarketOrderArgs, OrderType
        order_args = MarketOrderArgs(token_id=token_id, amount=shares, side="SELL")
        signed_order = client.create_market_order(order_args)
        resp = client.post_order(signed_order, OrderType.FOK)
        if resp and resp.get("success"):
            log.info(f"✅ Position closed @ {current_price}")
            return True
        else:
            log.warning(f"Could not close: {resp}")
            return False
    except Exception as e:
        log.error(f"Close failed: {e}")
        return False


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
            log.info(f"Position {entry['title'][:30]}... resolved — removing")
            del open_positions[token_id]
            continue
        entry_price   = entry["entry_price"]
        current_price = float(pos.get("curPrice", entry_price))
        shares        = float(pos.get("size", 0))
        pnl_pct       = ((current_price - entry_price) / entry_price) * 100
        log.info(f"📈 {entry['title'][:30]}... | Entry: {entry_price:.3f} | Now: {current_price:.3f} | P&L: {pnl_pct:+.1f}%")
        if pnl_pct >= EXIT_PROFIT_PCT:
            log.info(f"🎯 Profit target hit ({pnl_pct:.1f}%) — closing")
            if close_position(token_id, shares, current_price):
                del open_positions[token_id]
        elif pnl_pct <= -EXIT_LOSS_PCT:
            log.info(f"💀 Down {pnl_pct:.1f}% — holding to resolution")
            entry["dead"] = True


# MAIN LOOP

def process_new_trade(trade: dict, trader_label: str):
    title    = trade.get("title", "Unknown market")
    outcome  = trade.get("outcome", "?")
    side     = trade.get("side", "BUY")
    price    = float(trade.get("price", 0))
    token_id = trade.get("asset", "")

    # Skip old trades
    trade_time = trade.get("timestamp", "") or trade.get("createdAt", "")
    if trade_time:
        try:
            from datetime import datetime, timezone, timedelta
            t = datetime.fromisoformat(trade_time.replace("Z", "+00:00"))
            age_minutes = (datetime.now(timezone.utc) - t).total_seconds() / 60
            if age_minutes > float(os.environ.get("MAX_TRADE_AGE_MINUTES", "30")):
                log.info(f"⏭️  Skipped — trade is {age_minutes:.0f} mins old")
                return
        except:
            pass
            
    log.info(f"🔍 {trader_label}: [{outcome}] '{title}' @ {price:.3f}")

    if price < MIN_ODDS or price > MAX_ODDS:
        log.info(f"⏭️  Skipped — price {price:.3f} outside [{MIN_ODDS}, {MAX_ODDS}]")
        return

    if not token_id:
        log.info("⏭️  Skipped — no token_id")
        return

    # Get live price data (avoids /book ghost spread bug)
    price_data = get_token_price(token_id)
    if not price_data:
        log.info("⏭️  Skipped — could not fetch live price")
        return

    if not price_data["has_liquidity"]:
        log.info(f"⏭️  Skipped — spread too wide ({price_data['spread']:.3f} > {MAX_SPREAD})")
        return

    analysis = analyse_trade_with_claude(trade, price_data)
    log.info(f"🤖 Claude: approve={analysis['approve']} | {analysis.get('confidence')}/10 | {analysis.get('reason')}")

    if not analysis["approve"]:
        log.info("❌ Rejected by Claude")
        return

    log.info(f"✅ Approved — placing ${TRADE_SIZE_USDC} trade")
    entry_price = price_data["buy_price"] if side == "BUY" else price_data["sell_price"]
    order_id = place_trade(token_id, side, entry_price, TRADE_SIZE_USDC)

    if order_id:
        open_positions[token_id] = {
            "title": title,
            "outcome": outcome,
            "entry_price": entry_price,
            "exit_target": analysis.get("exit_target", 0),
            "order_id": order_id,
            "trader": trader_label,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "dead": False
        }
        log.info(f"📌 Tracked | Target: {analysis.get('exit_target', 0):.3f}")


def run():
    traders = [
        (TRADER_1, "Trader1"),
        (TRADER_2, "Trader2"),
    ]
    log.info("🚀 Polymarket Claude Bot starting...")
    log.info(f"   Traders: {TRADER_1[:10]}... & {TRADER_2[:10]}...")
    log.info(f"   Size: ${TRADE_SIZE_USDC} | Poll: {POLL_INTERVAL}s | Max spread: {MAX_SPREAD}")

    while True:
        try:
            for wallet, label in traders:
                trades = get_recent_trades(wallet, limit=50)
                for trade in trades:
                    tx = trade.get("transactionHash", "")
                    if tx and tx not in seen_trades:
                        seen_trades.add(tx)
                        process_new_trade(trade, label)
            monitor_open_positions()
        except KeyboardInterrupt:
            log.info("Bot stopped")
            break
        except Exception as e:
            log.error(f"Main loop error: {e}")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
