"""
Polymarket Claude Copy Trading Bot
- Monitors 2 traders for new positions
- Sends each trade to Claude for analysis
- Places $1 copy trades if approved
- Monitors and exits positions based on rules
"""

import os
import time
import json
import logging
import requests
from datetime import datetime, timezone
from typing import Optional

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ── Config from environment variables ──────────────────────────────────────────
ANTHROPIC_API_KEY   = os.environ["ANTHROPIC_API_KEY"]
PK_PRIVATE_KEY      = os.environ["PK_PRIVATE_KEY"]        # Polymarket trading private key
PK_API_KEY          = os.environ["PK_API_KEY"]            # Polymarket API key
PK_API_SECRET       = os.environ["PK_API_SECRET"]         # Polymarket API secret
PK_PASSPHRASE       = os.environ["PK_PASSPHRASE"]         # Polymarket API passphrase
MY_PROXY_WALLET     = os.environ["MY_PROXY_WALLET"]       # Your 0xD98E... address

TRADER_1            = os.environ.get("TRADER_1", "0x2005d16a84ceefa912d4e380cd32e7ff827875ea")
TRADER_2            = os.environ.get("TRADER_2", "0x6e1d5040d0ac73709b0621f620d2a60b80d2d0fa")

POLL_INTERVAL       = int(os.environ.get("POLL_INTERVAL", "30"))   # seconds
TRADE_SIZE_USDC     = float(os.environ.get("TRADE_SIZE_USDC", "1.0"))  # $ per trade
MIN_ODDS            = float(os.environ.get("MIN_ODDS", "0.05"))    # skip if price < 5¢
MAX_ODDS            = float(os.environ.get("MAX_ODDS", "0.95"))    # skip if price > 95¢
EXIT_PROFIT_PCT     = float(os.environ.get("EXIT_PROFIT_PCT", "50"))   # exit at +50%
EXIT_LOSS_PCT       = float(os.environ.get("EXIT_LOSS_PCT", "70"))     # hold at -70% (dead)

# ── API base URLs ──────────────────────────────────────────────────────────────
GAMMA_API   = "https://gamma-api.polymarket.com"
DATA_API    = "https://data-api.polymarket.com"
CLOB_API    = "https://clob.polymarket.com"

# ── State ──────────────────────────────────────────────────────────────────────
seen_trades: set = set()          # track already-processed trade hashes
open_positions: dict = {}         # token_id -> entry info


# ══════════════════════════════════════════════════════════════════════════════
# POLYMARKET DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def get_recent_trades(wallet: str, limit: int = 20) -> list:
    """Fetch recent trades for a wallet from Polymarket Data API."""
    try:
        r = requests.get(
            f"{DATA_API}/trades",
            params={"user": wallet, "limit": limit},
            timeout=10
        )
        r.raise_for_status()
        return r.json() if isinstance(r.json(), list) else []
    except Exception as e:
        log.error(f"Failed to fetch trades for {wallet[:10]}...: {e}")
        return []


def get_market_orderbook(token_id: str) -> Optional[dict]:
    """Fetch order book for a market token to assess liquidity."""
    try:
        r = requests.get(
            f"{CLOB_API}/book",
            params={"token_id": token_id},
            timeout=10
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"Failed to fetch orderbook for {token_id[:10]}...: {e}")
        return None


def get_my_positions() -> list:
    """Fetch current open positions for our wallet."""
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


def check_liquidity(token_id: str, side: str) -> dict:
    """
    Check order book liquidity.
    Returns dict with: has_liquidity, best_price, spread, depth_usdc
    """
    book = get_market_orderbook(token_id)
    if not book:
        return {"has_liquidity": False, "best_price": 0, "spread": 1, "depth_usdc": 0}

    bids = book.get("bids", [])
    asks = book.get("asks", [])

    if not bids or not asks:
        return {"has_liquidity": False, "best_price": 0, "spread": 1, "depth_usdc": 0}

    best_bid = float(bids[0]["price"]) if bids else 0
    best_ask = float(asks[0]["price"]) if asks else 1
    spread = best_ask - best_bid

    # Depth = sum of top 5 levels on the relevant side
    relevant = asks if side == "BUY" else bids
    depth = sum(float(l["size"]) * float(l["price"]) for l in relevant[:5])

    best_price = best_ask if side == "BUY" else best_bid

    return {
        "has_liquidity": spread < 0.15 and depth > 5,
        "best_price": best_price,
        "spread": spread,
        "depth_usdc": depth
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLAUDE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyse_trade_with_claude(trade: dict, liquidity: dict) -> dict:
    """
    Send trade details to Claude for analysis.
    Returns: { approve: bool, confidence: int, reason: str, exit_target: float }
    """
    prompt = f"""You are an expert Polymarket prediction market analyst helping filter copy trades.

Analyse this trade and decide whether to copy it with $1.

TRADE DETAILS:
- Market: {trade.get('title', 'Unknown')}
- Outcome: {trade.get('outcome', 'Unknown')}
- Side: {trade.get('side', 'Unknown')}
- Price paid by copied trader: {trade.get('price', 0):.3f} (implies {trade.get('price', 0)*100:.1f}% probability)
- Trader's position size: ${float(trade.get('size', 0)) * float(trade.get('price', 0)):.2f} USDC

MARKET LIQUIDITY:
- Has adequate liquidity: {liquidity['has_liquidity']}
- Best available price: {liquidity['best_price']:.3f}
- Bid-ask spread: {liquidity['spread']:.3f}
- Order book depth: ${liquidity['depth_usdc']:.2f} USDC

ASSESSMENT CRITERIA:
1. Price sanity — is the price between 5¢ and 95¢? (avoid near-certain outcomes)
2. Liquidity — is there enough depth to enter AND exit later?
3. Spread — is it under 15¢? Wide spreads mean poor execution
4. Value — does the price seem reasonable for this type of event?
5. Market type — sports and crypto price markets preferred over obscure political markets

RULES:
- If price < 0.05 or > 0.95: REJECT (no value, can't exit)
- If spread > 0.15: REJECT (too illiquid)
- If depth < $5: REJECT (can't exit position)
- If market title is unclear or very niche: REJECT

Respond in this exact JSON format only, no other text:
{{
  "approve": true or false,
  "confidence": 1-10,
  "reason": "one sentence explanation",
  "exit_target": 0.0 to 1.0 (suggested price to take profit, or 0 if rejecting)
}}"""

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
        r.raise_for_status()
        raw = r.json()["content"][0]["text"].strip()

        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        return result

    except Exception as e:
        log.error(f"Claude analysis failed: {e}")
        return {"approve": False, "confidence": 0, "reason": f"Claude error: {e}", "exit_target": 0}


# ══════════════════════════════════════════════════════════════════════════════
# TRADE EXECUTION (via py-clob-client)
# ══════════════════════════════════════════════════════════════════════════════

def get_clob_client():
    """Initialise and return authenticated CLOB client."""
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds

        creds = ApiCreds(
            api_key=PK_API_KEY,
            api_secret=PK_API_SECRET,
            api_passphrase=PK_PASSPHRASE
        )
        client = ClobClient(
            host=CLOB_API,
            key=PK_PRIVATE_KEY,
            chain_id=137,  # Polygon
            creds=creds
        )
        return client
    except Exception as e:
        log.error(f"Failed to init CLOB client: {e}")
        return None


def place_trade(token_id: str, side: str, price: float, size_usdc: float) -> Optional[str]:
    """
    Place a market order on Polymarket.
    size_usdc = dollar amount to spend
    Returns order ID if successful, None if failed.
    """
    client = get_clob_client()
    if not client:
        return None

    try:
        from py_clob_client.clob_types import MarketOrderArgs, OrderType

        # Calculate shares from USDC amount
        shares = round(size_usdc / price, 4)

        order_args = MarketOrderArgs(
            token_id=token_id,
            amount=shares,
        )

        signed_order = client.create_market_order(order_args)
        resp = client.post_order(signed_order, OrderType.FOK)  # Fill or Kill

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
    """Attempt to sell/close a position."""
    client = get_clob_client()
    if not client:
        return False

    try:
        from py_clob_client.clob_types import MarketOrderArgs, OrderType

        # To close a YES position, sell it back
        order_args = MarketOrderArgs(
            token_id=token_id,
            amount=shares,
            side="SELL"
        )

        signed_order = client.create_market_order(order_args)
        resp = client.post_order(signed_order, OrderType.FOK)

        if resp and resp.get("success"):
            log.info(f"✅ Position closed: {token_id[:10]}... @ {current_price}")
            return True
        else:
            log.warning(f"Could not close position: {resp}")
            return False

    except Exception as e:
        log.error(f"Close position failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# POSITION MONITORING & EXIT LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def monitor_open_positions():
    """Check all open positions and exit if profit/loss targets hit."""
    if not open_positions:
        return

    my_positions = get_my_positions()
    current_map = {p["asset"]: p for p in my_positions}

    for token_id, entry in list(open_positions.items()):
        pos = current_map.get(token_id)
        if not pos:
            # Position no longer exists — resolved or already closed
            log.info(f"Position {token_id[:10]}... no longer open — removing")
            del open_positions[token_id]
            continue

        entry_price   = entry["entry_price"]
        current_price = float(pos.get("curPrice", entry_price))
        shares        = float(pos.get("size", 0))
        pnl_pct       = ((current_price - entry_price) / entry_price) * 100

        log.info(f"Position {entry['title'][:30]}... | Entry: {entry_price:.3f} | Now: {current_price:.3f} | P&L: {pnl_pct:+.1f}%")

        # Exit rules
        if pnl_pct >= EXIT_PROFIT_PCT:
            log.info(f"🎯 Profit target hit ({pnl_pct:.1f}%) — attempting to close")
            if close_position(token_id, shares, current_price):
                del open_positions[token_id]

        elif pnl_pct <= -EXIT_LOSS_PCT:
            log.info(f"💀 Position down {pnl_pct:.1f}% — likely dead, holding to resolution")
            # Don't try to close — no liquidity at this price
            # Mark as dead so we stop logging it noisily
            entry["dead"] = True


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def process_new_trade(trade: dict, trader_label: str):
    """Full pipeline: detect → analyse → execute."""
    tx_hash = trade.get("transactionHash", "")
    title   = trade.get("title", "Unknown market")
    outcome = trade.get("outcome", "?")
    side    = trade.get("side", "BUY")
    price   = float(trade.get("price", 0))
    token_id = trade.get("asset", "")

    log.info(f"🔍 New trade from {trader_label}: [{outcome}] on '{title}' @ {price:.3f}")

    # Basic sanity checks before calling Claude
    if price < MIN_ODDS or price > MAX_ODDS:
        log.info(f"⏭️  Skipped — price {price:.3f} outside range [{MIN_ODDS}, {MAX_ODDS}]")
        return

    if not token_id:
        log.info("⏭️  Skipped — no token_id")
        return

    # Check liquidity
    liquidity = check_liquidity(token_id, side)
    log.info(f"📊 Liquidity — spread: {liquidity['spread']:.3f} | depth: ${liquidity['depth_usdc']:.2f}")

    # Claude analysis
    analysis = analyse_trade_with_claude(trade, liquidity)
    log.info(f"🤖 Claude: approve={analysis['approve']} | confidence={analysis.get('confidence')}/10 | {analysis.get('reason')}")

    if not analysis["approve"]:
        log.info(f"❌ Trade rejected by Claude")
        return

    # Place $1 trade
    log.info(f"✅ Approved — placing ${TRADE_SIZE_USDC} trade")
    order_id = place_trade(token_id, side, liquidity["best_price"], TRADE_SIZE_USDC)

    if order_id:
        open_positions[token_id] = {
            "title": title,
            "outcome": outcome,
            "entry_price": liquidity["best_price"],
            "exit_target": analysis.get("exit_target", 0),
            "order_id": order_id,
            "trader": trader_label,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "dead": False
        }
        log.info(f"📌 Position tracked | Target exit: {analysis.get('exit_target', 0):.3f}")


def run():
    """Main polling loop."""
    traders = [
        (TRADER_1, "Trader1-Crypto"),
        (TRADER_2, "Trader2-Sports"),
    ]

    log.info("🚀 Polymarket Claude Bot starting...")
    log.info(f"   Monitoring: {TRADER_1[:10]}... & {TRADER_2[:10]}...")
    log.info(f"   Trade size: ${TRADE_SIZE_USDC} USDC")
    log.info(f"   Poll interval: {POLL_INTERVAL}s")
    log.info(f"   Exit targets: +{EXIT_PROFIT_PCT}% profit / -{EXIT_LOSS_PCT}% stop")

    while True:
        try:
            # Check each trader for new trades
            for wallet, label in traders:
                trades = get_recent_trades(wallet, limit=10)
                for trade in trades:
                    tx = trade.get("transactionHash", "")
                    if tx and tx not in seen_trades:
                        seen_trades.add(tx)
                        process_new_trade(trade, label)

            # Monitor our open positions
            monitor_open_positions()

        except KeyboardInterrupt:
            log.info("🛑 Bot stopped by user")
            break
        except Exception as e:
            log.error(f"Main loop error: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
