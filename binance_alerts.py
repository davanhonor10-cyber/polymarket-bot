"""
Binance Long Setup Alert Bot v3
- 30 minute interval
- 1H + 4H candle analysis
- Real crypto news via cryptocurrency.cv (no API key)
- Score threshold: 6.5/10
- Daily status at 11 UTC (7am Trinidad)
- Uses BINANCE_TELEGRAM_TOKEN
"""

import os
import time
import json
import logging
import requests
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
TELEGRAM_TOKEN    = os.environ.get("BINANCE_TELEGRAM_TOKEN") or os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID  = os.environ["TELEGRAM_CHAT_ID"]

COINS = ["BTCUSDT", "XRPUSDT", "SOLUSDT"]
MIN_SCORE       = float(os.environ.get("MIN_SCORE", "6.5"))
CHECK_INTERVAL  = int(os.environ.get("CHECK_INTERVAL", "1800"))
STATUS_HOUR_UTC = int(os.environ.get("STATUS_HOUR_UTC", "11"))

BINANCE_FUTURES = "https://fapi.binance.com"

alerts_sent_today = 0
last_status_date = None
analysis_count_today = 0


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


# NEWS

def get_news(coin: str) -> str:
    """Fetch recent news headlines from cryptocurrency.cv — no API key needed."""
    coin_slug = {"BTCUSDT": "bitcoin", "XRPUSDT": "xrp", "SOLUSDT": "solana"}
    slug = coin_slug.get(coin, "bitcoin")
    try:
        r = requests.get(
            f"https://cryptocurrency.cv/api/v1/news",
            params={"coin": slug, "limit": 3},
            timeout=8
        )
        if r.ok:
            articles = r.json().get("data", [])
            headlines = [a.get("title", "") for a in articles[:3] if a.get("title")]
            if headlines:
                return " | ".join(headlines)
    except:
        pass

    # Fallback: try RSS-style endpoint
    try:
        r = requests.get(
            f"https://cryptocurrency.cv/api/v1/news/latest",
            params={"q": slug, "limit": 3},
            timeout=8
        )
        if r.ok:
            data = r.json()
            if isinstance(data, list):
                headlines = [a.get("title", "") for a in data[:3] if a.get("title")]
                if headlines:
                    return " | ".join(headlines)
    except:
        pass

    return "No recent news available"


# BINANCE DATA

def get_klines(symbol: str, interval: str = "1h", limit: int = 60) -> list:
    try:
        r = requests.get(
            f"{BINANCE_FUTURES}/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"Failed to fetch {interval} klines for {symbol}: {e}")
        return []


def get_current_price(symbol: str) -> float:
    try:
        r = requests.get(
            f"{BINANCE_FUTURES}/fapi/v1/premiumIndex",
            params={"symbol": symbol},
            timeout=10
        )
        r.raise_for_status()
        return float(r.json()["markPrice"])
    except Exception as e:
        log.error(f"Failed to fetch price for {symbol}: {e}")
        return 0.0


def get_funding_rate(symbol: str) -> float:
    """Get current funding rate — negative = longs being paid = bullish."""
    try:
        r = requests.get(
            f"{BINANCE_FUTURES}/fapi/v1/premiumIndex",
            params={"symbol": symbol},
            timeout=10
        )
        r.raise_for_status()
        return float(r.json().get("lastFundingRate", 0))
    except:
        return 0.0


def get_btc_dominance() -> float:
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        r.raise_for_status()
        return r.json()["data"]["market_cap_percentage"]["btc"]
    except Exception as e:
        log.error(f"Failed to fetch BTC dominance: {e}")
        return 0.0


def calculate_indicators(klines: list) -> dict:
    if len(klines) < 20:
        return {}

    closes  = [float(k[4]) for k in klines]
    highs   = [float(k[2]) for k in klines]
    lows    = [float(k[3]) for k in klines]
    volumes = [float(k[5]) for k in klines]

    current_close = closes[-1]

    # EMA 20
    ema20 = closes[-20]
    k20 = 2 / 21
    for c in closes[-19:]:
        ema20 = c * k20 + ema20 * (1 - k20)

    # EMA 50
    ema50 = closes[0]
    k50 = 2 / 51
    for c in closes[1:]:
        ema50 = c * k50 + ema50 * (1 - k50)

    # RSI 14
    gains, losses = [], []
    for i in range(-14, 0):
        diff = closes[i] - closes[i-1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains) / 14
    avg_loss = sum(losses) / 14
    rs = avg_gain / avg_loss if avg_loss > 0 else 100
    rsi = 100 - (100 / (1 + rs))

    avg_volume     = sum(volumes[-20:-1]) / 19
    current_volume = volumes[-1]
    volume_ratio   = current_volume / avg_volume if avg_volume > 0 else 1

    recent_support    = min(lows[-21:-1])
    recent_resistance = max(highs[-21:-1])
    above_ema20       = current_close > ema20
    above_ema50       = current_close > ema50
    change_pct        = ((current_close - closes[-2]) / closes[-2] * 100) if len(closes) >= 2 else 0
    change_24h        = ((current_close - closes[-25]) / closes[-25] * 100) if len(closes) >= 25 else 0

    return {
        "current_price": current_close,
        "ema20": round(ema20, 6),
        "ema50": round(ema50, 6),
        "rsi": round(rsi, 1),
        "volume_ratio": round(volume_ratio, 2),
        "recent_support": round(recent_support, 6),
        "recent_resistance": round(recent_resistance, 6),
        "above_ema20": above_ema20,
        "above_ema50": above_ema50,
        "change_pct": round(change_pct, 2),
        "change_24h": round(change_24h, 2),
    }


# CLAUDE ANALYSIS

def analyse_coin(symbol: str, ind_1h: dict, ind_4h: dict, btc_dominance: float,
                 funding_rate: float, news: str, btc_1h: dict, btc_4h: dict) -> dict:

    coin_name = {"BTCUSDT": "Bitcoin (BTC)", "XRPUSDT": "XRP", "SOLUSDT": "Solana (SOL)"}
    name = coin_name.get(symbol, symbol)
    is_btc = symbol == "BTCUSDT"

    btc_context = ""
    if not is_btc:
        btc_context = f"""
BTC CONTEXT:
- BTC Price: ${btc_1h.get('current_price', 0):,.2f}
- BTC 1H RSI: {btc_1h.get('rsi', 0)} | 4H RSI: {btc_4h.get('rsi', 0)}
- BTC above 1H EMA20: {btc_1h.get('above_ema20', False)} | 4H EMA20: {btc_4h.get('above_ema20', False)}
- BTC 24h change: {btc_1h.get('change_24h', 0)}%
"""

    funding_note = ""
    if funding_rate < -0.0001:
        funding_note = f"⚠️ BULLISH: Negative funding rate ({funding_rate:.4f}) — longs being paid"
    elif funding_rate > 0.0005:
        funding_note = f"⚠️ BEARISH: High positive funding ({funding_rate:.4f}) — market overleveraged long"
    else:
        funding_note = f"Neutral funding rate: {funding_rate:.4f}"

    prompt = f"""You are an expert crypto trader analyzing {name} for a LONG entry on Binance perpetual futures with 20x leverage.

COIN: {name}

1H TIMEFRAME:
- Price: ${ind_1h.get('current_price', 0):,.6f}
- EMA20: ${ind_1h.get('ema20', 0):,.6f} ({'ABOVE ✅' if ind_1h.get('above_ema20') else 'BELOW ❌'})
- EMA50: ${ind_1h.get('ema50', 0):,.6f} ({'ABOVE ✅' if ind_1h.get('above_ema50') else 'BELOW ❌'})
- RSI(14): {ind_1h.get('rsi', 0)}
- Volume: {ind_1h.get('volume_ratio', 0)}x avg
- Support: ${ind_1h.get('recent_support', 0):,.6f}
- Resistance: ${ind_1h.get('recent_resistance', 0):,.6f}
- 1H change: {ind_1h.get('change_pct', 0)}% | 24H: {ind_1h.get('change_24h', 0)}%

4H TIMEFRAME:
- EMA20: ${ind_4h.get('ema20', 0):,.6f} ({'ABOVE ✅' if ind_4h.get('above_ema20') else 'BELOW ❌'})
- EMA50: ${ind_4h.get('ema50', 0):,.6f} ({'ABOVE ✅' if ind_4h.get('above_ema50') else 'BELOW ❌'})
- RSI(14): {ind_4h.get('rsi', 0)}
- Support: ${ind_4h.get('recent_support', 0):,.6f}
- 4H change: {ind_4h.get('change_24h', 0)}%
{btc_context}
MARKET CONDITIONS:
- BTC Dominance: {btc_dominance:.1f}%
- Funding Rate: {funding_note}

RECENT NEWS:
{news}

YOUR 6-FACTOR SCORING FRAMEWORK:
1. Technical Analysis (1H + 4H combined) — EMA alignment, RSI, support/resistance
2. News & Catalysts — positive/negative news impact
3. Market Sentiment — fear/greed, dominance, overall conditions  
4. Whale/Volume Activity — volume spikes, unusual activity
5. Bitcoin Position — BTC trend and dominance (for alts: is BTC supportive?)
6. Timeframe Alignment — do 1H and 4H agree on direction?

TRADING RULES:
- LONGS ONLY
- Ideal: price near support on BOTH 1H and 4H
- RSI below 65 preferred (not overbought)
- For alts: BTC must be stable or bullish on 4H
- 4H alignment adds significant conviction
- Minimum 1:2 risk/reward required
- Negative funding = extra bullish bonus
- Score 6.5+ = send alert

Respond in JSON only, no other text:
{{
  "score": 0.0-10.0,
  "factor_scores": {{
    "technical": 0-10,
    "news": 0-10,
    "sentiment": 0-10,
    "whale": 0-10,
    "btc_position": 0-10,
    "timeframe": 0-10
  }},
  "summary": "2-3 sentences covering key reasons",
  "entry_zone": "specific price range",
  "invalidation": "specific price level",
  "target": "specific price target",
  "risk_reward": "e.g. 1:2.5",
  "timeframe_alignment": "1H and 4H both bullish / mixed / conflicting"
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
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        if not r.ok:
            log.error(f"Claude API error: {r.status_code}")
            return {}
        raw = r.json()["content"][0]["text"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        # Extract just the JSON object
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            log.error(f"No JSON found: {raw[:100]}")
            return {}
        return json.loads(raw[start:end])
    except Exception as e:
        log.error(f"Claude analysis failed for {symbol}: {e}")
        return {}


def format_alert(symbol: str, ind_1h: dict, ind_4h: dict, analysis: dict, funding_rate: float) -> str:
    coin_name = {"BTCUSDT": "BTC/USDT", "XRPUSDT": "XRP/USDT", "SOLUSDT": "SOL/USDT"}
    name = coin_name.get(symbol, symbol)
    score = analysis.get("score", 0)
    factors = analysis.get("factor_scores", {})
    funding_emoji = "🟢" if funding_rate < 0 else "🔴" if funding_rate > 0.0005 else "⚪"

    return (
        f"🟢 <b>{name} LONG SETUP — {score}/10</b>\n\n"
        f"💰 <b>Price:</b> ${ind_1h.get('current_price', 0):,.6f}\n"
        f"📊 <b>1H RSI:</b> {ind_1h.get('rsi', 0)} | <b>4H RSI:</b> {ind_4h.get('rsi', 0)}\n"
        f"📈 <b>1H:</b> EMA20 {'✅' if ind_1h.get('above_ema20') else '❌'} EMA50 {'✅' if ind_1h.get('above_ema50') else '❌'}\n"
        f"📈 <b>4H:</b> EMA20 {'✅' if ind_4h.get('above_ema20') else '❌'} EMA50 {'✅' if ind_4h.get('above_ema50') else '❌'}\n"
        f"{funding_emoji} <b>Funding:</b> {funding_rate:.4f}\n\n"
        f"<b>Scores:</b> Tech:{factors.get('technical',0)} News:{factors.get('news',0)} "
        f"Sent:{factors.get('sentiment',0)} Whale:{factors.get('whale',0)} "
        f"BTC:{factors.get('btc_position',0)} TF:{factors.get('timeframe',0)}\n\n"
        f"🔗 <b>TF Alignment:</b> {analysis.get('timeframe_alignment', 'N/A')}\n\n"
        f"📝 {analysis.get('summary', '')}\n\n"
        f"✅ <b>Entry:</b> {analysis.get('entry_zone', 'N/A')}\n"
        f"🛑 <b>Invalidation:</b> {analysis.get('invalidation', 'N/A')}\n"
        f"🎯 <b>Target:</b> {analysis.get('target', 'N/A')}\n"
        f"⚖️ <b>R:R:</b> {analysis.get('risk_reward', 'N/A')}\n\n"
        f"⚠️ <i>Run full checklist. 20x leverage — manage risk.</i>"
    )


def send_status_update(prices: dict, btc_dominance: float):
    now_utc = datetime.now(timezone.utc)
    send_telegram(
        f"📊 <b>Daily Status Update</b>\n"
        f"🕖 7:00 AM Trinidad | {now_utc.strftime('%Y-%m-%d')}\n\n"
        f"✅ <b>Bot:</b> Running\n"
        f"⏱ <b>Interval:</b> Every 30 mins\n"
        f"📈 <b>Analyses today:</b> {analysis_count_today}\n"
        f"🔔 <b>Alerts sent:</b> {alerts_sent_today}\n"
        f"🎯 <b>Min score:</b> {MIN_SCORE}/10\n\n"
        f"<b>Prices:</b>\n"
        f"  BTC: ${prices.get('BTCUSDT', 0):,.2f}\n"
        f"  XRP: ${prices.get('XRPUSDT', 0):,.4f}\n"
        f"  SOL: ${prices.get('SOLUSDT', 0):,.2f}\n\n"
        f"🌐 <b>BTC Dominance:</b> {btc_dominance:.1f}%"
    )


def run_analysis():
    global alerts_sent_today, analysis_count_today

    now = datetime.now(timezone.utc)
    analysis_count_today += 1
    log.info(f"🔍 Analysis #{analysis_count_today} at {now.strftime('%H:%M UTC')}")

    btc_dominance = get_btc_dominance()
    log.info(f"BTC Dominance: {btc_dominance:.1f}%")

    # BTC data for context
    btc_1h_klines = get_klines("BTCUSDT", "1h", 60)
    btc_4h_klines = get_klines("BTCUSDT", "4h", 60)
    btc_1h = calculate_indicators(btc_1h_klines) if btc_1h_klines else {}
    btc_4h = calculate_indicators(btc_4h_klines) if btc_4h_klines else {}

    prices = {}
    for symbol in COINS:
        prices[symbol] = get_current_price(symbol)

    for symbol in COINS:
        log.info(f"Analyzing {symbol}...")

        klines_1h = get_klines(symbol, "1h", 60)
        klines_4h = get_klines(symbol, "4h", 60)

        if not klines_1h:
            log.warning(f"No 1H data for {symbol}")
            continue

        ind_1h = calculate_indicators(klines_1h)
        ind_4h = calculate_indicators(klines_4h) if klines_4h else {}

        if not ind_1h:
            continue

        funding_rate = get_funding_rate(symbol)
        news = get_news(symbol)

        log.info(f"{symbol} — Price: ${ind_1h.get('current_price', 0):,.4f} | 1H RSI: {ind_1h.get('rsi', 0)} | 4H RSI: {ind_4h.get('rsi', 0) if ind_4h else 'N/A'} | Funding: {funding_rate:.4f}")

        b1h = btc_1h if symbol != "BTCUSDT" else {}
        b4h = btc_4h if symbol != "BTCUSDT" else {}

        analysis = analyse_coin(symbol, ind_1h, ind_4h, btc_dominance, funding_rate, news, b1h, b4h)

        if not analysis:
            continue

        score = analysis.get("score", 0)
        log.info(f"{symbol}: {score}/10")

        if score >= MIN_SCORE:
            log.info(f"✅ {symbol} alert!")
            send_telegram(format_alert(symbol, ind_1h, ind_4h, analysis, funding_rate))
            alerts_sent_today += 1
        else:
            log.info(f"❌ {symbol} score {score} below {MIN_SCORE}")

        time.sleep(2)

    return prices, btc_dominance


def run():
    global last_status_date, alerts_sent_today, analysis_count_today

    log.info("🚀 Binance Alert Bot v3 starting...")
    send_telegram(
        f"🚀 <b>Binance Alert Bot v3!</b>\n"
        f"✅ 1H + 4H analysis\n"
        f"✅ Live crypto news\n"
        f"✅ Funding rate signal\n"
        f"✅ Min score: {MIN_SCORE}/10\n"
        f"⏱ Every 30 mins"
    )

    last_prices = {}
    last_dominance = 0.0

    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            today = now_utc.date()

            if last_status_date != today:
                alerts_sent_today = 0
                analysis_count_today = 0

            if (now_utc.hour == STATUS_HOUR_UTC and
                now_utc.minute < 31 and
                last_status_date != today):
                send_status_update(last_prices, last_dominance)
                last_status_date = today

            last_prices, last_dominance = run_analysis()

        except Exception as e:
            log.error(f"Analysis error: {e}")
            send_telegram(f"⚠️ Binance bot error: {e}")

        log.info(f"Sleeping {CHECK_INTERVAL}s...")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    run()
