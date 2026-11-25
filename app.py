
import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse

# ======================= Configuration =======================

SYMBOL_A = "SKHYNIXUSDT"
SYMBOL_B = "VRTUSDT"

TIMEFRAMES = ["5m", "15m", "30m", "1h", "2h", "4h", "1d"]

BETA_WINDOWS = {
    "5m": 504,
    "15m": 168,
    "30m": 84,
    "1h": 42,
    "2h": 21,
    "4h": 20,
    "1d": 60,
}

# SPREAD_WINDOWS = {
#     "5m": 2016,
#     "15m": 672,
#     "30m": 336,
#     "1h": 168,
#     "2h": 84,
#     "4h": 42,
#     "1d": 30,
# }
BETA_HISTORY_POINTS = 200

# 每个周期对应 7 天的 K 线数量
SPREAD_WINDOWS = {
    "5m": 2016,
    "15m": 672,
    "30m": 336,
    "1h": 168,
    "2h": 84,
    "4h": 42,
    "1d": 7
}

# 每 30 根新 K 线更新一次模型
MODEL_REFRESH_BARS = 30

# 每隔 30 秒检查一次是否需要更新
REFRESH_MODEL_SECONDS = 30
POLL_SECONDS = 1.0

HOST = "127.0.0.1"
PORT = 8000

REST_URL = "https://fapi.binance.com/fapi/v1/klines"
BOOK_URL = "https://fapi.binance.com/fapi/v1/ticker/bookTicker"

PROXY = "http://127.0.0.1:7897"
PROXIES = {"http": PROXY, "https": PROXY}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("pair-monitor")


# ======================= Global State =======================

def initial_state(symbol_a, symbol_b):
    return {
        "symbols": [symbol_a, symbol_b],
        "prices": {symbol_a: None, symbol_b: None},
        "price_times": {symbol_a: None, symbol_b: None},
        "quotes": {symbol_a: None, symbol_b: None},
        "active_period": "5m",
        "models": {},
        "beta": None,
        "beta_change": None,
        "beta_std": None,
        "intercept": None,
        "spread_mean": None,
        "spread_std": None,
        "spread": None,
        "zscore": None,
        "model_time": None,
        "last_tick_time": None,
        "status": "Starting",
        "updates": 0
    }


state = initial_state(SYMBOL_A, SYMBOL_B)

clients = set()
background_tasks = []

model_locks = {
    tf: asyncio.Lock() for tf in TIMEFRAMES
}

pair_lock = asyncio.Lock()


# ======================= Market Data =======================

def get_window_config(interval):
    return BETA_WINDOWS[interval], SPREAD_WINDOWS[interval]


def get_interval_seconds(interval):
    units = {"m": 60, "h": 3600, "d": 86400}
    return int(interval[:-1]) * units[interval[-1]]


def fetch_klines(symbol, interval):
    beta_window, spread_window = get_window_config(interval)
    target = beta_window + spread_window + 5
    end_time = None
    all_rows = []

    while len(all_rows) < target:
        params = {"symbol": symbol, "interval": interval, "limit": 1500}
        if end_time is not None:
            params["endTime"] = end_time

        response = requests.get(
            REST_URL,
            params=params,
            proxies=PROXIES,
            timeout=20
        )
        response.raise_for_status()
        rows = response.json()

        if not rows:
            break

        all_rows = rows + all_rows
        oldest_time = int(rows[0][0])
        end_time = oldest_time - 1

        if len(rows) < 1500:
            break

    if not all_rows:
        raise RuntimeError(f"No klines returned: {symbol} {interval}")

    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trade_count",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]

    df = pd.DataFrame(all_rows, columns=columns)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    now = pd.Timestamp.now(tz="UTC")
    df = df.loc[
        df["close_time"] < now,
        ["open_time", "close"]
    ]

    df = (
        df.dropna()
        .drop_duplicates("open_time")
        .set_index("open_time")
        .sort_index()
    )

    if len(df) < target:
        raise RuntimeError(
            f"{symbol} {interval}: only {len(df)} closed bars; need {target}"
        )

    return df.tail(target).rename(columns={"close": symbol})

def fetch_book_ticker(symbol):
    response = requests.get(
        BOOK_URL,
        params={"symbol": symbol},
        proxies=PROXIES,
        timeout=10
    )

    response.raise_for_status()
    data = response.json()

    bid = float(data["bidPrice"])
    ask = float(data["askPrice"])

    if bid <= 0 or ask <= 0 or ask < bid:
        raise RuntimeError(f"Invalid book ticker: {symbol}")

    return {
        "bid": bid,
        "ask": ask,
        "mid": (bid + ask) / 2,
        "bid_qty": float(data["bidQty"]),
        "ask_qty": float(data["askQty"])
    }


# ======================= Model =======================
def fetch_latest_closed_kline_time(symbol, interval):
    response = requests.get(
        REST_URL,
        params={"symbol": symbol, "interval": interval, "limit": 2},
        proxies=PROXIES,
        timeout=15
    )
    response.raise_for_status()

    rows = response.json()
    if not rows:
        raise RuntimeError(f"No klines returned: {symbol} {interval}")

    now_ms = int(time.time() * 1000)
    closed = [row for row in rows if int(row[6]) < now_ms]

    if not closed:
        raise RuntimeError(f"No completed candle: {symbol} {interval}")

    return pd.to_datetime(int(closed[-1][0]), unit="ms", utc=True)


def build_timeframe_model(symbol_a, symbol_b, interval):
    beta_window, spread_window = get_window_config(interval)

    a = fetch_klines(symbol_a, interval)
    b = fetch_klines(symbol_b, interval)

    joined = a.join(b, how="inner").dropna().sort_index()

    required = beta_window + spread_window + 5
    if len(joined) < required:
        raise RuntimeError(
            f"{interval}: insufficient aligned history "
            f"({len(joined)} bars; need {required})"
        )

    if (joined[[symbol_a, symbol_b]] <= 0).any().any():
        raise RuntimeError("Non-positive prices found")

    log_a = np.log(joined[symbol_a].astype(float))
    log_b = np.log(joined[symbol_b].astype(float))

    # 每根 K 线使用过去 504 根数据估计滚动 OLS
    rolling_mean_a = log_a.rolling(
        beta_window, min_periods=beta_window
    ).mean()

    rolling_mean_b = log_b.rolling(
        beta_window, min_periods=beta_window
    ).mean()

    rolling_cov = log_a.rolling(
        beta_window, min_periods=beta_window
    ).cov(log_b)

    rolling_var = log_b.rolling(
        beta_window, min_periods=beta_window
    ).var()

    rolling_beta = rolling_cov / rolling_var.replace(0, np.nan)
    rolling_intercept = rolling_mean_a - rolling_beta * rolling_mean_b

    # 每个时点使用当时的 alpha、beta 计算残差
    rolling_spread = (
        log_a - rolling_intercept - rolling_beta * log_b
    )

    valid = pd.DataFrame({
        "beta": rolling_beta,
        "intercept": rolling_intercept,
        "spread": rolling_spread
    })

    valid = (
        valid.replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(valid) < spread_window:
        raise RuntimeError(
            f"{interval}: insufficient valid residuals"
        )

    # 当前模型参数：最近一根已收盘 K 线的滚动 OLS
    beta = float(valid["beta"].iloc[-1])
    intercept = float(valid["intercept"].iloc[-1])

    # 使用最近 7 天残差计算均值和标准差
    baseline = valid["spread"].tail(spread_window)
    spread_mean = float(baseline.mean())
    spread_std = float(baseline.std(ddof=1))

    if not np.isfinite(spread_std) or spread_std < 1e-12:
        raise RuntimeError(
            f"{interval}: spread standard deviation too small"
        )

    beta_series = valid["beta"].tail(BETA_HISTORY_POINTS)

    beta_change = float(
        beta_series.iloc[-1]
        - beta_series.iloc[-min(21, len(beta_series))]
    )

    beta_std = float(
        beta_series.std(ddof=1)
    ) if len(beta_series) > 1 else 0.0

    history = []

    for idx, value in baseline.items():
        value = float(value)
        history.append({
            "time": idx.isoformat(),
            "spread": value,
            "mean": spread_mean,
            "upper1": spread_mean + spread_std,
            "lower1": spread_mean - spread_std,
            "upper2": spread_mean + 2 * spread_std,
            "lower2": spread_mean - 2 * spread_std,
            "zscore": (value - spread_mean) / spread_std
        })

    beta_history = [
        {
            "time": idx.isoformat(),
            "beta": float(value)
        }
        for idx, value in beta_series.items()
    ]

    current_spread = float(valid["spread"].iloc[-1])
    current_zscore = float(
        (current_spread - spread_mean) / spread_std
    )

    return {
        "interval": interval,
        "beta_window": beta_window,
        "spread_window": spread_window,
        "beta": beta,
        "beta_change": beta_change,
        "beta_std": beta_std,
        "intercept": intercept,
        "spread_mean": spread_mean,
        "spread_std": spread_std,
        "spread": current_spread,
        "zscore": current_zscore,
        "model_time": valid.index[-1].isoformat(),
        "history": history,
        "beta_history": beta_history,
        "live_spread": None,
        "live_zscore": None
    }

async def rebuild_timeframe(interval, symbols=None):
    async with model_locks[interval]:
        if symbols is None:
            symbols = tuple(state["symbols"])

        symbol_a, symbol_b = symbols

        model = await asyncio.to_thread(
            build_timeframe_model,
            symbol_a,
            symbol_b,
            interval
        )

        # 防止切换交易对期间旧模型覆盖新模型
        if tuple(state["symbols"]) != tuple(symbols):
            return

        state["models"][interval] = model

        log.info(
            "Model refreshed | %s | %s/%s | beta=%.6f | zscore=%.3f | bars=%s",
            interval,
            symbol_a,
            symbol_b,
            model["beta"],
            model["zscore"],
            len(model["history"])
        )


async def rebuild_all_models():
    symbols = tuple(state["symbols"])

    results = await asyncio.gather(
        *(
            rebuild_timeframe(tf, symbols)
            for tf in TIMEFRAMES
        ),
        return_exceptions=True
    )

    for tf, result in zip(TIMEFRAMES, results):
        if isinstance(result, Exception):
            log.error(
                "Model initialization failed | %s | %s",
                tf,
                result
            )

    update_live_spread()

    if state["models"]:
        state["status"] = "Market data connected / models ready"
    else:
        state["status"] = "Model initialization failed"

    await broadcast_state()


def update_live_spread():
    symbol_a, symbol_b = state["symbols"]

    pa = state["prices"].get(symbol_a)
    pb = state["prices"].get(symbol_b)

    if pa is None or pb is None or pa <= 0 or pb <= 0:
        return

    for interval, model in state["models"].items():
        beta = model["beta"]
        intercept = model["intercept"]

        mu = model["spread_mean"]
        sd = model["spread_std"]

        spread = float(
            np.log(pa) - intercept - beta * np.log(pb)
        )

        zscore = float((spread - mu) / sd)

        model["live_spread"] = spread
        model["live_zscore"] = zscore

    active = state.get("active_period", "5m")
    model = state["models"].get(active)

    if model:
        state.update({
            "beta": model["beta"],
            "beta_change": model["beta_change"],
            "beta_std": model["beta_std"],
            "intercept": model["intercept"],
            "spread_mean": model["spread_mean"],
            "spread_std": model["spread_std"],
            "spread": model["live_spread"],
            "zscore": model["live_zscore"],
            "model_time": model["model_time"]
        })


# ======================= WebSocket =======================

async def broadcast_state():
    if not clients:
        return

    payload = json.dumps(state, ensure_ascii=False)
    dead = []

    for ws in list(clients):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)

    for ws in dead:
        clients.discard(ws)


# ======================= Background Tasks =======================

async def market_stream():
    while True:
        start = time.monotonic()
        symbols = tuple(state["symbols"])

        try:
            a, b = await asyncio.gather(
                asyncio.to_thread(fetch_book_ticker, symbols[0]),
                asyncio.to_thread(fetch_book_ticker, symbols[1])
            )

            # 如果请求过程中切换了交易对，丢弃旧行情
            if tuple(state["symbols"]) != symbols:
                await asyncio.sleep(0.1)
                continue

            now = datetime.now(timezone.utc).isoformat()

            state["prices"][symbols[0]] = a["mid"]
            state["prices"][symbols[1]] = b["mid"]

            state["price_times"][symbols[0]] = now
            state["price_times"][symbols[1]] = now

            state["quotes"] = {
                symbols[0]: a,
                symbols[1]: b
            }

            state["last_tick_time"] = now
            state["updates"] += 1

            update_live_spread()

            state["status"] = "REST market data connected"

            await broadcast_state()

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            state["status"] = (
                f"REST polling warning: {type(exc).__name__}"
            )

            log.warning("REST polling error: %s", exc)

        elapsed = time.monotonic() - start

        await asyncio.sleep(
            max(0.1, POLL_SECONDS - elapsed)
        )


async def model_refresh_loop():
    while True:
        try:
            await asyncio.sleep(REFRESH_MODEL_SECONDS)

            async with pair_lock:
                symbols = tuple(state["symbols"])

                latest = await asyncio.gather(
                    *(
                        asyncio.to_thread(
                            fetch_latest_closed_kline_time,
                            symbols[0],
                            tf
                        )
                        for tf in TIMEFRAMES
                    ),
                    return_exceptions=True
                )

                changed = False

                for tf, newest in zip(TIMEFRAMES, latest):
                    if isinstance(newest, Exception):
                        log.warning(
                            "Latest candle check failed | %s | %s",
                            tf,
                            newest
                        )
                        continue

                    old_model = state["models"].get(tf)

                    if old_model is None:
                        should_refresh = True
                    else:
                        old_time = pd.Timestamp(old_model["model_time"])
                        interval_seconds = get_interval_seconds(tf)

                        elapsed_bars = (
                            newest - old_time
                        ).total_seconds() / interval_seconds

                        should_refresh = elapsed_bars >= MODEL_REFRESH_BARS

                    if should_refresh:
                        try:
                            await rebuild_timeframe(tf, symbols)
                            changed = True
                            log.info(
                                "Scheduled model refresh | %s | latest=%s",
                                tf,
                                newest
                            )
                        except Exception as exc:
                            log.warning(
                                "Model refresh failed | %s | %s",
                                tf,
                                exc
                            )

                if changed:
                    update_live_spread()
                    await broadcast_state()

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            log.warning(
                "Model refresh loop failed: %s",
                exc
            )

# ======================= Change Trading Pair =======================

async def change_pair(symbol_a, symbol_b):
    symbol_a = symbol_a.strip().upper()
    symbol_b = symbol_b.strip().upper()

    if not symbol_a or not symbol_b:
        return False, "Symbols cannot be empty"

    if symbol_a == symbol_b:
        return False, "Two symbols must be different"

    # 先验证两个合约是否可以获取历史数据
    try:
        await asyncio.gather(
            asyncio.to_thread(fetch_klines, symbol_a, "5m"),
            asyncio.to_thread(fetch_klines, symbol_b, "5m")
        )

    except Exception as exc:
        return False, f"Symbol validation failed: {exc}"

    async with pair_lock:
        state.clear()
        state.update(initial_state(symbol_a, symbol_b))

        state["status"] = "Building models..."

        log.info(
            "Trading pair changed: %s / %s",
            symbol_a,
            symbol_b
        )

        await broadcast_state()

        await rebuild_all_models()

        if not state["models"]:
            return False, "No timeframe model could be built"

        return True, (
            f"Pair changed successfully: {symbol_a} / {symbol_b}"
        )


# ======================= FastAPI Lifecycle =======================

@asynccontextmanager
async def lifespan(app):
    log.info("Initializing pair monitor...")

    await rebuild_all_models()

    background_tasks.append(
        asyncio.create_task(market_stream())
    )

    background_tasks.append(
        asyncio.create_task(model_refresh_loop())
    )

    yield

    for task in background_tasks:
        task.cancel()

    if background_tasks:
        await asyncio.gather(
            *background_tasks,
            return_exceptions=True
        )


app = FastAPI(
    title="Binance Multi-Timeframe Pair Monitor",
    lifespan=lifespan
)


# ======================= API =======================

@app.post("/api/pair")
async def api_change_pair(request: Request):
    try:
        data = await request.json()

        symbol_a = str(data.get("symbol_a", ""))
        symbol_b = str(data.get("symbol_b", ""))

        success, message = await change_pair(
            symbol_a,
            symbol_b
        )

        return JSONResponse({
            "success": success,
            "message": message,
            "symbols": state["symbols"]
        })

    except Exception as exc:
        log.exception("Pair change API error")

        return JSONResponse({
            "success": False,
            "message": str(exc)
        }, status_code=500)


@app.get("/api/state")
async def api_state():
    return state


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)

    try:
        await ws.send_text(
            json.dumps(state, ensure_ascii=False)
        )

        while True:
            await ws.receive_text()

    except WebSocketDisconnect:
        pass

    finally:
        clients.discard(ws)


# ======================= Dashboard =======================

HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Binance Pair Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
:root{
--bg:#0b1020;
--panel:#121a2b;
--muted:#8ea0bd;
--text:#e8eef9;
--line:#26334b;
--green:#22c55e;
--red:#ef4444;
--blue:#60a5fa;
--orange:#f59e0b;
--purple:#a78bfa
}

*{box-sizing:border-box}

body{
margin:0;
background:var(--bg);
color:var(--text);
font:14px/1.5 Inter,Segoe UI,Arial,sans-serif
}

main{
max-width:1350px;
margin:28px auto;
padding:0 18px
}

header{
display:flex;
justify-content:space-between;
align-items:center;
gap:12px;
margin-bottom:22px
}

h1{font-size:23px;margin:0}
h2{font-size:15px;margin:0 0 12px}
.sub{color:var(--muted);font-size:12px;margin-top:4px}

.status{
padding:7px 11px;
border:1px solid var(--line);
border-radius:20px;
color:var(--muted);
font-size:12px
}

.card{
background:var(--panel);
border:1px solid var(--line);
border-radius:13px;
padding:16px;
min-width:0
}

.quote-grid{
display:grid;
grid-template-columns:repeat(2,minmax(0,1fr));
gap:13px
}

.grid{
display:grid;
grid-template-columns:repeat(4,minmax(0,1fr));
gap:13px
}

.metric-grid{
display:grid;
grid-template-columns:repeat(3,minmax(0,1fr));
gap:13px
}

.label{color:var(--muted);font-size:12px}

.value{
font-size:24px;
font-weight:650;
margin-top:8px;
font-variant-numeric:tabular-nums;
overflow-wrap:anywhere
}

.small{
font-size:12px;
color:var(--muted);
margin-top:5px;
overflow-wrap:anywhere
}

.section{margin-top:15px}

table{
width:100%;
border-collapse:collapse
}

td,th{
text-align:left;
padding:9px 8px;
border-bottom:1px solid var(--line);
font-variant-numeric:tabular-nums
}

th{color:var(--muted);font-size:12px;font-weight:500}

.quote-table{margin-top:10px;font-size:13px}
.quote-table td:first-child{color:var(--muted)}
.quote-table td:last-child{text-align:right}

.chartbox{height:300px;position:relative}

.period-selector{
display:flex;
flex-wrap:wrap;
gap:8px;
margin:12px 0
}

.period-btn,.action-btn{
padding:8px 14px;
border:1px solid var(--line);
border-radius:6px;
background:#1f2937;
color:#d1d5db;
cursor:pointer
}

.period-btn.active,.action-btn{
background:#2563eb;
border-color:#2563eb;
color:white
}

.pair-form{
display:flex;
flex-wrap:wrap;
gap:10px;
align-items:center;
margin-top:12px
}

.pair-input{
background:#0b1020;
border:1px solid var(--line);
border-radius:6px;
padding:10px 12px;
color:white;
width:180px;
outline:none
}

.pair-input:focus{border-color:var(--blue)}

.pos{color:var(--green)}
.neg{color:var(--red)}

.footer{
color:var(--muted);
font-size:11px;
margin:16px 2px
}

@media(max-width:900px){
.grid{grid-template-columns:repeat(2,minmax(0,1fr))}
}

@media(max-width:600px){
main{margin:18px auto;padding:0 12px}
.quote-grid,.metric-grid{grid-template-columns:1fr}
header{align-items:flex-start;flex-direction:column}
.value{font-size:20px}
.chartbox{height:250px}
.pair-input{width:100%}
}
</style>
</head>

<body>
<main>

<header>
<div>
<h1>Binance Pair Monitor</h1>
<div class="sub" id="pair-title">Multi-Timeframe Rolling OLS</div>
</div>
<div class="status" id="status">Connecting...</div>
</header>

<section class="card section">
<h2>Change Trading Pair</h2>

<div class="pair-form">
<input id="symbol-a-input" class="pair-input" value="SKHYNIXUSDT" placeholder="Symbol A">
<span style="color:#8ea0bd">/</span>
<input id="symbol-b-input" class="pair-input" value="VRTUSDT" placeholder="Symbol B">
<button class="action-btn" id="change-pair-btn" onclick="changePair()">Apply Pair</button>
</div>

<div class="small" id="pair-message">
Enter two Binance USDⓈ-M perpetual symbols.
</div>
</section>

<section class="quote-grid section">

<div class="card">
<div class="label" id="symbol-a-title">Symbol A · Mid Price</div>
<div class="value" id="price-a">—</div>

<table class="quote-table">
<tr><td>Best Bid</td><td id="a-bid">—</td></tr>
<tr><td>Best Ask</td><td id="a-ask">—</td></tr>
<tr><td>Bid-Ask Spread</td><td id="a-spread">—</td></tr>
<tr><td>Bid Qty / Ask Qty</td><td id="a-qty">—</td></tr>
</table>

<div class="small" id="a-time">Waiting for quote</div>
</div>

<div class="card">
<div class="label" id="symbol-b-title">Symbol B · Mid Price</div>
<div class="value" id="price-b">—</div>

<table class="quote-table">
<tr><td>Best Bid</td><td id="b-bid">—</td></tr>
<tr><td>Best Ask</td><td id="b-ask">—</td></tr>
<tr><td>Bid-Ask Spread</td><td id="b-spread">—</td></tr>
<tr><td>Bid Qty / Ask Qty</td><td id="b-qty">—</td></tr>
</table>

<div class="small" id="b-time">Waiting for quote</div>
</div>

</section>

<section class="section card">
<h2>Model Timeframe</h2>

<div class="period-selector">
<button class="period-btn active" data-period="5m">5m</button>
<button class="period-btn" data-period="15m">15m</button>
<button class="period-btn" data-period="30m">30m</button>
<button class="period-btn" data-period="1h">1H</button>
<button class="period-btn" data-period="2h">2H</button>
<button class="period-btn" data-period="4h">4H</button>
<button class="period-btn" data-period="1d">1D</button>
</div>

<div class="small" id="period-info">
Each timeframe has its own independent regression model.
</div>
</section>

<section class="section grid">

<div class="card">
<div class="label">Hedge Ratio · Beta</div>
<div class="value" id="beta">—</div>
<div class="small" id="beta-info">Waiting for model</div>
</div>

<div class="card">
<div class="label">Live Spread Z-score</div>
<div class="value" id="z">—</div>
<div class="small" id="signal">Waiting for prices</div>
</div>

<div class="card">
<div class="label">Live Log Spread</div>
<div class="value" id="spread">—</div>
<div class="small">Current midpoint-based residual</div>
</div>

<div class="card">
<div class="label">Model Refresh Time</div>
<div class="value" id="model-time">—</div>
<div class="small" id="model-period">Latest completed candle</div>
</div>

</section>

<section class="section metric-grid">

<div class="card">
<div class="label">Rolling Spread Mean</div>
<div class="value" id="mean">—</div>
</div>

<div class="card">
<div class="label">Rolling Spread Std</div>
<div class="value" id="std">—</div>
</div>

<div class="card">
<div class="label">Successful Polling Cycles</div>
<div class="value" id="updates">0</div>
</div>

</section>

<section class="section card">
<h2 id="spread-title">5m · Rolling Spread</h2>
<div class="chartbox"><canvas id="spreadChart"></canvas></div>
<div class="small">
Historical rolling residuals with mean and ±1σ / ±2σ reference levels.
Live spread uses current midpoint prices.
</div>
</section>

<section class="section card">
<h2 id="beta-title">5m · Rolling Hedge Ratio</h2>
<div class="chartbox"><canvas id="betaChart"></canvas></div>
<div class="small" id="beta-window-info">
Rolling Beta estimated from completed candles.
</div>
</section>

<section class="section card">
<h2>Model & Feed Details</h2>

<table>
<tr><th>Item</th><th>Value</th></tr>
<tr><td>Regression</td><td id="regression-formula">—</td></tr>
<tr><td>Selected Timeframe</td><td id="detail-period">5m</td></tr>
<tr><td>Beta Window</td><td id="beta-window-detail">—</td></tr>
<tr><td>Z-score Baseline</td><td id="spread-window-detail">—</td></tr>
<tr><td>Beta Change</td><td id="beta-change">—</td></tr>
<tr><td>Beta Standard Deviation</td><td id="beta-std">—</td></tr>
<tr><td>Last Quote Received (UTC)</td><td id="last-tick">—</td></tr>
<tr><td>Model Refresh Time (UTC)</td><td id="model-time-2">—</td></tr>
</table>
</section>

<div class="footer">
For research and monitoring only. No orders are submitted.
Z-score is a statistical deviation indicator, not proof of cointegration
or a guaranteed trading signal.
</div>

</main>

<script>

let chart = null;
let betaChart = null;
let selectedPeriod = "5m";
let latestState = null;
let ws = null;
let reconnectTimer = null;

const fmt = (x, n=6) => {
    if (x === null || x === undefined || !Number.isFinite(Number(x))) return "—";
    return Number(x).toFixed(n);
};

const fmtTime = x => {
    return x ? new Date(x).toLocaleString() : "—";
};


function initChart() {
    chart = new Chart(
        document.getElementById("spreadChart"),
        {
            type: "line",
            data: {
                labels: [],
                datasets: []
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: {
                    mode: "index",
                    intersect: false
                },
                plugins: {
                    legend: {
                        labels: {color: "#cbd5e1"}
                    }
                },
                scales: {
                    x: {
                        ticks: {color: "#8ea0bd", maxTicksLimit: 8},
                        grid: {color: "#26334b"}
                    },
                    y: {
                        ticks: {color: "#8ea0bd"},
                        grid: {color: "#26334b"}
                    }
                }
            }
        }
    );
}


function initBetaChart() {
    betaChart = new Chart(
        document.getElementById("betaChart"),
        {
            type: "line",
            data: {
                labels: [],
                datasets: [{
                    label: "Rolling Beta",
                    data: [],
                    borderColor: "#a78bfa",
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: {
                    mode: "index",
                    intersect: false
                },
                plugins: {
                    legend: {
                        labels: {color: "#cbd5e1"}
                    }
                },
                scales: {
                    x: {
                        ticks: {color: "#8ea0bd", maxTicksLimit: 8},
                        grid: {color: "#26334b"}
                    },
                    y: {
                        ticks: {color: "#8ea0bd"},
                        grid: {color: "#26334b"}
                    }
                }
            }
        }
    );
}


function updateQuote(symbol, prefix, d) {
    const q = d.quotes?.[symbol];

    const priceId = prefix === "a" ? "price-a" : "price-b";

    document.getElementById(priceId).textContent =
        fmt(d.prices?.[symbol], 4);

    document.getElementById(prefix + "-bid").textContent =
        q ? fmt(q.bid, 4) : "—";

    document.getElementById(prefix + "-ask").textContent =
        q ? fmt(q.ask, 4) : "—";

    document.getElementById(prefix + "-spread").textContent =
        q && q.mid > 0
        ? fmt((q.ask - q.bid) / q.mid * 10000, 2) + " bps"
        : "—";

    document.getElementById(prefix + "-qty").textContent =
        q ? fmt(q.bid_qty, 3) + " / " + fmt(q.ask_qty, 3) : "—";

    document.getElementById(prefix + "-time").textContent =
        d.price_times?.[symbol]
        ? "Updated: " + fmtTime(d.price_times[symbol]) + " UTC"
        : "Waiting for quote";
}


function updateCharts(model, period) {
    const hist = model.history || [];

    const labels = hist.map(x => fmtTime(x.time));

    chart.data.labels = labels.concat(["LIVE"]);

    chart.data.datasets = [
        {
            label: "Historical Spread",
            data: hist.map(x => x.spread).concat([null]),
            borderColor: "#60a5fa",
            borderWidth: 1.5,
            pointRadius: 0
        },
        {
            label: "Mean",
            data: hist.map(x => x.mean).concat([null]),
            borderColor: "#f59e0b",
            borderDash: [5, 5],
            borderWidth: 1,
            pointRadius: 0
        },
        {
            label: "+1σ",
            data: hist.map(x => x.upper1).concat([null]),
            borderColor: "#22c55e",
            borderDash: [3, 3],
            borderWidth: 1,
            pointRadius: 0
        },
        {
            label: "-1σ",
            data: hist.map(x => x.lower1).concat([null]),
            borderColor: "#22c55e",
            borderDash: [3, 3],
            borderWidth: 1,
            pointRadius: 0
        },
        {
            label: "+2σ",
            data: hist.map(x => x.upper2).concat([null]),
            borderColor: "#ef4444",
            borderDash: [3, 3],
            borderWidth: 1,
            pointRadius: 0
        },
        {
            label: "-2σ",
            data: hist.map(x => x.lower2).concat([null]),
            borderColor: "#ef4444",
            borderDash: [3, 3],
            borderWidth: 1,
            pointRadius: 0
        },
        {
            label: "Live Spread",
            data: hist.map(() => null).concat([model.live_spread]),
            borderColor: "#f59e0b",
            backgroundColor: "#f59e0b",
            pointRadius: 5,
            showLine: false
        }
    ];

    chart.update("none");

    const bh = model.beta_history || [];

    betaChart.data.labels = bh.map(x => fmtTime(x.time));
    betaChart.data.datasets[0].data = bh.map(x => x.beta);

    betaChart.update("none");

    document.getElementById("spread-title").textContent =
        period + " · Rolling Spread";

    document.getElementById("beta-title").textContent =
        period + " · Rolling Hedge Ratio";
}


function updateSelectedModel(d) {
    const model = d.models?.[selectedPeriod];

    if (!model) {
        return;
    }

    document.querySelectorAll(".period-btn").forEach(btn => {
        btn.classList.toggle(
            "active",
            btn.dataset.period === selectedPeriod
        );
    });

    document.getElementById("detail-period").textContent =
        selectedPeriod;

    document.getElementById("model-period").textContent =
        "Latest completed " + selectedPeriod + " candle";

    document.getElementById("beta").textContent =
        fmt(model.beta, 6);

    document.getElementById("beta-info").textContent =
        "20-bar change: " + fmt(model.beta_change, 6) +
        " | Beta std: " + fmt(model.beta_std, 6);

    document.getElementById("beta-change").textContent =
        fmt(model.beta_change, 6);

    document.getElementById("beta-std").textContent =
        fmt(model.beta_std, 6);

    document.getElementById("model-time").textContent =
        fmtTime(model.model_time);

    document.getElementById("model-time-2").textContent =
        model.model_time || "—";

    document.getElementById("spread").textContent =
        fmt(model.live_spread, 8);

    document.getElementById("mean").textContent =
        fmt(model.spread_mean, 8);

    document.getElementById("std").textContent =
        fmt(model.spread_std, 8);

    document.getElementById("beta-window-detail").textContent =
        model.beta_window + " completed bars";

    document.getElementById("spread-window-detail").textContent =
        "Last " + model.spread_window + " residuals";

    document.getElementById("beta-window-info").textContent =
        "Rolling Beta estimated from " +
        model.beta_window + " completed candles.";

    const z = model.live_zscore;

    const zEl = document.getElementById("z");

    zEl.textContent = fmt(z, 3);
    zEl.className = "value " +
        (z > 0 ? "pos" : z < 0 ? "neg" : "");

    let sig = "Waiting for prices";

    if (z !== null && Number.isFinite(Number(z))) {
        if (z >= 2) sig = "Positive deviation ≥ +2σ";
        else if (z <= -2) sig = "Negative deviation ≤ -2σ";
        else if (z >= 1) sig = "Positive deviation";
        else if (z <= -1) sig = "Negative deviation";
        else sig = "Within ±1σ";
    }

    document.getElementById("signal").textContent = sig;

    updateCharts(model, selectedPeriod);
}


function update(d) {
    latestState = d;

    document.getElementById("status").textContent =
        d.status || "—";

    const symbolA = d.symbols?.[0];
    const symbolB = d.symbols?.[1];

    if (!symbolA || !symbolB) return;

    document.getElementById("pair-title").textContent =
        "USDⓈ-M Perpetual · " + symbolA + " / " +
        symbolB + " · Multi-Timeframe Rolling OLS";

    document.getElementById("symbol-a-title").textContent =
        symbolA + " · Mid Price";

    document.getElementById("symbol-b-title").textContent =
        symbolB + " · Mid Price";

    document.getElementById("symbol-a-input").value = symbolA;
    document.getElementById("symbol-b-input").value = symbolB;

    document.getElementById("regression-formula").textContent =
        "log(" + symbolA + ") = intercept + beta × log(" +
        symbolB + ")";

    updateQuote(symbolA, "a", d);
    updateQuote(symbolB, "b", d);

    document.getElementById("updates").textContent =
        (d.updates || 0).toLocaleString();

    document.getElementById("last-tick").textContent =
        d.last_tick_time || "—";

    if (!d.models?.[selectedPeriod] && d.models) {
        const first = Object.keys(d.models)[0];

        if (first) selectedPeriod = first;
    }

    updateSelectedModel(d);
}


async function changePair() {
    const symbolA = document
        .getElementById("symbol-a-input")
        .value.trim().toUpperCase();

    const symbolB = document
        .getElementById("symbol-b-input")
        .value.trim().toUpperCase();

    const btn = document.getElementById("change-pair-btn");
    const msg = document.getElementById("pair-message");

    if (!symbolA || !symbolB) {
        msg.textContent = "Please enter both symbols.";
        return;
    }

    if (symbolA === symbolB) {
        msg.textContent = "Two symbols must be different.";
        return;
    }

    btn.disabled = true;
    btn.textContent = "Loading...";
    msg.textContent = "Validating symbols and rebuilding models...";

    try {
        const response = await fetch("/api/pair", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                symbol_a: symbolA,
                symbol_b: symbolB
            })
        });

        const result = await response.json();

        msg.textContent = result.message || "Request completed.";

        if (result.success) {
            selectedPeriod = "5m";
            document.querySelectorAll(".period-btn").forEach(btn => {
                btn.classList.toggle(
                    "active",
                    btn.dataset.period === selectedPeriod
                );
            });
        }

    } catch (err) {
        msg.textContent = "Request failed: " + err.message;
    } finally {
        btn.disabled = false;
        btn.textContent = "Apply Pair";
    }
}


document.querySelectorAll(".period-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        selectedPeriod = btn.dataset.period;

        if (latestState) {
            updateSelectedModel(latestState);
        }
    });
});


function connect() {
    if (ws && (
        ws.readyState === WebSocket.OPEN ||
        ws.readyState === WebSocket.CONNECTING
    )) {
        return;
    }

    const proto =
        location.protocol === "https:" ? "wss" : "ws";

    ws = new WebSocket(
        proto + "://" + location.host + "/ws"
    );

    ws.onmessage = event => {
        try {
            update(JSON.parse(event.data));
        } catch (err) {
            console.error("WebSocket message error:", err);
        }
    };

    ws.onclose = () => {
        document.getElementById("status").textContent =
            "Dashboard disconnected; retrying...";

        ws = null;

        if (reconnectTimer === null) {
            reconnectTimer = setTimeout(() => {
                reconnectTimer = null;
                connect();
            }, 2000);
        }
    };

    ws.onerror = () => {
        if (ws) ws.close();
    };
}


initChart();
initBetaChart();
connect();

</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


# ======================= Main =======================

if __name__ == "__main__":
    print(f"Starting dashboard: http://{HOST}:{PORT}")

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )
