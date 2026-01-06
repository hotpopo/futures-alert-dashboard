import re
import time
import math
from datetime import datetime, timezone, timedelta
from collections import deque

import requests
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# =========================
# 基础配置
# =========================
st.set_page_config(page_title="期货实时提示看板", layout="wide")

TZ_JST = timezone(timedelta(hours=9))
TZ_CST = timezone(timedelta(hours=8))

# ✅ 关键：用 http，更稳定（Cloud 下 https 经常空返回）
SINA_QUOTE_URL = "http://hq.sinajs.cn/list="

CONTRACT_GROUPS = {
    "2605": {"Y": "nf_y2605", "P": "nf_p2605", "OI": "nf_oi2605", "M": "nf_m2605"},
    "2609": {"Y": "nf_y2609", "P": "nf_p2609", "OI": "nf_oi2609", "M": "nf_m2609"},
}

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://finance.sina.com.cn/",
    "Accept": "*/*",
    "Connection": "keep-alive",
}

# =========================
# 交易时段判断（按CST）
# =========================
def is_trading_time_cst(dt_cst: datetime) -> bool:
    if dt_cst.weekday() >= 5:
        return False
    hm = dt_cst.hour * 60 + dt_cst.minute

    def in_range(a, b):
        return a <= hm <= b

    day_1 = in_range(9 * 60, 11 * 60 + 30)
    day_2 = in_range(13 * 60 + 30, 15 * 60)
    night = in_range(21 * 60, 23 * 60)

    return day_1 or day_2 or night

# =========================
# 新浪行情：抓取/解析（带调试）
# =========================
def fetch_sina_quotes(symbols: list[str]):
    """
    返回:
      quotes: dict[symbol]->fields(list)
      debug: dict 里面有 status_code / text_head / url
    """
    if not symbols:
        return {}, {"status_code": None, "text_head": "", "url": ""}

    url = SINA_QUOTE_URL + ",".join(symbols)
    try:
        r = requests.get(url, headers=HEADERS, timeout=8, allow_redirects=True)
        r.encoding = "gbk"
        text = r.text
        status = r.status_code
    except Exception as e:
        return {}, {"status_code": "EXC", "text_head": str(e)[:300], "url": url}

    out = {}
    for m in re.finditer(r'var\s+hq_str_(\w+)\s*=\s*"([^"]*)";', text):
        sym = m.group(1)                 # nf_y2605
        payload = m.group(2).strip()     # 逗号分隔字段
        out[sym] = payload.split(",") if payload else []

    debug = {
        "status_code": status,
        "url": url,
        "text_head": text[:300].replace("\n", "\\n"),
        "matched_symbols": list(out.keys())[:10],
    }
    return out, debug


def parse_nf(fields: list[str]) -> dict:
    def fnum(x):
        try:
            return float(x)
        except Exception:
            return float("nan")

    name = fields[0] if len(fields) > 0 else ""
    open_ = fnum(fields[1]) if len(fields) > 1 else math.nan
    last  = fnum(fields[3]) if len(fields) > 3 else math.nan
    high  = fnum(fields[4]) if len(fields) > 4 else math.nan
    low   = fnum(fields[5]) if len(fields) > 5 else math.nan

    if not np.isfinite(last):
        for idx in [2, 1, 6, 7]:
            if len(fields) > idx:
                v = fnum(fields[idx])
                if np.isfinite(v):
                    last = v
                    break

    return {"name": name, "open": open_, "high": high, "low": low, "last": last}

# =========================
# 统计：Z-score / ATR代理 / 突破确认
# =========================
def zscore_from_list(values: list[float]) -> float:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 20:
        return float("nan")
    mu = arr.mean()
    sd = arr.std(ddof=1)
    if sd == 0:
        return float("nan")
    return (arr[-1] - mu) / sd


def atr_proxy_from_prices(prices: list[float], lookback: int) -> float:
    arr = np.array(prices, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < lookback + 2:
        return float("nan")
    diffs = np.abs(np.diff(arr[-(lookback + 1):]))
    return float(np.mean(diffs)) if len(diffs) else float("nan")


def breakout_signal(prices: deque, window: int, confirm_k: int, buffer: float,
                   atr_lookback: int, atr_mult_stop: float, rr_take: float):
    series = [p for p in list(prices) if np.isfinite(p)]
    if len(series) < window + confirm_k + 5:
        return None, None, None, None, None, "样本不足（开盘后需要积累一段数据）"

    base = series[: -confirm_k]
    recent = series[-confirm_k:]
    base_window = base[-window:]
    H = float(np.max(base_window))
    L = float(np.min(base_window))

    last = float(series[-1])
    atrp = atr_proxy_from_prices(series, atr_lookback)

    long_ok = all(x > (H + buffer) for x in recent)
    short_ok = all(x < (L - buffer) for x in recent)

    if not (long_ok or short_ok):
        return None, None, None, None, None, "未触发突破确认"

    direction = "LONG" if long_ok else "SHORT"
    entry = last

    if direction == "LONG":
        stop1 = H
        stop2 = (H - atr_mult_stop * atrp) if np.isfinite(atrp) else stop1
        stop = min(stop1, stop2)
        R = entry - stop
        tp = entry + rr_take * R if R > 0 else float("nan")
        return direction, entry, stop, tp, H, f"突破上沿H={H:.0f}，连续{confirm_k}次确认"
    else:
        stop1 = L
        stop2 = (L + atr_mult_stop * atrp) if np.isfinite(atrp) else stop1
        stop = max(stop1, stop2)
        R = stop - entry
        tp = entry - rr_take * R if R > 0 else float("nan")
        return direction, entry, stop, tp, L, f"跌破下沿L={L:.0f}，连续{confirm_k}次确认"

# =========================
# UI
# =========================
st.title("📊 期货实时提示看板（2605 / 2609）")

with st.sidebar:
    st.header("参数")
    group = st.selectbox("合约组", ["2605", "2609"], index=0)

    refresh_trading = st.slider("交易时段刷新间隔（秒）", 1, 10, 2)
    refresh_off = st.slider("非交易时段刷新间隔（秒）", 30, 600, 120, step=30)

    only_trade_time = st.checkbox("仅在交易时段请求行情（推荐）", value=True)
    pause_fetch = st.checkbox("暂停抓取（我现在不盯盘）", value=False)

    st.divider()
    st.subheader("突破确认信号（模板1）")
    signal_symbol = st.selectbox("信号品种", ["Y", "P", "OI", "M"], index=0)
    win = st.slider("区间窗口N（样本点）", 30, 600, 180, step=30)
    confirm_k = st.slider("确认次数K", 2, 8, 3)
    buffer = st.number_input("突破缓冲（点）", value=1.0, min_value=0.0, step=1.0)
    atr_lb = st.slider("波动代理窗口（样本点）", 20, 300, 60, step=20)
    atr_mult_stop = st.number_input("止损ATR倍数（代理）", value=0.5, min_value=0.0, step=0.1)
    rr_take = st.number_input("止盈R倍（TP = entry ± R*倍数）", value=2.0, min_value=0.5, step=0.5)
    cooldown_sec = st.slider("同向信号冷却（秒）", 30, 600, 120, step=30)

    st.divider()
    st.subheader("结构价差（Z-score）")
    z_win = st.slider("Z-score 窗口（样本点）", 60, 600, 180, step=30)
    z_th = st.slider("告警阈值 |Z| ≥", 1.0, 3.0, 2.0, step=0.1)

    st.divider()
    show_debug = st.checkbox("显示调试信息（可选）", value=False)

symbols_map = CONTRACT_GROUPS[group]
symbols = list(symbols_map.values())

# Session state
if "hist_spread" not in st.session_state:
    st.session_state.hist_spread = {
        "Y-P": deque(maxlen=5000),
        "OI-Y": deque(maxlen=5000),
        "OI-P": deque(maxlen=5000),
    }
if "price_hist" not in st.session_state:
    st.session_state.price_hist = {sym: deque(maxlen=8000) for sym in symbols}
if "last_signal_ts" not in st.session_state:
    st.session_state.last_signal_ts = {}
if "last_alert_ts" not in st.session_state:
    st.session_state.last_alert_ts = {}

# 时段与刷新
now_cst = datetime.now(TZ_CST)
trading_now = is_trading_time_cst(now_cst)
refresh_sec = refresh_trading if trading_now else refresh_off
st_autorefresh(interval=refresh_sec * 1000, key="tick")

now_jst = datetime.now(TZ_JST)
st.caption(
    f"更新时间：{now_jst:%Y-%m-%d %H:%M:%S JST}（{now_cst:%Y-%m-%d %H:%M:%S CST}）｜"
    f"{'🟢 交易时段' if trading_now else '⚪ 非交易时段'}｜刷新：{refresh_sec}s｜合约组：{group}"
)

# 是否请求行情
should_fetch = True
if pause_fetch:
    should_fetch = False
elif only_trade_time and (not trading_now):
    should_fetch = False

raw = {}
debug = {"status_code": None, "text_head": "", "url": "", "matched_symbols": []}

if should_fetch:
    raw, debug = fetch_sina_quotes(symbols)
    st.caption("已请求新浪期货行情（若仍无数据，请看调试区的 status_code 与 text_head）")
else:
    st.info("当前非交易时段/暂停抓取，已停止行情请求")

# ✅ 调试信息前置：你勾选后马上能看到
if show_debug:
    st.info(
        f"DEBUG｜status_code={debug.get('status_code')} ｜ matched={debug.get('matched_symbols')} \n\n"
        f"URL：{debug.get('url')}\n\n"
        f"text_head：{debug.get('text_head')}"
    )

# 解析 DataFrame
rows = []
for prod, sym in symbols_map.items():
    fields = raw.get(sym, None)
    if not fields:
        rows.append({"品种": prod, "合约": sym.replace("nf_", "").upper(), "名称": "-", "最新": np.nan, "今开": np.nan, "最高": np.nan, "最低": np.nan})
        continue

    info = parse_nf(fields)
    rows.append({
        "品种": prod,
        "合约": sym.replace("nf_", "").upper(),
        "名称": info["name"] if info["name"] else "-",
        "最新": info["last"],
        "今开": info["open"],
        "最高": info["high"],
        "最低": info["low"],
    })

df = pd.DataFrame(rows)

# 顶部卡片
c1, c2, c3, c4 = st.columns(4)
for col, prod in zip([c1, c2, c3, c4], ["Y", "P", "OI", "M"]):
    r = df[df["品种"] == prod].iloc[0]
    with col:
        st.metric(
            label=f"{prod}  {r['合约']}",
            value="-" if not np.isfinite(r["最新"]) else f"{r['最新']:.0f}",
            help=r["名称"],
        )

st.divider()
st.subheader("实时行情")
st.dataframe(df, width="stretch", hide_index=True)

# 更新历史
for prod, sym in symbols_map.items():
    v = df[df["品种"] == prod]["最新"].values[0]
    if np.isfinite(v):
        st.session_state.price_hist.setdefault(sym, deque(maxlen=8000)).append(float(v))

# 突破确认模板
st.subheader("单品种交易提示（突破确认模板）")
target_sym = symbols_map[signal_symbol]
prices = st.session_state.price_hist.get(target_sym, deque())

direction, entry, stop, tp, level, info = breakout_signal(
    prices, win, confirm_k, buffer, atr_lb, atr_mult_stop, rr_take
)

def can_emit_signal(group_: str, sym_: str, dir_: str, cooldown: int) -> bool:
    key = (group_, sym_, dir_)
    last_ts = st.session_state.last_signal_ts.get(key, 0.0)
    now_ts = time.time()
    if now_ts - last_ts >= cooldown:
        st.session_state.last_signal_ts[key] = now_ts
        return True
    return False

left, mid, right = st.columns([1.2, 1.0, 1.2])

with left:
    st.write(f"**标的：** {signal_symbol} / {target_sym.replace('nf_', '').upper()}")
    st.write(f"**状态：** {info}")

with mid:
    st.write("**参考位**")
    st.metric("突破参考位", "-" if level is None else f"{level:.0f}")

with right:
    st.write("**建议（仅提示，不构成投资建议）**")
    if direction is None:
        st.info("暂无触发信号")
    else:
        action = "考虑做多" if direction == "LONG" else "考虑做空"
        emoji = "🚀" if direction == "LONG" else "📉"
        if can_emit_signal(group, target_sym, direction, cooldown_sec):
            st.warning(
                f"{emoji}【突破确认】{target_sym.replace('nf_', '').upper()}：{action}\n\n"
                f"入场参考：{entry:.0f}\n"
                f"止损参考：{stop:.0f}\n"
                f"止盈参考：{tp:.0f}（{rr_take}R）"
            )
        else:
            st.info(f"信号仍有效（冷却中）：{action}｜入场 {entry:.0f}｜止损 {stop:.0f}｜止盈 {tp:.0f}")

st.caption("说明：模板1=区间突破+连续K次确认。止损以“回到突破位”为主，叠加波动代理保护；止盈按R倍给出。")

# 结构价差与Z-score（略：保持原逻辑）
def get_price(prod: str) -> float:
    v = df[df["品种"] == prod]["最新"].values[0]
    return float(v) if np.isfinite(v) else float("nan")

Y = get_price("Y")
P = get_price("P")
OI = get_price("OI")

spreads = {
    "Y-P": Y - P if np.isfinite(Y) and np.isfinite(P) else float("nan"),
    "OI-Y": OI - Y if np.isfinite(OI) and np.isfinite(Y) else float("nan"),
    "OI-P": OI - P if np.isfinite(OI) and np.isfinite(P) else float("nan"),
}

for name, val in spreads.items():
    if np.isfinite(val):
        st.session_state.hist_spread[name].append(val)

st.subheader("结构价差与提示（Z-score）")
s1, s2, s3 = st.columns(3)
for col, name in zip([s1, s2, s3], ["Y-P", "OI-Y", "OI-P"]):
    series = list(st.session_state.hist_spread[name])[-z_win:]
    z = zscore_from_list(series) if len(series) >= 20 else float("nan")
    val = spreads[name]
    with col:
        st.metric(label=f"{name}", value="-" if not np.isfinite(val) else f"{val:.0f}",
                  delta=None if not np.isfinite(z) else f"Z={z:.2f}")
