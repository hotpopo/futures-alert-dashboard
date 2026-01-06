import re
import time
from datetime import datetime, timezone, timedelta
from collections import deque

import requests
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# ---------------------------
# 基础配置
# ---------------------------
st.set_page_config(page_title="期货实时提示看板", layout="wide")

TZ_JST = timezone(timedelta(hours=9))   # 页面显示：日本时间
TZ_CST = timezone(timedelta(hours=8))   # 交易时段判断：中国时间

SINA_QUOTE_URL = "https://hq.sinajs.cn/list="

CONTRACT_GROUPS = {
    "2605": {"Y": "y2605", "P": "p2605", "OI": "oi2605", "M": "m2605"},
    "2609": {"Y": "y2609", "P": "p2609", "OI": "oi2609", "M": "m2609"},
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36",
    "Referer": "https://finance.sina.com.cn/",
    "Accept": "*/*",
}


# ---------------------------
# 交易时段判断（保守版：日盘 + 夜盘 21:00-23:00）
# ---------------------------
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


# ---------------------------
# 行情
# ---------------------------
def fetch_sina_quotes(symbols: list[str]) -> dict:
    if not symbols:
        return {}
    url = SINA_QUOTE_URL + ",".join(symbols)
    r = requests.get(url, headers=HEADERS, timeout=8)
    r.encoding = "gbk"
    text = r.text

    out = {}
    for m in re.finditer(r'var\s+hq_str_(\w+)\s*=\s*"([^"]*)";', text):
        sym = m.group(1)
        payload = m.group(2).strip()
        if not payload:
            continue
        out[sym] = payload.split(",")
    return out


def parse_common(fields: list[str]) -> dict:
    def fnum(x):
        try:
            return float(x)
        except Exception:
            return float("nan")

    name = fields[0] if len(fields) > 0 else ""
    open_ = fnum(fields[2]) if len(fields) > 2 else float("nan")
    high = fnum(fields[3]) if len(fields) > 3 else float("nan")
    low = fnum(fields[4]) if len(fields) > 4 else float("nan")
    last = fnum(fields[5]) if len(fields) > 5 else float("nan")

    return {"name": name, "open": open_, "high": high, "low": low, "last": last}


@st.cache_data(ttl=5, show_spinner=False)
def fetch_sina_quotes_cached(symbols: tuple[str, ...]) -> dict:
    return fetch_sina_quotes(list(symbols))


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


# ---------------------------
# 突破确认信号（模板1）
# ---------------------------
def atr_proxy_from_prices(prices: list[float], lookback: int) -> float:
    """没有OHLC时，用 |ΔP| 的均值近似波动（保守替代 ATR）"""
    arr = np.array(prices, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < lookback + 2:
        return float("nan")
    diffs = np.abs(np.diff(arr[-(lookback + 1):]))
    if len(diffs) == 0:
        return float("nan")
    return float(np.mean(diffs))


def breakout_signal(
    prices: deque,
    window: int,
    confirm_k: int,
    buffer: float,
    atr_lookback: int,
    atr_mult_stop: float,
    rr_take: float,
):
    """
    返回：
    - direction: "LONG"/"SHORT"/None
    - entry, stop, tp
    - level: 突破的参考位（上沿/下沿）
    - info: 文本解释
    """
    series = [p for p in list(prices) if np.isfinite(p)]
    if len(series) < window + confirm_k + 5:
        return None, None, None, None, None, "样本不足（开盘后需要累积一段数据）"

    # 用“过去window个点”作为区间，不包含最近confirm_k个点，以免自我引用
    base = series[: -(confirm_k)]
    recent = series[-confirm_k:]

    if len(base) < window:
        return None, None, None, None, None, "样本不足（base不足）"

    base_window = base[-window:]
    H = float(np.max(base_window))
    L = float(np.min(base_window))

    last = float(series[-1])
    atrp = atr_proxy_from_prices(series, atr_lookback)

    # 触发：最近 confirm_k 个点都站上/站下（带 buffer）
    long_ok = all(x > (H + buffer) for x in recent)
    short_ok = all(x < (L - buffer) for x in recent)

    if not (long_ok or short_ok):
        return None, None, None, None, None, "未触发突破确认"

    direction = "LONG" if long_ok else "SHORT"
    entry = last

    # 止损（两种思路融合：以突破位为主，ATR代理做保护）
    if direction == "LONG":
        stop1 = H  # 跌回区间上沿，突破失败
        stop2 = (H - atr_mult_stop * atrp) if np.isfinite(atrp) else stop1
        stop = min(stop1, stop2)  # 多单止损取更宽一点（更低）
        R = entry - stop
        tp = entry + rr_take * R if R > 0 else float("nan")
        level = H
        info = f"突破上沿 H={H:.0f}，连续{confirm_k}次确认站上（buffer={buffer:g}）"
    else:
        stop1 = L  # 反弹回区间下沿，突破失败
        stop2 = (L + atr_mult_stop * atrp) if np.isfinite(atrp) else stop1
        stop = max(stop1, stop2)  # 空单止损取更宽一点（更高）
        R = stop - entry
        tp = entry - rr_take * R if R > 0 else float("nan")
        level = L
        info = f"跌破下沿 L={L:.0f}，连续{confirm_k}次确认站下（buffer={buffer:g}）"

    return direction, entry, stop, tp, level, info


# ---------------------------
# UI
# ---------------------------
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
    z_win = st.slider("价差Z-score 窗口（样本点）", 60, 600, 180, step=30)
    z_th = st.slider("价差告警阈值 |Z| ≥", 1.0, 3.0, 2.0, step=0.1)
    show_debug = st.checkbox("显示调试信息（可选）", value=False)

symbols_map = CONTRACT_GROUPS[group]
symbols = tuple(symbols_map.values())

# state：价差历史 + 单品种价格历史 + 信号冷却
if "hist_spread" not in st.session_state:
    st.session_state.hist_spread = {
        "Y-P": deque(maxlen=2000),
        "OI-Y": deque(maxlen=2000),
        "OI-P": deque(maxlen=2000),
    }
if "price_hist" not in st.session_state:
    st.session_state.price_hist = {sym: deque(maxlen=5000) for sym in symbols}
if "last_signal_ts" not in st.session_state:
    st.session_state.last_signal_ts = {}  # key: (group, sym, dir) -> ts

# 交易时段
now_cst = datetime.now(TZ_CST)
trading_now = is_trading_time_cst(now_cst)
refresh_sec = refresh_trading if trading_now else refresh_off

st_autorefresh(interval=refresh_sec * 1000, key="tick")

now_jst = datetime.now(TZ_JST).strftime("%Y-%m-%d %H:%M:%S JST")
now_cst_str = now_cst.strftime("%Y-%m-%d %H:%M:%S CST")
status = "🟢 交易时段" if trading_now else "⚪️ 非交易时段"
st.caption(f"更新时间：{now_jst}（{now_cst_str}）｜{status}｜刷新：{refresh_sec}s｜合约组：{group}")

# 是否请求行情
should_fetch = True
if pause_fetch:
    should_fetch = False
elif only_trade_time and (not trading_now):
    should_fetch = False

raw = {}
if should_fetch:
    try:
        raw = fetch_sina_quotes_cached(symbols)
        st.caption("已请求新浪行情（带缓存限频）")
    except Exception as e:
        st.error(f"拉取行情失败：{e}")
        st.stop()
else:
    st.caption("当前未请求行情（暂停或非交易时段）")

# 解析 DataFrame
rows = []
for k, sym in symbols_map.items():
    fields = raw.get(sym)
    if not fields:
        rows.append({"品种": k, "合约": sym, "名称": "-", "最新": np.nan, "今开": np.nan, "最高": np.nan, "最低": np.nan})
        continue
    info = parse_common(fields)
    rows.append({
        "品种": k,
        "合约": sym,
        "名称": info["name"],
        "最新": info["last"],
        "今开": info["open"],
        "最高": info["high"],
        "最低": info["low"],
    })

df = pd.DataFrame(rows)

# 顶部卡片
c1, c2, c3, c4 = st.columns(4)
for col, k in zip([c1, c2, c3, c4], ["Y", "P", "OI", "M"]):
    r = df[df["品种"] == k].iloc[0]
    with col:
        st.metric(
            label=f"{k}  {r['合约']}",
            value="-" if not np.isfinite(r["最新"]) else f"{r['最新']:.0f}",
            help=r["名称"],
        )

st.divider()

st.subheader("实时行情")
st.dataframe(df, width="stretch", hide_index=True)

# 更新单品种价格历史（用于突破信号）
for k, sym in symbols_map.items():
    v = df[df["品种"] == k]["最新"].values[0]
    if np.isfinite(v):
        # 如果切换合约组后 symbols 变化，确保 key 存在
        if sym not in st.session_state.price_hist:
            st.session_state.price_hist[sym] = deque(maxlen=5000)
        st.session_state.price_hist[sym].append(float(v))

# ---------------------------
# 突破信号区
# ---------------------------
st.subheader("单品种交易提示（突破确认模板）")

target_sym = symbols_map[signal_symbol]
prices = st.session_state.price_hist.get(target_sym, deque())

direction, entry, stop, tp, level, info = breakout_signal(
    prices=prices,
    window=win,
    confirm_k=confirm_k,
    buffer=buffer,
    atr_lookback=atr_lb,
    atr_mult_stop=atr_mult_stop,
    rr_take=rr_take,
)

def can_emit_signal(group_: str, sym_: str, dir_: str, cooldown: int) -> bool:
    key = (group_, sym_, dir_)
    last_ts = st.session_state.last_signal_ts.get(key, 0)
    now_ts = time.time()
    if now_ts - last_ts >= cooldown:
        st.session_state.last_signal_ts[key] = now_ts
        return True
    return False

left, mid, right = st.columns([1.2, 1.0, 1.2])
with left:
    st.write(f"**标的：** {signal_symbol} / {target_sym}")
    st.write(f"**状态：** {info}")

with mid:
    st.write("**参考位**")
    st.metric("突破参考位", "-" if level is None else f"{level:.0f}")

with right:
    st.write("**建议（仅提示，不构成投资建议）**")
    if direction is None:
        st.info("暂无触发信号")
    else:
        # 给出明确的“入场/止损/止盈”
        if direction == "LONG":
            action = "考虑做多"
            emoji = "🚀"
        else:
            action = "考虑做空"
            emoji = "📉"

        # 冷却防刷屏：只在冷却窗口内第一次显示“强提示”，否则用普通提示
        if can_emit_signal(group, target_sym, direction, cooldown_sec):
            st.warning(
                f"{emoji}【突破确认】{group} {target_sym}：{action}\n\n"
                f"入场参考：{entry:.0f}\n"
                f"止损参考：{stop:.0f}\n"
                f"止盈参考：{tp:.0f}（{rr_take}R）"
            )
        else:
            st.info(
                f"信号仍然有效（冷却中）：{action}｜入场 {entry:.0f}｜止损 {stop:.0f}｜止盈 {tp:.0f}"
            )

st.caption("说明：本模块使用“区间突破 + 连续K次确认”。止损以“跌回突破位”为主，叠加“波动代理(类似ATR)”做保护；止盈按 R 倍给出。")

# ---------------------------
# 结构价差（你原来的）
# ---------------------------
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

st.subheader("结构价差与提示")

s1, s2, s3 = st.columns(3)
for col, name in zip([s1, s2, s3], ["Y-P", "OI-Y", "OI-P"]):
    series = list(st.session_state.hist_spread[name])[-z_win:]
    z = zscore_from_list(series) if len(series) >= 20 else float("nan")
    val = spreads[name]
    with col:
        st.metric(
            label=f"{name}",
            value="-" if not np.isfinite(val) else f"{val:.0f}",
            delta=None if not np.isfinite(z) else f"Z={z:.2f}",
        )

alerts = []
for name in ["Y-P", "OI-Y", "OI-P"]:
    series = list(st.session_state.hist_spread[name])[-z_win:]
    z = zscore_from_list(series)
    if np.isfinite(z) and abs(z) >= z_th:
        direction = "偏高" if z > 0 else "偏低"
        alerts.append((name, z, direction, spreads[name]))

# 去抖：同一告警 60 秒内不重复刷屏
if "last_alert" not in st.session_state:
    st.session_state.last_alert = {}

def should_alert(key: str, cooldown_sec: int = 60) -> bool:
    last_ts = st.session_state.last_alert.get(key, 0)
    if time.time() - last_ts >= cooldown_sec:
        st.session_state.last_alert[key] = time.time()
        return True
    return False

if alerts:
    for name, z, direction, val in alerts:
        key = f"{group}-{name}-{direction}"
        if should_alert(key):
            st.warning(f"⚠️【价差极值提示】{group}  {name} {direction}｜当前 {val:.0f}｜Z={z:.2f}")
else:
    st.success("✅ 当前无价差极值告警（你可以在左侧调整 Z-score 窗口与阈值）")

if show_debug:
    st.write("trading_now(CST):", trading_now)
    st.write("signal target:", target_sym)
    st.write("price_hist_len:", len(prices))
