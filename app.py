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

# 时区：显示用JST；交易时段判断用CST（中国时间，避免Cloud误判）
TZ_JST = timezone(timedelta(hours=9))
TZ_CST = timezone(timedelta(hours=8))

SINA_QUOTE_URL = "https://hq.sinajs.cn/list="

# ✅ 新浪期货：必须 nf_ + 小写
CONTRACT_GROUPS = {
    "2605": {"Y": "nf_y2605", "P": "nf_p2605", "OI": "nf_oi2605", "M": "nf_m2605"},
    "2609": {"Y": "nf_y2609", "P": "nf_p2609", "OI": "nf_oi2609", "M": "nf_m2609"},
}

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://finance.sina.com.cn/",
    "Accept": "*/*",
}


# =========================
# 交易时段判断（按中国时间CST）
# =========================
def is_trading_time_cst(dt_cst: datetime) -> bool:
    # 周末不交易（这里不考虑法定节假日）
    if dt_cst.weekday() >= 5:
        return False

    hm = dt_cst.hour * 60 + dt_cst.minute

    def in_range(a, b):
        return a <= hm <= b

    day_1 = in_range(9 * 60, 11 * 60 + 30)
    day_2 = in_range(13 * 60 + 30, 15 * 60)

    # 夜盘：豆油/棕榈油/菜油通常到23:00；豆粕有的可到23:00或更晚
    # 为了稳妥，这里用 21:00-23:00（你需要可再扩）
    night = in_range(21 * 60, 23 * 60)

    return day_1 or day_2 or night


# =========================
# 新浪行情：抓取/解析
# =========================
def fetch_sina_quotes(symbols: list[str]) -> dict:
    if not symbols:
        return {}
    url = SINA_QUOTE_URL + ",".join(symbols)
    r = requests.get(url, headers=HEADERS, timeout=8)
    r.encoding = "gbk"
    text = r.text

    out = {}
    # 支持 nf_xxx / 任意code
    for m in re.finditer(r'var\s+hq_str_(\w+)\s*=\s*"([^"]*)";', text):
        sym = m.group(1)                 # 例如 nf_y2605
        payload = m.group(2).strip()     # 逗号分隔字段
        if payload == "":
            out[sym] = []
        else:
            out[sym] = payload.split(",")
    return out


@st.cache_data(ttl=5, show_spinner=False)
def fetch_sina_quotes_cached(symbols: tuple[str, ...]) -> dict:
    return fetch_sina_quotes(list(symbols))


def parse_nf(fields: list[str]) -> dict:
    """
    nf_ 期货字段在不同品种可能略有差异。
    我们做“尽量稳”的解析：优先常见顺序：
      name, open, prev_close, last, high, low, ...
    """
    def fnum(x):
        try:
            return float(x)
        except Exception:
            return float("nan")

    name = fields[0] if len(fields) > 0 else ""

    # 常见字段位
    open_ = fnum(fields[1]) if len(fields) > 1 else float("nan")
    last  = fnum(fields[3]) if len(fields) > 3 else float("nan")
    high  = fnum(fields[4]) if len(fields) > 4 else float("nan")
    low   = fnum(fields[5]) if len(fields) > 5 else float("nan")

    # 兜底：如果 last 解析不到，但 fields[2]/[1]能用，就尝试换位
    if not np.isfinite(last):
        cand = []
        for idx in [2, 1, 6, 7]:
            if len(fields) > idx:
                v = fnum(fields[idx])
                if np.isfinite(v):
                    cand.append(v)
        if cand:
            last = cand[0]

    return {"name": name, "open": open_, "high": high, "low": low, "last": last}


# =========================
# 统计：Z-score / ATR代理 / 突破确认（模板1）
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
    """没有OHLC，用 |ΔP| 均值近似波动（保守替代）"""
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
    模板1：区间突破 + K次确认
    返回：direction("LONG"/"SHORT"/None), entry, stop, tp, level, info
    """
    series = [p for p in list(prices) if np.isfinite(p)]
    if len(series) < window + confirm_k + 5:
        return None, None, None, None, None, "样本不足（开盘后需要积累一段数据）"

    base = series[: -(confirm_k)]
    recent = series[-confirm_k:]

    if len(base) < window:
        return None, None, None, None, None, "样本不足（base不足）"

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
        level = H
        info = f"突破上沿 H={H:.0f}，连续{confirm_k}次确认站上（buffer={buffer:g}）"
    else:
        stop1 = L
        stop2 = (L + atr_mult_stop * atrp) if np.isfinite(atrp) else stop1
        stop = max(stop1, stop2)
        R = stop - entry
        tp = entry - rr_take * R if R > 0 else float("nan")
        level = L
        info = f"跌破下沿 L={L:.0f}，连续{confirm_k}次确认站下（buffer={buffer:g}）"

    return direction, entry, stop, tp, level, info


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


symbols_map = CONTRACT_GROUPS[group]  # {"Y": "nf_y2605", ...}
symbols = tuple(symbols_map.values())

# =========================
# Session State：历史数据
# =========================
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

# =========================
# 时段与刷新
# =========================
now_cst = datetime.now(TZ_CST)
trading_now = is_trading_time_cst(now_cst)

refresh_sec = refresh_trading if trading_now else refresh_off
st_autorefresh(interval=refresh_sec * 1000, key="tick")

now_jst_str = datetime.now(TZ_JST).strftime("%Y-%m-%d %H:%M:%S JST")
now_cst_str = now_cst.strftime("%Y-%m-%d %H:%M:%S CST")
status = "🟢 交易时段" if trading_now else "⚪️ 非交易时段"

st.caption(
    f"更新时间：{now_jst_str}（{now_cst_str}）｜{status}｜刷新：{refresh_sec}s｜合约组：{group}"
)

# =========================
# 是否请求行情
# =========================
should_fetch = True
if pause_fetch:
    should_fetch = False
elif only_trade_time and (not trading_now):
    should_fetch = False

raw = {}
if should_fetch:
    try:
        raw = fetch_sina_quotes_cached(symbols)
        st.caption("已请求新浪期货行情（带缓存限频）")
    except Exception as e:
        st.error(f"拉取行情失败：{e}")
        st.stop()
else:
    st.info("当前非交易时段/暂停抓取，已停止行情请求")


# =========================
# 解析 DataFrame（实时行情）
# =========================
rows = []
for prod, sym in symbols_map.items():
    fields = raw.get(sym, None)  # sym 例如 nf_y2605
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

# 更新单品种价格历史
for prod, sym in symbols_map.items():
    v = df[df["品种"] == prod]["最新"].values[0]
    if np.isfinite(v):
        if sym not in st.session_state.price_hist:
            st.session_state.price_hist[sym] = deque(maxlen=8000)
        st.session_state.price_hist[sym].append(float(v))


# =========================
# 突破确认模板1
# =========================
st.subheader("单品种交易提示（突破确认模板）")

target_sym = symbols_map[signal_symbol]      # nf_y2605
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
        if direction == "LONG":
            action = "考虑做多"
            emoji = "🚀"
        else:
            action = "考虑做空"
            emoji = "📉"

        # 冷却防刷屏
        if can_emit_signal(group, target_sym, direction, cooldown_sec):
            st.warning(
                f"{emoji}【突破确认】{target_sym.replace('nf_', '').upper()}：{action}\n\n"
                f"入场参考：{entry:.0f}\n"
                f"止损参考：{stop:.0f}\n"
                f"止盈参考：{tp:.0f}（{rr_take}R）"
            )
        else:
            st.info(
                f"信号仍有效（冷却中）：{action}｜入场 {entry:.0f}｜止损 {stop:.0f}｜止盈 {tp:.0f}"
            )

st.caption("说明：模板1=区间突破+连续K次确认。止损以“回到突破位”为主，叠加波动代理保护；止盈按R倍给出。")


# =========================
# 结构价差与Z-score
# =========================
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
        st.metric(
            label=f"{name}",
            value="-" if not np.isfinite(val) else f"{val:.0f}",
            delta=None if not np.isfinite(z) else f"Z={z:.2f}",
        )

def should_alert(key: str, cooldown: int = 60) -> bool:
    last_ts = st.session_state.last_alert_ts.get(key, 0.0)
    now_ts = time.time()
    if now_ts - last_ts >= cooldown:
        st.session_state.last_alert_ts[key] = now_ts
        return True
    return False

alerts = []
for name in ["Y-P", "OI-Y", "OI-P"]:
    series = list(st.session_state.hist_spread[name])[-z_win:]
    z = zscore_from_list(series)
    if np.isfinite(z) and abs(z) >= z_th:
        direction_txt = "偏高" if z > 0 else "偏低"
        alerts.append((name, z, direction_txt, spreads[name]))

if alerts:
    for name, z, direction_txt, val in alerts:
        key = f"{group}-{name}-{direction_txt}"
        if should_alert(key, cooldown=60):
            st.warning(f"⚠️【价差极值提示】{group}  {name} {direction_txt}｜当前 {val:.0f}｜Z={z:.2f}")
else:
    st.success("✅ 当前无价差极值告警（可在左侧调整窗口与阈值）")


# =========================
# 调试信息（可选）
# =========================
if show_debug:
    st.divider()
    st.subheader("调试信息")
    st.write("Cloud 判断交易时段（CST）：", trading_now)
    st.write("should_fetch：", should_fetch)
    st.write("symbols：", symbols)
    # 输出一个品种的原始字段长度，便于定位解析问题
    sample_sym = symbols_map["Y"]
    st.write("样例 raw key：", sample_sym)
    st.write("样例 fields len：", len(raw.get(sample_sym, [])))
    st.write("price_hist_len：", len(st.session_state.price_hist.get(sample_sym, [])))
