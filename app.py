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

TZ_JST = timezone(timedelta(hours=9))  # 你在日本，界面显示用 JST
TZ_CST = timezone(timedelta(hours=8))  # 交易时段判断用中国时间 CST

SINA_QUOTE_URL = "https://hq.sinajs.cn/list="  # 多个用逗号拼接

# 固定合约：2605 / 2609
CONTRACT_GROUPS = {
    "2605": {"Y": "Y2605", "P": "P2605", "OI": "OI2605", "M": "M2605"},
    "2609": {"Y": "Y2609", "P": "P2609", "OI": "OI2609", "M": "M2609"},
}

# 为了尽量避免 403，带上常见 headers（新浪接口有时会校验来源/UA）
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122 Safari/537.36",
    "Referer": "https://finance.sina.com.cn/",
    "Accept": "*/*",
}


# ---------------------------
# 交易时段判断（DCE 常见：日盘 + 夜盘）
# 注意：不同合约/节假日会变，这里做“安全保守版”
# - 周一到周五
# - 日盘：09:00-11:30, 13:30-15:00
# - 夜盘：21:00-23:00（保守写到23:00；有些品种到23:00/23:30/01:00）
# 你可以后续再精细化到具体品种
# ---------------------------
def is_trading_time_cst(dt_cst: datetime) -> bool:
    # 周末直接 false
    if dt_cst.weekday() >= 5:
        return False

    hm = dt_cst.hour * 60 + dt_cst.minute

    def in_range(start_hm, end_hm):
        return start_hm <= hm <= end_hm

    # 日盘
    day_1 = in_range(9 * 60, 11 * 60 + 30)
    day_2 = in_range(13 * 60 + 30, 15 * 60)

    # 夜盘（保守：21:00-23:00）
    night = in_range(21 * 60, 23 * 60)

    return day_1 or day_2 or night


# ---------------------------
# 取行情：Sina hq.sinajs.cn
# 返回格式类似：var hq_str_Y2605="豆油2605,....";
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

    volume = float("nan")
    oi = float("nan")
    for idx in [12, 13, 14, 15]:
        if len(fields) > idx and (volume != volume):
            volume = fnum(fields[idx])
        if len(fields) > idx + 1 and (oi != oi):
            oi = fnum(fields[idx + 1])

    dt_text = ""
    if len(fields) >= 2:
        tail = fields[-2:]
        if re.match(r"\d{4}-\d{2}-\d{2}", tail[0]):
            dt_text = " ".join(tail)

    return {
        "name": name,
        "open": open_,
        "high": high,
        "low": low,
        "last": last,
        "volume": volume,
        "oi": oi,
        "dt_text": dt_text,
    }


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
# 缓存：限制访问频率（关键：避免一直刷新浪）
# ttl_seconds 内多次调用只会真的请求一次
# ---------------------------
@st.cache_data(ttl=5, show_spinner=False)
def fetch_sina_quotes_cached(symbols: tuple[str, ...]) -> dict:
    return fetch_sina_quotes(list(symbols))


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

    z_win = st.slider("Z-score 窗口（样本点）", 60, 600, 180, step=30)
    z_th = st.slider("告警阈值 |Z| ≥", 1.0, 3.0, 2.0, step=0.1)
    show_debug = st.checkbox("显示调试信息（可选）", value=False)

symbols_map = CONTRACT_GROUPS[group]
symbols = tuple(symbols_map.values())

# session_state：保存价差历史，用于 z-score
if "hist" not in st.session_state:
    st.session_state.hist = {
        "Y-P": deque(maxlen=2000),
        "OI-Y": deque(maxlen=2000),
        "OI-P": deque(maxlen=2000),
    }
if "last_alert" not in st.session_state:
    st.session_state.last_alert = {}

# 判断当前是否交易时段（用 CST）
now_cst = datetime.now(TZ_CST)
trading_now = is_trading_time_cst(now_cst)

# 决定刷新间隔
refresh_sec = refresh_trading if trading_now else refresh_off

# 自动刷新
st_autorefresh(interval=refresh_sec * 1000, key="tick")

now_jst = datetime.now(TZ_JST).strftime("%Y-%m-%d %H:%M:%S JST")
now_cst_str = now_cst.strftime("%Y-%m-%d %H:%M:%S CST")
status = "🟢 交易时段" if trading_now else "⚪️ 非交易时段"
st.caption(f"更新时间：{now_jst}（{now_cst_str}）｜{status}｜当前刷新：{refresh_sec}s｜合约组：{group}")

# 是否请求行情
should_fetch = True
if pause_fetch:
    should_fetch = False
elif only_trade_time and (not trading_now):
    should_fetch = False

raw = {}
fetch_note = ""
if should_fetch:
    try:
        # 缓存 + 限频：ttl=5 秒（你可以按需改大，比如 8~10）
        raw = fetch_sina_quotes_cached(symbols)
        fetch_note = "已请求新浪行情（带缓存限频）"
    except Exception as e:
        st.error(f"拉取行情失败：{e}")
        st.stop()
else:
    fetch_note = "当前未请求行情（暂停或非交易时段）"

st.caption(fetch_note)

# 解析
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
st.dataframe(df, use_container_width=True, hide_index=True)

# 计算价差（如果没行情则是 NaN）
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

# 更新历史（只有在“有有效价差”时才追加）
for name, val in spreads.items():
    if np.isfinite(val):
        st.session_state.hist[name].append(val)

st.subheader("结构价差与提示")

s1, s2, s3 = st.columns(3)
for col, name in zip([s1, s2, s3], ["Y-P", "OI-Y", "OI-P"]):
    series = list(st.session_state.hist[name])[-z_win:]
    z = zscore_from_list(series) if len(series) >= 20 else float("nan")
    val = spreads[name]
    with col:
        st.metric(
            label=f"{name}",
            value="-" if not np.isfinite(val) else f"{val:.0f}",
            delta=None if not np.isfinite(z) else f"Z={z:.2f}",
        )

# 告警：|Z| >= 阈值
alerts = []
for name in ["Y-P", "OI-Y", "OI-P"]:
    series = list(st.session_state.hist[name])[-z_win:]
    z = zscore_from_list(series)
    if np.isfinite(z) and abs(z) >= z_th:
        direction = "偏高" if z > 0 else "偏低"
        alerts.append((name, z, direction, spreads[name]))

# 去抖：同一告警 60 秒内不重复刷屏
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
    st.write("symbols:", symbols)
    st.write("raw keys:", list(raw.keys()))
