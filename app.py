import re
import time
from datetime import datetime, timezone, timedelta
from collections import deque

import requests
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------
# 基础配置
# ---------------------------
st.set_page_config(page_title="期货实时提示看板", layout="wide")

TZ_JST = timezone(timedelta(hours=9))
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
# 取行情：Sina hq.sinajs.cn
# 返回格式类似：var hq_str_Y2605="豆油2605,....";
# ---------------------------
def fetch_sina_quotes(symbols: list[str]) -> dict:
    if not symbols:
        return {}
    url = SINA_QUOTE_URL + ",".join(symbols)
    r = requests.get(url, headers=HEADERS, timeout=8)
    r.encoding = "gbk"  # 新浪返回常见为 gbk
    text = r.text

    out = {}
    # 匹配：var hq_str_XXXX="....";
    for m in re.finditer(r'var\s+hq_str_(\w+)\s*=\s*"([^"]*)";', text):
        sym = m.group(1)
        payload = m.group(2).strip()
        if not payload:
            continue
        fields = payload.split(",")
        out[sym] = fields
    return out


def parse_common(fields: list[str]) -> dict:
    """
    字段在不同品种/版本可能略有差异。我们这里做“稳健解析”：
    - name: fields[0]
    - open/high/low/last: 尽量从常见位置取，取不到就 NaN
    - volume/oi: 取不到就 NaN
    """
    def fnum(x):
        try:
            return float(x)
        except:
            return float("nan")

    name = fields[0] if len(fields) > 0 else ""

    # 常见：open=2, high=3, low=4, last=5
    open_ = fnum(fields[2]) if len(fields) > 2 else float("nan")
    high = fnum(fields[3]) if len(fields) > 3 else float("nan")
    low = fnum(fields[4]) if len(fields) > 4 else float("nan")
    last = fnum(fields[5]) if len(fields) > 5 else float("nan")

    # 成交量、持仓量常见在 12/13 或 13/14 一带，存在差异，做兜底：
    volume = float("nan")
    oi = float("nan")
    for idx in [12, 13, 14, 15]:
        if len(fields) > idx and volume != volume:  # NaN check
            v = fnum(fields[idx])
            # 成交量通常很大且为整数；这里不强校验，能转就收
            volume = v
        if len(fields) > idx + 1 and oi != oi:
            o = fnum(fields[idx + 1])
            oi = o

    # 日期/时间（末尾常有 date 或 date,time）
    dt_text = ""
    if len(fields) >= 2:
        # 有些是 ... , 2025-01-06, 14:01:02
        tail = fields[-2:]
        if re.match(r"\d{4}-\d{2}-\d{2}", tail[0]):
            dt_text = " ".join(tail)
        elif re.match(r"\d{4}-\d{2}-\d{2}", fields[-1]):
            dt_text = fields[-1]

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


def zscore(series: list[float]) -> float:
    arr = np.array(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 20:
        return float("nan")
    mu = arr.mean()
    sd = arr.std(ddof=1)
    if sd == 0:
        return float("nan")
    return (arr[-1] - mu) / sd


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📊 期货实时提示看板（2605 / 2609）")

with st.sidebar:
    st.header("参数")
    group = st.selectbox("合约组", ["2605", "2609"], index=0)
    refresh_sec = st.slider("刷新间隔（秒）", 1, 10, 2)
    z_win = st.slider("Z-score 窗口（样本点）", 60, 600, 180, step=30)
    z_th = st.slider("告警阈值 |Z| ≥", 1.0, 3.0, 2.0, step=0.1)
    show_debug = st.checkbox("显示调试信息（可选）", value=False)

symbols_map = CONTRACT_GROUPS[group]
symbols = list(symbols_map.values())

# session_state：保存价差历史，用于 z-score
if "hist" not in st.session_state:
    st.session_state.hist = {
        "Y-P": deque(maxlen=2000),
        "OI-Y": deque(maxlen=2000),
        "OI-P": deque(maxlen=2000),
    }
if "last_alert" not in st.session_state:
    st.session_state.last_alert = {}

# 自动刷新（不写死循环，避免云端卡死）
st.autorefresh(interval=refresh_sec * 1000, key="tick")

# 拉行情
now = datetime.now(TZ_JST).strftime("%Y-%m-%d %H:%M:%S JST")
st.caption(f"更新时间：{now}｜合约组：{group}（{', '.join(symbols)}）")

try:
    raw = fetch_sina_quotes(symbols)
except Exception as e:
    st.error(f"拉取行情失败：{e}")
    st.stop()

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

# 表格
st.subheader("实时行情")
st.dataframe(df, use_container_width=True, hide_index=True)

# 计算价差 & 告警
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

# 更新历史
for name, val in spreads.items():
    if np.isfinite(val):
        st.session_state.hist[name].append(val)

# 展示价差 + zscore
st.subheader("结构价差与提示")
s1, s2, s3 = st.columns(3)
for col, name in zip([s1, s2, s3], ["Y-P", "OI-Y", "OI-P"]):
    series = list(st.session_state.hist[name])[-z_win:]
    z = zscore(series) if len(series) >= 20 else float("nan")
    val = spreads[name]
    with col:
        st.metric(
            label=f"{name}",
            value="-" if not np.isfinite(val) else f"{val:.0f}",
            delta=None if not np.isfinite(z) else f"Z={z:.2f}",
        )

# 告警逻辑：|Z| >= 阈值
alerts = []
for name in ["Y-P", "OI-Y", "OI-P"]:
    series = list(st.session_state.hist[name])[-z_win:]
    z = zscore(series)
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
    st.write("Raw keys:", list(raw.keys()))
