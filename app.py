import time
import math
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, time as dtime

# =========================
# 页面设置
# =========================
st.set_page_config(
    page_title="期货实时提示看板",
    layout="wide"
)

st.title("📊 期货实时提示看板（2605 / 2609）")

# =========================
# 合约配置（⚠️ 新浪期货必须 nf_ + 小写）
# =========================
CONTRACT_GROUPS = {
    "2605": {
        "Y": "nf_y2605",
        "P": "nf_p2605",
        "OI": "nf_oi2605",
        "M": "nf_m2605",
    },
    "2609": {
        "Y": "nf_y2609",
        "P": "nf_p2609",
        "OI": "nf_oi2609",
        "M": "nf_m2609",
    }
}

# =========================
# 工具函数
# =========================
def is_trading_time():
    """国内商品期货常规交易时间（不含夜盘细分）"""
    now = datetime.now().time()
    sessions = [
        (dtime(9, 0), dtime(11, 30)),
        (dtime(13, 30), dtime(15, 0)),
        (dtime(21, 0), dtime(23, 59)),
        (dtime(0, 0), dtime(2, 30)),
    ]
    return any(start <= now <= end for start, end in sessions)


def fetch_sina_quotes(codes: list[str]) -> dict:
    """从新浪获取行情"""
    url = "https://hq.sinajs.cn/list=" + ",".join(codes)
    headers = {"Referer": "https://finance.sina.com.cn"}
    r = requests.get(url, headers=headers, timeout=5)
    r.encoding = "gbk"
    data = {}
    for line in r.text.splitlines():
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        symbol = key.split("_")[-1]
        raw = val.strip().strip('";')
        data[symbol] = raw
    return data


def parse_common(fields: list[str]) -> dict:
    """通用期货字段解析（兼容 nf_）"""

    def fnum(x):
        try:
            return float(x)
        except Exception:
            return float("nan")

    name = fields[0] if len(fields) > 0 else ""
    open_ = fnum(fields[1]) if len(fields) > 1 else math.nan
    last = fnum(fields[3]) if len(fields) > 3 else math.nan
    high = fnum(fields[4]) if len(fields) > 4 else math.nan
    low = fnum(fields[5]) if len(fields) > 5 else math.nan

    return {
        "name": name,
        "open": open_,
        "last": last,
        "high": high,
        "low": low,
    }


# =========================
# Sidebar 参数
# =========================
with st.sidebar:
    st.header("参数")

    contract_group = st.selectbox("合约组", ["2605", "2609"])

    refresh_trade = st.slider("交易时段刷新间隔（秒）", 1, 10, 2)
    refresh_idle = st.slider("非交易时段刷新间隔（秒）", 30, 300, 120)

    only_trade = st.checkbox("仅在交易时段请求行情（推荐）", value=True)

    st.divider()
    st.subheader("突破确认信号（模板1）")

    signal_symbol = st.selectbox("信号品种", ["Y", "P", "OI", "M"])
    lookback_n = st.slider("区间窗口 N（样本点）", 60, 300, 180)
    confirm_k = st.slider("确认次数 K", 2, 5, 3)

# =========================
# 行情获取
# =========================
codes = list(CONTRACT_GROUPS[contract_group].values())
symbol_map = {v.split("_")[-1]: k for k, v in CONTRACT_GROUPS[contract_group].items()}

now = datetime.now()
trade_flag = is_trading_time()

if only_trade and not trade_flag:
    st.info("⏸ 当前非交易时段，暂停行情请求")
    time.sleep(refresh_idle)
    st.stop()

raw = fetch_sina_quotes(codes)

rows = []
for raw_code, raw_text in raw.items():
    if not raw_text:
        rows.append({
            "品种": symbol_map.get(raw_code.upper(), raw_code),
            "合约": raw_code.upper(),
            "最新": None,
            "今开": None,
            "最高": None,
            "最低": None,
        })
        continue

    fields = raw_text.split(",")
    parsed = parse_common(fields)

    rows.append({
        "品种": symbol_map.get(raw_code.upper(), raw_code),
        "合约": raw_code.upper(),
        "最新": parsed["last"],
        "今开": parsed["open"],
        "最高": parsed["high"],
        "最低": parsed["low"],
    })

df = pd.DataFrame(rows)

# =========================
# 顶部状态
# =========================
st.caption(
    f"更新时间：{now:%Y-%m-%d %H:%M:%S} ｜ "
    f"{'🟢 交易时段' if trade_flag else '⚪ 非交易时段'} ｜ "
    f"刷新：{refresh_trade if trade_flag else refresh_idle}s ｜ 合约组：{contract_group}"
)

# =========================
# 实时行情表
# =========================
st.subheader("实时行情")
st.dataframe(df, width="stretch", hide_index=True)

# =========================
# 突破确认信号（模板1）
# =========================
st.subheader("单品种交易提示（突破确认模板）")

target_row = df[df["品种"] == signal_symbol]
if target_row.empty or pd.isna(target_row.iloc[0]["最新"]):
    st.warning("暂无有效行情数据")
else:
    price = float(target_row.iloc[0]["最新"])

    hist_key = f"hist_{signal_symbol}_{contract_group}"
    history = st.session_state.get(hist_key, [])
    history.append(price)
    history = history[-lookback_n:]
    st.session_state[hist_key] = history

    if len(history) < lookback_n:
        st.info("样本不足（开盘后需要积累一段数据）")
    else:
        high_n = max(history[:-1])
        above = sum(1 for p in history[-confirm_k:] if p > high_n)

        if above == confirm_k:
            stop = high_n
            risk = price - stop
            target = price + 2 * risk if risk > 0 else None

            st.success(
                f"🚀 突破确认 · 做多\n\n"
                f"标的：{signal_symbol}{contract_group}\n\n"
                f"入场参考：{price:.2f}\n"
                f"止损：{stop:.2f}\n"
                f"目标：{target:.2f}" if target else "目标待确认"
            )
        else:
            st.info("暂未触发突破确认信号")

# =========================
# 自动刷新
# =========================
time.sleep(refresh_trade if trade_flag else refresh_idle)
st.experimental_rerun()
