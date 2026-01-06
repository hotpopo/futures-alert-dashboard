import streamlit as st
import time

st.set_page_config(page_title="期货实时提示看板", layout="wide")

st.title("📊 期货实时提示看板（测试版）")

st.markdown("用于后续接入：豆油 / 棕榈油 / 菜油 / 豆粕 实时提示")

placeholder = st.empty()

for i in range(5):
    with placeholder.container():
        st.metric("示例指标", i)
        st.info("这是测试页面，用于确认 Streamlit 能正常运行")
    time.sleep(1)