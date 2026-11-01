"""Trang Streamlit mẫu – dùng để kiểm tra deploy. Sau đó thay bằng demo thật của đề tài."""
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Demo mẫu – ELC3008", page_icon="🚀", layout="wide")
st.title("🚀 Demo mẫu – Dự án AI trong kinh doanh")
st.caption("Nếu bạn thấy trang này trên link công khai, việc deploy đã thành công.")


@st.cache_data
def make_data(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"ngày": pd.date_range("2026-01-01", periods=n),
                         "doanh_số": rng.integers(50, 150, n).cumsum()})


with st.sidebar:
    st.header("Tham số")
    n = st.slider("Số ngày", 10, 120, 60)
    seed = st.number_input("Seed", 0, 999, 42)

df = make_data(n, int(seed))
c1, c2 = st.columns(2)
c1.metric("Tổng doanh số (dữ liệu giả)", f"{df['doanh_số'].iloc[-1]:,}")
c2.metric("Số ngày", n)
st.line_chart(df.set_index("ngày"))
with st.expander("Xem dữ liệu"):
    st.dataframe(df, hide_index=True)
