# ============================================================
# [역할]
#   - Phase 1에서 수행한 EDA 결과를 시각화
#   - CPU / 메모리 분포, 시간대별 사용량 패턴 차트 표시
#   - processed 데이터를 샘플링해서 직접 시각화
#   - 다크모드 친화적 색상 적용 (plotly dark 테마)
# ============================================================

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.loader import load_config

st.set_page_config(page_title="데이터 탐색 | EcoTracing", page_icon="📊", layout="wide")

config = load_config("config/config.yaml")
st.title("📊 데이터 탐색 (EDA)")
st.markdown("Google Cluster Trace 2019 — `instance_usage` 데이터 분석 결과")
st.divider()

@st.cache_data   # 동일 데이터 반복 로드 방지 (캐싱)
def load_sample_data(config: dict, sample_n: int = 50000) -> pd.DataFrame:
    """
    전처리된 parquet 파일에서 샘플 데이터 로드

    Args:
        config  : config 딕셔너리
        sample_n: 샘플링할 행 수 (기본 5만 행)

    Returns:
        샘플 DataFrame
    """
    processed_path = os.path.join(
        config["paths"]["processed_data"],
        "instance_usage_full_processed.parquet"
    )

    if not os.path.exists(processed_path):
        return None

    df = pd.read_parquet(processed_path)

    # 전체 데이터가 sample_n보다 많으면 샘플링
    if len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=config["model"]["random_state"])

    return df

df = load_sample_data(config)

if df is None:
    st.warning("⚠️ 전처리된 데이터 파일이 없어요. Colab에서 03_preprocessing.ipynb를 먼저 실행해주세요.")
    st.stop()

st.markdown("### 📋 데이터 기본 정보")

info_col1, info_col2, info_col3, info_col4 = st.columns(4)

with info_col1:
    st.metric("총 행 수 (샘플)", f"{len(df):,}")
with info_col2:
    st.metric("CPU 평균", f"{df['cpu'].mean():.4f}")
with info_col3:
    st.metric("메모리 평균", f"{df['memory'].mean():.4f}")
with info_col4:
    st.metric("에너지 평균 (kWh)", f"{df['energy_kwh'].mean():.6f}")

st.divider()

st.markdown("### ⚙️ CPU 사용률 분포")

fig_cpu = px.histogram(
    df,
    x="cpu",
    nbins=50,
    title="CPU 사용률 분포",
    labels={"cpu": "CPU 사용률 (0~1)"},
    template="plotly_dark",   # 다크모드 테마
    color_discrete_sequence=["#00d4ff"]  # 밝은 청록 — 다크모드에서 잘 보임
)
fig_cpu.update_layout(bargap=0.05)
st.plotly_chart(fig_cpu, use_container_width=True)

st.markdown("### 💾 메모리 사용률 분포")

fig_mem = px.histogram(
    df,
    x="memory",
    nbins=50,
    title="메모리 사용률 분포",
    labels={"memory": "메모리 사용률 (0~1)"},
    template="plotly_dark",
    color_discrete_sequence=["#7bed9f"]  # 밝은 초록
)
fig_mem.update_layout(bargap=0.05)
st.plotly_chart(fig_mem, use_container_width=True)

st.markdown("### 🕐 시간대별 평균 사용량")

hourly = df.groupby("hour")[["cpu", "memory"]].mean().reset_index()

fig_hourly = go.Figure()

fig_hourly.add_trace(go.Scatter(
    x=hourly["hour"],
    y=hourly["cpu"],
    mode="lines+markers",
    name="CPU 평균",
    line=dict(color="#00d4ff", width=2)
))

fig_hourly.add_trace(go.Scatter(
    x=hourly["hour"],
    y=hourly["memory"],
    mode="lines+markers",
    name="메모리 평균",
    line=dict(color="#7bed9f", width=2)
))

fig_hourly.update_layout(
    title="시간대별 평균 CPU / 메모리 사용률",
    xaxis_title="시간 (0~23)",
    yaxis_title="평균 사용률 (0~1)",
    template="plotly_dark",
    legend=dict(orientation="h", y=1.1)
)

st.plotly_chart(fig_hourly, use_container_width=True)

# ----------------------------------------
# CPU vs Memory 산점도
# 두 변수 간 관계 파악
# ----------------------------------------
st.markdown("### 🔗 CPU vs 메모리 사용률 상관관계")

# 산점도는 10,000개만 샘플링 (렌더링 속도)
scatter_df = df.sample(n=min(10000, len(df)), random_state=42)

fig_scatter = px.scatter(
    scatter_df,
    x="cpu",
    y="memory",
    color="energy_kwh",
    title="CPU vs 메모리 (색상: 에너지 소비량)",
    labels={"cpu": "CPU 사용률", "memory": "메모리 사용률", "energy_kwh": "에너지(kWh)"},
    template="plotly_dark",
    color_continuous_scale="Viridis",
    opacity=0.5
)
st.plotly_chart(fig_scatter, use_container_width=True)