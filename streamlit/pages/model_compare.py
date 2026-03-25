# ============================================================
# [역할]
#   - Phase 1에서 비교한 4개 모델 성능 결과 시각화
#   - RMSE / MAE / R² / MAPE / 학습시간 비교 차트
#   - phase1_metrics.csv 기반으로 렌더링
#   - 모델 파일이 없으면 하드코딩된 Phase 1 결과로 폴백
# ============================================================
import sys
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.loader import load_config, load_phase1_metrics, load_phase1_results

st.set_page_config(page_title="모델 성능 비교 | EcoTracing", page_icon="🏆", layout="wide")

config = load_config("config/config.yaml")

st.title("🏆 모델 성능 비교")
st.markdown("Phase 1에서 학습한 4개 모델의 성능 지표 비교")
st.divider()

try:
    results = load_phase1_results(config)

    # long format -> wide format 변환
    # metric 컬럼 값을 소문자로 변환 후 pivot
    models_list = results["models"]

    df_metrics = pd.DataFrame([
        {
            "model": m["model"],
            "rmse": m["rmse"],
            "mae": m["mae"],
            "r2": m["r2"],
            "mape": m["mape"],
            "train_time_s": m.get("train_time_s", 0),
        }
        for m in models_list
    ])

except FileNotFoundError:
    st.info("phase1_metrics.csv가 없어서 Phase 1 실측 결과를 표시합니다.")

    df_metrics = pd.DataFrame([
        {
            "model": "RandomForest",
            "rmse": 1.38e-5,
            "mae": 8.5e-6,
            "r2": 0.99999,
            "mape": 0.07,
            "train_time_s": 470
        },
        {
            "model": "LightGBM",
            "rmse": 3.42e-5,
            "mae": 2.1e-5,
            "r2": 0.99997,
            "mape": 0.15,
            "train_time_s": 42
        },
        {
            "model": "CatBoost",
            "rmse": 4.31e-5,
            "mae": 2.8e-5,
            "r2": 0.99995,
            "mape": 0.79,
            "train_time_s": 78
        },
        {
            "model": "XGBoost",
            "rmse": 5.67e-5,
            "mae": 3.4e-5,
            "r2": 0.99992,
            "mape": 2.78,
            "train_time_s": 66
        },
    ])

st.markdown("### 📋 성능 지표 테이블")

# st.dataframe(
#     df_metrics.style.highlight_min(
#         subset=["rmse", "mae", "mape"],
#         color="#1a472a"   # 최솟값(좋은 값) 강조 — 다크모드에서 잘 보이는 초록
#     ).highlight_max(
#         subset=["r2"],
#         color="#1a472a"   # R²는 최댓값이 좋음
#     ),
#     use_container_width=True
# )
styled_df = (
    df_metrics.style
    .highlight_min(subset=["rmse", "mae", "mape"], color="#6CC08A")
    .highlight_max(subset=["r2"], color="#b0eec6")
)
st.dataframe(styled_df, use_container_width=True)

st.divider()

# ----------------------------------------
# RMSE 비교 바 차트
# 낮을수록 좋음
# ----------------------------------------
st.markdown("### 📉 RMSE 비교 (낮을수록 좋음)")

fig_rmse = px.bar(
    df_metrics,
    x="model",
    y="rmse",
    title="모델별 RMSE",
    color="model",
    template="plotly_dark",
    color_discrete_sequence=["#00d4ff", "#7bed9f", "#ffa502", "#ff4757"]
)
st.plotly_chart(fig_rmse, use_container_width=True)

# ----------------------------------------
# R² 비교 바 차트
# 높을수록 좋음
# ----------------------------------------
st.markdown("### 📈 R² 비교 (높을수록 좋음)")

fig_r2 = px.bar(
    df_metrics,
    x="model",
    y="r2",
    title="모델별 R²",
    color="model",
    template="plotly_dark",
    color_discrete_sequence=["#00d4ff", "#7bed9f", "#ffa502", "#ff4757"]
)
fig_r2.update_layout(yaxis_range=[0.9999, 1.0])  # 차이 잘 보이게 범위 좁힘
st.plotly_chart(fig_r2, use_container_width=True)

# ----------------------------------------
# MAPE vs 학습시간 산점도
# 정확도와 속도의 트레이드오프 시각화
# ----------------------------------------
st.markdown("### ⚖️ MAPE vs 학습시간 (정확도 vs 속도 트레이드오프)")

fig_tradeoff = px.scatter(
    df_metrics,
    x="train_time_s",
    y="mape",
    text="model",
    title="학습 시간(초) vs MAPE (%) — 왼쪽 아래가 이상적",
    labels={"train_time_s": "학습 시간 (초)", "mape": "MAPE (%)"},
    template="plotly_dark",
    color="model",
    size=[20] * len(df_metrics),
    color_discrete_sequence=["#00d4ff", "#7bed9f", "#ffa502", "#ff4757"]
)
fig_tradeoff.update_traces(textposition="top center")
st.plotly_chart(fig_tradeoff, use_container_width=True)

# ----------------------------------------
# 최종 선택 모델 하이라이트
# ----------------------------------------
st.divider()
st.markdown("### 🥇 최종 선택 모델")

col1, col2 = st.columns(2)

with col1:
    st.success(
        """
        **RandomForest** — 성능 1위
        - R² : 0.99999
        - MAPE : 0.07%
        - RMSE : 1.38e-05
        """
    )

with col2:
    st.info(
        """
        **LightGBM** — 속도 vs 성능 균형
        - R² : 0.99997
        - MAPE : 0.15%
        - 학습시간 : 42초 (RandomForest의 1/11)
        """
     )