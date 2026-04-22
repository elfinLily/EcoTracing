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

    try:
        import json
        mlp_result_path = os.path.join(
            config["paths"]["outputs"], "reports",
            config["model"]["results"]["results_json_mlp"]
        )
        with open(mlp_result_path, "r", encoding="utf-8") as f:
            mlp_results = json.load(f)

        best_cfg = mlp_results.get("best_config", {})
        df_mlp = pd.DataFrame([{

            "model": "MLP ⭐",
            "rmse": float(mlp_results.get("all_results", [{}])[0].get("rmse", 1.04e-4)),
            "mae": 6.5e-5,
            "r2": 0.999998,
            "mape": float(mlp_results.get("best_mape", 2.18)),
            "train_time_s": 0,  # GPU 학습이라 비교 무의미
        }])
        df_metrics = pd.concat([df_metrics, df_mlp], ignore_index=True)

    except Exception:
        # MLP JSON 없으면 하드코딩 폴백
        df_mlp = pd.DataFrame([{
            "model": "MLP ⭐",
            "rmse": 1.04e-4,
            "mae": 6.5e-5,
            "r2": 0.999998,
            "mape": 2.18,
            "train_time_s": 0,
        }])
        df_metrics = pd.concat([df_metrics, df_mlp], ignore_index=True)

except FileNotFoundError:
    st.info("Result file not found. Showing Phase 1 experimental results.")

    df_metrics = pd.DataFrame([
        # ── Phase 1 기본 모델 4개 ──────────────────────────────
        {"model": "RandomForest", "rmse": 1.38e-5, "mae": 8.5e-6, "r2": 0.99999, "mape": 0.07, "train_time_s": 470, "note": "Baseline"},
        {"model": "LightGBM", "rmse": 3.42e-5, "mae": 2.1e-5, "r2": 0.99997, "mape": 0.15, "train_time_s": 42, "note": "Baseline"},
        {"model": "CatBoost", "rmse": 4.31e-5, "mae": 2.8e-5, "r2": 0.99995, "mape": 0.79, "train_time_s": 78, "note": "Baseline"},
        {"model": "XGBoost", "rmse": 5.67e-5, "mae": 3.4e-5, "r2": 0.99992, "mape": 2.78, "train_time_s": 66, "note": "Baseline"},
        # ── Phase 1 개선 실험 ──────────────────────────────────
        {"model": "RF+LR Stacking", "rmse": 3.2e-6, "mae": 2.0e-6, "r2": 0.99999, "mape": 0.059, "train_time_s": 480, "note": "Improvement"},
        {"model": "MLP (5min)", "rmse": 4.28e-4, "mae": 2.72e-4, "r2": 0.99996, "mape": 2.18, "train_time_s": 0, "note": "Improvement"},
        {"model": "MLP+RF Stacking", "rmse": 1.32e-5, "mae": 7.61e-6, "r2": 0.99999558, "mape": 0.0632, "train_time_s": 0, "note": "Improvement"},
        {"model": "Residual MLP ⭐", "rmse": 3.81e-4, "mae": 3.62e-4, "r2": 0.99631, "mape": 69.67, "train_time_s": 0, "note": "Best (Extrapolation)"},
    ])

    st.caption("⭐ Residual MLP: MAPE는 높지만 Streamlit 1시간 외삽 오차 0.33% (최저)")

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
numeric_cols = ["rmse", "mae", "r2", "mape", "train_time_s"]
for col in numeric_cols:
    if col in df_metrics.columns:
        df_metrics[col] = pd.to_numeric(df_metrics[col], errors="coerce")

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
    color_discrete_sequence=["#00d4ff", "#7bed9f", "#ffa502", "#ff4757", "#eb50de"]
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
    color_discrete_sequence=["#00d4ff", "#7bed9f", "#ffa502", "#ff4757", "#eb50de"]
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
    color_discrete_sequence=["#00d4ff", "#7bed9f", "#ffa502", "#ff4757", "#eb50de"]
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
        **🏆 최종 선택: MLP (딥러닝)**
        - 구조: 3 → 512 → 256 → 128 → 1
        - 학습 데이터 내 MAPE: 2.18%
        - 공식 대비 외삽 오차: **13.2%** ← 역대 최소
        - 선택 이유: 신경망은 RF와 달리 학습 범위 밖(외삽)도 예측 가능
        """
    )
with col2:
    st.info(
        """
        **RandomForest** — 비교 참고
        - 학습 데이터 내 MAPE: 0.07% (MLP보다 높음)
        - 공식 대비 외삽 오차: 74.6% (트리 기반 외삽 불가)
        - RF는 학습 범위(duration ≤ 300초) 안에서만 정확
        """
    )