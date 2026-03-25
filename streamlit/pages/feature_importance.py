# ============================================================
# [역할]
#   - Phase 1 모델의 피처 중요도(Feature Importance) 시각화
#   - duration > cpu > memory > hour 순서 확인
#   - 각 피처가 에너지 예측에 미치는 영향 해석 제공
#   - phase1_results.json 기반으로 렌더링
# ============================================================
import sys
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.loader import load_config, load_phase1_results, load_feature_importance

st.set_page_config(page_title="피처 중요도 | EcoTracing", page_icon="🔍", layout="wide")

config = load_config("config/config.yaml")

st.title("🔍 피처 중요도 (Feature Importance)")
st.markdown("Phase 1 모델에서 에너지 예측에 가장 영향을 미치는 피처 분석")
st.divider()

try:
    # config 기반으로 feature_importance_comparison.csv 로드
    df_fi = load_feature_importance(config)
    # df_fi columns: feature, LightGBM, XGBoost, RandomForest, CatBoost

except FileNotFoundError as e:
    st.warning(f"결과 파일을 찾을 수 없어요: {e}")
    st.stop()

# 중요도 계산: 4개 모델 평균으로 대표값 생성
model_cols = ["LightGBM", "XGBoost", "RandomForest", "CatBoost"]
df_fi["importance"] = df_fi[model_cols].mean(axis=1)
df_fi = df_fi.sort_values("importance", ascending=False)

total_importance = df_fi["importance"].sum()
df_fi["importance_pct"] = (df_fi["importance"] / total_importance * 100).round(2)

st.markdown("### 📋 피처 중요도 수치")

for _, row in df_fi.iterrows():
    st.metric(
        label=f"{'🥇' if row['feature'] == df_fi.iloc[0]['feature'] else '  '} {row['feature']}",
        value=f"{row['importance_pct']}%",
        help=f"raw importance: {row['importance']}"
    )

st.divider()

st.markdown("### 📊 피처 중요도 차트")

fig_bar = px.bar(
    df_fi,
    x="importance_pct",
    y="feature",
    orientation="h",        # 가로 바 차트
    title="Feature Importance (%)",
    labels={"importance_pct": "중요도 (%)", "feature": "피처"},
    template="plotly_dark",
    color="importance_pct",
    color_continuous_scale="Teal",   # 다크모드에서 잘 보이는 색상
    text="importance_pct"
)
fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig_bar.update_layout(yaxis=dict(autorange="reversed"))   # 중요도 높은 순 정렬
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("### 🥧 피처 중요도 비율")

fig_pie = px.pie(
    df_fi,
    names="feature",
    values="importance_pct",
    title="Feature Importance 비율",
    template="plotly_dark",
    color_discrete_sequence=["#00d4ff", "#7bed9f", "#ffa502", "#ff4757"]
)
fig_pie.update_traces(textinfo="percent+label")
st.plotly_chart(fig_pie, use_container_width=True)

st.divider()
st.markdown("### 💡 피처별 해석")

st.markdown("""
| 피처 | 중요도 | 해석 |
|------|--------|------|
| **duration** | 1위 (~51%) | 에너지 = 전력 x **시간** 이므로 가장 큰 영향 |
| **cpu** | 2위 (~41%) | 공식에서 CPU 계수(300W)가 메모리(50W)보다 6배 커서 영향 큼 |
| **memory** | 3위 (~8%) | CPU보다 낮지만 무시할 수 없는 수준 |
| **hour** | 4위 (~<1%) | 현재 공식에 시간대 변수가 없어서 낮음 (시간대 패턴은 존재) |
""")

st.info(
    "💡 **인사이트**: `hour`의 중요도가 낮은 이유는 현재 에너지 공식에 "
    "시간대 변수가 포함되지 않아서입니다. "
    "Phase 2에서 지역별 탄소 집약도(Carbon Intensity)는 시간대에 따라 달라지므로 "
    "중요도가 올라갈 수 있어요."
)