import streamlit as st
from utils.loader import load_config

st.set_page_config(
    page_title="EcoTracing",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

config = load_config("config/config.yaml")

with st.sidebar:
    st.markdown("## 🌿 EcoTracing")
    st.markdown("데이터센터 에너지 & 탄소 예측 대시보드")
    st.divider()
    st.markdown(
        """
        **📄 Pages**
        - 🔋 에너지 예측기
        - 📊 데이터 탐색 (EDA)
        - 🏆 모델 성능 비교
        - 🔍 피처 중요도
        """
    )
    st.divider()
    st.caption(f"Project: `{config['project_name']}`")
    st.caption("Phase 1.5 — Streamlit Dashboard")

st.title("🌿 EcoTracing")
st.subheader("데이터센터 에너지 소비 & 탄소 배출 예측 대시보드")

st.divider()

# 프로젝트 개요 카드
col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        """
        **⚡ Phase 1 (완료)**
        CPU / 메모리 사용량으로
        에너지 소비량(kWh) 예측
        """
    )

with col2:
    st.warning(
        """
        **🌍 Phase 2 (예정)**
        에너지(kWh) →
        탄소 배출량(kg CO₂) 변환
        """
    )

with col3:
    st.error(
        """
        **🤖 Phase 3 (예정)**
        AI 학습/추론의
        탄소 발자국 분석
        """
    )

st.divider()

# 에너지 공식 표시
st.markdown("### ⚡ 에너지 계산 공식")
st.code(
    """
전력 (W) = 200 + (CPU_usage x 300) + (Memory_usage x 50)
에너지 (kWh) = 전력 (W) x 측정시간 (h) / 1000
    """,
    language="text"
)

# Phase 1 핵심 결과 요약
st.markdown("### 📈 Phase 1 핵심 결과")

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric(label="Best Model", value="RandomForest")
with metric_col2:
    st.metric(label="R²", value="0.99999")
with metric_col3:
    st.metric(label="MAPE", value="0.07%")
with metric_col4:
    st.metric(label="학습 데이터", value="약 2천만 행")

st.divider()
st.caption("왼쪽 사이드바에서 각 페이지로 이동하세요 👈")