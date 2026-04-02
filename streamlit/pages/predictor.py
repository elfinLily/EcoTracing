# ============================================================
# [역할]
#   - CPU / 메모리 슬라이더로 값 입력받아 실시간 에너지 예측
#   - 공식 기반 계산 결과 표시
#   - 학습된 모델 기반 예측 결과와 비교
#   - 직관적 비교 (LED 전구, 스마트폰 충전 등) 시각화
# ============================================================

from datetime import datetime
import sys
import os
import streamlit as st
from utils.loader import load_config, load_model, load_best_model, load_stacking_models, load_residual_models, load_rf_hourly
from utils.predictor import calc_energy_by_formula, predict_energy_by_model, energy_to_analogy, predict_energy_by_stacking, predict_energy_by_residual, predict_energy_by_rf_hourly

st.set_page_config(page_title="에너지 예측기 | EcoTracing", page_icon="🔋", layout="wide")

config = load_config("config/config.yaml")

st.title("🔋 실시간 에너지 예측기")
st.markdown("CPU / 메모리 사용률과 측정 시간을 입력하면 에너지 소비량을 예측합니다.")
st.divider()

with st.sidebar:
    st.markdown("## ⚙️ 입력값 설정")

    cpu_usage = st.slider(
        label="CPU 사용률 (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=1,
        help="서버의 CPU 사용률을 설정하세요 (0~100%)"
    ) / 100.0   # 0~1 범위로 변환

    memory_usage = st.slider(
        label="메모리 사용률 (%)",
        min_value=0,
        max_value=100,
        value=30,
        step=1,
        help="서버의 메모리 사용률을 설정하세요 (0~100%)"
    ) / 100.0   # 0~1 범위로 변환

    duration_min = st.number_input(
        label="측정 시간 (분)",
        min_value=1,
        max_value=1440,
        value=60,
        step=1,
        help="에너지를 측정할 시간(분)을 입력하세요"
    )
    # duration_sec = duration_min * 60   # 초 단위로 변환

    duration_h = st.number_input(
        label="측정 시간 (시간)",
        min_value=0.1,
        max_value=24.0,
        value=1.0,
        step=0.5,
        help="에너지를 측정할 시간 (시간 단위)"
    )

    # duration_sec = duration_h * 3600

    # hour = st.slider(
    #     label="시간대 (0~23시)",
    #     min_value=0,
    #     max_value=23,
    #     value=12,
    #     step=1,
    #     help="현재 시간대를 설정하세요"
    # )
    
    hour = datetime.now().hour
# ----------------------------------------
# 계산
# ----------------------------------------
formula_result = calc_energy_by_formula(cpu_usage, memory_usage, duration_h, config)
analogy = energy_to_analogy(formula_result["energy_kwh"])

# ----------------------------------------
# 결과 표시 — 핵심 수치
# ----------------------------------------
st.markdown("### 📊 예측 결과 (공식 기반)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="⚡ 전력",
        value=f"{formula_result['power_w']} W",
        help="Power(W) = 200 + (CPU x 300) + (Mem x 50)"
    )
with col2:
    st.metric(
        label="🕐 측정 시간",
        value=f"{formula_result['duration_h']:.4f} h",
    )
with col3:
    st.metric(
        label="🔋 에너지 소비량",
        value=f"{formula_result['energy_kwh']:.6f} kWh",
    )
with col4:
    st.metric(
        label="🌍 탄소 배출량",
        value=f"{formula_result['carbon_kg']:.6f} kg CO₂",
        help=f"emission_factor: {config['carbon']['emission_factor']} kg CO₂/kWh"
    )

st.divider()


# ----------------------------------------
# 직관적 비교 시각화
# 에너지를 일상적인 단위로 표현
# ----------------------------------------
st.markdown("### 💡 이 에너지로 할 수 있는 것")

ana_col1, ana_col2, ana_col3 = st.columns(3)

with ana_col1:
    st.info(f"💡 LED 전구(10W)\n\n**{analogy['led_bulb_hours']}시간** 켤 수 있어요")

with ana_col2:
    st.info(f"📱 스마트폰 충전(12Wh)\n\n**{analogy['smartphone_charge']}회** 충전 가능해요")

with ana_col3:
    st.info(f"💨 CO₂ 배출\n\n**{analogy['co2_grams']}g** CO₂ 배출돼요")

st.divider()

# ----------------------------------------
# 모델 기반 예측 (모델 파일이 있을 때만 표시)
# ----------------------------------------
st.markdown("### 🤖 모델 기반 예측")

try:
    # model = load_best_model(config)
    # rf, lr, meta_model = load_stacking_models(config)
    # rf, lr = load_residual_models(config)
    # model_pred = predict_energy_by_stacking(rf, lr, meta_model, cpu_usage, memory_usage, duration_h)
    # model_pred = predict_energy_by_residual(rf, lr, cpu_usage, memory_usage, duration_h)
    # model_pred = predict_energy_by_model(model, cpu_usage, memory_usage, duration_h)
    # model_pred = predict_energy_by_model(model, cpu_usage, memory_usage, duration_sec, hour)

    rf = load_rf_hourly(config)
    print(config["model"]["model_names"])
    model_pred = predict_energy_by_rf_hourly(rf, cpu_usage, memory_usage, duration_h)
    
    st.success(
        f"**Best Model 예측값**: `{model_pred:.8f} kWh` "
        f"| **공식 계산값**: `{formula_result['energy_kwh']:.8f} kWh`"
    )

    diff = abs(model_pred - formula_result["energy_kwh"])
    st.caption(f"두 값의 차이: {diff:.8f} kWh")

except FileNotFoundError as e:
    # 모델 파일이 없으면 안내 메시지 표시
    st.warning(f"⚠️ 모델 파일이 없어서 공식 기반 계산만 표시했어요.\n\n`{e}`")


# ----------------------------------------
# 사용된 공식 표시
# ----------------------------------------
st.divider()
st.markdown("### 📐 사용된 공식")
st.code(
    f"""
CPU 사용률  : {cpu_usage * 100:.0f}%  →  {cpu_usage}
메모리 사용률: {memory_usage * 100:.0f}%  →  {memory_usage}
측정 시간   : {duration_min}분  →   {formula_result['duration_h']:.6f}h

전력 (W) = 200 + ({cpu_usage} x 300) + ({memory_usage} x 50)
              = 200 + {cpu_usage * 300:.1f} + {memory_usage * 50:.1f}
              = {formula_result['power_w']} W

에너지 (kWh) = {formula_result['power_w']} W x {formula_result['duration_h']:.6f} h / 1000
                     = {formula_result['energy_kwh']} kWh

탄소 (kg CO₂) = {formula_result['energy_kwh']} kWh x {config['carbon']['emission_factor']}
                      = {formula_result['carbon_kg']} kg CO₂
    """,
    language="text"
)