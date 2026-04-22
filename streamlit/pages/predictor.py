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
from utils.loader import load_config
from utils.predictor import calc_energy_by_formula, energy_to_analogy
import torch
import torch.nn as nn
import numpy as np
import pickle
import streamlit as st

st.set_page_config(page_title="에너지 예측기 | EcoTracing", page_icon="🔋", layout="wide")

config = load_config("config/config.yaml")

st.title("🔋 실시간 에너지 예측기")
st.markdown("CPU / 메모리 사용률과 측정 시간을 입력하면 에너지 소비량을 예측합니다.")
st.divider()

with st.sidebar:
    st.markdown("## ⚙️  Input Settings")

    cpu_usage = st.slider(
        label="CPU Usage (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=1,
        help="Set server CPU usage (0~100%)"
    ) / 100.0   # 0~1 범위로 변환

    memory_usage = st.slider(
        label="Memory Usage (%)",
        min_value=0,
        max_value=100,
        value=30,
        step=1,
        help="Set server memory usage (0~100%)"
    ) / 100.0   # 0~1 범위로 변환

    duration_h = st.number_input(
            label="Duration (hours)",
            min_value=0.1,
            max_value=24.0,
            value=1.0,
            step=0.5,
            help="Set measurement duration in hours"
    )
    
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
    # EnergyMLP 클래스 정의 (저장된 state_dict 구조와 동일)
    class EnergyMLP(nn.Module):
        def __init__(self, input_size=3, hidden_sizes=[512, 256, 128], dropout=0.0):
            super().__init__()
            layers = []
            prev = input_size
            for h in hidden_sizes:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x).squeeze(1)

    # config에서 파일명 읽기
    mlp_path = os.path.join(config['paths']['models'], config['model']['model_names']['mlp_residual'])
    scaler_x_path = os.path.join(config['paths']['models'], config['model']['model_names']['scaler_x_residual'])

    # 모델 + 스케일러 로드
    mlp = EnergyMLP(input_size=3, hidden_sizes=[512, 256, 128])
    mlp.load_state_dict(torch.load(mlp_path, map_location='cpu'))
    mlp.eval()

    with open(scaler_x_path, 'rb') as f:
        scaler_X = pickle.load(f)

    # Residual 예측
    # y_final = y_formula + MLP_residual
    duration_sec = duration_h * 3600
    X_infer = np.array([[cpu_usage, memory_usage, duration_sec]], dtype=np.float32)
    X_infer_sc = scaler_X.transform(X_infer)
    X_tensor = torch.tensor(X_infer_sc, dtype=torch.float32)

    with torch.no_grad():
        residual_pred = mlp(X_tensor).item()

    model_pred = formula_result['energy_kwh'] + residual_pred

    st.success(
        f"**Residual MLP 예측값**: `{model_pred:.8f} kWh` "
        f"| **공식 계산값**: `{formula_result['energy_kwh']:.8f} kWh`"
    )

    diff = abs(model_pred - formula_result["energy_kwh"])
    st.caption(f"두 값의 차이(residual): {residual_pred:.8f} kWh")

except FileNotFoundError as e:
    st.warning(f"⚠️ 모델 파일이 없어서 공식 기반 계산만 표시했어요.\n\n`{e}`")

# try:
#     # model = load_best_model(config)
#     # rf, lr, meta_model = load_stacking_models(config)
#     # rf, lr = load_residual_models(config)
#     # model_pred = predict_energy_by_stacking(rf, lr, meta_model, cpu_usage, memory_usage, duration_h)
#     # model_pred = predict_energy_by_residual(rf, lr, cpu_usage, memory_usage, duration_h)
#     # model_pred = predict_energy_by_model(model, cpu_usage, memory_usage, duration_h)
#     # model_pred = predict_energy_by_model(model, cpu_usage, memory_usage, duration_sec, hour)

#     # rf = load_rf_hourly(config)
#     # print(config["model"]["model_names"])
#     # model_pred = predict_energy_by_rf_hourly(rf, cpu_usage, memory_usage, duration_h)
#     model, scaler_X, scaler_y = load_mlp_model(config)
#     model_pred = predict_energy_by_mlp(model, scaler_X, scaler_y, cpu_usage, memory_usage, duration_h)
    
#     st.success(
#         f"**Best Model 예측값**: `{model_pred:.8f} kWh` "
#         f"| **공식 계산값**: `{formula_result['energy_kwh']:.8f} kWh`"
#     )

#     diff = abs(model_pred - formula_result["energy_kwh"])
#     st.caption(f"두 값의 차이: {diff:.8f} kWh")

# except FileNotFoundError as e:
#     # 모델 파일이 없으면 안내 메시지 표시
#     st.warning(f"⚠️ 모델 파일이 없어서 공식 기반 계산만 표시했어요.\n\n`{e}`")


# ----------------------------------------
# 사용된 공식 표시
# ----------------------------------------
st.divider()
st.markdown("### 📐 사용된 공식")
st.code(
    f"""
CPU 사용률  : {cpu_usage * 100:.0f}%  →  {cpu_usage}
메모리 사용률: {memory_usage * 100:.0f}%  →  {memory_usage}
측정 시간   :  {formula_result['duration_h']:.6f}h

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