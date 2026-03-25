import numpy as np
import pandas as pd

def calc_energy_by_formula(
    cpu_usage: float,
    memory_usage: float,
    duration_sec: float,
    config: dict
) -> dict: 
    """
    에너지 소비량을 공식으로 직접 계산

    Args:
        cpu_usage   : CPU 사용률 (0.0 ~ 1.0)
        memory_usage : 메모리 사용률 (0.0 ~ 1.0)
        duration_sec : 측정 시간 (초)
        config  : load_config()로 읽은 config 딕셔너리

    Returns:
        dict: {
            "power_w"     : 전력(W),
            "duration_h"   : 시간(h),
            "energy_kwh"   : 에너지(kWh),
            "carbon_kg"    : 탄소 배출량(kg CO2)  ← carbon.emission_factor 사용
        }
    """
    # 전력 계산 (W)
    power_w = 200 + (cpu_usage * 300) + (memory_usage * 50)

    # 시간 변환 (초 → 시간)
    duration_h = duration_sec / 3600

    # 에너지 계산 (kWh)
    energy_kwh = power_w * duration_h / 1000

    # 탄소 배출량 계산 (kg CO2) — config의 emission_factor 사용
    emission_factor = config["carbon"]["emission_factor"]  # kg CO2 per kWh
    carbon_kg = energy_kwh * emission_factor

    return {
        "power_w": round(power_w, 4),
        "duration_h": round(duration_h, 6),
        "energy_kwh": round(energy_kwh, 8),
        "carbon_kg": round(carbon_kg, 8),
    }

def predict_energy_by_model(
    model,
    cpu_usage: float,
    memory_usage: float,
    duration_sec: float,
    hour: int
) -> float:
    """
    학습된 모델로 에너지 소비량 예측

    Args:
        model  : load_model()로 불러온 sklearn/lgbm 모델
        cpu_usage : CPU 사용률 (0.0 ~ 1.0)
        memory_usage : 메모리 사용률 (0.0 ~ 1.0)
        duration_sec : 측정 시간 (초)
        hour  : 시간대 (0 ~ 23)

    Returns:
        예측된 에너지 소비량 (kWh, float)
    """
    # 모델 입력 피처 순서: [duration, cpu, memory, hour]
    # Phase 1 학습 시 피처 순서와 반드시 일치해야 함
    features = pd.DataFrame([{
        "duration": duration_sec,
        "cpu": cpu_usage,
        "memory": memory_usage,
        "hour": hour
    }])

    prediction = model.predict(features)[0]
    return round(float(prediction), 8)

def energy_to_analogy(energy_kwh: float) -> dict:
    """
    에너지 소비량을 직관적인 비교 수치로 변환

    Args:
        energy_kwh: 에너지 소비량 (kWh)

    Returns:
        {
            "led_bulb_hours"   : LED 전구(10W) 켤 수 있는 시간(h),
            "smartphone_charge": 스마트폰(12Wh) 충전 횟수,
            "co2_grams"        : CO2 배출량(g)
        }
    """
    led_bulb_hours = energy_kwh / 0.01        # LED 10W 기준
    smartphone_charge = energy_kwh / 0.012    # 스마트폰 12Wh 기준
    co2_grams = energy_kwh * 500 * 1000         # 평균 500g CO2/kWh → g 단위

    return {
        "led_bulb_hours": round(led_bulb_hours, 2),
        "smartphone_charge": round(smartphone_charge, 2),
        "co2_grams": round(co2_grams, 4),
    }