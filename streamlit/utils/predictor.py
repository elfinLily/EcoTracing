# ============================================================
# [입력값 계산 방식]
# RandomForest = 트리 기반 모델
#   -> 학습 범위 밖 값 외삽 불가
#   -> duration max = 300초이므로 3600초 넣어도 300초와 동일하게 예측
#   -> 그래서 duration 얼마를 넣어도 예측값이 똑같이 나옴
#   모델에는 duration=300 고정으로 넘기고
#   사용자 입력 시간 비율로 스케일링

#   예시) CPU=0.5, Mem=0.3, 1시간 입력
#     모델 예측 (300초 기준) = 0.02290934 kWh
#     scale_factor = 3600 / 300 = 12
#     최종 예측 = 0.02290934 * 12 = 0.27491 kWh
# ============================================================
import numpy as np
import pandas as pd
TRAIN_DURATION_SEC = 300

def calc_energy_by_formula(
    cpu_usage: float,
    memory_usage: float,
    duration_h: float,
    config: dict
) -> dict: 
    """
    에너지 소비량을 공식으로 직접 계산

    Args:
        cpu_usage   : CPU 사용률 (0.0 ~ 1.0)
        memory_usage : 메모리 사용률 (0.0 ~ 1.0)
        duration_h : 측정 시간 (시간)
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

    # 에너지 계산(kWh)
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

def predict_energy_by_model(model, cpu_usage, memory_usage, duration_h):
    """
    학습된 모델로 에너지 소비량 예측

    Args:
        model : load_best_model()로 불러온 모델
        cpu_usage : CPU 사용률 (0.0 ~ 1.0)
        memory_usage : 메모리 사용률 (0.0 ~ 1.0)
        duration_h  : 측정 시간 (시간) - UI 입력값

    Returns:
        예측된 에너지 소비량 (kWh)
    """
    # 학습 데이터와 동일한 단위(초)로 변환해서 그대로 전달
    # 모델이 이미 duration_sec 기반으로 energy_kwh를 학습했음
    features = pd.DataFrame(
        [[cpu_usage, memory_usage, TRAIN_DURATION_SEC]],
        columns=["cpu", "memory", "duration"]
    )

    base_prediction = model.predict(features)[0]
    # 사용자 입력 시간에 맞게 스케일링
    # scale_factor = 실제 초 / 학습 기준 초
    duration_sec = duration_h * 3600
    scale_factor = duration_sec / TRAIN_DURATION_SEC
    prediction = base_prediction * scale_factor

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

def predict_energy_by_stacking(rf, lr, meta_model, cpu_usage, memory_usage, duration_h):
    """
    Args:
        rf, lr, meta_model : load_stacking_models()로 불러온 모델들
        cpu_usage          : CPU 사용률 (0.0 ~ 1.0)
        memory_usage       : 메모리 사용률 (0.0 ~ 1.0)
        duration_h         : 측정 시간 (시간)

    Note:
        피처: power_w + duration
        power_w = 200 + cpu*300 + memory*50 (전처리와 동일한 공식)
        RF는 트리 기반 외삽 불가 -> 300초 고정 + 스케일링
        LR은 선형 외삽 가능 -> duration_sec 직접 전달
    """
    duration_sec = duration_h * 3600

    # power_w 계산 (전처리와 동일한 공식)
    power_w = 200 + (cpu_usage * 300) + (memory_usage * 50)

    # RF: duration=300 고정 + 스케일링
    features_300 = pd.DataFrame(
        [[power_w, TRAIN_DURATION_SEC]],
        columns=["power_w", "duration"]
    )
    rf_pred = rf.predict(features_300)[0]
    scale_factor = duration_sec / TRAIN_DURATION_SEC
    rf_pred_scaled = rf_pred * scale_factor

    # LR: duration_sec 직접 전달 (선형 외삽 가능)
    features_full = pd.DataFrame(
        [[power_w, duration_sec]],
        columns=["power_w", "duration"]
    )
    lr_pred = lr.predict(features_full)[0]

    # 메타모델로 최종 예측
    meta_input = np.array([[rf_pred_scaled, lr_pred]])
    prediction = meta_model.predict(meta_input)[0]

    return round(float(prediction), 8)


TRAIN_DURATION_MAX = 300.0
MAX_DURATION = 3600.0
TRAIN_DURATION_SEC = 300

def dynamic_weight_rf(duration_sec: float) -> float:
    """
    duration에 따라 RF 가중치 동적 계산
    - duration <= 300초 : RF = 1.0
    - 300 < duration <= 3600초: 선형 감소
    - duration > 3600초 : LR = 1.0
    """
    if duration_sec <= TRAIN_DURATION_MAX:
        return 1.0
    return max(0.0, 1.0 - (duration_sec - TRAIN_DURATION_MAX) / (MAX_DURATION - TRAIN_DURATION_MAX))


def predict_energy_by_dynamic_blending(rf, lr, cpu_usage, memory_usage, duration_h):
    """
    Dynamic Weighted Blending으로 에너지 예측

    Args:
        rf  : RandomForest 모델
        lr  : LinearRegression 모델
        cpu_usage  : CPU 사용률 (0.0 ~ 1.0)
        memory_usage : 메모리 사용률 (0.0 ~ 1.0)
        duration_h   : 측정 시간 (시간)

    Returns:
        예측된 에너지 소비량 (kWh)
    """
    duration_sec = duration_h * 3600
    rf_w = dynamic_weight_rf(duration_sec)
    lr_w = 1.0 - rf_w

    features = pd.DataFrame(
        [[cpu_usage, memory_usage, duration_sec]],
        columns=["cpu", "memory", "duration"]
    )

    # RF: 300초 고정 + 스케일링
    features_300 = features.copy()
    features_300['duration'] = TRAIN_DURATION_SEC
    rf_pred = rf.predict(features_300)[0] * (duration_sec / TRAIN_DURATION_SEC)

    # LR: duration 직접 전달
    lr_pred = lr.predict(features)[0]

    prediction = rf_w * rf_pred + lr_w * lr_pred
    return round(float(prediction), 8)

def predict_energy_by_residual(rf, lr, cpu_usage, memory_usage, duration_h):
    """
    Residual Learning으로 에너지 예측
    최종 예측 = RF 예측 + LR이 예측한 RF 오차

    Args:
        rf         : RandomForest 모델
        lr         : LinearRegression 모델
        cpu_usage  : CPU 사용률 (0.0 ~ 1.0)
        memory_usage : 메모리 사용률 (0.0 ~ 1.0)
        duration_h   : 측정 시간 (시간)

    Returns:
        예측된 에너지 소비량 (kWh)

    Note:
        RF는 트리 기반 외삽 불가 -> 300초 고정 + 스케일링
        LR은 RF의 오차를 duration 직접 넘겨서 보정
    """
    duration_sec = duration_h * 3600

    features_300 = pd.DataFrame(
        [[cpu_usage, memory_usage, TRAIN_DURATION_SEC]],
        columns=["cpu", "memory", "duration"]
    )
    features_full = pd.DataFrame(
        [[cpu_usage, memory_usage, duration_sec]],
        columns=["cpu", "memory", "duration"]
    )

    # RF: 300초 고정 + 스케일링
    rf_pred = rf.predict(features_300)[0] * (duration_sec / TRAIN_DURATION_SEC)

    # LR: RF 오차 보정 (duration 직접 전달)
    lr_pred = lr.predict(features_full)[0]

    prediction = rf_pred + lr_pred
    return round(float(prediction), 8)

def predict_energy_by_rf_hourly(rf, cpu_usage, memory_usage, duration_h):
    """
    hourly RF 모델로 에너지 예측
    피처: cpu, memory, duration, hour
    타겟: energy_kwh (직접 예측)

    Args:
        rf  : hourly RF 모델
        cpu_usage  : CPU 사용률 (0.0 ~ 1.0)
        memory_usage : 메모리 사용률 (0.0 ~ 1.0)
        duration_h : 측정 시간 (시간)
    """
    from datetime import datetime
    hour = datetime.now().hour
    duration_sec = duration_h * 3600

    # 피처 4개: cpu, memory, duration, hour
    feat = pd.DataFrame(
        [[cpu_usage, memory_usage, duration_sec, hour]],
        columns=["cpu", "memory", "duration", "hour"]
    )

    # energy_kwh 직접 예측 (power_w 변환 불필요)
    prediction = rf.predict(feat)[0]
    return round(float(prediction), 8)