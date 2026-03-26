import os
import yaml
import pickle
import json
import pandas as pd

def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    config.yaml 파일을 읽어서 딕셔너리로 반환

    Args:
        config_path: config 파일 경로 (기본값: config/config.yaml)

    Returns:
        config 딕셔너리
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def load_model(model_key: str, config: dict):
    """
    config의 model.model_names에서 파일명을 읽어 모델 로드

    Args:
        model_key : config model_names의 키 (예: "lightgbm", "randomforest")
        config    : load_config()로 읽은 config 딕셔너리

    Returns:
        로드된 모델 객체
    """
    file_name = config["model"]["model_names"][model_key]
    model_path = os.path.join(config["paths"]["models"], file_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없어요: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model

def load_phase1_results(config: dict) -> dict:
    """
    Phase 1 모델 평가 결과 JSON 파일 로드

    Args:
        config: load_config()로 읽은 config 딕셔너리

    Returns:
        결과 딕셔너리 (RMSE, MAE, R2, MAPE, feature_importance 등)
    """
    file_name = config["model"]["results"]["results_json"]     # <- config에서 읽기
    result_path = os.path.join(
        config["paths"]["outputs"], "reports", file_name
    )

    if not os.path.exists(result_path):
        raise FileNotFoundError(f"결과 파일을 찾을 수 없어요: {result_path}")

    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_phase1_metrics(config: dict) -> pd.DataFrame:
    """
    Phase 1 모델 비교 메트릭 CSV 파일 로드

    Args:
        config: load_config()로 읽은 config 딕셔너리

    Returns:
        모델별 성능 지표 DataFrame (RMSE, MAE, R2, MAPE, 학습시간 등)
    """
    output_dir = config["paths"]["outputs"]
    metrics_path = os.path.join(output_dir, "reports", "phase1_metrics.csv")

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"메트릭 파일을 찾을 수 없어요: {metrics_path}")

    df = pd.read_csv(metrics_path)
    return df

def load_feature_importance(config: dict) -> pd.DataFrame:
    """
    config의 model.results.feature_importance_csv에서
    파일명을 읽어 피처 중요도 CSV 로드

    Args:
        config: load_config()로 읽은 config 딕셔너리

    Returns:
        DataFrame (columns: feature, LightGBM, XGBoost, RandomForest, CatBoost)
    """
    file_name = config["model"]["results"]["feature_importance_csv"]
    fi_path = os.path.join(
        config["paths"]["outputs"], "reports", file_name
    )

    if not os.path.exists(fi_path):
        raise FileNotFoundError(f"파일을 찾을 수 없어요: {fi_path}")

    return pd.read_csv(fi_path)

def load_best_model(config: dict):
    """
    phase1_full_results.json의 best_model 키를 읽어
    해당 모델 자동 로드

    Args:
        config: load_config()로 읽은 config 딕셔너리

    Returns:
        로드된 best 모델 객체
    """
    # results JSON에서 best_model 이름 읽기
    results = load_phase1_results(config)
    best_model_key = results["best_model"].lower()   # 예: "RandomForest" -> "randomforest"

    return load_model(best_model_key, config)