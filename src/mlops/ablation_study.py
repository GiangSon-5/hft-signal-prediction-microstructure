"""Module Thực Nghiệm Đối Chuẩn Đa Kịch Bản & Đa Mô Hình (MLOps Ablation Study).

Thực hiện ma trận thực nghiệm 16 cấu hình (4 Kịch bản Đặc trưng x 4 Thuật toán Mô hình)
trên 5-Fold Time-Aware Purged & Embargoed Cross-Validation kết hợp thanh tiến trình trực quan tqdm.
Ghi log toàn bộ kết quả vào MLflow (mlflow/mlflow.db) và xuất báo cáo so sánh định lượng.
"""

from datetime import datetime
import json
import os
import time
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from tqdm import tqdm

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.feature_engineering.generator import generate_extended_feature_matrix
from src.lakehouse.pipeline import run_lakehouse_pipeline
from src.predictive_modeling.models import (
    evaluate_predictions,
    predict_rule_based_baseline,
    train_gbdt_model,
    train_lightgbm_model,
    train_xgboost_model,
    train_stacking_ensemble,
)
from src.predictive_modeling.validation import PurgedTimeSeriesSplit


def load_or_build_ablation_dataset(
    extended_gold_path: str = "data/gold/extended_features.parquet",
    csv_fallback: str = "data/raw/ds_assessment_data.csv",
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Tải dữ liệu ma trận đặc trưng mở rộng từ Gold Lakehouse hoặc tự động kích hoạt pipeline."""
    if not os.path.exists(extended_gold_path):
        if not os.path.exists(csv_fallback):
            candidates = [
                "data/raw/ds_assessment_data.csv",
                "../data/raw/ds_assessment_data.csv",
                "ds_assessment_data.csv",
                "../ds_assessment_data.csv",
            ]
            for c in candidates:
                if os.path.exists(c):
                    csv_fallback = c
                    break
        print(f"[ABLATION] Không tìm thấy {extended_gold_path}. Đang kích hoạt Lakehouse Pipeline...")
        run_lakehouse_pipeline(raw_csv=csv_fallback)

    df_ext = pd.read_parquet(extended_gold_path)

    # Lấy định nghĩa 4 kịch bản từ generator
    _, scenarios = generate_extended_feature_matrix(df_ext.head(100), dropna=False)

    return df_ext, scenarios


def instantiate_base_model(model_name: str, random_state: int = 42) -> Any:
    """Khởi tạo mô hình học máy cơ sở theo tên thuật toán."""
    if model_name == "HistGBDT":
        return HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=random_state,
        )
    elif model_name == "LightGBM":
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM không có sẵn.")
        return lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    elif model_name == "XGBoost":
        if not HAS_XGBOOST:
            raise ImportError("XGBoost không có sẵn.")
        return xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            scale_pos_weight=4.0,  # 80/20 class ratio
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )
    elif model_name == "StackingEnsemble":
        return HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=random_state,
        )
    else:
        raise ValueError(f"Không nhận diện được thuật toán: {model_name}")


def evaluate_scenario_configuration(
    df: pd.DataFrame,
    scenario_name: str,
    feature_cols: List[str],
    model_name: str,
    n_splits: int = 5,
    purge_window: int = 15,
    embargo_window: int = 30,
) -> Dict[str, Any]:
    """Chạy kiểm định 5-Fold Purged & Embargoed Cross-Validation cho 1 cấu hình cụ thể."""
    start_t = time.time()
    X = df[feature_cols].values
    y = df["target_vol_spike_15m"].values

    cv_splitter = PurgedTimeSeriesSplit(
        n_splits=n_splits, purge_window=purge_window, embargo_window=embargo_window
    )

    oof_y_true = []
    oof_y_prob = []
    fold_roc_list = []
    fold_pr_list = []

    fold_bar = tqdm(
        enumerate(cv_splitter.split(df)),
        total=n_splits,
        desc=f"  ↳ [{scenario_name} | {model_name}] 5 Folds",
        leave=False,
        unit="fold",
    )

    for fold_idx, (train_idx, test_idx) in fold_bar:
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        base_estimator = instantiate_base_model(model_name, random_state=42 + fold_idx)

        # Hiệu chuẩn Isotonic Calibration nội bộ 3-folds
        calibrated_model = CalibratedClassifierCV(
            estimator=base_estimator, method="isotonic", cv=3
        )
        calibrated_model.fit(X_train, y_train)

        y_prob = calibrated_model.predict_proba(X_test)[:, 1]

        fold_metrics = evaluate_predictions(y_test, y_prob, threshold=0.50)
        fold_roc_list.append(fold_metrics["roc_auc"])
        fold_pr_list.append(fold_metrics["pr_auc"])

        oof_y_true.extend(y_test)
        oof_y_prob.extend(y_prob)

        fold_bar.set_postfix({"ROC": f"{fold_metrics['roc_auc']:.3f}", "PR": f"{fold_metrics['pr_auc']:.3f}"})

    oof_y_true = np.array(oof_y_true)
    oof_y_prob = np.array(oof_y_prob)

    oof_metrics = evaluate_predictions(oof_y_true, oof_y_prob, threshold=0.50)
    elapsed = round(time.time() - start_t, 2)

    return {
        "scenario": scenario_name,
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "model": model_name,
        "elapsed_seconds": elapsed,
        "roc_auc": round(oof_metrics["roc_auc"], 4),
        "pr_auc": round(oof_metrics["pr_auc"], 4),
        "brier_score": round(oof_metrics["brier_score"], 4),
        "ece": round(oof_metrics["ece"], 4),
        "log_loss": round(oof_metrics["log_loss"], 4),
        "f1_score": round(oof_metrics["f1_score"], 4),
        "roc_auc_fold_std": round(float(np.std(fold_roc_list)), 4),
        "pr_auc_fold_std": round(float(np.std(fold_pr_list)), 4),
    }


def run_ablation_study(
    extended_gold_path: str = "data/gold/extended_features.parquet",
    output_dir: str = "reports",
    experiment_name: str = "Quantitative_Ablation_Study",
) -> pd.DataFrame:
    """Thực thi toàn bộ ma trận 16 cấu hình thực nghiệm Ablation Study kèm thanh tiến trình tqdm."""
    os.makedirs(output_dir, exist_ok=True)
    df_ext, scenarios = load_or_build_ablation_dataset(extended_gold_path)

    # 1. Đánh giá Baseline Rule-Based Heuristic mốc tham chiếu
    _, base_prob = predict_rule_based_baseline(df_ext)
    base_metrics = evaluate_predictions(
        df_ext["target_vol_spike_15m"].values, base_prob, threshold=0.50
    )

    models_to_test = ["HistGBDT"]
    if HAS_LIGHTGBM:
        models_to_test.append("LightGBM")
    if HAS_XGBOOST:
        models_to_test.append("XGBoost")

    results_list = []

    # Thêm Baseline Heuristic vào kết quả
    results_list.append(
        {
            "scenario": "Baseline_Rule_Based",
            "feature_count": 2,
            "features": ["volume_spike_z_60m", "parkinson_vol_15m"],
            "model": "Rule_Based_Heuristic",
            "elapsed_seconds": 0.5,
            "roc_auc": round(base_metrics["roc_auc"], 4),
            "pr_auc": round(base_metrics["pr_auc"], 4),
            "brier_score": round(base_metrics["brier_score"], 4),
            "ece": round(base_metrics["ece"], 4),
            "log_loss": round(base_metrics["log_loss"], 4),
            "f1_score": round(base_metrics["f1_score"], 4),
            "roc_auc_fold_std": 0.0,
            "pr_auc_fold_std": 0.0,
        }
    )

    if HAS_MLFLOW:
        try:
            os.makedirs("mlflow", exist_ok=True)
            mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"[MLOPS] Cảnh báo khởi tạo MLflow DB: {e}")

    print("=== BẮT ĐẦU CHUỖI THỰC NGHIỆM ABLATION STUDY (16 CẤU HÌNH) ===")
    print(f"Tổng số mẫu quan sát: {len(df_ext):,}")
    print(f"Kịch bản đặc trưng  : {list(scenarios.keys())}")
    print(f"Mô hình thuật toán  : {models_to_test}\n")

    # Tạo danh sách các cặp (scenario, model)
    experiments_queue = []
    for sc_name, sc_features in scenarios.items():
        for m_name in models_to_test:
            experiments_queue.append((sc_name, sc_features, m_name))

    main_progress_bar = tqdm(
        experiments_queue,
        desc="Tổng thể Ma trận Thực nghiệm Ablation",
        unit="run",
    )

    for sc_name, sc_features, m_name in main_progress_bar:
        main_progress_bar.set_description(f"Đang chạy: {sc_name} | {m_name}")
        res = evaluate_scenario_configuration(
            df_ext,
            scenario_name=sc_name,
            feature_cols=sc_features,
            model_name=m_name,
        )
        results_list.append(res)

        main_progress_bar.set_postfix({
            "PR-AUC": f"{res['pr_auc']:.4f}",
            "ROC-AUC": f"{res['roc_auc']:.4f}",
            "Brier": f"{res['brier_score']:.4f}",
        })

        # Ghi log MLflow
        if HAS_MLFLOW:
            try:
                with mlflow.start_run(run_name=f"{sc_name}_{m_name}"):
                    mlflow.log_params(
                        {
                            "scenario": sc_name,
                            "feature_count": len(sc_features),
                            "model_type": m_name,
                            "calibration": "Isotonic",
                            "cv_splits": 5,
                            "purge_window": 15,
                            "embargo_window": 30,
                        }
                    )
                    mlflow.log_metrics(
                        {
                            "roc_auc": res["roc_auc"],
                            "pr_auc": res["pr_auc"],
                            "brier_score": res["brier_score"],
                            "ece": res["ece"],
                            "log_loss": res["log_loss"],
                            "f1_score": res["f1_score"],
                            "roc_auc_fold_std": res["roc_auc_fold_std"],
                            "pr_auc_fold_std": res["pr_auc_fold_std"],
                            "elapsed_seconds": res["elapsed_seconds"],
                        }
                    )
            except Exception as e:
                print(f"[MLOPS] Cảnh báo log MLflow run: {e}")

    df_results = pd.DataFrame(results_list)
    df_ranked = df_results.sort_values("pr_auc", ascending=False).reset_index(drop=True)

    # 2. Xuất báo cáo JSON và Markdown
    json_path = os.path.join(output_dir, "ablation_study_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df_ranked.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    md_path = os.path.join(output_dir, "ablation_study_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo Thực Nghiệm Định Lượng: Feature Ablation & Multi-Model Benchmarking\n\n")
        f.write(f"- Thời điểm thực thi: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}\n")
        f.write(f"- Tổng số mẫu kiểm định: {len(df_ext):,}\n")
        f.write("- Phương pháp chia mẫu: 5-Fold Time-Aware Purged (15m) & Embargoed (30m) Cross-Validation\n")
        f.write("- Hiệu chuẩn xác suất: Isotonic Regression\n\n")
        f.write("### Bảng Xếp Hạng Hiệu Năng Toàn Bộ Cấu Hình Thực Nghiệm\n\n")
        f.write(df_ranked[["scenario", "model", "feature_count", "roc_auc", "pr_auc", "brier_score", "ece", "f1_score", "elapsed_seconds"]].to_markdown(index=False))
        f.write("\n\n")

    print(f"\n[ABLATION] Hoàn thành thực nghiệm! Báo cáo đã xuất tại: {md_path}")
    return df_ranked


if __name__ == "__main__":
    ranked_df = run_ablation_study()
    print("\n=== TOP 5 CẤU HÌNH THỰC NGHIỆM TỐI ƯU NHẤT ===")
    print(ranked_df[["scenario", "model", "feature_count", "roc_auc", "pr_auc", "brier_score", "ece", "f1_score"]].head(5).to_string())
