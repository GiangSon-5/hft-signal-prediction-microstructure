"""Pipeline huấn luyện mô hình dự báo biến động và đăng ký Model Registry (MLflow & Joblib).

Quy trình:
1. Đọc dữ liệu từ Gold Lakehouse (data/gold/features.parquet).
2. Thực hiện 5-Fold Time-Aware Purged & Embargoed Cross-Validation.
3. Ghi log các chỉ số đánh giá (ROC-AUC, PR-AUC, Brier Score, F1, Log Loss) vào MLflow.
4. Huấn luyện Champion Model với hiệu chuẩn Isotonic Probability Calibration.
5. Đóng gói và lưu trữ models/champion_model.pkl kèm models/model_metadata.json.
"""

from datetime import datetime
import json
import os
import time
from typing import Any, Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from tqdm import tqdm

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from src.lakehouse.pipeline import run_lakehouse_pipeline
from src.predictive_modeling.models import evaluate_predictions, predict_rule_based_baseline
from src.predictive_modeling.validation import PurgedTimeSeriesSplit


def load_or_build_training_data(
    gold_path: str = "data/gold/features.parquet",
    csv_fallback: str = "data/raw/ds_assessment_data.csv",
) -> Tuple[pd.DataFrame, list]:
    """Tải dữ liệu ma trận đặc trưng từ Gold Lakehouse hoặc tự động kích hoạt Lakehouse pipeline."""
    if not os.path.exists(gold_path):
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
        print(f"[MLOPS] Không tìm thấy {gold_path}. Đang kích hoạt Medallion Lakehouse Pipeline...")
        run_lakehouse_pipeline(raw_csv=csv_fallback)

    df_gold = pd.read_parquet(gold_path)
    feature_cols = [
        "parkinson_vol_15m",
        "garman_klass_vol_15m",
        "ofi_ratio",
        "trade_density",
        "volume_spike_z_60m",
        "return_momentum_15m",
        "rolling_vol_60m",
        "spread_ratio_15m",
    ]
    return df_gold, feature_cols


def train_and_register_champion_model(
    gold_path: str = "data/gold/features.parquet",
    model_output_dir: str = "models",
    experiment_name: str = "hft_volatility_prediction",
    n_splits: int = 5,
    purge_window: int = 15,
    embargo_window: int = 30,
) -> Dict[str, Any]:
    """Huấn luyện mô hình Champion GBDT qua Purged CV, log vào MLflow và lưu artifact."""
    start_time = time.time()
    os.makedirs(model_output_dir, exist_ok=True)

    df_gold, feature_cols = load_or_build_training_data(gold_path=gold_path)
    X = df_gold[feature_cols].values
    y = df_gold["target_vol_spike_15m"].values

    cv_splitter = PurgedTimeSeriesSplit(
        n_splits=n_splits, purge_window=purge_window, embargo_window=embargo_window
    )

    print(f"=== BẮT ĐẦU HUẤN LUYỆN MODEL REGISTRY ({n_splits}-FOLD TIME-SERIES CV) ===")
    oof_y_true = []
    oof_y_prob = []
    oof_y_base_prob = []
    fold_metrics_list = []

    fold_iterator = tqdm(
        enumerate(cv_splitter.split(df_gold)),
        total=n_splits,
        desc="Đang chạy Purged & Embargoed Cross-Validation",
        unit="fold",
    )

    for fold_idx, (train_idx, test_idx) in fold_iterator:
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Huấn luyện GBDT Fold
        base_gbdt = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=42,
        )
        base_gbdt.fit(X_train, y_train)

        # Dự đoán xác suất
        y_prob = base_gbdt.predict_proba(X_test)[:, 1]

        # Baseline
        df_test_fold = df_gold.iloc[test_idx]
        _, base_prob = predict_rule_based_baseline(df_test_fold)

        metrics = evaluate_predictions(y_test, y_prob, threshold=0.50)
        metrics["fold"] = fold_idx + 1
        fold_metrics_list.append(metrics)

        oof_y_true.extend(y_test)
        oof_y_prob.extend(y_prob)
        oof_y_base_prob.extend(base_prob)

        fold_iterator.set_postfix({
            "ROC-AUC": f"{metrics['roc_auc']:.4f}",
            "PR-AUC": f"{metrics['pr_auc']:.4f}",
        })

    oof_y_true = np.array(oof_y_true)
    oof_y_prob = np.array(oof_y_prob)
    oof_y_base_prob = np.array(oof_y_base_prob)

    # Đánh giá toàn cục OOF
    oof_metrics = evaluate_predictions(oof_y_true, oof_y_prob, threshold=0.50)
    baseline_oof_metrics = evaluate_predictions(
        oof_y_true, oof_y_base_prob, threshold=0.50
    )

    # 2. Huấn luyện Mô Hình Sản Phẩm (Champion Model) kèm Isotonic Calibration
    print("\n[MLOPS] Đang huấn luyện Champion Model toàn cục với Isotonic Probability Calibration...")
    final_base_gbdt = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
    )
    final_calibrated_model = CalibratedClassifierCV(
        estimator=final_base_gbdt, method="isotonic", cv=3
    )
    final_calibrated_model.fit(X, y)

    # 3. Lưu trữ Model Artifact
    model_path = os.path.join(model_output_dir, "champion_model.pkl")
    metadata_path = os.path.join(model_output_dir, "model_metadata.json")

    joblib.dump(final_calibrated_model, model_path)

    metadata = {
        "model_name": "Champion_HistGBDT_Calibrated",
        "model_type": "CalibratedClassifierCV(HistGradientBoostingClassifier)",
        "version": "1.0.0",
        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"),
        "features": feature_cols,
        "feature_count": len(feature_cols),
        "target": "target_vol_spike_15m",
        "target_definition": "I(Forward_15m_Std >= Quantile_0.80)",
        "sample_size": len(df_gold),
        "cross_validation": {
            "strategy": "PurgedTimeSeriesSplit",
            "n_splits": n_splits,
            "purge_window_bars": purge_window,
            "embargo_window_bars": embargo_window,
        },
        "metrics_out_of_fold": {
            "roc_auc": round(oof_metrics["roc_auc"], 4),
            "pr_auc": round(oof_metrics["pr_auc"], 4),
            "brier_score": round(oof_metrics["brier_score"], 4),
            "ece": round(oof_metrics["ece"], 4),
            "log_loss": round(oof_metrics["log_loss"], 4),
            "f1_score": round(oof_metrics["f1_score"], 4),
            "precision": round(oof_metrics["precision"], 4),
            "recall": round(oof_metrics["recall"], 4),
        },
        "baseline_comparison": {
            "baseline_type": "Rule_Based_Heuristic",
            "roc_auc": round(baseline_oof_metrics["roc_auc"], 4),
            "pr_auc": round(baseline_oof_metrics["pr_auc"], 4),
            "brier_score": round(baseline_oof_metrics["brier_score"], 4),
            "f1_score": round(baseline_oof_metrics["f1_score"], 4),
            "roc_auc_lift_pct": round(
                (oof_metrics["roc_auc"] - baseline_oof_metrics["roc_auc"])
                / baseline_oof_metrics["roc_auc"]
                * 100.0,
                2,
            ),
            "pr_auc_lift_pct": round(
                (oof_metrics["pr_auc"] - baseline_oof_metrics["pr_auc"])
                / baseline_oof_metrics["pr_auc"]
                * 100.0,
                2,
            ),
        },
        "hyperparameters": {
            "max_iter": 150,
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_samples_leaf": 50,
            "class_weight": "balanced",
            "calibration_method": "isotonic",
            "calibration_cv": 3,
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    if HAS_MLFLOW:
        try:
            # Thiết lập tracking SQLite Database trong thư mục mlflow/
            os.makedirs("mlflow", exist_ok=True)
            mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name="champion_gbdt_calibrated"):
                # Ghi nhận Parameters
                mlflow.log_params({
                    "model_family": "HistGradientBoostingClassifier",
                    "max_iter": 150,
                    "learning_rate": 0.05,
                    "max_depth": 5,
                    "min_samples_leaf": 50,
                    "class_weight": "balanced",
                    "calibration_method": "isotonic",
                    "n_splits": n_splits,
                    "purge_window": purge_window,
                    "embargo_window": embargo_window,
                })

                # Ghi nhận Metrics OOF
                mlflow.log_metrics({
                    "oof_roc_auc": oof_metrics["roc_auc"],
                    "oof_pr_auc": oof_metrics["pr_auc"],
                    "oof_brier_score": oof_metrics["brier_score"],
                    "oof_ece": oof_metrics["ece"],
                    "oof_f1_score": oof_metrics["f1_score"],
                    "oof_log_loss": oof_metrics["log_loss"],
                    "baseline_roc_auc": baseline_oof_metrics["roc_auc"],
                    "baseline_pr_auc": baseline_oof_metrics["pr_auc"],
                })

                # Ghi nhận Fold-level Metrics
                for fm in fold_metrics_list:
                    f_num = fm["fold"]
                    mlflow.log_metric(f"fold_{f_num}_roc_auc", fm["roc_auc"])
                    mlflow.log_metric(f"fold_{f_num}_pr_auc", fm["pr_auc"])

                # Log artifacts
                mlflow.log_artifact(metadata_path)
                print("[MLOPS] Đã ghi log thành công vào MLflow Tracking Registry (mlflow/mlflow.db).")
        except Exception as e:
            print(f"[MLOPS] Cảnh báo ghi log MLflow: {e}")

    elapsed_total = round(time.time() - start_time, 2)
    print("\n=== KẾT QUẢ HUẤN LUYỆN CHAMPION MODEL ===")
    print(f"Tổng thời gian  : {elapsed_total}s")
    print(f"OOF ROC-AUC     : {oof_metrics['roc_auc']:.4f} (Baseline: {baseline_oof_metrics['roc_auc']:.4f}, Lift: +{metadata['baseline_comparison']['roc_auc_lift_pct']}%)")
    print(f"OOF PR-AUC      : {oof_metrics['pr_auc']:.4f} (Baseline: {baseline_oof_metrics['pr_auc']:.4f}, Lift: +{metadata['baseline_comparison']['pr_auc_lift_pct']}%)")
    print(f"OOF Brier Score : {oof_metrics['brier_score']:.4f} (ECE: {oof_metrics['ece']:.4f})")
    print(f"Artifact Model  : {model_path}")
    print(f"Artifact Metadata: {metadata_path}")

    return {
        "status": "SUCCESS",
        "elapsed_seconds": elapsed_total,
        "oof_metrics": oof_metrics,
        "metadata": metadata,
    }


if __name__ == "__main__":
    train_and_register_champion_model()
