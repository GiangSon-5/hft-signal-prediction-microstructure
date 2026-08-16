"""Module huấn luyện mô hình dự báo biến động và hiệu chỉnh xác suất (Predictive Modeling).

Cung cấp các hàm huấn luyện Rule-based Baseline, GBDT (HistGradientBoosting/XGBoost),
đánh giá hiệu năng đa chiều (ROC-AUC, PR-AUC, Brier Score), hiệu chỉnh xác suất (Probability Calibration)
và phân tích độ quan trọng đặc trưng (Permutation Feature Importance).
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    brier_score_loss,
    log_loss,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance


def evaluate_predictions(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Tính toán các chỉ số đánh giá toàn diện cho mô hình phân loại xác suất.

    Args:
        y_true: Mảng nhãn thực tế nhị phân (0 hoặc 1).
        y_prob: Mảng xác suất dự đoán (trong khoảng [0, 1]).
        threshold: Ngưỡng phân loại nhị phân để tính F1, Precision, Recall.

    Returns:
        Dict[str, float]: Từ điển chứa các chỉ số ROC-AUC, PR-AUC, Brier Score, Log Loss, F1, Precision, Recall.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_prob)

    # PR-AUC (Precision-Recall Area Under Curve - rất quan trọng cho bài toán lệch lớp 80/20)
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_arr, precision_arr)

    # Brier Score & Log Loss
    brier = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, y_prob)

    # Binary metrics at threshold
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "log_loss": float(ll),
        "f1_score": float(f1),
        "precision": float(prec),
        "recall": float(rec),
    }


def predict_rule_based_baseline(
    df: pd.DataFrame,
    vol_z_threshold: float = 1.5,
    parkinson_quantile: float = 0.80,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dự đoán Target bằng quy tắc trực giác (Rule-Based Baseline).

    Quy tắc:
        Dự đoán Y_hat = 1 nếu:
        - Volume Z-Score > vol_z_threshold HOẶC
        - Parkinson Volatility 15m vượt phân vị 80% trong lịch sử ngắn hạn.

    Args:
        df: DataFrame chứa các đặc trưng 'volume_spike_z_60m' và 'parkinson_vol_15m'.
        vol_z_threshold: Ngưỡng Z-score khối lượng (mặc định 1.5).
        parkinson_quantile: Ngưỡng phân vị độ biến động Parkinson (mặc định 0.80).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Mảng nhãn nhị phân và mảng điểm số xác suất xấp xỉ của Baseline.
    """
    p_cutoff = df["parkinson_vol_15m"].quantile(parkinson_quantile)
    cond_vol = df["volume_spike_z_60m"] > vol_z_threshold
    cond_park = df["parkinson_vol_15m"] > p_cutoff

    y_pred = (cond_vol | cond_park).astype(int).values

    # Xây dựng xác suất đại diện từ mức độ vượt ngưỡng
    z_norm = np.clip((df["volume_spike_z_60m"].values + 2.0) / 6.0, 0.0, 1.0)
    p_norm = np.clip(df["parkinson_vol_15m"].values / (p_cutoff * 2.0 + 1e-8), 0.0, 1.0)
    score = np.maximum(z_norm, p_norm)

    return y_pred, score


def train_gbdt_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    max_iter: int = 150,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    min_samples_leaf: int = 50,
) -> HistGradientBoostingClassifier:
    """Huấn luyện mô hình GBDT xử lý mất cân bằng lớp (80/20).

    Args:
        X_train: Ma trận đặc trưng tập huấn luyện.
        y_train: Mảng nhãn tập huấn luyện.
        random_state: Seed ngẫu nhiên.
        max_iter: Số lượng cây tối đa.
        learning_rate: Tốc độ học.
        max_depth: Độ sâu tối đa của cây.
        min_samples_leaf: Số lượng mẫu tối thiểu trên mỗi lá.

    Returns:
        HistGradientBoostingClassifier: Mô hình đã được huấn luyện.
    """
    model = HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def calibrate_probability_predictions(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    method: str = "isotonic",
    cv: int = 3,
) -> np.ndarray:
    """Hiệu chỉnh xác suất dự đoán bằng Isotonic Regression hoặc Sigmoid (Platt Scaling).

    Args:
        model: Mô hình cơ sở (base estimator).
        X_train: Tập đặc trưng huấn luyện.
        y_train: Tập nhãn huấn luyện.
        X_test: Tập đặc trưng kiểm thử.
        method: Phương pháp hiệu chỉnh ('isotonic' hoặc 'sigmoid').
        cv: Số folds cross-validation nội bộ của calibrator.

    Returns:
        np.ndarray: Mảng xác suất đã được hiệu chỉnh trên tập kiểm thử.
    """
    calibrator = CalibratedClassifierCV(estimator=model, method=method, cv=cv)
    calibrator.fit(X_train, y_train)
    return calibrator.predict_proba(X_test)[:, 1]


def calculate_permutation_feature_importance(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 5,
    random_state: int = 42,
    scoring: str = "roc_auc",
) -> pd.DataFrame:
    """Tính toán Permutation Feature Importance trên tập kiểm thử Out-of-Fold.

    Args:
        model: Mô hình đã huấn luyện.
        X_test: Ma trận đặc trưng tập test.
        y_test: Mảng nhãn tập test.
        feature_names: Danh sách tên các đặc trưng.
        n_repeats: Số lần xáo trộn ngẫu nhiên.
        random_state: Seed ngẫu nhiên.
        scoring: Thước đo đánh giá mức giảm điểm số (mặc định 'roc_auc').

    Returns:
        pd.DataFrame: Bảng xếp hạng độ quan trọng đặc trưng sắp xếp giảm dần.
    """
    perm_res = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
    )

    feat_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance_Mean": perm_res.importances_mean,
            "Importance_Std": perm_res.importances_std,
        }
    ).sort_values("Importance_Mean", ascending=False).reset_index(drop=True)

    return feat_df
