"""Module huấn luyện mô hình dự báo biến động và hiệu chỉnh xác suất (Predictive Modeling).

Cung cấp các hàm huấn luyện:
- Rule-based Baseline
- HistGradientBoostingClassifier (Isotonic Calibrated)
- LightGBM Classifier
- XGBoost Classifier
- Stacking Ensemble Classifier

Đánh giá hiệu năng đa chiều (ROC-AUC, PR-AUC, Brier Score, ECE, Log Loss, F1),
hiệu chỉnh xác suất (Probability Calibration) và phân tích độ quan trọng đặc trưng.
"""

from typing import Any, Dict, List, Tuple
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
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

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


def calculate_expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    r"""Tính toán chỉ số Expected Calibration Error (ECE).

    Đo lường sai số kỳ vọng tuyệt đối giữa xác suất dự đoán và tần suất thực nghiệm.

    Công thức:
        $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$

    Args:
        y_true: Mảng nhãn thực tế nhị phân.
        y_prob: Mảng xác suất dự đoán trong [0, 1].
        n_bins: Số lượng thùng phân vị xác suất (mặc định 10).

    Returns:
        float: Giá trị ECE trong khoảng [0, 1].
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_predictions(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Tính toán các chỉ số đánh giá toàn diện cho mô hình phân loại xác suất.

    Args:
        y_true: Mảng nhãn thực tế nhị phân (0 hoặc 1).
        y_prob: Mảng xác suất dự đoán (trong khoảng [0, 1]).
        threshold: Ngưỡng phân loại nhị phân để tính F1, Precision, Recall.

    Returns:
        Dict[str, float]: Từ điển chứa các chỉ số ROC-AUC, PR-AUC, Brier Score, ECE, Log Loss, F1, Precision, Recall.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_prob)

    # PR-AUC (Precision-Recall Area Under Curve - rất quan trọng cho bài toán lệch lớp 80/20)
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_arr, precision_arr)

    # Brier Score & ECE & Log Loss
    brier = brier_score_loss(y_true, y_prob)
    ece = calculate_expected_calibration_error(y_true, y_prob, n_bins=10)
    ll = log_loss(y_true, y_prob)

    # Binary metrics at threshold
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "ece": float(ece),
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
    """Huấn luyện mô hình HistGradientBoostingClassifier xử lý mất cân bằng lớp (80/20).

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


def train_lightgbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    n_estimators: int = 150,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    num_leaves: int = 31,
) -> Any:
    """Huấn luyện mô hình LightGBM Classifier xử lý mất cân bằng lớp.

    Args:
        X_train: Ma trận đặc trưng tập huấn luyện.
        y_train: Mảng nhãn tập huấn luyện.
        random_state: Seed ngẫu nhiên.
        n_estimators: Số lượng cây.
        learning_rate: Tốc độ học.
        max_depth: Độ sâu tối đa.
        num_leaves: Số lượng lá tối đa.

    Returns:
        LGBMClassifier: Mô hình LightGBM đã huấn luyện.
    """
    if not HAS_LIGHTGBM:
        raise ImportError("LightGBM chưa được cài đặt trong môi trường.")

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    n_estimators: int = 150,
    learning_rate: float = 0.05,
    max_depth: int = 5,
) -> Any:
    """Huấn luyện mô hình XGBoost Classifier với trọng số cân bằng lớp scale_pos_weight.

    Args:
        X_train: Ma trận đặc trưng tập huấn luyện.
        y_train: Mảng nhãn tập huấn luyện.
        random_state: Seed ngẫu nhiên.
        n_estimators: Số lượng cây.
        learning_rate: Tốc độ học.
        max_depth: Độ sâu tối đa.

    Returns:
        XGBClassifier: Mô hình XGBoost đã huấn luyện.
    """
    if not HAS_XGBOOST:
        raise ImportError("XGBoost chưa được cài đặt trong môi trường.")

    neg_count = np.sum(y_train == 0)
    pos_count = max(np.sum(y_train == 1), 1)
    scale_pos = neg_count / pos_count

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        scale_pos_weight=scale_pos,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


def train_stacking_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> StackingClassifier:
    """Huấn luyện Stacking Ensemble kết hợp HistGradientBoosting, LightGBM và LogisticRegression Meta-Learner.

    Args:
        X_train: Ma trận đặc trưng tập huấn luyện.
        y_train: Mảng nhãn tập huấn luyện.
        random_state: Seed ngẫu nhiên.

    Returns:
        StackingClassifier: Mô hình Stacking Ensemble đã huấn luyện.
    """
    estimators = [
        (
            "hist_gbdt",
            HistGradientBoostingClassifier(
                max_iter=100, learning_rate=0.05, max_depth=5, class_weight="balanced", random_state=random_state
            ),
        )
    ]

    if HAS_LIGHTGBM:
        estimators.append(
            (
                "lightgbm",
                lgb.LGBMClassifier(
                    n_estimators=100, learning_rate=0.05, max_depth=5, class_weight="balanced", random_state=random_state, n_jobs=-1, verbose=-1
                ),
            )
        )

    if HAS_XGBOOST:
        neg_count = np.sum(y_train == 0)
        pos_count = max(np.sum(y_train == 1), 1)
        estimators.append(
            (
                "xgboost",
                xgb.XGBClassifier(
                    n_estimators=100, learning_rate=0.05, max_depth=5, scale_pos_weight=neg_count / pos_count, random_state=random_state, n_jobs=-1, eval_metric="logloss"
                ),
            )
        )

    meta_learner = LogisticRegression(class_weight="balanced", max_iter=500, random_state=random_state)
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=3,
        n_jobs=-1,
    )
    stacking.fit(X_train, y_train)
    return stacking


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
