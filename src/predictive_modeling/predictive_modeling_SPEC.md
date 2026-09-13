# Đặc Tả Kỹ Thuật (Module Specification): Mô Hình Dự Đoán & Kiểm Lỗi Chuỗi Thời Gian (`src/predictive_modeling`)

## 1. Tổng Quan Mô-đun

Mô-đun `predictive_modeling` cung cấp các thuật toán máy học và khung kiểm định chuỗi thời gian định lượng:
- **Biến mục tiêu**: Dự báo bùng nổ biến động 15 phút tới $Y_t = \mathbb{I}\left(\sigma_{fwd, 15m} \ge Q_{0.80}\right)$.
- **Chiến lược kiểm định**: 5-Fold Time-Aware Purged (15m) & Embargoed (30m) Cross-Validation (ngăn chặn 100% rò rỉ dữ liệu).
- **Các họ thuật toán**: HistGradientBoosting, LightGBM, XGBoost, Stacking Ensemble và Rule-Based Baseline Heuristic.
- **Hiệu chuẩn xác suất**: Isotonic Regression (3-fold inner CV), đánh giá Brier Score và Expected Calibration Error (ECE).
- **Giải thích mô hình**: Permutation Feature Importance & SHAP Values.

---

## 2. Thuật Toán & Kiểm Lỗi Chống Rò Rỉ Dữ Liệu (No Lookahead Leakage)

### 2.1 Quy Trình Time-Aware Purged & Embargoed Cross-Validation
Chia dữ liệu $N = 264,886$ dòng thành $K=5$ khối thời gian liên tục.
1. **Purging (15 phút)**: Tại ranh giới tập Train và Test, loại bỏ 15 nến liền kề trước Test để tránh rò rỉ target nhìn về tương lai.
2. **Embargoing (30 phút)**: Chèn khoảng đệm 30 nến sau tập Test để triệt tiêu hiện tượng tự tương quan chuỗi thời gian.

```
Train Set | Purge (15m) | Test Set (Fold i) | Embargo (30m) | Train Set Tiếp Theo
[=======] | [X]         | [---------------] | [E]           | [=================]
```

---

## 3. Hiệu Chuẩn Xác Suất & Thước Đo Hiệu Năng Đa Chiều

### 3.1 Chỉ Số Hiệu Năng
- **PR-AUC (Precision-Recall Area Under Curve)**: Thước đo quan trọng nhất cho bài toán mất cân bằng lớp 80/20.
- **ROC-AUC**: Năng lực phân tách giữa hai lớp trạng thái thị trường.
- **Brier Score**: Đo lường sai số bình phương trung bình của xác suất dự đoán $BS = \frac{1}{N}\sum (p_t - y_t)^2$.
- **Expected Calibration Error (ECE)**: Đo lường sai số kỳ vọng tuyệt đối giữa độ tin cậy và tần suất thực nghiệm qua 10 bins.

---

## 4. Kết Quả Đối Chuẩn Thực Nghiệm (Ablation Benchmark)

| Thuật Toán | Kịch Bản | PR-AUC (OOF) | ROC-AUC (OOF) | Brier Score | ECE | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **HistGBDT (Calibrated)** | `scenario_c_multiscale_16` | **0.7674** | **0.9091** | **0.0869** | **0.0183** | **0.6630** |
| **LightGBM (Calibrated)** | `scenario_c_multiscale_16` | **0.7670** | **0.9089** | **0.0869** | **0.0184** | **0.6636** |
| **XGBoost (Calibrated)** | `scenario_c_multiscale_16` | **0.7668** | **0.9089** | **0.0870** | **0.0188** | **0.6624** |
| **Rule-Based Baseline** | Heuristic 2 Features | **0.5894** | **0.8455** | **0.1718** | **0.2359** | **0.5904** |
