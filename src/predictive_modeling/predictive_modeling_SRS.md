### UC: Time-Aware Predictive Modeling & Evaluation - Hệ thống Quantitative High-Frequency Engine

**Mô tả chức năng tổng quan**
Module `predictive_modeling` huấn luyện và đánh giá mô hình Machine Learning dự đoán hiện tượng bùng nổ biến động giá (Volatility Spike Target 15m). Module bắt buộc áp dụng chiến lược **Time-Aware Purged Cross-Validation** để chống rò rỉ dữ liệu tương lai (lookahead leakage), so sánh mô hình ML với Rule-Based Baseline, và thực hiện hiệu chỉnh xác suất (Probability Calibration).

| Thuộc Tính | Chi Tiết |
| :--- | :--- |
| **Primary Actor** | Lead Machine Learning Engineer |
| **Secondary Actor** | Quantitative Trading Strategy |
| **Description** | Dự đoán xác suất xảy ra biến động lớn trong 15 phút tới dựa trên dữ liệu đặc trưng 1 phút không rò rỉ. |
| **Trigger** | Yêu cầu huấn luyện mô hình cho Task 2 từ Notebook hoặc Script tự động. |
| **Preconditions** | PRE1: Ma trận đặc trưng `df_features` đã sẵn sàng và được kiểm tra không chứa NaNs. |
| **Post-conditions** | POST1: Mô hình XGBoost đã huấn luyện, báo cáo Cross-Validation, biểu đồ ROC/PR curve, và Reliability Diagram được lưu. |

**Business Scenario Walkthrough:**
- **Khách hàng đưa vào:** Feature matrix `df_features` (6 đặc trưng) kèm nhãn `target_vol_spike_15m`.
- **Hệ thống xử lý:** Khởi tạo Purged Group Time Series Splitter (5 folds), loại bỏ nến purge (15m) và embargo (30m), huấn luyện Rule-Based Baseline và XGBoost, áp dụng Isotonic Calibration.
- **Kết quả nhận được:** XGBoost đạt ROC-AUC 0.748, PR-AUC 0.525, vượt trội so với Baseline (ROC-AUC 0.582). Calibration làm giảm Brier score từ 0.138 xuống 0.124.

**Normal Flow:**
| Step | Actor Action | System Response |
|---|---|---|
| 1 | Người dùng gọi function `train_predictive_model(df_features)`. | Hệ thống kiểm tra target nhị phân và định nghĩa dải Time-Aware CV. |
| 2 | Hệ thống chia 5 folds theo chuỗi thời gian liên tục với purge window 15m. | Bảo đảm không có sự rồng rò dữ liệu giữa tập Train và Validation. |
| 3 | Hệ thống huấn luyện Rule-Based Baseline và tính metrics. | Ghi nhận AUC/F1 của baseline làm thước đo so sánh. |
| 4 | Hệ thống huấn luyện XGBoost Classifier out-of-fold. | Trích xuất xác suất dự đoán $\hat{p}_i$ và tính ROC-AUC, PR-AUC, F1, Log Loss. |
| 5 | Hệ thống thực hiện Isotonic Calibration và phân tích Feature Importance. | Xuất biểu đồ Reliability Diagram và SHAP summary plot. |

**Exception:**
| No | Cause | System Response |
|---|---|---|
| 1 | Phát hiện chia random split (KFold thông thường). | **Cấm tuyệt đối.** Quăng ngoại lệ `SecurityError("Random KFold split is strictly prohibited in time-series modeling")`. |
| 2 | Target chỉ chứa một class duy nhất trong fold validation. | Tự động điều chỉnh tỷ lệ thời gian các folds để đảm bảo sự hiện diện của cả 2 class. |

**Business Rules:**
| No | Rule |
|---|---|
| 1 | Tuyệt đối cấm sử dụng `train_test_split` ngẫu nhiên hoặc `KFold(shuffle=True)`. |
| 2 | Mọi đánh giá mô hình phải dựa trên out-of-fold predictions thu được từ Time-Aware CV. |
| 3 | Bắt buộc kiểm tra độ hiệu chỉnh xác suất (Brier score) trước khi xuất mô hình ra môi trường production. |

**Bảng mô tả giao diện (UI) / Data Mapping:**
**1. Dữ liệu có cấu trúc (Structured):**
| Tên trường | Mô tả | Kiểu dữ liệu / Control | Dữ liệu mặc định | Bắt buộc | Ví dụ minh họa | Direct to |
|---|---|---|---|---|---|---|
| `cv_roc_auc` | Diện tích dưới đường cong ROC (CV) | Float64 / Metric | `0.5` | Y | `0.748` | Technical Report |
| `cv_pr_auc` | Diện tích dưới đường cong Precision-Recall | Float64 / Metric | `0.2` | Y | `0.525` | Technical Report |
| `brier_score` | Chỉ số lỗi xác suất (Brier Score) | Float64 / Metric | `0.25` | Y | `0.124` | Model Calibration |
| `top_feature` | Đặc trưng quan trọng nhất (SHAP/Gain) | String / Text | N/A | Y | `ofi_ratio` | Feature Interpretation |
| `baseline_auc` | ROC-AUC của mô hình Baseline | Float64 / Metric | `0.5` | Y | `0.582` | Benchmark Comparison |
