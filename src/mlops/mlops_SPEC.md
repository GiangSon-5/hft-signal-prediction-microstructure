# Đặc Tả Kỹ Thuật: MLOps Training & Ablation Study (`src/mlops`)

## 1. Tổng Quan Mô-đun

Mô-đun `mlops` đóng vai trò điều phối quá trình nghiên cứu thực nghiệm đối chuẩn (Ablation Study) và quản lý vòng đời mô hình sản phẩm (Model Registry & Tracking) thông qua MLflow:

1. **`ablation_study.py` (Pipeline Thử Nghiệm Đối Chuẩn)**:
   - Chạy toàn bộ ma trận 16 cấu hình ($4\text{ Scenarios} \times 4\text{ Models}$).
   - Đánh giá Out-of-Fold metrics qua 5-Fold Time-Aware Purged & Embargoed Cross-Validation.
   - Ghi nhận 12 runs vào MLflow experiment `Quantitative_Ablation_Study`.
   - Xuất báo cáo `reports/ablation_study_summary.md` và `reports/ablation_study_results.json`.

2. **`train_mlflow.py` (Pipeline Chính Đóng Gói Champion Model)**:
   - Huấn luyện mô hình Champion GBDT với Isotonic Probability Calibration.
   - Đóng gói artifact `models/champion_model.pkl`.
   - Lưu trữ metadata và chỉ số OOF vào `models/model_metadata.json`.
   - Đăng ký run vào MLflow experiment `hft_volatility_prediction`.

---

## 2. Lưu Trữ MLflow Cục Bộ

- Cơ sở dữ liệu SQLite: `mlflow/mlflow.db`
- Lệnh mở giao diện trực quan:
  ```powershell
  python -m mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5000
  ```
