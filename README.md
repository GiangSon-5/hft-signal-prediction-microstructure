# High-Frequency Market Data Architecture & Quantitative Machine Learning
## Phân Tích Dữ Liệu Thị Trường Tần Suất Cao & Kiến Trúc Dự Báo Biến Động Định Lượng (H2 2024)

---

## 1. Tổng Quan Hệ Thống & Bối Cảnh Nghiên Cứu

Hệ thống được thiết kế dưới dạng một **Kiến Trúc Kỹ Thuật Tài Chính Định Lượng Toàn Diện (Quantitative Financial Engineering Architecture)**, có khả năng tái lập thực nghiệm 100%, phục vụ nghiên cứu chuỗi thời gian thị trường tần suất cao (nến 1 phút) trong nửa cuối năm 2024 ($264,961$ mẫu quan sát chuỗi thời gian liên tục từ `01/07/2024` đến `31/12/2024`).

### Các Thành Phần Kiến Trúc Cốt Lõi:
1. **Medallion Data Lakehouse**: Chuyển đổi và chuẩn hóa dữ liệu qua 3 tầng lưu trữ Apache Parquet (Snappy compression):
   - **Bronze Layer** (`data/bronze/raw.parquet`): Dữ liệu thô nguyên bản ($264,961$ dòng).
   - **Silver Layer** (`data/silver/cleaned.parquet`): Dữ liệu làm sạch tái lập lưới 1m, forward-fill giá và zero-fill volume ($264,961$ dòng).
   - **Gold Layer** (`data/gold/features.parquet` & `data/gold/extended_features.parquet`): Bảng rộng chứa 8 đến 16 đặc trưng cấu trúc vi mô và biến mục tiêu nhị phân 15m ($264,886$ dòng).
2. **Kỹ Thuật Đặc Trưng Đa Quy Mô (Multi-Scale Feature Engineering)**: Khai thác 100% các cột dữ liệu thô (`quote_volume`, `trades`, `taker_buy_volume`), xây dựng 16 đặc trưng vi cấu trúc và thiết lập 4 kịch bản đối chuẩn ($8, 12, 16, 10$ features).
3. **MLOps Ablation Study & Benchmarking**: Khảo sát toàn diện ma trận 16 cấu hình ($4\text{ Kịch bản} \times 4\text{ Mô hình}$) thông qua kỹ thuật **5-Fold Time-Aware Purged (15m) & Embargoed (30m) Cross-Validation** kết hợp **Isotonic Probability Calibration**.
4. **Model Registry & Tracking**: Lưu trữ mô hình sản phẩm (`models/champion_model.pkl`), siêu dữ liệu (`models/model_metadata.json`) và ghi log toàn bộ vào MLflow Tracking Registry cục bộ (`mlflow/mlflow.db`).

---

## 2. Công Nghệ & Lý Do Kiến Trúc (Tech Stack)

| Tầng Công Nghệ | Thư Viện / Công Cụ | Vai Trò & Lý Do Kiến Trúc |
| :--- | :--- | :--- |
| **Ngôn Ngữ Lõi** | Python 3.11+ | Nền tảng chuẩn cho Data Science và Kỹ thuật Tài chính Định lượng. |
| **Xử Lý Dữ Liệu & Lakehouse** | `pandas`, `pyarrow` | Biến đổi vector chuỗi thời gian tốc độ cao, định dạng lưu trữ cột nén Apache Parquet (Snappy). |
| **Phân Tích Thống Kê** | `scipy.stats`, `statsmodels` | Kiểm định Jarque-Bera, khớp phân phối Student-t, phân tích tự tương quan ACF/PACF. |
| **Mô Hình Học Máy** | `scikit-learn`, `lightgbm`, `xgboost` | Huấn luyện GBDT xử lý bất đối xứng lớp (80/20), kết hợp Stacking Ensemble. |
| **Hiệu Chuẩn Xác Suất** | `CalibratedClassifierCV` (Isotonic) | Hiệu chuẩn xác suất dự báo, tối ưu sai số kỳ vọng tuyệt đối (ECE) và Brier Score. |
| **MLOps & Quản Trị Thực Nghiệm** | `mlflow` | Tracking parameters, metrics từng fold, model lineage và lưu trữ artifact trong SQLite Database. |
| **Trực Quan Hóa & Thanh Tiến Trình** | `matplotlib`, `seaborn`, `plotly`, `tqdm` | Xuất đồ thị phân tích chất lượng cao và thanh tiến trình thời gian thực. |

---

## 3. Cấu Trúc Thư Mục Dự Án (Project Tree)

```
├── README.md                               # Báo cáo Kiến trúc Tổng quan (Master Architecture Doc)
├── requirements.txt                        # Khai báo các thư viện Python
├── data/
│   ├── raw/                                # Dữ liệu thô gốc (data/raw/ds_assessment_data.csv)
│   ├── bronze/                             # Lakehouse Bronze Parquet (raw.parquet)
│   ├── silver/                             # Lakehouse Silver Parquet (cleaned.parquet)
│   └── gold/                               # Lakehouse Gold Parquet (features.parquet, extended_features.parquet)
├── mlflow/                                 # Kho lưu trữ MLflow SQLite Database (mlflow.db)
├── models/
│   ├── champion_model.pkl                  # Champion Model Artifact (GBDT + Isotonic Calibration)
│   └── model_metadata.json                 # Thông số cấu hình, metrics OOF và siêu dữ liệu
├── notebooks/
│   ├── 01_task1_signal_characterization.ipynb  # Phân tích đặc trưng hóa tín hiệu & thống kê
│   ├── 02_task2_predictive_modeling.ipynb       # Nghiên cứu mô hình hóa dự báo & SHAP
│   └── 03_task3_deep_dive.ipynb                # Nghiên cứu sâu vi cấu trúc & Kyle's Lambda
├── reports/
│   ├── ablation_study_summary.md           # Bảng xếp hạng định lượng 16 cấu hình thực nghiệm
│   ├── ablation_study_results.json         # Chi tiết kết quả từng fold và từng cấu hình
│   └── figures/                            # Biểu đồ phân tích và SHAP Summary
└── src/
    ├── data_quality/                       # Làm sạch chuỗi thời gian 1m & kiểm định OHLC
    │   ├── cleaner.py
    │   ├── data_quality_SPEC.md
    │   └── data_quality_SRS.md
    ├── signal_characterization/            # Thống kê phân phối, regimes, ACF/PACF
    │   ├── analyzer.py
    │   ├── signal_characterization_SPEC.md
    │   └── signal_characterization_SRS.md
    ├── feature_engineering/                # Trích xuất 16 đặc trưng vi cấu trúc & 4 kịch bản
    │   ├── generator.py
    │   ├── feature_engineering_SPEC.md
    │   └── feature_engineering_SRS.md
    ├── lakehouse/                          # Pipeline Medallion 3 tầng (DuckDB/Pandas Parquet)
    │   └── pipeline.py
    ├── predictive_modeling/                # Thuật toán ML, Calibration & Time-Aware CV
    │   ├── models.py
    │   ├── validation.py
    │   ├── predictive_modeling_SPEC.md
    │   └── predictive_modeling_SRS.md
    └── mlops/                              # Huấn luyện Champion Model & Ablation Study
        ├── train_mlflow.py
        └── ablation_study.py
```

---

## 4. Không Gian 16 Đặc Trưng Vi Cấu Trúc & 4 Kịch Bản Nghiên Cứu

### 4.1 Bảng 16 Đặc Trưng Vi Cấu Trúc Thị Trường

| Nhóm Đặc Trưng | Tên Biến | Công Thức Toán Học / Định Nghĩa Định Lượng | Ý Nghĩa Tài Chính Vi Cấu Trúc |
| :--- | :--- | :--- | :--- |
| **Biến động Extreme-Value** | `parkinson_vol_15m` | $\sqrt{\frac{1}{15 \cdot 4 \ln 2} \sum \ln(H/L)^2}$ | Đo lường độ biến động qua khoảng High/Low 15m (hiệu quả gấp 5 lần so với Close-to-Close). |
| | `garman_klass_vol_15m` | $\sqrt{\frac{1}{15} \sum \left[ 0.5 \ln(H/L)^2 - (2\ln 2 - 1)\ln(C/O)^2 \right]}$ | Đo lường độ biến động kết hợp khoảng nhảy Open/Close và High/Low. |
| **Dòng lệnh & Mật độ** | `ofi_ratio` | $\frac{\text{taker\_buy\_volume}}{\text{volume} + \epsilon}$ | Tỷ lệ mất cân bằng dòng lệnh mua/bán chủ động (Order Flow Imbalance). |
| | `trade_density` | $\frac{\text{volume}}{\text{trades} + \epsilon}$ | Khối lượng trung bình mỗi lượt khớp lệnh (phân biệt tổ chức vs nhỏ lẻ). |
| | `normalized_net_flow` | $\frac{2 \cdot \text{taker\_buy\_vol} - \text{vol}}{\text{vol} + \epsilon} \in [-1, 1]$ | Dòng tiền ròng chủ động chuẩn hóa trong đoạn $[-1, 1]$. |
| **Bùng nổ Khối lượng & Lệnh** | `volume_spike_z_60m` | $\frac{\text{volume} - \mu_{60m}}{\sigma_{60m} + \epsilon}$ | Z-Score phát hiện các cú bùng nổ khối lượng đột biến so với nền 60 phút. |
| | `trades_z_60m` | $\frac{\text{trades} - \mu_{60m}}{\sigma_{60m} + \epsilon}$ | Z-Score đo lường sự bùng nổ đột biến về tần suất giao dịch của thị trường. |
| **Giá trị giao dịch & Lệch VWAP** | `vwap_dev_15m` | $\frac{P_t - \text{VWAP}_{15m}}{P_t}$ với $\text{VWAP} = \frac{\sum \text{quote\_volume}}{\sum \text{volume}}$ | Độ phân kỳ giữa giá hiện tại và giá bình quân gia quyền khối lượng. |
| | `dollar_trade_size` | $\frac{\text{quote\_volume}}{\text{trades} + \epsilon}$ | Giá trị định danh USD trung bình mỗi lệnh (nhận diện dòng tiền tổ chức). |
| **Cấu trúc kỳ hạn & Đa quy mô**| `vol_term_structure_15_60`| $\frac{\sigma_{\text{GK}, 15m}}{\sigma_{\text{GK}, 60m}}$ | Tỷ số giữa biến động ngắn hạn 15m và trung hạn 60m (phát hiện xung lực bùng nổ). |
| | `parkinson_vol_5m` | $\sqrt{\frac{1}{5 \cdot 4 \ln 2} \sum \ln(H/L)^2}$ | Biến động Parkinson siêu ngắn hạn 5 phút. |
| | `parkinson_vol_30m` | $\sqrt{\frac{1}{30 \cdot 4 \ln 2} \sum \ln(H/L)^2}$ | Biến động Parkinson trung hạn 30 phút. |
| **Bước nhảy & Động lượng** | `jump_intensity_15m` | $\frac{\text{Parkinson Vol}_{15m}}{\text{Realized Vol}_{15m}}$ | Đo lường mức độ ảnh hưởng của bước nhảy giá so với biến động liên tục. |
| | `return_momentum_15m` | $\ln(P_t / P_{t-15})$ | Động lượng log-return 15 phút. |
| | `rolling_vol_60m` | $\sigma(\text{return}, 60m) \times \sqrt{525,600}$ | Độ biến động độ lệch chuẩn giá đóng cửa 60m quy năm. |
| | `spread_ratio_15m` | $\text{mean}_{15m}\left(\frac{\text{High} - \text{Low}}{\text{Open}}\right)$ | Tỷ số biên độ nến trung bình trượt 15 phút. |

---

### 4.2 Định Nghĩa 4 Kịch Bản Thực Nghiệm (Feature Scenarios)

1. **`scenario_a_baseline_8` (8 biến)**: 8 đặc trưng cấu trúc vi mô cơ sở.
2. **`scenario_b_full_raw_12` (12 biến)**: Khai thác 100% cột dữ liệu thô (bổ sung `vwap_dev_15m`, `dollar_trade_size`, `normalized_net_flow`, `trades_z_60m`).
3. **`scenario_c_multiscale_16` (16 biến)**: Toàn diện 16 đặc trưng kết hợp cấu trúc kỳ hạn đa khung thời gian và cường độ bước nhảy.
4. **`scenario_d_optimal_10` (10 biến)**: Top 10 đặc trưng tinh gọn tối ưu được chọn lọc theo Permutation Importance.

---

## 5. Kết Quả Thực Nghiệm Đối Chuẩn (Ablation Study Matrix)

Tất cả các cấu hình được đánh giá trên **5-Fold Time-Aware Purged (15m) & Embargoed (30m) Cross-Validation** kết hợp **Isotonic Probability Calibration**:

| Xếp Hạng | Kịch Bản Đặc Trưng | Thuật Toán Mô Hình | Số Biến | PR-AUC (OOF) | ROC-AUC (OOF) | Brier Score | ECE | F1-Score | Thời Gian (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Champion** | **`scenario_c_multiscale_16`** | **HistGBDT** | **16** | **0.7674** | **0.9091** | **0.0869** | **0.0183** | **0.6630** | 110.51 |
| 🥈 **Top 2** | `scenario_c_multiscale_16` | LightGBM | 16 | **0.7670** | **0.9089** | **0.0869** | **0.0184** | **0.6636** | **12.71** |
| 🥉 **Top 3** | `scenario_c_multiscale_16` | XGBoost | 16 | **0.7668** | **0.9089** | **0.0870** | **0.0188** | **0.6624** | 13.20 |
| **4** | `scenario_b_full_raw_12` | HistGBDT | 12 | **0.7638** | **0.9080** | **0.0876** | **0.0184** | **0.6623** | 118.05 |
| **5** | `scenario_b_full_raw_12` | LightGBM | 12 | **0.7634** | **0.9079** | **0.0877** | **0.0188** | **0.6605** | 12.65 |
| **6** | `scenario_b_full_raw_12` | XGBoost | 12 | **0.7632** | **0.9079** | **0.0877** | **0.0191** | **0.6603** | 19.20 |
| **7** | `scenario_a_baseline_8` | HistGBDT | 8 | **0.7611** | **0.9075** | **0.0881** | **0.0182** | **0.6594** | 86.43 |
| **8** | `scenario_a_baseline_8` | LightGBM | 8 | **0.7610** | **0.9075** | **0.0881** | **0.0186** | **0.6592** | 10.26 |
| **9** | `scenario_a_baseline_8` | XGBoost | 8 | **0.7610** | **0.9075** | **0.0881** | **0.0181** | **0.6577** | 10.13 |
| **10** | `scenario_d_optimal_10` | XGBoost | 10 | **0.7599** | **0.9077** | **0.0881** | **0.0200** | **0.6613** | 10.82 |
| **11** | `scenario_d_optimal_10` | LightGBM | 10 | **0.7595** | **0.9076** | **0.0882** | **0.0196** | **0.6604** | 11.31 |
| **12** | `scenario_d_optimal_10` | HistGBDT | 10 | **0.7594** | **0.9075** | **0.0882** | **0.0198** | **0.6594** | 56.08 |
| **Mốc Tham Chiếu**| `Baseline_Rule_Based` | Heuristic | 2 | **0.5894** | **0.8455** | **0.1718** | **0.2359** | **0.5904** | 0.50 |

---

## 6. Hướng Dẫn Thực Thi Hệ Thống (Execution Guide)

### 6.1 Khởi Tạo Môi Trường
```powershell
# 1. Kích hoạt môi trường ảo
.\.venv\Scripts\activate

# 2. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

### 6.2 Thực Thi Tuần Tự Các Pipeline
```powershell
# Bước 1: Xây dựng Medallion Data Lakehouse (CSV -> Bronze -> Silver -> Gold)
python -m src.lakehouse.pipeline

# Bước 2: Huấn luyện và đóng gói Champion Model vào Model Registry
python -m src.mlops.train_mlflow

# Bước 3: Chạy chuỗi thực nghiệm đối chuẩn Ablation Study (16 cấu hình)
python -m src.mlops.ablation_study

# Bước 4: Khởi động giao diện trực quan hóa MLflow Dashboard
python -m mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5000
```
Sau đó truy cập trình duyệt tại địa chỉ: `http://127.0.0.1:5000`.
