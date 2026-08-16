# High-Frequency Market Data Architecture & Analytics
## Phân Tích Dữ Liệu Thị Trường Tần Suất Cao & Mô Hình Dự Báo Biến Động (H2 2024)

---

## 1. Tổng Quan Hệ Thống & Bối Cảnh Bài Toán

Dự án này là một hệ thống Khoa học Dữ liệu Định lượng (Quantitative Data Science Architecture) hoàn chỉnh, có khả năng tái lập 100%, được thiết kế chuyên biệt để phân tích dữ liệu thị trường tần suất cao khung thời gian 1 phút trong nửa cuối năm 2024 (Tháng 7 – Tháng 12 năm 2024, bao gồm ~260,000 nến chuỗi thời gian liên tục).

### Các Mô-Đun Phân Tích Cốt Lõi
1. **Task 1 — Đặc Trưng Hóa Tín Hiệu (Signal Characterization):** Phân tích thống kê chi tiết phân phối tỷ suất lợi nhuận 1 phút (close-to-close), đo lường độ nhọn (kurtosis) và đặc tính đuôi béo (fat-tailed), xác định 2 chế độ biến động (volatility regimes) trên cửa sổ trượt 60 phút, phân tích mối quan hệ giữa Khối lượng (`volume`), Số giao dịch (`trades`) và Biên độ giá, kiểm tra tự tương quan (autocorrelation) và đảm bảo chất lượng dữ liệu.
2. **Task 2 — Mô Hình Dự Đoán (Predictive Modeling):** Định nghĩa mục tiêu nhị phân (Binary Target: Bùng nổ biến động 15 phút tới), tạo ít nhất 6 đặc trưng kỹ thuật domain-informed từ nến OHLCV, áp dụng chiến lược kiểm lỗi chéo theo thời gian **Time-Aware Cross-Validation (Purged & Embargoed CV — tuyệt đối không rò rỉ dữ liệu tương lai)**, huấn luyện mô hình XGBoost/LightGBM so sánh với Rule-based Baseline, và đánh giá hiệu chỉnh xác suất (Probability Calibration).
3. **Task 3 — Phân Tích Chuyên Sâu (Deep Dive):** Đề xuất giả thuyết định lượng về hiện tượng cấu trúc thị trường (Order Flow Toxicity & Tác động giá bất đối xứng Kyle's Lambda), thực hiện kiểm định thống kê kèm định lượng độ bất định (Uncertainty Quantification bằng 95% Confidence Interval từ Block Bootstrap), và nêu rõ hướng phát triển khi có dữ liệu tick/order book L2.

### Luồng Đầu Vào & Đầu Ra (Input / Output)
- **Đầu vào (Input):** File `data/raw/ds_assessment_data.csv` (~260,000 dòng x 9 cột: `timestamp` (UTC), `open`, `high`, `low`, `close`, `volume`, `quote_volume`, `trades`, `taker_buy_volume`).
- **Đầu ra (Output):**
  1. Mô-đun mã nguồn Python chuẩn hóa (`src/`) & Các Notebook thực thi (`notebooks/`).
  2. Hệ thống biểu đồ trực quan hóa cao cấp (`reports/figures/`).
  3. Báo cáo Kỹ thuật cô đọng 3 trang A4 (`reports/technical_report.html` / PDF).

---

## 2. Lựa Chọn Công Nghệ & Lý Do Kiến Trúc (Tech Stack Decisions)

| Tầng Công Nghệ | Công Cụ / Thư Viện | Lý Do Lựa Chọn & Vai Trò |
| :--- | :--- | :--- |
| **Ngôn ngữ Lõi** | Python 3.11+ | Chuẩn mực ngành cho Quantitative Finance & Data Science. |
| **Xử lý Dữ liệu** | `pandas`, `numpy` | Biến đổi chuỗi thời gian vectorized tốc độ cao, tính toán cửa sổ trượt (rolling window) tối ưu bộ nhớ. |
| **Phân Tích Thống Kê** | `scipy.stats`, `statsmodels` | Kiểm định Jarque-Bera, khớp phân phối Student-t, tính toán hệ số tự tương quan ACF/PACF và kiểm định giả thuyết. |
| **Mô Hình Học Máy** | `scikit-learn`, `xgboost`, `lightgbm` | Mô hình cây quyết định tăng cường gradient (GBDT) học quan hệ phi tuyến; kiểm lỗi Time-Series CV chống lookahead leakage. |
| **Trực Quan Hóa** | `matplotlib`, `seaborn`, `plotly` | Tạo biểu đồ tĩnh chất lượng cao (High-DPI) và đồ thị tương tác phục vụ EDA. |
| **Tạo Báo Cáo** | `jinja2`, HTML5 / CSS3 Paged Media | Tự động hóa việc đóng gói dữ liệu và biểu đồ thành Báo cáo Kỹ thuật 3 trang A4 hoàn chỉnh. |

---

## 3. Cấu Trúc Thư Mục Dự Án (Project Tree)

```
├── README.md                               # Tài liệu Kiến trúc Master (Tiếng Việt)
├── requirements.txt                        # Khai báo thư viện Python bắt buộc
├── data/
│   ├── raw/                                # Lưu trữ file thô ban đầu (data/raw/ds_assessment_data.csv)
│   ├── bronze/                             # Lakehouse Bronze Parquet (raw.parquet)
│   ├── silver/                             # Lakehouse Silver Parquet (cleaned.parquet)
│   └── gold/                               # Lakehouse Gold Parquet (features.parquet)
├── notebooks/
│   ├── 01_task1_signal_characterization.ipynb  # Notebook phân tích Task 1
│   ├── 02_task2_predictive_modeling.ipynb       # Notebook mô hình hóa Task 2
│   └── 03_task3_deep_dive.ipynb                # Notebook nghiên cứu sâu Task 3
├── src/
│   ├── data_quality/                       # Mô-đun Kiểm tra & Làm sạch Dữ liệu
│   │   ├── __init__.py
│   │   ├── data_quality_SPEC.md            # Đặc tả Kỹ thuật (Dev Spec)
│   │   ├── data_quality_SRS.md             # Đặc tả Nghiệp vụ (Business SRS)
│   │   └── cleaner.py                      # Code xử lý gap, outlier, forward fill
│   ├── signal_characterization/           # Mô-đun Đặc trưng hóa Tín hiệu (Task 1)
│   │   ├── __init__.py
│   │   ├── signal_characterization_SPEC.md
│   │   ├── signal_characterization_SRS.md
│   │   └── analyzer.py                     # Code phân phối, regimes, ACF
│   ├── feature_engineering/               # Mô-đun Tạo Đặc Trưng (Features)
│   │   ├── __init__.py
│   │   ├── feature_engineering_SPEC.md
│   │   ├── feature_engineering_SRS.md
│   │   └── generator.py                    # Code tính Parkinson vol, OFI, Trade density...
│   ├── predictive_modeling/               # Mô-đun Mô Hình Dự Đoán (Task 2)
│   │   ├── __init__.py
│   │   ├── predictive_modeling_SPEC.md
│   │   ├── predictive_modeling_SRS.md
│   │   ├── validation.py                   # Time-Aware Purged/Embargoed CV
│   │   └── models.py                       # Huấn luyện XGBoost, Baseline, Calibration
│   ├── deep_dive/                         # Mô-đun Phân Tích Chuyên Sâu (Task 3)
│   │   ├── __init__.py
│   │   ├── deep_dive_SPEC.md
│   │   ├── deep_dive_SRS.md
│   │   └── microstructure.py               # Code hồi quy Kyle's Lambda & Bootstrap 95% CI
│   └── reporting/                         # Mô-đun Sinh Báo Cáo Kỹ Thuật 3 Trang
│       ├── __init__.py
│       ├── reporting_SPEC.md
│       ├── reporting_SRS.md
│       └── generate_report.py              # Script biên dịch báo cáo HTML/PDF
└── reports/
    ├── figures/                            # Thư mục chứa biểu đồ đã xuất
    └── technical_report.html               # Báo cáo Kỹ thuật 3 trang hoàn chỉnh
```

---

## 4. Kiến Trúc Pipeline & Luồng Dữ Liệu (Data Flow)

```
+------------------------------------+
|  data/raw/ds_assessment_data.csv   | (Dữ liệu thô ~260,000 dòng nến 1 phút)
+------------------------------------+
             |
             v
+------------------------------------+
|         src.data_quality           | --> Lấp khoảng trống (gap), phát hiện bất thường, forward-fill giá
+------------------------------------+
             |
             +---------------------------------------+
             |                                       |
             v                                       v
+--------------------------+           +--------------------------+
|src.signal_characterization|           | src.feature_engineering  |
|  (Task 1: Phân phối,     |           | (Tính Parkinson Vol, OFI,|
|   Vol Regimes, ACF)      |           |  Garman-Klass, Density)  |
+--------------------------+           +--------------------------+
             |                                       |
             v                                       v
+--------------------------+           +--------------------------+
|      reports/figures     |           | src.predictive_modeling  |
| (Biểu đồ xuất tự động)   |           | (Task 2: Time-Aware CV,  |
+--------------------------+           |  XGBoost vs Baseline)    |
             ^                         +--------------------------+
             |                                       |
             |                                       v
             |                         +--------------------------+
             |                         |      src.deep_dive       |
             |                         |  (Task 3: Kyle's Lambda  |
             |                         |   & Bootstrap 95% CI)    |
             |                         +--------------------------+
             |                                       |
             +---------------------------------------+
             |
             v
+--------------------------+
|      src.reporting       | --> Sinh file reports/technical_report.html (Tối đa 3 trang A4)
+--------------------------+
```

---

## 5. Quy Chuẩn Dữ Liệu Toàn Cục (Global Data Contracts)

### 5.1 Schema Dữ Liệu Thô (`data/raw/ds_assessment_data.csv`)
| Tên Cột | Kiểu Dữ Liệu | Ý Nghĩa / Mô Tả | Đơn Vị / Định Dạng |
| :--- | :--- | :--- | :--- |
| `timestamp` | Datetime (UTC) | Thời điểm mở nến 1 phút | `YYYY-MM-DD HH:MM:SS` |
| `open` | Float64 | Giá mở cửa trong phút | USD / Quote currency |
| `high` | Float64 | Giá cao nhất trong phút | USD / Quote currency |
| `low` | Float64 | Giá thấp nhất trong phút | USD / Quote currency |
| `close` | Float64 | Giá đóng cửa trong phút | USD / Quote currency |
| `volume` | Float64 | Tổng khối lượng giao dịch đồng cơ sở | BTC / Base asset |
| `quote_volume` | Float64 | Tổng giá trị giao dịch đồng định giá | USDT / USD |
| `trades` | Float64 | Số lượng khớp lệnh riêng lẻ | Số lượt (Count) |
| `taker_buy_volume` | Float64 | Khối lượng khớp bởi lệnh mua chủ động | BTC / Base asset |

### 5.2 Schema Ma Trận Đặc Trưng (`df_features`)
| Tên Feature | Công Thức / Nguồn Tính | Ý Nghĩa Tài Chính & Kỳ Vọng Hữu Ích |
| :--- | :--- | :--- |
| `log_return` | $\ln(Close_t / Close_{t-1})$ | Tỷ suất lợi nhuận 1 phút close-to-close. |
| `rolling_vol_60m` | $\sigma(r, N=60) \times \sqrt{525600}$ | Độ biến động trượt 60 phút quy năm. |
| `parkinson_vol_15m` | $\sqrt{\frac{1}{4 \ln 2} \sum \ln(H/L)^2}$ | Ước lượng độ biến động qua khoảng High/Low 15 phút (hiệu quả hơn close-to-close). |
| `garman_klass_vol` | $0.5 \ln(H/L)^2 - (2\ln 2-1)\ln(C/O)^2$ | Độ biến động bao gồm cả khoảng nhảy nến Open/Close. |
| `ofi_ratio` | $\frac{taker\_buy\_volume}{volume + \epsilon}$ | Tỷ lệ mất cân bằng dòng lệnh (OFI: >0.5 lực mua áp đảo, <0.5 lực bán áp đảo). |
| `trade_density` | $\frac{volume}{trades + \epsilon}$ | Khối lượng trung bình mỗi lệnh (phân biệt tổ chức vs nhỏ lẻ). |
| `volume_spike_z` | $\frac{volume - \mu_{vol, 60m}}{\sigma_{vol, 60m}}$ | Z-score khối lượng nến hiện tại so với 60 phút trước (phát hiện dòng tiền bất thường). |
| `target_vol_spike_15m` | $\mathbb{I}\left(vol_{t+15m} \ge Q_{0.80}\right)$ | Target nhị phân: Bùng nổ biến động giá trong 15 phút tới. |

---

## 6. Hướng Dẫn Cài Đặt & Thực Thi Hệ Thống (Installation & Quickstart Guide)

### 6.1 Cài Đặt Môi Trường Với `uv` (Khuyên Dùng - Siêu Tốc)

```bash
# 1. Clone repository về máy local
git clone https://github.com/GiangSon-5/hft-signal-prediction-microstructure.git
cd hft-signal-prediction-microstructure

# 2. Tạo môi trường ảo .venv với Python 3.11 sử dụng uv
uv venv .venv --python 3.11

# 3. Kích hoạt môi trường (Windows PowerShell)
.\.venv\Scripts\activate

# 3b. Kích hoạt môi trường (Linux / macOS)
source .venv/bin/activate

# 4. Cài đặt toàn bộ thư viện từ requirements.txt qua uv pip
uv pip install -r requirements.txt

# 5. Đăng ký Kernel cho Jupyter Notebook
python -m ipykernel install --user --name hft_ds_py311 --display-name "Python 3.11 (.venv)"
```

---

### 6.2 Hướng Dẫn Thực Thi Pipeline & Jupyter Notebooks

1. **Thực thi phân tích Task 1 (Signal Characterization):**
   Mở và chạy file [notebooks/01_task1_signal_characterization.ipynb](file:///c:/Users/Admin/Desktop/Data%20Scientist/notebooks/01_task1_signal_characterization.ipynb) để tính toán đặc trưng phân phối, volatility regimes và xuất biểu đồ vào `reports/figures/`.

2. **Thực thi phân tích Task 2 (Predictive Modeling):**
   Mở và chạy file [notebooks/02_task2_predictive_modeling.ipynb](file:///c:/Users/Admin/Desktop/Data%20Scientist/notebooks/02_task2_predictive_modeling.ipynb) để trích xuất tập feature, thực hiện Time-Aware Purged/Embargoed CV và huấn luyện XGBoost/LightGBM.

3. **Thực thi phân tích Task 3 (Deep Dive & Microstructure):**
   Mở và chạy file [notebooks/03_task3_deep_dive.ipynb](file:///c:/Users/Admin/Desktop/Data%20Scientist/notebooks/03_task3_deep_dive.ipynb) để kiểm định hiện tượng Kyle's Lambda và chạy Block Bootstrap 95% Confidence Interval.

4. **Tự động sinh Báo cáo Kỹ thuật (Technical Report 3 trang):**
   ```bash
   python -m src.reporting.generate_report
   ```

---

## 7. Bảng Tóm Tắt Kết Quả Thực Nghiệm Task 1 & Lộ Trình Chuyển Giao Task 2

### 7.1 Kết Quả Thực Nghiệm Định Lượng Task 1 (Task 1 Empirical Results Summary)

| Hạng Mục Kiểm Định | Thuật Toán & Phương Pháp | Kết Quả Định Lượng Thực Nghiệm | Kết Luận & Ý Nghĩa Quản Trị Rủi Ro |
| :--- | :--- | :--- | :--- |
| **Data Integrity Audit** | Grid 1m Audit & Forward-fill | **100% Sạch** ($N = 264,961$ nến, 0 Gap, 0 Lỗi OHLCV) | Dữ liệu đạt độ toàn vẹn tuyệt đối H2 2024, không cần tạo nến ảo. |
| **Phân Phối Return 1m** | Jarque-Bera Test & Student-t | $JB = 47,347,809.45$ ($p = 0.0$), $df = 2.665 < 3.0$ | Bác bỏ phân phối chuẩn; 100% thuộc tính đuôi béo (Fat-tails). |
| **Volatility Regimes** | Rolling Std 60m & Quantile $Q_{0.75}$ | Low Vol: 75% ($<55.61\%$), High Vol: 25% ($\ge 55.61\%$) | Xác nhận hiện tượng cụm biến động (Volatility Clustering). |
| **Volume - Trades - Price** | Spearman Rank Correlation | $r_{\text{Volume}} = +0.7601$, $r_{\text{Trades}} = +0.7184$ | Biến động giá lớn bắt buộc đi kèm CẢ Volume lớn LẪN Trades dồn dập. |
| **Regime Fat-Tails** | Sub-sample Student-t Fit | Low Vol $df = 3.90$ vs High Vol $df = 1.990 \le 2.0$ | High Vol Regime rơi vào ranh giới Phương sai vô hạn (Infinite Variance Hazard). |
| **Autocorrelation (ACF)** | ACF/PACF Lags 1-30m (95% CI) | Lag 1m $r_1 = -0.0057 < -0.0038$ ($p < 0.05$) | Hiện tượng Đảo chiều vi mô (Bid-Ask Bounce) & Chu kỳ bot TWAP (7-13m, 30m). |

### 7.2 Lộ Trình Chuyển Giao Sang Task 2 (Task 2 Readiness & Feature Pipeline)

1. **Định nghĩa Binary Target ($Y_t$):** Bùng nổ biến động 15m tới $Y_t = \mathbb{I}\left(\sigma_{fwd, 15m} \ge Q_{0.80}\right)$.
2. **Bộ Đặc Trưng Vi Mô ($8$ Features):** `parkinson_vol_15m`, `garman_klass_vol_15m`, `ofi_ratio`, `trade_density`, `volume_spike_z_60m`, `return_momentum_15m`, `rolling_vol_60m`, `spread_ratio_15m`.
3. **Chiến Lược CV Chống Rò Rỉ:** Time-Aware Purged & Embargoed TimeSeries Split 5-Fold (Purge 15m, Embargo 30m).
4. **Mô Hình & Hiệu Chỉnh:** Huấn luyện GBDT (HistGradientBoosting/XGBoost) vs Rule-based Baseline, đánh giá ROC-AUC, PR-AUC, F1-Score và Isotonic Probability Calibration (Brier Score).

---

### 7.3 Hướng Dẫn Tái Sử Dụng Thư Viện Cốt Lõi (`cleaner.py` & `analyzer.py`) Cho Task 2 & Task 3

Toàn bộ logic xử lý dữ liệu và thuật toán toán học của Task 1 đã được đóng gói thành các hàm chuẩn trong tầng `src/` để tái sử dụng xuyên suốt dự án:

#### 1. Thư Viện Tiền Xử Lý Dữ Liệu ([`src/data_quality/cleaner.py`](file:///c:/Users/Admin/Desktop/Data%20Scientist/src/data_quality/cleaner.py)):
- **`audit_data_quality(df: pd.DataFrame) -> dict`**: Kiểm định toàn diện số lượng NaN, phân tích gaps $\Delta t > 1\text{m}$, phát hiện vi phạm logic OHLCV.
- **`validate_and_clean_time_series(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]`**: Chuẩn hóa `timestamp` sang `Datetime64[ns]`, tái lập 1-min grid, lấp nến có điều kiện (Forward-fill Close, Zero-fill Volume).
- *Cách dùng trong Task 2 & Task 3*:
  ```python
  from src.data_quality.cleaner import validate_and_clean_time_series
  df_clean, audit = validate_and_clean_time_series(pd.read_csv('data/raw/ds_assessment_data.csv'))
  ```

#### 2. Thư Viện Phân Tích Định Lượng ([`src/signal_characterization/analyzer.py`](file:///c:/Users/Admin/Desktop/Data%20Scientist/src/signal_characterization/analyzer.py)):
- **`calculate_log_returns(df, col='close')`**: Tính tỷ suất lợi nhuận Log 1m: $r_t = \ln(P_t / P_{t-1})$.
- **`analyze_returns_distribution(returns)`**: Tính Mean, Std, Skewness, Kurtosis, kiểm định Jarque-Bera và khớp Student-t $df$.
- **`compute_rolling_volatility(returns, window=60)`**: Tính độ biến động trượt 60m quy năm ($\times \sqrt{525,600}$).
- **`detect_volatility_regimes(rolling_vol, threshold_quantile=0.75)`**: Phân tách 2 Chế độ biến động Low Vol vs High Vol tại ngưỡng $Q_{0.75}$.
- **`compare_volatility_regimes(returns, regimes)`**: So sánh Kurtosis và bậc tự do Student-t giữa các chế độ biến động.
- **`analyze_volume_trades_range(df)`**: Tính ma trận tương quan Spearman giữa Volume, Trades và Price Range.
- **`analyze_autocorrelation(returns, nlags=30)`**: Tính ACF/PACF 30 lags và dải tin cậy 95%.
- *Cách dùng trong Task 2 & Task 3*:
  ```python
  from src.signal_characterization.analyzer import (
      calculate_log_returns,
      compute_rolling_volatility,
      detect_volatility_regimes
  )
  df_clean['log_return'] = calculate_log_returns(df_clean)
  df_clean['rolling_vol_60m'] = compute_rolling_volatility(df_clean['log_return'], window=60)
  df_clean['regime'], cutoff = detect_volatility_regimes(df_clean['rolling_vol_60m'], threshold_quantile=0.75)
  ```

