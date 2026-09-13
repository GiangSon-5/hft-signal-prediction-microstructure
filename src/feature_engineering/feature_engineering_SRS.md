### UC: High-Frequency Feature Extraction - Hệ thống Quantitative High-Frequency Engine

**Mô tả chức năng tổng quan**
Module `feature_engineering` trích xuất và biến đổi các thuộc tính nến OHLCV thô thành ma trận đặc trưng (feature matrix) sẵn sàng cho mô hình Machine Learning. Module tính toán các chỉ số độ biến động nâng cao (Parkinson, Garman-Klass), chỉ số dòng lệnh chủ động (Order Flow Imbalance), mật độ giao dịch, z-score khối lượng bùng nổ, và động lượng tỷ suất lợi nhuận.

| Primary Actor: | Quantitative Developer | Secondary Actor: | Predictive Modeling Module |
|---|---|---|---|
| **Description:** | Tạo ra ít nhất 6 đặc trưng domain-informed từ nến 1 phút mà không bị rò rỉ dữ liệu tương lai. |
| **Trigger:** | Yêu cầu trích xuất đặc trưng cho Task 2 từ Notebook hoặc Script tự động. |
| **Preconditions:** | PRE1: Dữ liệu nến 1 phút sạch `clean_ohlcv.parquet` đã sẵn sàng. |
| **Post-conditions:** | POST1: Ma trận đặc trưng `df_features` được khởi tạo và kiểm tra tính dừng (stationarity). |

**Business Scenario Walkthrough:**
- **Khách hàng đưa vào:** DataFrame nến 1 phút liên tục.
- **Hệ thống xử lý:** Chạy các hàm vectorized rolling calculation (cửa sổ 15m và 60m), tính OFI ratio, Parkinson vol, Garman-Klass vol, Trade Density, Volume Z-score.
- **Kết quả nhận được:** DataFrame chứa 6 đặc trưng mới kèm theo target dự đoán nhị phân (forward 15m volatility spike).

**Normal Flow:**
| Step | Actor Action | System Response |
|---|---|---|
| 1 | Người dùng gọi function `generate_features(df_clean)`. | Hệ thống kiểm tra cấu trúc cột đầu vào. |
| 2 | Hệ thống tính toán các chỉ số biến động Parkinson & Garman-Klass (15m window). | Tạo các cột `parkinson_vol_15m` và `garman_klass_vol_15m`. |
| 3 | Hệ thống tính Order Flow Imbalance Ratio và Trade Density. | Tạo các cột `ofi_ratio` và `trade_density`. |
| 4 | Hệ thống tính Volume Spike Z-Score (60m window) và Momentum (15m return). | Tạo các cột `volume_spike_z_60m` và `return_momentum_15m`. |
| 5 | Hệ thống drop bớt $N=59$ nến đầu tiên do khoảng khởi động rolling window (warmup period). | Trả về ma trận `df_features` hoàn chỉnh không chứa NaNs. |

**Exception:**
| No | Cause | System Response |
|---|---|---|
| 1 | Dữ liệu đầu vào chứa giá trị âm hoặc $High < Low$. | Quăng lỗi `AssertionError("High price must be >= Low price")`. |
| 2 | Cột `taker_buy_volume` bị khuyết. | Tự động giả định $OFI = 0.5$ (trung tính) và ghi warning log. |

**Business Rules:**
| No | Rule |
|---|---|
| 1 | Mọi đặc trưng rolling phải strictly lookback (chỉ dùng dữ liệu từ $t-K$ đến $t$), tuyệt đối cấm dùng dữ liệu $t+1$. |
| 2 | Mọi phép chia phải được bảo vệ bằng hằng số $\epsilon = 10^{-8}$ để tránh lỗi divide-by-zero. |
| 3 | Loại bỏ các dòng nến warmup ban đầu để đảm bảo mô hình ML nhận đầu vào sạch 100%. |

**Bảng mô tả giao diện (UI) / Data Mapping:**
**1. Dữ liệu có cấu trúc (Structured):**
| Tên trường | Mô tả | Kiểu dữ liệu / Control | Dữ liệu mặc định | Bắt buộc | Ví dụ minh họa | Direct to |
|---|---|---|---|---|---|---|
| `parkinson_vol_15m` | Độ biến động High/Low (15m) | Float64 / Numeric | N/A | Y | `0.00150` | Predictive Modeling |
| `garman_klass_vol_15m`| Độ biến động OHLC (15m) | Float64 / Numeric | N/A | Y | `0.00162` | Predictive Modeling |
| `ofi_ratio` | Tỷ lệ mua chủ động (OFI) | Float64 / Numeric | `0.5` | Y | `0.700` | Feature Importance |
| `trade_density` | Khối lượng trung bình/giao dịch | Float64 / Numeric | `0.0` | Y | `0.250` | Feature Importance |
| `volume_spike_z_60m` | Z-score khối lượng (60m) | Float64 / Numeric | `0.0` | Y | `2.15` | Predictive Modeling |
| `return_momentum_15m` | Lợi nhuận tích lũy 15 phút | Float64 / Numeric | `0.0` | Y | `0.00133` | Predictive Modeling |
