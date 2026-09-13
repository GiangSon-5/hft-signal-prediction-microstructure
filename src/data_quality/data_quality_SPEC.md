# Đặc Tả Kỹ Thuật (Module Specification): Kiểm Tra & Làm Sạch Dữ Liệu (`src/data_quality`)

## 1. Tổng Quan Mô-đun

Mô-đun `src/data_quality/cleaner.py` chịu trách nhiệm nạp, kiểm định độ toàn vẹn, phát hiện bất thường logic OHLCV, phân tích khoảng trống thời gian (time gaps), và thực thi đường ống làm sạch có điều kiện (conditional cleaning pipeline) cho dữ liệu chuỗi thời gian nến 1 phút (`ds_assessment_data.csv`).

Mô-đun được chuẩn hóa $100\%$ theo các bước thực nghiệm của Task 1 để tái sử dụng xuyên suốt cho Task 2 (Predictive Modeling), Task 3 (Deep Dive Microstructure) và Pipeline Báo Cáo Tự Động (`src/reporting`).

---

## 2. Giao Diện Lập Trình Ứng Dụng (API Contracts)

Mô-đun [`src/data_quality/cleaner.py`](file:///c:/Users/Admin/Desktop/Data%20Scientist/src/data_quality/cleaner.py) cung cấp 2 hàm cốt lõi:

### 2.1 `audit_data_quality(df: pd.DataFrame) -> Dict[str, Any]`
Kiểm tra toàn diện các chỉ số chất lượng dữ liệu thô:
- **Missing Values**: Đếm số lượng Null/NaN trên 9 biến số.
- **Timestamp & Gaps**: Phân tích khoảng trống thời gian $\Delta t > 1\text{m}$ và tổng số phút khuyết thiếu.
- **Invalid Prices**: Đếm số mẫu có giá $\le 0$.
- **OHLC Logic Violations**: Đếm số mẫu vi phạm $(High < Low) \lor (High < Open/Close) \lor (Low > Open/Close)$.

### 2.2 `validate_and_clean_time_series(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]`
Thực hiện chuẩn hóa chuỗi thời gian có điều kiện:
- **Chuẩn hóa Timestamp**: Ép kiểu `pd.to_datetime(utc=True).dt.tz_localize(None)` về `Datetime64[ns]` timezone-naive.
- **Tạo Lưới Thời Gian 1m**: Tái lập `pd.date_range(start=min_time, end=max_time, freq='1min')`.
- **Lấp Nến Có Điều Kiện (Conditional Gap Filling)**:
  - Nếu phát hiện nến khuyết (`missing_count > 0`): Forward-fill giá Close, đồng bộ Open/High/Low theo Close, gán $0.0$ cho toàn bộ các cột khối lượng (`volume`, `quote_volume`, `trades`, `taker_buy_volume`).
  - Nếu chuỗi đã đạt $100\%$ liên tục (`missing_count == 0`): Giữ nguyên $100\%$ dữ liệu gốc, không phát sinh nến ảo.
- *Alias tương thích ngược*: `clean_market_data = validate_and_clean_time_series`.

---

## 3. Quy Chuẩn Dữ Liệu Đầu Vào & Đầu Ra

### 3.1 Cấu Trúc Dữ Liệu Đầu Vào (Input Schema)
| Tên Cột | Kiểu Dữ Liệu | Ràng Buộc Kỹ Thuật | Ý Nghĩa Tài Chính |
| :--- | :--- | :--- | :--- |
| `timestamp` | `object / datetime64` | Chuỗi thời gian | Thời điểm đóng/mở nến 1 phút |
| `open` | `float64` | $Open > 0$ | Giá mở cửa |
| `high` | `float64` | $High \ge \max(Open, Close)$ | Giá cao nhất trong phút |
| `low` | `float64` | $Low \le \min(Open, Close)$ | Giá thấp nhất trong phút |
| `close` | `float64` | $Close > 0$ | Giá đóng cửa |
| `volume` | `float64` | $Volume \ge 0$ | Khối lượng coin cơ sở (BTC) |
| `quote_volume` | `float64` | $QuoteVolume \ge 0$ | Khối lượng định giá (USDT) |
| `trades` | `float64` | $Trades \ge 0$ | Số lượng giao dịch khớp lệnh |
| `taker_buy_volume` | `float64` | $0 \le TakerBuy \le Volume$ | Khối lượng mua chủ động (Market Buy) |

### 3.2 Kết Quả Kiểm Định Thực Nghiệm Trên Tập Dữ Liệu Gốc (H2 2024)
```json
{
  "total_records": 264961,
  "start_timestamp": "2024-06-30 17:00:00",
  "end_timestamp": "2024-12-31 17:00:00",
  "missing_values_count": 0,
  "detected_time_gaps_gt_1m": 0,
  "missing_minutes_total": 0,
  "ohlc_logic_violations": 0,
  "is_fully_continuous": true
}
```

---

## 4. Hướng Dẫn Tái Sử Dụng Cho Task 2 & Task 3

### Cách Tích Hợp Vào Task 2 (Predictive Modeling):
```python
import pandas as pd
from src.data_quality.cleaner import validate_and_clean_time_series

# Tải dữ liệu thô
df_raw = pd.read_csv('../ds_assessment_data.csv')

# Tiền xử lý dữ liệu sạch
df_clean, summary = validate_and_clean_time_series(df_raw)
df_clean = df_clean.set_index('timestamp')
```

### Cách Tích Hợp Vào Task 3 (Deep Dive & Microstructure):
```python
import pandas as pd
from src.data_quality.cleaner import validate_and_clean_time_series

# Làm sạch dữ liệu phục vụ phân tích Kyle's Lambda và Order Flow Toxicity
df_clean, _ = validate_and_clean_time_series(pd.read_csv('ds_assessment_data.csv'))
```
