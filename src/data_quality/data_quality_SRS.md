### UC: Data Quality Assurance & Sanitization - Hệ thống Quantitative High-Frequency Engine

**Mô tả chức năng tổng quan**
Module `data_quality` tiếp nhận dữ liệu OHLCV 1 phút thô từ file CSV, kiểm tra tính toàn vẹn về thời gian và giá trị, xử lý missing data/gap, loại bỏ các bất thường (anomalies/outliers), và cung cấp tập dữ liệu chuẩn hóa cho các pipeline EDA và Machine Learning.

| Thuộc Tính | Chi Tiết |
| :--- | :--- |
| **Primary Actor** | Quantitative Data Engineer |
| **Secondary Actor** | Automated Data Pipeline / Storage |
| **Description** | Chuẩn hóa chuỗi thời gian 1 phút H2 2024, làm sạch dữ liệu nhiễu và lấp khoảng trống timestamp. |
| **Trigger** | Khởi chạy notebook EDA hoặc script pipeline xử lý dữ liệu. |
| **Preconditions** | PRE1: File `ds_assessment_data.csv` tồn tại và đúng định dạng CSV. |
| **Post-conditions** | POST1: Tập dữ liệu sạch được lưu tại `data/processed/clean_ohlcv.parquet` kèm log báo cáo chất lượng. |

**Business Scenario Walkthrough:**
- **Khách hàng đưa vào:** File `ds_assessment_data.csv` chứa 264,963 dòng dữ liệu 1 phút thô.
- **Hệ thống xử lý:** Parser quét timestamp, phát hiện 12 điểm gap, thực hiện forward-fill giá và zero-fill khối lượng, kiểm tra vi phạm $High \ge Low$.
- **Kết quả nhận được:** DataFrame sạch 100% liên tục theo từng phút, bổ sung cột đánh dấu `is_imputed`.

**Normal Flow:**
| Step | Actor Action | System Response |
|---|---|---|
| 1 | Người dùng gọi function `clean_market_data(raw_csv_path)`. | Hệ thống nạp CSV vào memory và parse cột timestamp sang UTC Datetime. |
| 2 | Hệ thống tự động kiểm tra tính liên tục của chuỗi thời gian. | Phát hiện các khoảng trống phút và tự động reindex tạo lưới thời gian đầy đủ. |
| 3 | Hệ thống áp dụng quy tắc forward-fill giá và zero-fill volume. | Gán nhãn `is_imputed = True` cho các dòng được lấp lỗ trống. |
| 4 | Hệ thống xuất file dữ liệu đã làm sạch. | Báo cáo tóm tắt số lượng dòng thô, dòng lấp trống, và các outliers đã xử lý. |

**Exception:**
| No | Cause | System Response |
|---|---|---|
| 1 | File CSV bị lỗi định dạng hoặc thiếu cột bắt buộc. | Quăng ngoại lệ `ValueError("Missing required OHLCV column")` và dừng pipeline. |
| 2 | Cột timestamp bị lặp lại (duplicate timestamps). | Tự động deduplicate giữ lại bản ghi cuối cùng (`keep='last'`) và ghi warning log. |

**Business Rules:**
| No | Rule |
|---|---|
| 1 | Không bao giờ nội suy (interpolate) giá trị volume cho khoảng trống thời gian; volume phải được gán bằng 0.0. |
| 2 | Giá mở cửa của nến lấp trống phải bằng giá đóng cửa của nến liền trước. |
| 3 | Taker buy volume không bao giờ được phép vượt quá Total Volume ($TakerBuy \le Volume$). |

**Bảng mô tả giao diện (UI) / Data Mapping:**
**1. Dữ liệu có cấu trúc (Structured):**
| Tên trường | Mô tả | Kiểu dữ liệu / Control | Dữ liệu mặc định | Bắt buộc | Ví dụ minh họa | Direct to |
|---|---|---|---|---|---|---|
| `timestamp` | Thời gian nến (UTC) | Datetime64 / Index | N/A | Y | `2024-07-01 00:00:00` | Data Cleaning Pipeline |
| `open` | Giá mở cửa | Float64 / Numeric | N/A | Y | `61603.54` | Feature Engineering |
| `high` | Giá cao nhất | Float64 / Numeric | N/A | Y | `61638.00` | Feature Engineering |
| `low` | Giá thấp nhất | Float64 / Numeric | N/A | Y | `61603.54` | Feature Engineering |
| `close` | Giá đóng cửa | Float64 / Numeric | N/A | Y | `61638.00` | Signal Analysis & Modeling |
| `volume` | Khối lượng base asset | Float64 / Numeric | `0.0` | Y | `5.83527` | Feature Engineering |
| `quote_volume` | Giá trị quote asset | Float64 / Numeric | `0.0` | Y | `359571.18` | Feature Engineering |
| `trades` | Số lượng giao dịch | Float64 / Numeric | `0` | Y | `591.0` | Microstructure Analysis |
| `taker_buy_volume` | Khối lượng lệnh mua chủ động | Float64 / Numeric | `0.0` | Y | `5.51656` | Order Flow Analysis |
