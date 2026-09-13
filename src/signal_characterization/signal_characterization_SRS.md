### UC: Quantitative Signal Characterization - Hệ thống Quantitative High-Frequency Engine

**Mô tả chức năng tổng quan**
Module `signal_characterization` thực hiện phân tích thống kê chuyên sâu trên chuỗi thời gian tỷ suất lợi nhuận 1 phút. Chức năng bao gồm đánh giá tính chuẩn của phân phối, tính toán độ biến động trượt 60 phút, phân cụm chế độ biến động (volatility regimes), phân tích quan hệ giữa khối lượng/số lượng giao dịch với biên độ giá, và kiểm định tự tương quan (autocorrelation).

| Primary Actor: | Quantitative Researcher | Secondary Actor: | Reporting & Dashboard Module |
|---|---|---|---|
| **Description:** | Trích xuất các thuộc tính thống kê cốt lõi của thị trường 1 phút H2 2024 để làm cơ sở cho chiến lược giao dịch và quản trị rủi ro. |
| **Trigger:** | Yêu cầu thực thi Task 1 từ Notebook hoặc Script tự động. |
| **Preconditions:** | PRE1: Tập dữ liệu sạch `clean_ohlcv.parquet` đã được tạo từ module `data_quality`. |
| **Post-conditions:** | POST1: Các chỉ số thống kê, biểu đồ phân phối, biểu đồ 2 regime biến động và ma trận tương quan được xuất ra thư mục `reports/figures/`. |

**Business Scenario Walkthrough:**
- **Khách hàng đưa vào:** Dữ liệu nến 1 phút đã làm sạch (~260,000 nến).
- **Hệ thống xử lý:** Lập biểu đồ histogram phân phối log return, chạy kiểm định Jarque-Bera, tính rolling std 60 phút, phân cụm GMM thành 2 regime (Low Vol vs High Vol), tính hệ số ACF từ lag 1 đến 60.
- **Kết quả nhận được:** Báo cáo xác nhận phân phối không chuẩn (leptokurtic, fat-tailed), trực quan hóa rõ ràng 2 chế độ biến động, xác nhận biên độ giá lớn luôn đi kèm đồng thời cả khối lượng cao và số lượng giao dịch nhiều.

**Normal Flow:**
| Step | Actor Action | System Response |
|---|---|---|
| 1 | Nhà nghiên cứu gọi function `analyze_signals(df_clean)`. | Hệ thống tính log return 1 phút $r_t = \ln(Close_t / Close_{t-1})$. |
| 2 | Hệ thống tính toán skewness, kurtosis và thực hiện kiểm định Jarque-Bera. | Trả về kết quả bác bỏ giả thuyết phân phối chuẩn ($p < 0.001$). |
| 3 | Hệ thống tính độ biến động trượt 60 phút và áp dụng GMM 2 cụm. | Phân loại toàn bộ các khoảng thời gian thành Regime 0 (Biến động thấp) hoặc Regime 1 (Biến động cao). |
| 4 | Hệ thống phân tích tương quan Price Movement vs Volume vs Trades. | Xuất ma trận tương quan Spearman/Pearson và đồ thị scatter 3D/bubble. |
| 5 | Hệ thống tính autocorrelation (ACF/PACF). | Xác định các lag có ý nghĩa thống kê (vượt dải phân cách 95% CI). |

**Exception:**
| No | Cause | System Response |
|---|---|---|
| 1 | Chuỗi dữ liệu quá ngắn (<60 phút) không đủ window rolling. | Thông báo lỗi `ValueError("Dataset length insufficient for 60m rolling window")`. |
| 2 | Thuật toán GMM không hội tụ. | Tự động switch sang phương pháp K-Means hoặc threshold quantiles (80th percentile) làm fallback. |

**Business Rules:**
| No | Rule |
|---|---|
| 1 | Độ biến động trượt phải được tính dựa trên log returns và quy đổi theo chuỗi chuẩn hóa (annualized hoặc 60m standard deviation). |
| 2 | Bắt buộc xác định ít nhất 2 chế độ biến động có sự khác biệt rõ rệt về mức variance trung bình ($Ratio \ge 2.0$). |
| 3 | Mọi kết luận về autocorrelation phải dựa trên mức ý nghĩa thống kê $\alpha = 0.05$. |

**Bảng mô tả giao diện (UI) / Data Mapping:**
**1. Dữ liệu có cấu trúc (Structured):**
| Tên trường | Mô tả | Kiểu dữ liệu / Control | Dữ liệu mặc định | Bắt buộc | Ví dụ minh họa | Direct to |
|---|---|---|---|---|---|---|
| `mean_return` | Giá trị trung bình lợi nhuận 1m | Float64 / Label | `0.0` | Y | `0.0000012` | Technical Report |
| `kurtosis` | Độ nhọn phân phối (Excess Kurtosis) | Float64 / Label | `0.0` | Y | `24.85` | Technical Report |
| `skewness` | Độ lệch phân phối | Float64 / Label | `0.0` | Y | `-0.42` | Technical Report |
| `jb_pvalue` | p-value của kiểm định Jarque-Bera | Float64 / Label | `0.0` | Y | `0.0000` | Risk Assessment |
| `vol_regime` | Nhãn chế độ biến động (0=Low, 1=High) | Int64 / Category | `0` | Y | `1` | Feature Engineering |
| `lag1_acf` | Hệ số tự tương quan tại Lag 1 | Float64 / Metric | `0.0` | Y | `-0.0415` | Market Microstructure |
