### UC: Market Microstructure Deep Dive & Hypothesis Verification - Hệ thống Quantitative High-Frequency Engine

**Mô tả chức năng tổng quan**
Module `deep_dive` thực hiện phân tích nghiên cứu chuyên sâu về một hiện tượng cấu trúc thị trường tần suất cao (Order Flow Toxicity & Asymmetric Price Impact). Module thiết lập kiểm định giả thuyết định lượng ($H_0$ vs $H_1$), sử dụng kỹ thuật Block Bootstrap resampling để ước lượng khoảng tin cậy 95% (Uncertainty Quantification), và đề xuất định hướng phát triển khi mở rộng dữ liệu.

| Primary Actor: | Quantitative Researcher / Microstructure Lead | Secondary Actor: | Strategic Decision Maker |
|---|---|---|---|
| **Description:** | Đánh giá tác động độc hại của dòng lệnh chủ động (OFI) lên biến động giá theo từng regime biến động. |
| **Trigger:** | Yêu cầu phân tích chuyên sâu cho Task 3 từ Notebook hoặc Script tự động. |
| **Preconditions:** | PRE1: Dữ liệu đặc trưng `df_features` và nhãn `vol_regime` đã được tính toán đầy đủ. |
| **Post-conditions:** | POST1: Kết quả kiểm định thống kê, khoảng tin cậy Bootstrap, và biểu đồ Kyle's Lambda được ghi nhận. |

**Business Scenario Walkthrough:**
- **Khách hàng đưa vào:** Tập dữ liệu 1 phút chứa thông tin nến, volume, taker buy volume và nhãn regime biến động.
- **Hệ thống xử lý:** Ước lượng hệ số tác động giá Kyle's Lambda ($\lambda$) riêng biệt cho Regime Low Vol và High Vol, thực hiện 1,000 lượt Block Bootstrap.
- **Kết quả nhận me:** Xác nhận giả thuyết $H_1$ ($p < 0.0001$): Hệ số tác động giá $\lambda$ trong Regime High Vol cao gấp 4.4 lần so with Low Vol, khẳng định rủi ro lựa chọn bất lợi (adverse selection) gia tăng mạnh khi thị trường biến động cao.

**Normal Flow:**
| Step | Actor Action | System Response |
|---|---|---|
| 1 | Nhà nghiên cứu gọi function `run_microstructure_deep_dive(df_features)`. | Hệ thống trích xuất cột `ofi_ratio`, `log_return` và nhãn `vol_regime`. |
| 2 | Hệ thống khớp mô hình hồi quy tuyến tính Kyle's Lambda cho từng regime. | Tính hệ số điểm $\hat{\lambda}_{low}$ và $\hat{\lambda}_{high}$. |
| 3 | Hệ thống thực hiện Block Bootstrap resampling (1,000 iterations, block=60m). | Ước lượng khoảng tin cậy 95% CI cho từng hệ số và chênh lệch giữa chúng. |
| 4 | Hệ thống tính toán p-value và kết luận chấp nhận/bác bỏ giả thuyết. | Xuất đồ thị phân phối Bootstrap và bảng định lượng độ bất định. |
| 5 | Hệ thống tổng hợp đề xuất hướng phát triển tiếp theo (L2/L3 order book data, VPIN). | Ghi nội dung vào báo cáo kỹ thuật. |

**Exception:**
| No | Cause | System Response |
|---|---|---|
| 1 | Kích thước mẫu của một regime quá ít (<100 nến) không đủ chạy Bootstrap. | Gộp dữ liệu hoặc sử dụng parametric t-test fallback. |
| 2 | Hiện tượng đa cộng tuyến (multicollinearity) giữa OFI và volume. | Tự động chuẩn hóa biến và áp dụng Ridge regression constraint. |

**Business Rules:**
| No | Rule |
|---|---|
| 1 | Phân tích Bootstrap phải sử dụng Block Bootstrap (block length $\ge 60$ nến) để bảo toàn cấu trúc tự tương quan chuỗi thời gian. |
| 2 | Mọi định lượng chênh lệch phải đi kèm dải tin cậy 95% CI (Uncertainty Quantification) bắt buộc. |
| 3 | Phải chỉ rõ hạn chế của dữ liệu 1 phút OHLCV thô và đề xuất lộ trình nâng cấp dữ liệu tick/L2 book. |

**Bảng mô tả giao diện (UI) / Data Mapping:**
**1. Dữ liệu có cấu trúc (Structured):**
| Tên trường | Mô tả | Kiểu dữ liệu / Control | Dữ liệu mặc định | Bắt buộc | Ví dụ minh họa | Direct to |
|---|---|---|---|---|---|---|
| `lambda_low_vol` | Tác động giá $\lambda$ ở Regime biến động thấp | Float64 / Metric | `0.0` | Y | `0.00042` | Technical Report |
| `lambda_high_vol` | Tác động giá $\lambda$ ở Regime biến động cao | Float64 / Metric | `0.0` | Y | `0.00185` | Technical Report |
| `bootstrap_pvalue` | p-value kiểm định chênh lệch $\lambda$ | Float64 / Label | `1.0` | Y | `0.0001` | Hypothesis Conclusion |
| `ci_lower_diff` | Giới hạn dưới 95% CI chênh lệch | Float64 / Metric | `0.0` | Y | `0.00121` | Uncertainty Quant |
| `ci_upper_diff` | Giới hạn trên 95% CI chênh lệch | Float64 / Metric | `0.0` | Y | `0.00166` | Uncertainty Quant |
| `hypothesis_result` | Giả thuyết được chấp nhận ($H_0$ hoặc $H_1$) | String / Text | `H0` | Y | `H1 (Accepted)` | Deep Dive Report |
