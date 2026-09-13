### UC: Automated Technical Report Compilation - Hệ thống Quantitative High-Frequency Engine

**Mô tả chức năng tổng quan**
Module `reporting` tự động tổng hợp toàn bộ kết quả phân tích thống kê, đồ thị trực quan, chỉ số mô hình ML, và kết quả kiểm định chuyên sâu thành một **Báo cáo Kỹ thuật (Technical Report)** chuẩn mực dài tối đa 3 trang định dạng HTML/PDF.

| Primary Actor: | Quantitative Researcher / Lead Data Scientist | Secondary Actor: | Hiring Manager / Executive Stakeholder |
|---|---|---|---|
| **Description:** | Đóng gói toàn bộ kết quả phân tích Task 1, Task 2, Task 3 thành báo cáo kỹ thuật 3 trang cô đọng, đẹp mắt, và tự giải thích. |
| **Trigger:** | Yêu cầu tạo báo cáo cuối cùng từ Notebook hoặc Script `generate_report.py`. |
| **Preconditions:** | PRE1: Tất cả các figures (.png) đã được xuất ra thư mục `reports/figures/`. |
| **Post-conditions:** | POST1: Báo cáo `reports/technical_report.html` được sinh ra hoàn chỉnh và sẵn sàng để in PDF. |

**Business Scenario Walkthrough:**
- **Khách hàng đưa vào:** Dictionary các metrics thống kê và danh sách file hình ảnh trực quan.
- **Hệ thống xử lý:** Nạp template HTML, mã hóa base64 hình ảnh, render các bảng số liệu, và áp dụng quy tắc CSS Paged Media.
- **Kết quả nhận được:** File HTML báo cáo độc lập (self-contained), vừa vặn 3 trang A4 khi xem trên trình duyệt hoặc in PDF.

**Normal Flow:**
| Step | Actor Action | System Response |
|---|---|---|
| 1 | Người dùng gọi script `python -m src.reporting.generate_report`. | Hệ thống kiểm tra sự tồn tại của các file hình ảnh trong `reports/figures/`. |
| 2 | Hệ thống đọc các file PNG và chuyển đổi sang dạng chuỗi Base64. | Bảo đảm báo cáo không phụ thuộc vào đường dẫn hình ảnh ngoài. |
| 3 | Hệ thống điền các giá trị thống kê vào template HTML Jinja2. | Tạo nội dung cho 3 trang báo cáo. |
| 4 | Hệ thống xuất file `reports/technical_report.html`. | Hiển thị thông báo thành công và kích thước file báo cáo. |

**Exception:**
| No | Cause | System Response |
|---|---|---|
| 1 | Thiếu file hình ảnh trực quan trong `reports/figures/`. | Quăng ngoại lệ `FileNotFoundError` chỉ rõ hình ảnh nào bị thiếu. |
| 2 | Nội dung vượt quá 3 trang A4. | Điều chỉnh kích thước CSS max-height của hình ảnh và margin để co gọn vừa 3 trang. |

**Business Rules:**
| No | Rule |
|---|---|
| 1 | Báo cáo bắt buộc có độ dài tối đa 3 trang A4, không được để lọt sang trang thứ 4. |
| 2 | Mọi hình ảnh phải nhúng trực tiếp dạng Base64 để file HTML hoàn toàn tự chứa (self-contained). |
| 3 | Báo cáo phải thể hiện đầy đủ 3 phần: Task 1 Signal Profiling, Task 2 Time-Aware Predictive ML, Task 3 Deep Dive & Limitations. |

**Bảng mô tả giao diện (UI) / Data Mapping:**
**1. Dữ liệu có cấu trúc (Structured):**
| Tên trường | Mô tả | Kiểu dữ liệu / Control | Dữ liệu mặc định | Bắt buộc | Ví dụ minh họa | Direct to |
|---|---|---|---|---|---|---|
| `report_title` | Tiêu đề báo cáo kỹ thuật | String / Text | N/A | Y | `Technical Assessment: High-Frequency Market Data` | Report Header |
| `author_name` | Tên tác giả / Data Scientist | String / Text | `Candidate` | Y | `Quantitative Data Scientist` | Report Header |
| `task1_figure` | Biểu đồ phân phối lợi nhuận & regimes | Base64 String / Img | N/A | Y | `data:image/png;base64,...` | Page 1 Layout |
| `task2_figure` | Biểu đồ ROC/PR curve & Reliability | Base64 String / Img | N/A | Y | `data:image/png;base64,...` | Page 2 Layout |
| `task3_figure` | Biểu đồ Kyle's Lambda Bootstrap CI | Base64 String / Img | N/A | Y | `data:image/png;base64,...` | Page 3 Layout |
| `limitations_text` | Tóm tắt hạn chế và hướng phát triển | String / Text Area | N/A | Y | `1-minute aggregated candles lack L2 order book depth...` | Page 3 Footer |
