# Đặc Tả Kỹ Thuật (Module Specification): Mô-đun Sinh Báo Cáo Kỹ Thuật 3 Trang (`src/reporting`)

## 1. Tổng Quan Mô-đun

Mô-đun `reporting` chịu trách nhiệm tổng hợp toàn bộ kết quả phân tích thống kê, đồ thị trực quan, chỉ số mô hình học máy, và kết quả kiểm định chuyên sâu thành một **Báo Cáo Kỹ Thuật (Technical Report) 3 Trang A4** tự động, định dạng HTML5 / CSS3 chuẩn mực (có thể in thành file PDF trực tiếp qua `weasyprint` hoặc trình duyệt).

---

## 2. Quy Chuẩn Đóng Gói Đầu Ra (Deliverable Spec)

### 2.1 File Đầu Ra (`reports/technical_report.html`)
- Định dạng: Single-file HTML5 độc lập (self-contained), tích hợp hình ảnh chuỗi mã hóa Base64.
- Đảm bảo giới hạn trang: Đúng 3 trang A4 (sử dụng CSS `@page` print boundaries).
- Bố cục từng trang:
  - **Trang 1:** Executive Summary, Dataset Overview, Task 1 Phân phối return, Kurtosis, Volatility Regimes.
  - **Trang 2:** Task 2 Mô hình hóa dự đoán, Features, Kết quả Time-Aware CV, ROC/PR curves & Calibration.
  - **Trang 3:** Task 3 Phân tích chuyên sâu Kyle's Lambda (Order Flow Toxicity, Bootstrap 95% CI), Hạn chế bài phân tích & Hướng phát triển.

---

## 3. Thuật Toán & Kỹ Thuật Đóng Gói

### 3.1 CSS Paged Media Quy Định Khung Trang
```css
@page {
    size: A4 portrait;
    margin: 1.5cm;
}
.page-break {
    page-break-after: always;
}
```

### 3.2 Nhúng Hình Ảnh Dạng Base64
Để file HTML hoạt động độc lập mà không lo bị lỗi đường dẫn hình ảnh khi chuyển sang máy khác:
```python
def embed_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"
```

### 3.3 Dynamic Template Processing
Sử dụng thư viện `jinja2` điền tự động các giá trị thống kê, bảng chỉ số AUC/F1, và hình ảnh Base64 vào khung HTML template.
