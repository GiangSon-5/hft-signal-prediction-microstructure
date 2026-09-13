# Đặc Tả Kỹ Thuật: Medallion Data Lakehouse (`src/lakehouse`)

## 1. Tổng Quan Kiến Trúc

Mô-đun `lakehouse` chịu trách nhiệm chuyển đổi và chuẩn hóa dữ liệu thị trường nến 1 phút qua kiến trúc 3 tầng Medallion Architecture sử dụng định dạng cột nén Apache Parquet (Snappy):

```
CSV Thô (data/raw/ds_assessment_data.csv)
   │
   ▼
Tầng Bronze (data/bronze/raw.parquet) ─── Dữ liệu thô nguyên bản 100% (264,961 dòng, 13.4 MB)
   │
   ▼
Tầng Silver (data/silver/cleaned.parquet) ─── Tái lập lưới 1m, forward-fill giá & zero-fill vol (264,961 dòng, 14.3 MB)
   │
   ▼
Tầng Gold (data/gold/) ─── Bảng rộng đặc trưng vi cấu trúc & nhãn mục tiêu nhị phân 15m (264,886 dòng)
   ├── features.parquet (8 đặc trưng cơ sở, 37.1 MB)
   └── extended_features.parquet (16 đặc trưng mở rộng, 54.1 MB)
```

---

## 2. Các Hàm Xử Lý Chính

- **`build_bronze_layer(csv_path, bronze_path) -> int`**: Chuyển đổi CSV sang Parquet nguyên bản.
- **`build_silver_layer(bronze_path, silver_path) -> dict`**: Làm sạch và kiểm định chuỗi thời gian nến 1m.
- **`build_gold_layer(silver_path, gold_path, extended_gold_path) -> dict`**: Trích xuất ma trận 8 và 16 đặc trưng, gắn nhãn mục tiêu $Y_t = \mathbb{I}(\sigma_{fwd, 15m} \ge Q_{0.80})$, loại bỏ 60 nến warmup và 15 nến tail.
- **`run_lakehouse_pipeline(raw_csv, base_data_dir) -> dict`**: Thực thi toàn bộ pipeline tự động kèm thanh tiến trình `tqdm`.
