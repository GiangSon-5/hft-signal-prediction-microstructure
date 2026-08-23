"""Pipeline Data Lakehouse 3 Tầng (Medallion Architecture) với Apache Parquet.

Pipeline thực hiện chuyển đổi và chuẩn hóa dữ liệu qua 3 tầng:
1. Bronze Layer (data/bronze/raw.parquet): Dữ liệu thô nguyên bản từ CSV.
2. Silver Layer (data/silver/cleaned.parquet): Dữ liệu làm sạch 1m grid, forward-fill giá và zero-fill volume.
3. Gold Layer (data/gold/features.parquet & data/gold/extended_features.parquet): Bảng rộng chứa các đặc trưng cấu trúc vi mô và biến mục tiêu nhị phân 15m.
"""

import os
import time
from typing import Any, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_quality.cleaner import validate_and_clean_time_series
from src.feature_engineering.generator import (
    create_binary_target,
    generate_feature_matrix,
    generate_extended_feature_matrix,
)


def build_bronze_layer(
    csv_path: str = "data/raw/ds_assessment_data.csv",
    bronze_path: str = "data/bronze/raw.parquet",
) -> int:
    """Chuyển đổi dữ liệu CSV thô sang Bronze Parquet nguyên bản 100%.

    Args:
        csv_path: Đường dẫn tới file CSV thô.
        bronze_path: Đường dẫn lưu trữ file Parquet tầng Bronze.

    Returns:
        int: Số lượng dòng dữ liệu đã nạp.
    """
    os.makedirs(os.path.dirname(bronze_path), exist_ok=True)

    # Tự động tìm kiếm vị trí file nếu chạy từ thư mục con
    if not os.path.exists(csv_path):
        candidates = [
            "data/raw/ds_assessment_data.csv",
            "../data/raw/ds_assessment_data.csv",
            "ds_assessment_data.csv",
            "../ds_assessment_data.csv",
        ]
        for c in candidates:
            if os.path.exists(c):
                csv_path = c
                break

    df_raw = pd.read_csv(csv_path)
    df_raw.to_parquet(bronze_path, index=False, compression="snappy")
    return len(df_raw)


def build_silver_layer(
    bronze_path: str = "data/bronze/raw.parquet",
    silver_path: str = "data/silver/cleaned.parquet",
) -> Dict[str, Any]:
    """Làm sạch dữ liệu chuỗi thời gian nến 1 phút và ghi vào Silver Parquet.

    Args:
        bronze_path: Đường dẫn file Parquet tầng Bronze.
        silver_path: Đường dẫn file Parquet tầng Silver.

    Returns:
        Dict[str, Any]: Nhật ký báo cáo kiểm định và kích thước tập dữ liệu sạch.
    """
    os.makedirs(os.path.dirname(silver_path), exist_ok=True)

    df_raw = pd.read_parquet(bronze_path)
    df_clean, summary = validate_and_clean_time_series(df_raw)
    df_clean.to_parquet(silver_path, index=False, compression="snappy")

    return {
        "silver_records": len(df_clean),
        "audit_summary": summary,
    }


def build_gold_layer(
    silver_path: str = "data/silver/cleaned.parquet",
    gold_path: str = "data/gold/features.parquet",
    extended_gold_path: str = "data/gold/extended_features.parquet",
) -> Dict[str, Any]:
    """Trích xuất ma trận đặc trưng cơ sở (8 features) và mở rộng (16 features), nhãn mục tiêu và ghi vào Gold Parquet.

    Args:
        silver_path: Đường dẫn file Parquet tầng Silver.
        gold_path: Đường dẫn file Parquet tầng Gold cơ sở (8 features).
        extended_gold_path: Đường dẫn file Parquet tầng Gold mở rộng (16 features).

    Returns:
        Dict[str, Any]: Thông tin số mẫu, danh sách đặc trưng và ngưỡng phân vị target.
    """
    os.makedirs(os.path.dirname(gold_path), exist_ok=True)

    df_clean = pd.read_parquet(silver_path)

    # 1. Trích xuất ma trận 8 đặc trưng cơ sở
    df_features, feature_cols = generate_feature_matrix(df_clean, dropna=False)

    # 2. Tạo nhãn biến mục tiêu nhị phân Y_t = I(sigma_fwd_15m >= Q0.80)
    df_gold, vol_cutoff = create_binary_target(
        df_features, forward_window=15, quantile_threshold=0.80
    )

    # 3. Loại bỏ Warmup và Tail NaNs
    df_gold_clean = df_gold.dropna(subset=feature_cols + ["target_vol_spike_15m"]).copy()
    df_gold_clean["target_vol_spike_15m"] = df_gold_clean["target_vol_spike_15m"].astype(int)
    df_gold_clean.to_parquet(gold_path, index=False, compression="snappy")

    # 4. Trích xuất ma trận đặc trưng mở rộng (16 features)
    df_ext, scenarios = generate_extended_feature_matrix(df_clean, dropna=False)
    df_ext_gold, _ = create_binary_target(df_ext, forward_window=15, quantile_threshold=0.80)
    all_ext_cols = scenarios["scenario_c_multiscale_16"]
    df_ext_gold_clean = df_ext_gold.dropna(subset=all_ext_cols + ["target_vol_spike_15m"]).copy()
    df_ext_gold_clean["target_vol_spike_15m"] = df_ext_gold_clean["target_vol_spike_15m"].astype(int)
    df_ext_gold_clean.to_parquet(extended_gold_path, index=False, compression="snappy")

    return {
        "gold_records": len(df_gold_clean),
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "extended_gold_records": len(df_ext_gold_clean),
        "extended_feature_count": len(all_ext_cols),
        "extended_features": all_ext_cols,
        "scenarios": {k: len(v) for k, v in scenarios.items()},
        "target_cutoff_q80_bps": float(vol_cutoff * 10000.0),
        "class_1_ratio_pct": float(
            df_gold_clean["target_vol_spike_15m"].mean() * 100.0
        ),
    }


def run_lakehouse_pipeline(
    raw_csv: str = "data/raw/ds_assessment_data.csv",
    base_data_dir: str = "data",
) -> Dict[str, Any]:
    """Thực thi toàn bộ chu trình Medallion Lakehouse: CSV -> Bronze -> Silver -> Gold kèm thanh tiến trình tqdm.

    Args:
        raw_csv: Đường dẫn file CSV dữ liệu thị trường thô.
        base_data_dir: Thư mục gốc lưu trữ các tầng Parquet.

    Returns:
        Dict[str, Any]: Báo cáo tổng hợp hiệu năng và kích thước của các tầng.
    """
    start_total = time.time()

    bronze_file = os.path.join(base_data_dir, "bronze", "raw.parquet")
    silver_file = os.path.join(base_data_dir, "silver", "cleaned.parquet")
    gold_file = os.path.join(base_data_dir, "gold", "features.parquet")
    extended_gold_file = os.path.join(base_data_dir, "gold", "extended_features.parquet")

    steps = [
        "1. Chuyển đổi CSV sang Bronze Parquet",
        "2. Làm sạch & tái lập lưới 1m Silver Parquet",
        "3. Trích xuất đặc trưng & gắn nhãn Gold Parquet",
    ]

    pbar = tqdm(steps, desc="Medallion Lakehouse Pipeline", unit="tầng")

    # 1. Bronze
    pbar.set_description("Đang xử lý: Tầng Bronze (CSV -> Parquet)")
    t0 = time.time()
    bronze_count = build_bronze_layer(raw_csv, bronze_file)
    t_bronze = time.time() - t0
    pbar.update(1)

    # 2. Silver
    pbar.set_description("Đang xử lý: Tầng Silver (Làm sạch 1m Grid)")
    t0 = time.time()
    silver_res = build_silver_layer(bronze_file, silver_file)
    t_silver = time.time() - t0
    pbar.update(1)

    # 3. Gold
    pbar.set_description("Đang xử lý: Tầng Gold (Feature Matrix & Target)")
    t0 = time.time()
    gold_res = build_gold_layer(silver_file, gold_file, extended_gold_file)
    t_gold = time.time() - t0
    pbar.update(1)
    pbar.close()

    total_time = time.time() - start_total

    report = {
        "pipeline_status": "SUCCESS",
        "total_elapsed_seconds": round(total_time, 4),
        "bronze": {
            "path": bronze_file,
            "records": bronze_count,
            "size_mb": round(os.path.getsize(bronze_file) / (1024 * 1024), 2),
            "elapsed_seconds": round(t_bronze, 4),
        },
        "silver": {
            "path": silver_file,
            "records": silver_res["silver_records"],
            "size_mb": round(os.path.getsize(silver_file) / (1024 * 1024), 2),
            "elapsed_seconds": round(t_silver, 4),
        },
        "gold": {
            "path": gold_file,
            "records": gold_res["gold_records"],
            "size_mb": round(os.path.getsize(gold_file) / (1024 * 1024), 2),
            "feature_count": gold_res["feature_count"],
            "features": gold_res["feature_names"],
            "extended_path": extended_gold_file,
            "extended_records": gold_res["extended_gold_records"],
            "extended_feature_count": gold_res["extended_feature_count"],
            "scenarios": gold_res["scenarios"],
            "target_q80_bps": gold_res["target_cutoff_q80_bps"],
            "class_1_ratio": gold_res["class_1_ratio_pct"],
            "elapsed_seconds": round(t_gold, 4),
        },
    }

    return report


if __name__ == "__main__":
    csv_input = "data/raw/ds_assessment_data.csv"
    if not os.path.exists(csv_input):
        candidates = [
            "../data/raw/ds_assessment_data.csv",
            "ds_assessment_data.csv",
            "../ds_assessment_data.csv",
        ]
        for c in candidates:
            if os.path.exists(c):
                csv_input = c
                break

    print("=== ĐANG KHỞI CHẠY PIPELINE MEDALLION DATA LAKEHOUSE ===")
    res = run_lakehouse_pipeline(raw_csv=csv_input)
    print(f"\nTrạng thái: {res['pipeline_status']}")
    print(f"Tổng thời gian: {res['total_elapsed_seconds']}s")
    print(f"Bronze: {res['bronze']['records']:,} dòng ({res['bronze']['size_mb']} MB)")
    print(f"Silver: {res['silver']['records']:,} dòng ({res['silver']['size_mb']} MB)")
    print(f"Gold  : {res['gold']['records']:,} dòng ({res['gold']['size_mb']} MB, {res['gold']['feature_count']} features)")
    print(f"Extended Gold: {res['gold']['extended_records']:,} dòng ({res['gold']['extended_feature_count']} features)")
