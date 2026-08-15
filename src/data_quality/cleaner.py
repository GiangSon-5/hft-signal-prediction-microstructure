"""Mô-đun kiểm định chất lượng dữ liệu và làm sạch chuỗi thời gian nến 1 phút.

Mô-đun này cung cấp các hàm kiểm tra độ toàn vẹn, phát hiện khoảng trống thời gian (Gaps),
kiểm tra tính hợp lệ của giá trị OHLCV và làm sạch chuỗi thời gian có điều kiện,
chuẩn hóa 100% theo các bước thực thi trong Notebook 1.
"""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd


def audit_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Kiểm tra toàn diện các chỉ số chất lượng dữ liệu thô (Data Quality Audit).

    Args:
        df: DataFrame chứa dữ liệu thô OHLCV.

    Returns:
        Dict[str, Any]: Báo cáo chi tiết về missing values, gaps thời gian và bất thường logic.
    """
    audit_report = {}

    # 1. Kiểm tra giá trị khuyết thiếu (Missing Values)
    audit_report["missing_values"] = df.isnull().sum().to_dict()

    # 2. Chuyển đổi timestamp sang Datetime chuẩn (timezone-naive) và phân tích Gaps
    df_sorted = df.copy()
    df_sorted["timestamp"] = pd.to_datetime(df_sorted["timestamp"], utc=True).dt.tz_localize(None)
    df_sorted = df_sorted.sort_values("timestamp").reset_index(drop=True)

    time_diffs = df_sorted["timestamp"].diff()
    expected_step = pd.Timedelta(minutes=1)
    gaps = time_diffs[time_diffs > expected_step]

    audit_report["start_time"] = str(df_sorted["timestamp"].min())
    audit_report["end_time"] = str(df_sorted["timestamp"].max())
    audit_report["total_gaps_count"] = int(len(gaps))
    audit_report["total_missing_minutes"] = (
        float((gaps.sum() - len(gaps) * expected_step).total_seconds() / 60.0)
        if len(gaps) > 0
        else 0.0
    )

    # 3. Kiểm tra bất thường giá (Negative or Zero prices)
    price_cols = ["open", "high", "low", "close"]
    invalid_prices = (df[price_cols] <= 0).sum().to_dict()
    audit_report["invalid_prices"] = invalid_prices

    # 4. Kiểm tra bất thường Logic OHLC (High >= Low, High >= Open/Close, Low <= Open/Close)
    ohlc_violations = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    ).sum()
    audit_report["ohlc_logic_violations"] = int(ohlc_violations)

    return audit_report


def validate_and_clean_time_series(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Kiểm tra tính liên tục của chuỗi thời gian 1 phút và thực hiện tiền xử lý làm sạch có điều kiện.

    Thực thi chính xác theo thuật toán tiền xử lý trong Notebook 1:
    1. Chuẩn hóa timestamp sang Datetime (UTC -> Localize None để đồng bộ).
    2. Khởi tạo lưới thời gian 1 phút chuẩn (1-min grid) từ min_time đến max_time.
    3. Kiểm tra số nến khuyết thiếu. Nếu có gaps:
       - Tái lập 1-min grid qua reindex.
       - Forward-fill giá Close, đồng bộ Open/High/Low theo Close.
       - Gán 0.0 cho các cột Volume, Quote Volume, Trades, Taker Buy Volume.
    4. Trả về DataFrame sạch và nhật ký báo cáo trạng thái tiền xử lý.

    Args:
        df: DataFrame chứa dữ liệu chuỗi thời gian OHLCV thô.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - df_cleaned: DataFrame đã được chuẩn hóa liên tục.
            - audit_summary: Nhật ký báo cáo trạng thái tiền xử lý.
    """
    df_clean = df.copy()

    # 1. Chuyển đổi timestamp sang Datetime chuẩn
    df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"], utc=True).dt.tz_localize(None)
    df_clean = df_clean.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # 2. Tạo lưới thời gian 1m chuẩn (1-min grid) từ min_time đến max_time
    min_time = df_clean["timestamp"].min()
    max_time = df_clean["timestamp"].max()
    full_time_index = pd.date_range(
        start=min_time, end=max_time, freq="1min", name="timestamp"
    )

    expected_count = len(full_time_index)
    actual_count = len(df_clean)
    missing_count = expected_count - actual_count

    audit_summary = {
        "expected_bars": expected_count,
        "actual_bars": actual_count,
        "missing_bars_padded": missing_count,
        "is_fully_continuous": (missing_count == 0),
    }

    # 3. Xử lý lấp nến có điều kiện (Conditional Gap Filling)
    if missing_count > 0:
        print(
            f"[TIỀN XỬ LÝ] Phát hiện {missing_count:,} mẫu quan sát 1m bị khuyết. "
            "Thực hiện tái lập 1-min grid & Forward-fill..."
        )
        df_clean = df_clean.set_index("timestamp").reindex(full_time_index)

        # Forward-fill giá đóng cửa (Close)
        df_clean["close"] = df_clean["close"].ffill().bfill()
        df_clean["open"] = df_clean["open"].fillna(df_clean["close"])
        df_clean["high"] = df_clean["high"].fillna(df_clean["close"])
        df_clean["low"] = df_clean["low"].fillna(df_clean["close"])

        # Gán khối lượng bằng 0.0 tại các nến lấp
        volume_cols = ["volume", "quote_volume", "trades", "taker_buy_volume"]
        for col in volume_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0.0)

        df_clean = df_clean.reset_index()
    else:
        print(
            f"[TIỀN XỬ LÝ] Chuỗi thời gian đạt độ toàn vẹn 100% "
            f"({actual_count:,} mẫu quan sát liên tục). Không cần xử lý lấp nến."
        )

    return df_clean, audit_summary


# Alias tương thích ngược
clean_market_data = validate_and_clean_time_series
