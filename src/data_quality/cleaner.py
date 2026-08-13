"""Data Quality Audit & Cleaning Module for High-Frequency Market Data.

Module này thực hiện kiểm tra chất lượng dữ liệu chuỗi thời gian (timestamps gaps, 
giá trị khuyết thiếu, outliers) và làm sạch dữ liệu nến 1 phút H2 2024.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


def audit_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Kiểm tra toàn diện các chỉ số chất lượng dữ liệu thô.

    Args:
        df (pd.DataFrame): DataFrame chứa dữ liệu thô OHLCV.

    Returns:
        Dict[str, Any]: Báo cáo chi tiết về missing values, gaps thời gian và bất thường.
    """
    audit_report = {}
    
    # 1. Kiểm tra giá trị khuyết thiếu (Missing Values)
    audit_report['missing_values'] = df.isnull().sum().to_dict()
    
    # 2. Kiểm tra tính liên tục của Timestamp (Gap Analysis)
    df_sorted = df.copy()
    df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'], utc=True)
    df_sorted = df_sorted.sort_values('timestamp')
    time_diffs = df_sorted['timestamp'].diff()
    expected_step = pd.Timedelta(minutes=1)
    gaps = time_diffs[time_diffs > expected_step]
    
    audit_report['total_gaps_count'] = len(gaps)
    audit_report['total_missing_minutes'] = (
        (gaps.sum() - len(gaps) * expected_step).total_seconds() / 60.0
        if len(gaps) > 0 else 0.0
    )
    
    # 3. Kiểm tra bất thường giá (Negative or Zero prices)
    price_cols = ['open', 'high', 'low', 'close']
    invalid_prices = (df[price_cols] <= 0).sum().to_dict()
    audit_report['invalid_prices'] = invalid_prices
    
    # 4. Kiểm tra bất thường Logic OHLC (High >= Low, High >= Open/Close, Low <= Open/Close)
    ohlc_violations = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    ).sum()
    audit_report['ohlc_logic_violations'] = int(ohlc_violations)
    
    return audit_report


def clean_market_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Xử lý lấp khoảng trống (gaps), forward fill giá và làm sạch dữ liệu chuỗi thời gian.

    Args:
        df (pd.DataFrame): DataFrame thô có cột 'timestamp'.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: DataFrame đã làm sạch và dict thông số xử lý.
    """
    cleaning_stats = {}
    
    # Đảm bảo timestamp ở dạng Datetime (UTC)
    df_clean = df.copy()
    df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], utc=True)
    df_clean = df_clean.sort_values('timestamp').drop_duplicates('timestamp')
    
    # Tạo chuỗi thời gian liên tục từng phút 1-min grid từ min_time tới max_time
    min_time = df_clean['timestamp'].min()
    max_time = df_clean['timestamp'].max()
    full_time_index = pd.date_range(start=min_time, end=max_time, freq='1min', tz='UTC', name='timestamp')
    
    cleaning_stats['raw_rows'] = len(df_clean)
    cleaning_stats['expected_rows'] = len(full_time_index)
    cleaning_stats['missing_rows_padded'] = len(full_time_index) - len(df_clean)
    
    # Reindex trên grid thời gian hoàn chỉnh
    df_clean = df_clean.set_index('timestamp').reindex(full_time_index)
    
    # Forward-fill giá đóng cửa (Close price) cho các nến bị khuyết
    df_clean['close'] = df_clean['close'].ffill().bfill()
    
    # Đặt Open, High, Low bằng Close nếu nến đó bị pad do thiếu dữ liệu
    df_clean['open'] = df_clean['open'].fillna(df_clean['close'])
    df_clean['high'] = df_clean['high'].fillna(df_clean['close'])
    df_clean['low'] = df_clean['low'].fillna(df_clean['close'])
    
    # Đặt Volume, Trades, Taker Buy Volume bằng 0 cho nến bị lấp khoảng trống
    volume_cols = ['volume', 'quote_volume', 'trades', 'taker_buy_volume']
    for col in volume_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0.0)
            
    df_clean = df_clean.reset_index()
    
    return df_clean, cleaning_stats
