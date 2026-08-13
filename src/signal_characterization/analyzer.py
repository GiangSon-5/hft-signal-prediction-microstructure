"""Signal Characterization Module for High-Frequency Market Data.

Module này thực hiện phân tích đặc trưng phân phối log returns, volatility regimes,
tương quan giữa Volume, Trades & Price Range, và kiểm định tự tương quan (ACF/PACF).
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
from typing import Dict, Any, Tuple


def calculate_log_returns(df: pd.DataFrame, col: str = 'close') -> pd.Series:
    """Tính tỷ suất lợi nhuận close-to-close dạng Logarithm (1-minute log return).

    Formula:
        $$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

    Args:
        df (pd.DataFrame): DataFrame chứa cột giá.
        col (str, optional): Tên cột giá đóng cửa. Mặc định là 'close'.

    Returns:
        pd.Series: Chuỗi log returns 1 phút (giá trị đầu tiên NaN).
    """
    return np.log(df[col] / df[col].shift(1))


def analyze_returns_distribution(returns: pd.Series) -> Dict[str, Any]:
    """Phân tích thống kê mô tả và kiểm định giả thuyết phân phối chuẩn cho Log Returns.

    Args:
        returns (pd.Series): Chuỗi log returns 1 phút.

    Returns:
        Dict[str, Any]: Thống kê mô tả (Mean, Std, Skewness, Kurtosis, Jarque-Bera test, Student-t params).
    """
    clean_returns = returns.dropna()
    
    # 1. Thống kê mô tả cơ bản
    mean_val = clean_returns.mean()
    std_val = clean_returns.std()
    skewness_val = clean_returns.skew()
    kurtosis_val = clean_returns.kurtosis() # Fisher kurtosis (Normal = 0)
    
    # 2. Kiểm định Jarque-Bera (H0: Phân phối chuẩn)
    jb_stat, jb_pvalue = stats.jarque_bera(clean_returns)
    
    # 3. Khớp phân phối Student-t (Fat-tailed distribution fitting)
    t_df, t_loc, t_scale = stats.t.fit(clean_returns)
    
    return {
        'count': len(clean_returns),
        'mean': float(mean_val),
        'std': float(std_val),
        'skewness': float(skewness_val),
        'kurtosis': float(kurtosis_val),
        'jarque_bera_stat': float(jb_stat),
        'jarque_bera_pvalue': float(jb_pvalue),
        'is_normal_5pct': bool(jb_pvalue > 0.05),
        'student_t_df': float(t_df),
        'student_t_loc': float(t_loc),
        'student_t_scale': float(t_scale)
    }


def compute_rolling_volatility(returns: pd.Series, window: int = 60) -> pd.Series:
    """Tính độ biến động trượt (Rolling Volatility) quy năm trên cửa sổ trượt N phút.

    Formula:
        $$\sigma_{\text{annualized}} = \sigma_{\text{rolling}}(r, N) \times \sqrt{525,600}$$

    Args:
        returns (pd.Series): Chuỗi log returns 1 phút.
        window (int, optional): Cửa sổ trượt (phút). Mặc định là 60.

    Returns:
        pd.Series: Chuỗi độ biến động trượt quy năm.
    """
    # 525,600 = Số phút trong 1 năm (365 ngày * 24 giờ * 60 phút)
    annualization_factor = np.sqrt(365 * 24 * 60)
    rolling_std = returns.rolling(window=window, min_periods=window//2).std()
    return rolling_std * annualization_factor


def detect_volatility_regimes(
    rolling_vol: pd.Series, 
    threshold_quantile: float = 0.75
) -> Tuple[pd.Series, float]:
    """Phân tách 2 chế độ biến động (Low Volatility Regime vs High Volatility Regime).

    Args:
        rolling_vol (pd.Series): Chuỗi độ biến động trượt.
        threshold_quantile (float, optional): Ngưỡng phân vị để phân tách High Vol. Mặc định là 0.75.

    Returns:
        Tuple[pd.Series, float]: Chuỗi nhãn Chế độ (0 = Low Vol, 1 = High Vol) và Giá trị ngưỡng cut-off.
    """
    threshold_value = float(rolling_vol.quantile(threshold_quantile))
    regimes = (rolling_vol >= threshold_value).astype(int)
    return regimes, threshold_value


def analyze_volume_trades_range(df: pd.DataFrame) -> Dict[str, Any]:
    """Phân tích mối quan hệ tương quan giữa Volume, Trades và Biên độ giá.

    Formula Biên độ giá:
        $$\text{Price Range Ratio} = \frac{\text{High}_t - \text{Low}_t}{\text{Close}_t}$$

    Args:
        df (pd.DataFrame): DataFrame chứa cột high, low, close, volume, trades.

    Returns:
        Dict[str, Any]: Ma trận tương quan Spearman và thống kê mô tả.
    """
    temp_df = pd.DataFrame()
    temp_df['price_range_ratio'] = (df['high'] - df['low']) / df['close']
    temp_df['volume'] = df['volume']
    temp_df['trades'] = df['trades']
    
    clean_df = temp_df.dropna()
    spearman_corr = clean_df.corr(method='spearman').to_dict()
    
    return {
        'spearman_correlation': spearman_corr,
        'corr_volume_vs_range': float(spearman_corr['volume']['price_range_ratio']),
        'corr_trades_vs_range': float(spearman_corr['trades']['price_range_ratio']),
        'corr_volume_vs_trades': float(spearman_corr['volume']['trades'])
    }


def analyze_autocorrelation(returns: pd.Series, nlags: int = 30) -> Dict[str, Any]:
    """Kiểm định tự tương quan ACF và PACF trên chuỗi log returns.

    Args:
        returns (pd.Series): Chuỗi log returns 1 phút.
        nlags (int, optional): Số lượng lags kiểm tra. Mặc định là 30.

    Returns:
        Dict[str, Any]: Mảng giá trị ACF, PACF và ngưỡng tin cậy 95%.
    """
    clean_returns = returns.dropna()
    acf_vals = acf(clean_returns, nlags=nlags, fft=True)
    pacf_vals = pacf(clean_returns, nlags=nlags, method='ywm')
    
    # Ngưỡng tin cậy 95% = +- 1.96 / sqrt(N)
    conf_interval = 1.96 / np.sqrt(len(clean_returns))
    
    return {
        'acf': acf_vals.tolist(),
        'pacf': pacf_vals.tolist(),
        'conf_interval_95': float(conf_interval),
        'significant_acf_lags': [
            int(lag) for lag in range(1, nlags + 1)
            if abs(acf_vals[lag]) > conf_interval
        ]
    }
