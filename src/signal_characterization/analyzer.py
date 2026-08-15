"""Signal Characterization Module for High-Frequency Market Data.

Module này thực hiện phân tích đặc trưng phân phối log returns, volatility regimes,
tương quan giữa Volume, Trades & Price Range, và kiểm định tự tương quan (ACF/PACF),
được đóng gói hoàn chỉnh từ các bước nghiên cứu thực nghiệm trong Notebook 1.
"""

from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf


def calculate_log_returns(df: pd.DataFrame, col: str = "close") -> pd.Series:
    """Tính tỷ suất lợi nhuận close-to-close dạng Logarithm (1-minute log return).

    Công thức:
        $$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

    Args:
        df: DataFrame chứa cột giá.
        col: Tên cột giá đóng cửa. Mặc định là 'close'.

    Returns:
        pd.Series: Chuỗi log returns 1 phút (giá trị đầu tiên là NaN).
    """
    return np.log(df[col] / df[col].shift(1))


def analyze_returns_distribution(returns: pd.Series) -> Dict[str, Any]:
    """Phân tích thống kê mô tả và kiểm định giả thuyết phân phối chuẩn cho Log Returns.

    Kiểm định Jarque-Bera (H0: Phân phối chuẩn) và ước lượng tham số phân phối Student-t.

    Args:
        returns: Chuỗi log returns 1 phút.

    Returns:
        Dict[str, Any]: Thống kê mô tả (Mean, Std, Skewness, Kurtosis, Jarque-Bera test, Student-t params).
    """
    clean_returns = returns.dropna()

    # 1. Thống kê mô tả cơ bản
    mean_val = clean_returns.mean()
    std_val = clean_returns.std()
    skewness_val = clean_returns.skew()
    kurtosis_val = clean_returns.kurtosis()  # Fisher excess kurtosis (Normal = 0)

    # 2. Kiểm định Jarque-Bera (H0: Phân phối chuẩn)
    jb_stat, jb_pvalue = stats.jarque_bera(clean_returns)

    # 3. Khớp phân phối Student-t (Fat-tailed distribution fitting)
    t_df, t_loc, t_scale = stats.t.fit(clean_returns)

    return {
        "count": int(len(clean_returns)),
        "mean": float(mean_val),
        "std": float(std_val),
        "skewness": float(skewness_val),
        "kurtosis": float(kurtosis_val),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_pvalue),
        "is_normal_5pct": bool(jb_pvalue > 0.05),
        "student_t_df": float(t_df),
        "student_t_loc": float(t_loc),
        "student_t_scale": float(t_scale),
    }


def compute_rolling_volatility(
    returns: pd.Series, window: int = 60
) -> pd.Series:
    """Tính độ biến động trượt (Rolling Volatility) quy năm trên cửa sổ trượt N phút.

    Công thức:
        $$\sigma_{\text{annualized}} = \sigma_{\text{rolling}}(r, N) \times \sqrt{525,600}$$
        (Trong đó 525,600 = 365 ngày * 24 giờ * 60 phút).

    Args:
        returns: Chuỗi log returns 1 phút.
        window: Cửa sổ trượt (phút). Mặc định là 60.

    Returns:
        pd.Series: Chuỗi độ biến động trượt quy năm.
    """
    annualization_factor = np.sqrt(365 * 24 * 60)
    rolling_std = returns.rolling(window=window, min_periods=window // 2).std()
    return rolling_std * annualization_factor


def detect_volatility_regimes(
    rolling_vol: pd.Series, threshold_quantile: float = 0.75
) -> Tuple[pd.Series, float]:
    """Phân tách 2 chế độ biến động (Low Volatility Regime vs High Volatility Regime).

    Args:
        rolling_vol: Chuỗi độ biến động trượt quy năm.
        threshold_quantile: Ngưỡng phân vị để phân tách High Vol. Mặc định là 0.75.

    Returns:
        Tuple[pd.Series, float]: Chuỗi nhãn Chế độ (0 = Low Vol, 1 = High Vol) và Giá trị ngưỡng cut-off.
    """
    threshold_value = float(rolling_vol.quantile(threshold_quantile))
    regimes = (rolling_vol >= threshold_value).astype(int)
    return regimes, threshold_value


def compare_volatility_regimes(
    returns: pd.Series, regimes: pd.Series
) -> pd.DataFrame:
    """So sánh chi tiết đặc tính đuôi béo (Kurtosis & Student-t df) giữa 2 chế độ biến động.

    Args:
        returns: Chuỗi log returns 1 phút.
        regimes: Chuỗi nhãn chế độ biến động (0/Low hoặc 1/High).

    Returns:
        pd.DataFrame: Bảng tổng hợp so sánh số mẫu, Kurtosis và bậc tự do Student-t.
    """
    temp_df = pd.DataFrame({"returns": returns, "regimes": regimes}).dropna()
    low_returns = temp_df[temp_df["regimes"] == 0]["returns"]
    high_returns = temp_df[temp_df["regimes"] == 1]["returns"]

    kurt_low = float(low_returns.kurtosis())
    kurt_high = float(high_returns.kurtosis())

    t_df_low, _, _ = stats.t.fit(low_returns)
    t_df_high, _, _ = stats.t.fit(high_returns)

    summary_df = pd.DataFrame(
        {
            "Regime": ["Low Volatility Regime", "High Volatility Regime"],
            "Count": [int(len(low_returns)), int(len(high_returns))],
            "Kurtosis": [kurt_low, kurt_high],
            "Student_t_df": [float(t_df_low), float(t_df_high)],
        }
    )
    return summary_df


def analyze_volume_trades_range(df: pd.DataFrame) -> Dict[str, Any]:
    """Phân tích mối quan hệ tương quan phi tuyến Spearman giữa Volume, Trades và Biên độ giá.

    Công thức Biên độ giá:
        $$\text{Price Range Ratio} = \frac{\text{High}_t - \text{Low}_t}{\text{Close}_t}$$

    Args:
        df: DataFrame chứa các cột high, low, close, volume, trades.

    Returns:
        Dict[str, Any]: Ma trận tương quan Spearman và các hệ số tương quan cặp đôi.
    """
    temp_df = pd.DataFrame()
    temp_df["price_range_ratio"] = (df["high"] - df["low"]) / df["close"]
    temp_df["volume"] = df["volume"]
    temp_df["trades"] = df["trades"]

    clean_df = temp_df.dropna()
    spearman_corr = clean_df.corr(method="spearman").to_dict()

    return {
        "spearman_correlation": spearman_corr,
        "corr_volume_vs_range": float(
            spearman_corr["volume"]["price_range_ratio"]
        ),
        "corr_trades_vs_range": float(
            spearman_corr["trades"]["price_range_ratio"]
        ),
        "corr_volume_vs_trades": float(spearman_corr["volume"]["trades"]),
    }


def analyze_autocorrelation(
    returns: pd.Series, nlags: int = 30
) -> Dict[str, Any]:
    """Kiểm định tự tương quan ACF và PACF trên chuỗi log returns.

    Args:
        returns: Chuỗi log returns 1 phút.
        nlags: Số lượng lags kiểm tra. Mặc định là 30.

    Returns:
        Dict[str, Any]: Mảng giá trị ACF, PACF, dải tin cậy 95% và các lags có ý nghĩa thống kê.
    """
    clean_returns = returns.dropna()
    acf_vals = acf(clean_returns, nlags=nlags, fft=True)
    pacf_vals = pacf(clean_returns, nlags=nlags, method="ywm")

    # Ngưỡng tin cậy 95% = +- 1.96 / sqrt(N)
    conf_interval = 1.96 / np.sqrt(len(clean_returns))

    return {
        "acf": acf_vals.tolist(),
        "pacf": pacf_vals.tolist(),
        "conf_interval_95": float(conf_interval),
        "significant_acf_lags": [
            int(lag)
            for lag in range(1, nlags + 1)
            if abs(acf_vals[lag]) > conf_interval
        ],
    }
