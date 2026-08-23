"""Module Feature Engineering cho dữ liệu thị trường tần suất cao OHLCV 1 phút.

Mô-đun này cung cấp các hàm tính toán đặc trưng cấu trúc vi mô (Microstructure Features),
khai thác toàn diện 100% các cột dữ liệu thô (bao gồm quote_volume, trades, taker_buy_volume),
và tạo nhãn biến mục tiêu nhị phân (Binary Target Formulation) không rò rỉ dữ liệu tương lai.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def calculate_parkinson_volatility(
    df: pd.DataFrame, window: int = 15
) -> pd.Series:
    r"""Tính độ biến động Parkinson qua giá Cao nhất (High) và Thấp nhất (Low).

    Độ biến động Parkinson sử dụng giá trị cực trị (Extreme Values) trong mỗi nến,
    đạt hiệu quả ước lượng phương sai cao gấp ~5 lần so với phương sai Close-to-Close.

    Công thức:
        $$\sigma_P = \sqrt{\frac{1}{4 \ln(2) \cdot W} \sum_{k=0}^{W-1} \left(\ln\frac{High_{t-k}}{Low_{t-k}}\right)^2}$$

    Args:
        df: DataFrame chứa các cột 'high' và 'low'.
        window: Kích thước cửa sổ trượt (mặc định 15 phút).

    Returns:
        pd.Series: Chuỗi độ biến động Parkinson theo cửa sổ trượt.
    """
    log_hl_sq = np.log(df["high"] / df["low"]) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    rolling_var = log_hl_sq.rolling(window=window, min_periods=window).mean() * factor
    return np.sqrt(rolling_var)


def calculate_garman_klass_volatility(
    df: pd.DataFrame, window: int = 15
) -> pd.Series:
    r"""Tính độ biến động Garman-Klass tích hợp cả High/Low và Open/Close.

    Công thức:
        $$GK = 0.5 \ln(H/L)^2 - (2\ln 2 - 1)\ln(C/O)^2$$

    Args:
        df: DataFrame chứa 'open', 'high', 'low', 'close'.
        window: Kích thước cửa sổ trượt (mặc định 15 phút).

    Returns:
        pd.Series: Chuỗi độ biến động Garman-Klass.
    """
    log_hl_sq = np.log(df["high"] / df["low"]) ** 2
    log_co_sq = np.log(df["close"] / df["open"]) ** 2
    gk_term = 0.5 * log_hl_sq - (2.0 * np.log(2.0) - 1.0) * log_co_sq
    rolling_gk = gk_term.rolling(window=window, min_periods=window).mean()
    return np.sqrt(np.maximum(rolling_gk, 0.0))


def calculate_order_flow_imbalance_ratio(
    df: pd.DataFrame, epsilon: float = 1e-8
) -> pd.Series:
    r"""Tính tỷ lệ mất cân bằng dòng lệnh (Order Flow Imbalance Ratio - OFI).

    Đo lường mức độ chủ động của bên mua so với tổng thanh khoản nến 1m.

    Công thức:
        $$\text{OFI} = \frac{taker\_buy\_volume}{volume + \epsilon}$$

    Args:
        df: DataFrame chứa 'taker_buy_volume' và 'volume'.
        epsilon: Hằng số chống chia cho 0.

    Returns:
        pd.Series: Tỷ lệ OFI trong khoảng [0, 1].
    """
    return df["taker_buy_volume"] / (df["volume"] + epsilon)


def calculate_trade_density(
    df: pd.DataFrame, epsilon: float = 1e-8
) -> pd.Series:
    r"""Tính khối lượng trung bình trên mỗi giao dịch (Trade Density / Block Size).

    Phản ánh sự tham gia của dòng tiền lớn (Institutional Block Orders)
    so với lệnh nhỏ lẻ (Retail Trading).

    Công thức:
        $$\text{TD} = \frac{volume}{trades + \epsilon}$$

    Args:
        df: DataFrame chứa 'volume' và 'trades'.
        epsilon: Hằng số chống chia cho 0.

    Returns:
        pd.Series: Khối lượng trung bình mỗi lượt khớp lệnh.
    """
    return df["volume"] / (df["trades"] + epsilon)


def calculate_volume_spike_zscore(
    df: pd.DataFrame, window: int = 60, epsilon: float = 1e-8
) -> pd.Series:
    r"""Tính Z-Score khối lượng giao dịch so với cửa sổ trượt 60 phút.

    Phát hiện các đột biến thanh khoản bất thường vượt ngưỡng lịch sử ngắn hạn.

    Công thức:
        $$Z_{vol} = \frac{volume - \mu_{vol, 60m}}{\sigma_{vol, 60m} + \epsilon}$$

    Args:
        df: DataFrame chứa cột 'volume'.
        window: Cửa sổ tính toán trung bình và độ lệch chuẩn (mặc định 60 phút).
        epsilon: Hằng số chống chia cho 0.

    Returns:
        pd.Series: Chuỗi Z-score chuẩn hóa của khối lượng.
    """
    roll_mean = df["volume"].rolling(window=window, min_periods=window).mean()
    roll_std = df["volume"].rolling(window=window, min_periods=window).std()
    return (df["volume"] - roll_mean) / (roll_std + epsilon)


def calculate_return_momentum(
    df: pd.DataFrame, window: int = 15
) -> pd.Series:
    r"""Tính động lượng lợi nhuận tích lũy (Return Momentum).

    Công thức:
        $$\text{Mom} = \ln\left(\frac{Close_t}{Close_{t-W}}\right)$$

    Args:
        df: DataFrame chứa cột 'close'.
        window: Độ trễ cửa sổ tích lũy (mặc định 15 phút).

    Returns:
        pd.Series: Tỷ suất lợi nhuận tích lũy log return.
    """
    return np.log(df["close"] / df["close"].shift(window))


def calculate_rolling_volatility_60m(
    df: pd.DataFrame, window: int = 60
) -> pd.Series:
    r"""Tính độ biến động trượt 60 phút quy năm (Annualized Rolling Volatility).

    Công thức:
        $$\sigma_{60m} = \text{std}(r_{t-59 \dots t}) \times \sqrt{525,600}$$

    Args:
        df: DataFrame chứa cột 'close' hoặc 'log_return'.
        window: Kích thước cửa sổ trượt (mặc định 60 phút).

    Returns:
        pd.Series: Độ biến động trượt quy năm.
    """
    if "log_return" in df.columns:
        ret = df["log_return"]
    else:
        ret = np.log(df["close"] / df["close"].shift(1))
    return ret.rolling(window=window, min_periods=window).std() * np.sqrt(525600.0)


def calculate_spread_ratio(
    df: pd.DataFrame, window: int = 15
) -> pd.Series:
    r"""Tính tỷ lệ mở rộng biên độ giá trung bình trong 15 phút (High-Low Spread Proxy).

    Công thức:
        $$\text{Spread}_{15m} = \text{mean}\left(\frac{High - Low}{Close}, 15m\right)$$

    Args:
        df: DataFrame chứa 'high', 'low', 'close'.
        window: Kích thước cửa sổ trượt (mặc định 15 phút).

    Returns:
        pd.Series: Chuỗi tỷ lệ spread trung bình trượt.
    """
    ratio = (df["high"] - df["low"]) / df["close"]
    return ratio.rolling(window=window, min_periods=window).mean()


def calculate_vwap_deviation(
    df: pd.DataFrame, window: int = 15, epsilon: float = 1e-8
) -> pd.Series:
    r"""Tính độ lệch giá đóng cửa so với giá bình quân gia quyền khối lượng (VWAP Deviation).

    Khai thác cột `quote_volume` kết hợp `volume` để đo lường độ phân kỳ giá vi mô.

    Công thức:
        $$\text{VWAP}_{W, t} = \frac{\sum_{k=0}^{W-1} \text{quote\_volume}_{t-k}}{\sum_{k=0}^{W-1} \text{volume}_{t-k} + \epsilon}$$
        $$\text{dev}_{VWAP} = \frac{\text{Close}_t - \text{VWAP}_{W, t}}{\text{Close}_t}$$

    Args:
        df: DataFrame chứa cột 'quote_volume', 'volume', 'close'.
        window: Cửa sổ tính toán trượt (mặc định 15 phút).
        epsilon: Hằng số chống chia cho 0.

    Returns:
        pd.Series: Chuỗi độ lệch tương đối của giá so với VWAP.
    """
    roll_quote = df["quote_volume"].rolling(window=window, min_periods=window).sum()
    roll_vol = df["volume"].rolling(window=window, min_periods=window).sum()
    vwap = roll_quote / (roll_vol + epsilon)
    return (df["close"] - vwap) / df["close"]


def calculate_dollar_trade_size(
    df: pd.DataFrame, epsilon: float = 1e-8
) -> pd.Series:
    r"""Tính quy mô giá trị định danh USD trung bình trên mỗi giao dịch (Dollar Trade Size).

    Phản ánh hành vi dòng tiền tổ chức (Block Trades) so với lệnh nhỏ lẻ.

    Công thức:
        $$\text{Dollar Trade Size} = \frac{\text{quote\_volume}}{\text{trades} + \epsilon}$$

    Args:
        df: DataFrame chứa cột 'quote_volume' và 'trades'.
        epsilon: Hằng số chống chia cho 0.

    Returns:
        pd.Series: Chuỗi giá trị quy mô USD bình quân mỗi lệnh.
    """
    return df["quote_volume"] / (df["trades"] + epsilon)


def calculate_normalized_net_order_flow(
    df: pd.DataFrame, epsilon: float = 1e-8
) -> pd.Series:
    r"""Tính tỷ lệ mất cân bằng dòng lệnh ròng hai chiều chuẩn hóa trong khoảng [-1, +1].

    Công thức:
        $$\text{Taker Sell} = \text{Volume} - \text{Taker Buy}$$
        $$\text{Net Flow} = \frac{\text{Taker Buy} - \text{Taker Sell}}{\text{Volume} + \epsilon} = \frac{2 \cdot \text{Taker Buy} - \text{Volume}}{\text{Volume} + \epsilon}$$

    Args:
        df: DataFrame chứa cột 'taker_buy_volume' và 'volume'.
        epsilon: Hằng số chống chia cho 0.

    Returns:
        pd.Series: Chuỗi tỷ lệ dòng lệnh ròng hai chiều.
    """
    return (2.0 * df["taker_buy_volume"] - df["volume"]) / (df["volume"] + epsilon)


def calculate_trades_zscore(
    df: pd.DataFrame, window: int = 60, epsilon: float = 1e-8
) -> pd.Series:
    r"""Tính Z-Score chuẩn hóa mức độ đột biến số lượng giao dịch trong 60 phút.

    Công thức:
        $$Z_{trades} = \frac{\text{trades} - \mu_{trades, 60m}}{\sigma_{trades, 60m} + \epsilon}$$

    Args:
        df: DataFrame chứa cột 'trades'.
        window: Cửa sổ trượt tính trung bình và độ lệch chuẩn (mặc định 60 phút).
        epsilon: Hằng số chống chia cho 0.

    Returns:
        pd.Series: Chuỗi Z-score chuẩn hóa của số lượng giao dịch.
    """
    roll_mean = df["trades"].rolling(window=window, min_periods=window).mean()
    roll_std = df["trades"].rolling(window=window, min_periods=window).std()
    return (df["trades"] - roll_mean) / (roll_std + epsilon)


def calculate_volatility_term_structure(
    df: pd.DataFrame, short_window: int = 15, long_window: int = 60, epsilon: float = 1e-8
) -> pd.Series:
    r"""Tính tỷ lệ cấu trúc kỳ hạn biến động (Volatility Term Structure / Ratio).

    Nhận diện sớm các pha nén biến động (Volatility Squeeze) và bùng nổ (Breakout).

    Công thức:
        $$\text{Vol Ratio} = \frac{\sigma_{P, 15m}}{\sigma_{P, 60m} + \epsilon}$$

    Args:
        df: DataFrame chứa cột 'high' và 'low'.
        short_window: Cửa sổ ngắn hạn (mặc định 15 phút).
        long_window: Cửa sổ dài hạn (mặc định 60 phút).
        epsilon: Hằng số chống chia cho 0.

    Returns:
        pd.Series: Chuỗi tỷ lệ biến động kỳ hạn.
    """
    vol_short = calculate_parkinson_volatility(df, window=short_window)
    vol_long = calculate_parkinson_volatility(df, window=long_window)
    return vol_short / (vol_long + epsilon)


def calculate_jump_intensity(
    df: pd.DataFrame, window: int = 15, epsilon: float = 1e-8
) -> pd.Series:
    r"""Tính cường độ bước nhảy giá mở cửa/đóng cửa bất thường so với nền biến động.

    Công thức:
        $$\text{Jump Intensity} = \frac{|\ln(\text{Open}_t / \text{Close}_{t-1})|}{\sigma_{P, 15m, t} + \epsilon}$$

    Args:
        df: DataFrame chứa 'open', 'close', 'high', 'low'.
        window: Cửa sổ tính biến động cơ sở (mặc định 15 phút).
        epsilon: Hằng số chống chia cho 0.

    Returns:
        pd.Series: Chuỗi cường độ bước nhảy giá.
    """
    gap = np.abs(np.log(df["open"] / df["close"].shift(1)))
    vol_base = calculate_parkinson_volatility(df, window=window)
    return gap / (vol_base + epsilon)


def create_binary_target(
    df: pd.DataFrame,
    forward_window: int = 15,
    quantile_threshold: float = 0.80,
) -> Tuple[pd.DataFrame, float]:
    r"""Tạo nhãn biến mục tiêu nhị phân dự báo bùng nổ biến động 15 phút tới.

    Công thức:
        $$Y_t = \mathbb{I}(\sigma_{fwd, 15m, t} \ge Q_{0.80}(\sigma_{fwd, 15m}))$$

    Args:
        df: DataFrame chứa cột 'close'.
        forward_window: Cửa sổ tính biến động tương lai (mặc định 15 phút).
        quantile_threshold: Ngưỡng phân vị xác định đột biến (mặc định 0.80).

    Returns:
        Tuple[pd.DataFrame, float]: DataFrame đã gán 'forward_vol_15m', 'target_vol_spike_15m' và giá trị ngưỡng cut-off.
    """
    out_df = df.copy()
    if "log_return" not in out_df.columns:
        out_df["log_return"] = np.log(out_df["close"] / out_df["close"].shift(1))

    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=forward_window)
    out_df["forward_vol_15m"] = (
        out_df["log_return"]
        .shift(-1)
        .rolling(window=indexer, min_periods=forward_window)
        .std()
    )

    vol_cutoff = float(out_df["forward_vol_15m"].quantile(quantile_threshold))
    out_df["target_vol_spike_15m"] = (
        out_df["forward_vol_15m"] >= vol_cutoff
    ).astype(float)
    out_df.loc[out_df["forward_vol_15m"].isna(), "target_vol_spike_15m"] = np.nan

    return out_df, vol_cutoff


def generate_feature_matrix(
    df: pd.DataFrame, dropna: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """Tạo ma trận 8 đặc trưng cấu trúc vi mô toàn diện từ dữ liệu OHLCV 1 phút (Kịch bản Baseline).

    Args:
        df: DataFrame dữ liệu nến 1 phút sạch.
        dropna: Có loại bỏ các dòng chứa NaN do khởi động cửa sổ trượt hay không.

    Returns:
        Tuple[pd.DataFrame, List[str]]: DataFrame đã gán features và danh sách tên các features.
    """
    out_df = df.copy()

    # 1. Log return 1m
    out_df["log_return"] = np.log(out_df["close"] / out_df["close"].shift(1))

    # 2. Parkinson Volatility 15m
    out_df["parkinson_vol_15m"] = calculate_parkinson_volatility(out_df, window=15)

    # 3. Garman-Klass Volatility 15m
    out_df["garman_klass_vol_15m"] = calculate_garman_klass_volatility(out_df, window=15)

    # 4. Order Flow Imbalance Ratio
    out_df["ofi_ratio"] = calculate_order_flow_imbalance_ratio(out_df)

    # 5. Trade Density
    out_df["trade_density"] = calculate_trade_density(out_df)

    # 6. Volume Spike Z-Score 60m
    out_df["volume_spike_z_60m"] = calculate_volume_spike_zscore(out_df, window=60)

    # 7. Return Momentum 15m
    out_df["return_momentum_15m"] = calculate_return_momentum(out_df, window=15)

    # 8. Rolling Volatility 60m Annualized
    out_df["rolling_vol_60m"] = calculate_rolling_volatility_60m(out_df, window=60)

    # 9. High-Low Spread Ratio 15m
    out_df["spread_ratio_15m"] = calculate_spread_ratio(out_df, window=15)

    feature_cols = [
        "parkinson_vol_15m",
        "garman_klass_vol_15m",
        "ofi_ratio",
        "trade_density",
        "volume_spike_z_60m",
        "return_momentum_15m",
        "rolling_vol_60m",
        "spread_ratio_15m",
    ]

    if dropna:
        out_df = out_df.dropna(subset=feature_cols)

    return out_df, feature_cols


def generate_extended_feature_matrix(
    df: pd.DataFrame, dropna: bool = True
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Tạo ma trận 16 đặc trưng cấu trúc vi mô mở rộng và định nghĩa 4 kịch bản nghiên cứu (Ablation Scenarios).

    Kịch bản:
    - scenario_a (8 features): Baseline vi cấu trúc chuẩn mực.
    - scenario_b (12 features): Khai thác 100% cột dữ liệu thô (tích hợp VWAP Dev, Dollar Trade Size, Net Flow, Trades Z-Score).
    - scenario_c (16 features): Cấu trúc kỳ hạn biến động đa khung thời gian và cường độ bước nhảy.
    - scenario_d (10 features): Bộ đặc trưng tinh gọn tối ưu (Optimal Parsimonious Subset).

    Args:
        df: DataFrame dữ liệu nến 1 phút sạch.
        dropna: Có loại bỏ các dòng NaN do khởi động rolling windows hay không.

    Returns:
        Tuple[pd.DataFrame, Dict[str, List[str]]]: DataFrame chứa tất cả đặc trưng và Dictionary định nghĩa danh sách feature theo từng scenario.
    """
    out_df, base_cols = generate_feature_matrix(df, dropna=False)

    # Nhóm mở rộng 1: Khai thác 100% cột thô
    out_df["vwap_dev_15m"] = calculate_vwap_deviation(out_df, window=15)
    out_df["dollar_trade_size"] = calculate_dollar_trade_size(out_df)
    out_df["normalized_net_flow"] = calculate_normalized_net_order_flow(out_df)
    out_df["trades_z_60m"] = calculate_trades_zscore(out_df, window=60)

    # Nhóm mở rộng 2: Cấu trúc kỳ hạn & Đa quy mô thời gian
    out_df["vol_term_structure_15_60"] = calculate_volatility_term_structure(out_df, 15, 60)
    out_df["parkinson_vol_5m"] = calculate_parkinson_volatility(out_df, window=5)
    out_df["parkinson_vol_30m"] = calculate_parkinson_volatility(out_df, window=30)
    out_df["jump_intensity_15m"] = calculate_jump_intensity(out_df, window=15)

    all_extended_cols = [
        # Nhóm A (Baseline - 8)
        "parkinson_vol_15m",
        "garman_klass_vol_15m",
        "ofi_ratio",
        "trade_density",
        "volume_spike_z_60m",
        "return_momentum_15m",
        "rolling_vol_60m",
        "spread_ratio_15m",
        # Nhóm B (Full Raw Columns - +4 = 12)
        "vwap_dev_15m",
        "dollar_trade_size",
        "normalized_net_flow",
        "trades_z_60m",
        # Nhóm C (Multi-Scale & Jump - +4 = 16)
        "vol_term_structure_15_60",
        "parkinson_vol_5m",
        "parkinson_vol_30m",
        "jump_intensity_15m",
    ]

    scenarios = {
        "scenario_a_baseline_8": base_cols,
        "scenario_b_full_raw_12": [
            "parkinson_vol_15m",
            "garman_klass_vol_15m",
            "ofi_ratio",
            "trade_density",
            "volume_spike_z_60m",
            "return_momentum_15m",
            "rolling_vol_60m",
            "spread_ratio_15m",
            "vwap_dev_15m",
            "dollar_trade_size",
            "normalized_net_flow",
            "trades_z_60m",
        ],
        "scenario_c_multiscale_16": all_extended_cols,
        "scenario_d_optimal_10": [
            "garman_klass_vol_15m",
            "parkinson_vol_15m",
            "vol_term_structure_15_60",
            "vwap_dev_15m",
            "volume_spike_z_60m",
            "normalized_net_flow",
            "dollar_trade_size",
            "trade_density",
            "return_momentum_15m",
            "ofi_ratio",
        ],
    }

    if dropna:
        out_df = out_df.dropna(subset=all_extended_cols)

    return out_df, scenarios
