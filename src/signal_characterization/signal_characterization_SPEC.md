# Đặc Tả Kỹ Thuật (Module Specification): Đặc Trưng Hóa Tín Hiệu & Phân Tích Thống Kê (`src/signal_characterization`)

## 1. Tổng Quan Mô-đun

Mô-đun `src/signal_characterization/analyzer.py` chịu trách nhiệm thực hiện các phân tích thống kê định lượng về tỷ suất lợi nhuận log 1 phút, phân tích phân phối đuôi béo (Jarque-Bera, Student-t fitting), tính toán độ biến động trượt 60 phút quy năm, phân tách 2 chế độ biến động (Low Vol vs High Vol Regimes), phân tích tương quan đa biến (Volume, Trades, Price Range) và kiểm định tự tương quan (ACF/PACF).

Toàn bộ thuật toán được đóng gói từ kết quả nghiên cứu trong Notebook 1 để phục vụ tái sử dụng cho Task 2, Task 3 và Pipeline Báo Cáo Tự Động (`src/reporting`).

---

## 2. Giao Diện Lập Trình Ứng Dụng (API Contracts)

Mô-đun [`src/signal_characterization/analyzer.py`](file:///c:/Users/Admin/Desktop/Data%20Scientist/src/signal_characterization/analyzer.py) cung cấp 7 hàm cốt lõi:

### 2.1 `calculate_log_returns(df: pd.DataFrame, col: str = 'close') -> pd.Series`
- **Chức năng**: Tính tỷ suất lợi nhuận Log Close-to-Close 1 phút.
- **Công thức**: $r_t = \ln(P_t / P_{t-1})$.

### 2.2 `analyze_returns_distribution(returns: pd.Series) -> Dict[str, Any]`
- **Chức năng**: Phân tích thống kê mô tả (Mean, Std, Skewness, Kurtosis) và kiểm định chuẩn Jarque-Bera, khớp Student-t $df$.
- **Kết quả thực nghiệm**: $Mean = 1.63 \times 10^{-6}$, $Std = 0.074\%$, $Skewness = -0.2693$, $Kurtosis = 65.49$, $JB = 47,347,809.45$ ($p = 0.0$), Student-t $df = 2.665$.

### 2.3 `compute_rolling_volatility(returns: pd.Series, window: int = 60) -> pd.Series`
- **Chức năng**: Tính độ biến động trượt 60m quy năm.
- **Công thức**: $\sigma_{\text{annualized}} = \sigma_{\text{rolling}}(r, 60) \times \sqrt{525,600}$ ($525,600 = 365 \times 24 \times 60$).

### 2.4 `detect_volatility_regimes(rolling_vol: pd.Series, threshold_quantile: float = 0.75) -> Tuple[pd.Series, float]`
- **Chức năng**: Phân tách 2 chế độ biến động thị trường tại ngưỡng phân vị $Q_{0.75}$.
- **Kết quả thực nghiệm**: Ngưỡng cut-off $Q_{0.75} = 55.61\%$ quy năm. Gán nhãn $0$ (Low Vol: $75\%$) và $1$ (High Vol: $25\%$).

### 2.5 `compare_volatility_regimes(returns: pd.Series, regimes: pd.Series) -> pd.DataFrame`
- **Chức năng**: So sánh đặc tính đuôi béo giữa 2 chế độ.
- **Kết quả thực nghiệm**:
  - Low Vol Regime: Kurtosis = $3.73$, Student-t $df = 3.903$.
  - High Vol Regime: Kurtosis = $36.01$, Student-t $df = 1.990 \le 2.0$ (Phương sai vô hạn - Infinite Variance).

### 2.6 `analyze_volume_trades_range(df: pd.DataFrame) -> Dict[str, Any]`
- **Chức năng**: Tính ma trận tương quan phi tuyến Spearman giữa Volume, Trades và Biên độ giá $\text{Price Range Ratio} = (High - Low) / Close$.
- **Kết quả thực nghiệm**: Spearman $\text{Trades vs Range} = +0.7601$, $\text{Volume vs Range} = +0.7490$, $\text{Volume vs Trades} = +0.8672$.

### 2.7 `analyze_autocorrelation(returns: pd.Series, nlags: int = 30) -> Dict[str, Any]`
- **Chức năng**: Kiểm định tự tương quan ACF & PACF với dải tin cậy 95% ($\pm 1.96/\sqrt{N}$).
- **Kết quả thực nghiệm**: Dải tin cậy $\pm 0.0038$. Lag 1 có $r_1 = -0.0057 < -0.0038$ (Đảo chiều vi mô có ý nghĩa thống kê).

---

## 3. Quy Chuẩn Schema Dữ Liệu Tóm Tắt (Summary JSON Schema)

```json
{
  "returns_distribution": {
    "count": 264960,
    "mean": 0.00000163,
    "std": 0.000740,
    "skewness": -0.2693,
    "kurtosis": 65.49,
    "jarque_bera_stat": 47347809.45,
    "jarque_bera_pvalue": 0.0,
    "is_normal": false,
    "student_t_df": 2.665
  },
  "volatility_regimes": {
    "cutoff_q75_annualized": 0.5561,
    "low_vol_count": 198727,
    "high_vol_count": 66233,
    "low_vol_kurtosis": 3.73,
    "high_vol_kurtosis": 36.01,
    "low_vol_student_t_df": 3.903,
    "high_vol_student_t_df": 1.990
  },
  "volume_trades_range_correlation": {
    "spearman_trades_range": 0.7601,
    "spearman_volume_range": 0.7490,
    "spearman_volume_trades": 0.8672
  },
  "autocorrelation": {
    "conf_interval_95": 0.0038,
    "lag1_acf": -0.0057,
    "is_lag1_significant": true
  }
}
```

---

## 4. Hướng Dẫn Tái Sử Dụng Cho Task 2 & Task 3

### 1. Tái sử dụng trong Task 2 (Predictive Modeling):
```python
from src.data_quality.cleaner import validate_and_clean_time_series
from src.signal_characterization.analyzer import calculate_log_returns, compute_rolling_volatility

# 1. Tính toán log returns phục vụ Target 15m spike
df_clean['log_return'] = calculate_log_returns(df_clean)

# 2. Tạo đặc trưng Rolling Volatility 60m Annualized
df_clean['rolling_vol_60m'] = compute_rolling_volatility(df_clean['log_return'], window=60)
```

### 2. Tái sử dụng trong Task 3 (Deep Dive & Market Microstructure):
```python
from src.signal_characterization.analyzer import (
    calculate_log_returns,
    compute_rolling_volatility,
    detect_volatility_regimes,
    analyze_volume_trades_range
)

# Phân tách Regimes để đánh giá tác động giá (Kyle's Lambda & OFI) theo từng trạng thái thị trường
df_clean['log_return'] = calculate_log_returns(df_clean)
rolling_vol = compute_rolling_volatility(df_clean['log_return'], window=60)
df_clean['regime'], cutoff = detect_volatility_regimes(rolling_vol, threshold_quantile=0.75)
```
