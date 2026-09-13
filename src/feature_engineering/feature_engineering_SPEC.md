# Đặc Tả Kỹ Thuật (Module Specification): Kỹ Thuật Tạo Đặc Trưng (`src/feature_engineering`)

## 1. Tổng Quan Mô-đun

Mô-đun `feature_engineering` trích xuất 16 đặc trưng cấu trúc vi mô từ nến OHLCV 1 phút thô, số lượng giao dịch `trades`, khối lượng giao dịch định danh `quote_volume` và khối lượng mua chủ động `taker_buy_volume`. Các đặc trưng được biến đổi thành các tín hiệu chuỗi thời gian dừng (stationary), không rò rỉ dữ liệu tương lai, và được đóng gói thành 4 kịch bản đối chuẩn (A, B, C, D) cho mô hình học máy.

---

## 2. Danh Sách 16 Đặc Trưng & Lý Do Kỳ Vọng Hữu Ích

### 2.1 Ma Trận 16 Đặc Trưng Vi Cấu Trúc
| Tên Feature | Kiểu Dữ Liệu | Công Thức & Ý Nghĩa Vi Cấu Trúc |
| :--- | :--- | :--- |
| `parkinson_vol_15m` | `float64` | Ước lượng độ biến động qua khoảng High/Low 15 phút. Sử dụng biên độ extreme-value giúp ước lượng variance hiệu quả gấp 5 lần so với close-to-close return. |
| `garman_klass_vol_15m`| `float64` | Ước lượng độ biến động bao gồm cả khoảng nhảy nến Open/Close và High/Low 15 phút. Bắt được các cú nhảy giá đột ngột giữa các nến 1 phút. |
| `ofi_ratio` | `float64` | Tỷ lệ Order Flow Imbalance ($\frac{\text{TakerBuyVol}}{\text{TotalVol}}$). Đo lường sự mất cân bằng giữa lực mua chủ động và lực bán chủ động trong sổ lệnh. |
| `trade_density` | `float64` | Khối lượng trung bình mỗi giao dịch ($\frac{\text{Volume}}{\text{Trades}}$). Phân biệt dòng tiền tổ chức (giao dịch lớn) và dòng tiền nhỏ lẻ (nhiều giao dịch nhỏ). |
| `volume_spike_z_60m` | `float64` | Z-score khối lượng nến hiện tại so với 60 phút trước. Đánh dấu các cú bùng nổ khối lượng đột biến (institutional sweeps). |
| `return_momentum_15m` | `float64` | Lợi nhuận tích lũy 15 phút $\ln(\text{Close}_t / \text{Close}_{t-15})$. Đo lường xu hướng và động lượng ngắn hạn của giá. |
| `rolling_vol_60m` | `float64` | Độ lệch chuẩn tỷ suất sinh lời 60 phút quy năm. Đo lường mức biến động nền tảng trung hạn. |
| `spread_ratio_15m` | `float64` | Tỷ số biên độ nến trung bình trượt 15 phút $\frac{\text{High} - \text{Low}}{\text{Open}}$. Phản ánh mức độ mở rộng của dải giá. |
| `vwap_dev_15m` | `float64` | Độ lệch giữa giá hiện tại và VWAP 15m $\frac{P_t - \text{VWAP}_{15m}}{P_t}$. Đo lường áp lực mua/bán tích cực đẩy giá rời xa mức cân bằng khối lượng. |
| `dollar_trade_size` | `float64` | Quy mô USD bình quân mỗi lệnh $\frac{\text{Quote Volume}}{\text{Trades}}$. Phát hiện sự xuất hiện của dòng tiền lớn (Whale orders). |
| `normalized_net_flow`| `float64` | Dòng tiền ròng chủ động chuẩn hóa trong đoạn $[-1, 1]$ $\frac{2 \cdot \text{TakerBuy} - \text{Vol}}{\text{Vol} + \epsilon}$. |
| `trades_z_60m` | `float64` | Z-Score bùng nổ số lượng giao dịch so với lịch sử 60 phút. |
| `vol_term_structure_15_60`| `float64`| Tỷ số cấu trúc kỳ hạn biến động $\frac{\sigma_{\text{GK}, 15m}}{\sigma_{\text{GK}, 60m}}$. Phát hiện sự bùng nổ biến động ngắn hạn so với nền trung hạn. |
| `parkinson_vol_5m` | `float64` | Biến động Parkinson siêu ngắn hạn 5 phút (bắt tín hiệu xung lực tức thời). |
| `parkinson_vol_30m` | `float64` | Biến động Parkinson trung hạn 30 phút (đo lường quán tính xu hướng). |
| `jump_intensity_15m` | `float64` | Cường độ bước nhảy giá $\frac{\text{Parkinson Vol}_{15m}}{\text{Realized Vol}_{15m}}$. Tách biệt biến động liên tục và biến động do tin tức đột ngột. |

---

## 3. Định Nghĩa 4 Kịch Bản Nghiên Cứu (Scenarios)

1. **`scenario_a_baseline_8`**: 8 đặc trưng cơ sở.
2. **`scenario_b_full_raw_12`**: 8 biến cơ sở + 4 biến khai thác 100% cột thô (`vwap_dev_15m`, `dollar_trade_size`, `normalized_net_flow`, `trades_z_60m`).
3. **`scenario_c_multiscale_16`**: Toàn bộ 16 biến đa quy mô thời gian và cấu trúc kỳ hạn.
4. **`scenario_d_optimal_10`**: Top 10 biến tinh gọn có Permutation Importance cao nhất.
