# Đặc Tả Kỹ Thuật (Module Specification): Phân Tích Chuyên Sâu Cấu Trúc Thị Trường (`src/deep_dive`)

## 1. Tổng Quan Mô-đun

Mô-đun `deep_dive` thực thi Task 3 (Phân tích mở rộng). Mô-đun tiến hành nghiên cứu định lượng hiện tượng cấu trúc thị trường: **Độ độc hại của dòng lệnh (Order Flow Toxicity) & Tác động giá bất đối xứng (Kyle's Lambda) trong các Chế độ Biến động khác nhau**. Mô-đun thiết lập giả thuyết thống kê ($H_0$ vs $H_1$), ước lượng định lượng bằng phương pháp Block Bootstrap resampling với khoảng tin cậy 95% CI (Uncertainty Quantification), và đưa ra định hướng phát triển tiếp theo.

---

## 2. Giả Thuyết Thống Kê & Quy Chuẩn Kết Quả

### 2.1 Hiện Tượng Nghiên Cứu & Giả Thuyết Thống Kê
- **Hiện tượng:** Tác động giá (Price Impact) của từng đơn vị khối lượng mua chủ động (OFI) sẽ tăng vọt bất đối xứng khi thị trường chuyển từ Regime Biến động thấp sang Regime Biến động cao do rủi ro lựa chọn bất lợi (adverse selection) của nhà tạo lập thị trường.
- **Giả thuyết $H_0$:** Hệ số tác động giá Kyle's Lambda trong Regime Biến động cao bằng với Regime Biến động thấp ($\lambda_{high} = \lambda_{low}$).
- **Giả thuyết $H_1$:** Hệ số tác động giá Kyle's Lambda trong Regime Biến động cao lớn hơn đáng kể so với Regime Biến động thấp ($\lambda_{high} > \lambda_{low}$).

### 2.2 Schema Kết Quả Đầu Ra (`dict` / `JSON`)
```json
{
  "investigated_phenomenon": "Order Flow Toxicity Asymmetry in High Volatility Regimes",
  "hypothesis": {
    "H0": "lambda_high_vol == lambda_low_vol",
    "H1": "lambda_high_vol > lambda_low_vol"
  },
  "quantitative_results": {
    "lambda_low_vol": 0.00042,
    "lambda_low_vol_95ci": [0.00039, 0.00045],
    "lambda_high_vol": 0.00185,
    "lambda_high_vol_95ci": [0.00168, 0.00204],
    "impact_ratio": 4.40,
    "p_value_bootstrap_difference": 0.0001,
    "hypothesis_accepted": "H1"
  }
}
```

---

## 3. Thuật Toán Hồi Quy & Định Lượng Độ Bất Định (Uncertainty Quantification)

### 3.1 Mô Hình Tác Động Giá Kyle's Lambda Theo Regime
Đo lường tác động giá thực nghiệm của dòng lệnh:
$$\Delta p_{t+1} = \alpha + \lambda \cdot (OFI_t - 0.5) + \epsilon_t$$
với $\Delta p_{t+1} = \frac{Close_{t+1} - Close_t}{Close_t}$ và $OFI_t = \frac{taker\_buy\_volume_t}{volume_t}$.

Hồi quy riêng biệt theo 2 regimes:
1. **Regime Biến động thấp ($S_t = 0$):** $\Delta p_{t+1} = \alpha_{low} + \lambda_{low} (OFI_t - 0.5) + \epsilon_{low}$
2. **Regime Biến động cao ($S_t = 1$):** $\Delta p_{t+1} = \alpha_{high} + \lambda_{high} (OFI_t - 0.5) + \epsilon_{high}$

### 3.2 Định Lượng Độ Bất Định Bằng Block Bootstrap (1,000 Lượt)
Do dữ liệu chuỗi thời gian có tự tương quan, áp dụng Block Bootstrap với kích thước block $B = 60$ nến:
1. Rút mẫu ngẫu nhiên có hoàn lại các khối 60 nến liên tục.
2. Ước lượng lại $\hat{\lambda}_{low}^{(b)}$ và $\hat{\lambda}_{high}^{(b)}$ cho $b = 1, \dots, 1000$.
3. Trích xuất khoảng tin cậy 95% CI: $\text{CI}_{95\%} = [Q_{0.025}, Q_{0.975}]$.
4. Tính p-value kiểm định chênh lệch: $p = \frac{1}{1000} \sum \mathbb{I}(\hat{\lambda}_{high}^{(b)} \le \hat{\lambda}_{low}^{(b)})$.

---

## 4. Hướng Phát Triển Mở Rộng (Future Extensions)

1. **Nâng Cấp Dữ Liệu Tick / Level 2 Order Book:** Mở rộng từ nến 1 phút sang dữ liệu sổ lệnh chi tiết để tính chỉ số VPIN (Volume-Synchronized Probability of Toxicity) và đo lường độ sâu sổ lệnh (Depth Imbalance).
2. **Mô Phỏng Thuật Toán Giao Dịch (Execution Engine):** Thiết lập mô phỏng thuật toán TWAP/VWAP điều chỉnh tốc độ khớp lệnh dựa trên hệ số Kyle's Lambda thực nghiệm.
