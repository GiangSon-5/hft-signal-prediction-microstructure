# Báo Cáo Thực Nghiệm Định Lượng: Feature Ablation & Multi-Model Benchmarking

- Thời điểm thực thi: 2026-08-23 14:14:05Z
- Tổng số mẫu kiểm định: 264,886
- Phương pháp chia mẫu: 5-Fold Time-Aware Purged (15m) & Embargoed (30m) Cross-Validation
- Hiệu chuẩn xác suất: Isotonic Regression

### Bảng Xếp Hạng Hiệu Năng Toàn Bộ Cấu Hình Thực Nghiệm

| scenario                 | model                |   feature_count |   roc_auc |   pr_auc |   brier_score |    ece |   f1_score |   elapsed_seconds |
|:-------------------------|:---------------------|----------------:|----------:|---------:|--------------:|-------:|-----------:|------------------:|
| scenario_c_multiscale_16 | HistGBDT             |              16 |    0.9091 |   0.7674 |        0.0869 | 0.0183 |     0.663  |            110.51 |
| scenario_c_multiscale_16 | LightGBM             |              16 |    0.9089 |   0.767  |        0.0869 | 0.0184 |     0.6636 |             12.71 |
| scenario_c_multiscale_16 | XGBoost              |              16 |    0.9089 |   0.7668 |        0.087  | 0.0188 |     0.6624 |             13.2  |
| scenario_b_full_raw_12   | HistGBDT             |              12 |    0.908  |   0.7638 |        0.0876 | 0.0184 |     0.6623 |            118.05 |
| scenario_b_full_raw_12   | LightGBM             |              12 |    0.9079 |   0.7634 |        0.0877 | 0.0188 |     0.6605 |             12.65 |
| scenario_b_full_raw_12   | XGBoost              |              12 |    0.9079 |   0.7632 |        0.0877 | 0.0191 |     0.6603 |             19.2  |
| scenario_a_baseline_8    | HistGBDT             |               8 |    0.9075 |   0.7611 |        0.0881 | 0.0182 |     0.6594 |             86.43 |
| scenario_a_baseline_8    | LightGBM             |               8 |    0.9075 |   0.761  |        0.0881 | 0.0186 |     0.6592 |             10.26 |
| scenario_a_baseline_8    | XGBoost              |               8 |    0.9075 |   0.761  |        0.0881 | 0.0181 |     0.6577 |             10.13 |
| scenario_d_optimal_10    | XGBoost              |              10 |    0.9077 |   0.7599 |        0.0881 | 0.02   |     0.6613 |             10.82 |
| scenario_d_optimal_10    | LightGBM             |              10 |    0.9076 |   0.7595 |        0.0882 | 0.0196 |     0.6604 |             11.31 |
| scenario_d_optimal_10    | HistGBDT             |              10 |    0.9075 |   0.7594 |        0.0882 | 0.0198 |     0.6594 |             56.08 |
| Baseline_Rule_Based      | Rule_Based_Heuristic |               2 |    0.8455 |   0.5894 |        0.1718 | 0.2359 |     0.5904 |              0.5  |

