"""Mô-đun phân tích chuyên sâu vi cấu trúc thị trường (Market Microstructure Deep Dive).

Mô-đun thực hiện nghiên cứu định lượng hiện tượng:
Tác động giá bất đối xứng của dòng lệnh chủ động (Order Flow Toxicity & Kyle's Lambda)
giữa các Chế độ Biến động khác nhau (Low Volatility vs High Volatility Regimes).
Áp dụng phương pháp Block Bootstrap resampling (1,000 lượt, block length = 60 nến)
để lượng hóa độ bất định (Uncertainty Quantification) với khoảng tin cậy 95% CI.
"""

from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _fast_ols_slope(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Tính toán nhanh hệ số góc (slope) và hệ số chặn (intercept) OLS qua vectorization.

    Args:
        x: Mảng numpy 1D biến độc lập.
        y: Mảng numpy 1D biến phụ thuộc.

    Returns:
        Tuple chứa (slope, intercept).
    """
    n = len(x)
    if n < 2:
        return 0.0, 0.0
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xx = np.dot(x, x)
    sum_xy = np.dot(x, y)
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-14:
        return 0.0, 0.0
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    return float(slope), float(intercept)


class MicrostructureDeepDiveAnalyzer:
    """Bộ phân tích chuyên sâu vi cấu trúc thị trường và tác động giá Kyle's Lambda.

    Lớp này thiết lập mô hình hồi quy thực nghiệm đo lường hệ số Kyle's Lambda
    trên các chế độ biến động khác nhau, áp dụng kỹ thuật Block Bootstrap để bảo toàn
    cấu trúc chuỗi thời gian và kiểm định giả thuyết thống kê H0 vs H1.
    """

    def __init__(self, random_state: int = 42) -> None:
        """Khởi tạo bộ phân tích vi cấu trúc thị trường.

        Args:
            random_state: Hạt giống sinh số ngẫu nhiên phục vụ tái lập thực nghiệm.
        """
        self.random_state = random_state
        np.random.seed(random_state)

    def prepare_microstructure_data(
        self,
        df: pd.DataFrame,
        vol_regime_col: str = "vol_regime",
    ) -> pd.DataFrame:
        """Chuẩn bị và làm sạch chuỗi dữ liệu phục vụ phân tích Kyle's Lambda.

        Args:
            df: DataFrame chứa các cột OHLCV, taker_buy_volume và nhãn regime biến động.
            vol_regime_col: Tên cột chứa nhãn chế độ biến động (0: Low, 1: High).

        Returns:
            pd.DataFrame đã bổ sung các cột OFI và lợi nhuận 1 bước phía trước (forward return).

        Raises:
            KeyError: Nếu thiếu các cột dữ liệu bắt buộc.
        """
        required_cols = ["close", "volume", "taker_buy_volume"]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Thiếu cột dữ liệu bắt buộc: {col}")

        df_work = df.copy()

        # Tính toán Order Flow Imbalance (OFI Ratio)
        epsilon = 1e-8
        df_work["ofi"] = df_work["taker_buy_volume"] / (df_work["volume"] + epsilon)
        df_work["ofi_centered"] = df_work["ofi"] - 0.5

        # Tính toán tỷ suất lợi nhuận 1 bước phía trước (Forward Return 1m)
        df_work["fwd_return_1m"] = df_work["close"].pct_change().shift(-1)

        # Nếu chưa có cột regime biến động, xác định bằng ngưỡng phân vị 75% của Parkinson/CC vol
        if vol_regime_col not in df_work.columns:
            log_hl = np.log(df_work["high"] / df_work["low"]) ** 2
            parkinson = np.sqrt(log_hl.rolling(60).mean() / (4 * np.log(2)))
            threshold = parkinson.quantile(0.75)
            df_work["vol_regime"] = (parkinson >= threshold).astype(int)
        else:
            df_work["vol_regime"] = df_work[vol_regime_col].astype(int)

        df_clean = df_work.dropna(subset=["ofi_centered", "fwd_return_1m", "vol_regime"]).copy()
        return df_clean

    def estimate_kyles_lambda(
        self,
        df: pd.DataFrame,
    ) -> Tuple[float, float, float, float]:
        """Ước lượng điểm hệ số Kyle's Lambda qua hồi quy OLS trên 2 Regime.

        Phương trình hồi quy:
            Delta p_{t+1} = alpha + lambda * (OFI_t - 0.5) + epsilon_t

        Args:
            df: DataFrame đã chuẩn bị chứa 'ofi_centered', 'fwd_return_1m', 'vol_regime'.

        Returns:
            Tuple chứa (lambda_low, alpha_low, lambda_high, alpha_high).
        """
        # Regime Biến động thấp (S = 0)
        df_low = df[df["vol_regime"] == 0]
        lambda_low, alpha_low = _fast_ols_slope(df_low["ofi_centered"].values, df_low["fwd_return_1m"].values)

        # Regime Biến động cao (S = 1)
        df_high = df[df["vol_regime"] == 1]
        lambda_high, alpha_high = _fast_ols_slope(df_high["ofi_centered"].values, df_high["fwd_return_1m"].values)

        return lambda_low, alpha_low, lambda_high, alpha_high

    def run_block_bootstrap(
        self,
        df: pd.DataFrame,
        n_iterations: int = 1000,
        block_size: int = 60,
    ) -> Dict[str, np.ndarray]:
        """Thực thi kỹ thuật Block Bootstrap resampling tối ưu hóa để lượng hóa độ bất định.

        Args:
            df: DataFrame chứa dữ liệu chuỗi thời gian đã làm sạch.
            n_iterations: Số lượng lượt lặp bootstrap (mặc định: 1,000).
            block_size: Kích thước khối tính bằng số nến (mặc định: 60 nến = 1 giờ).

        Returns:
            Dict chứa mảng các ước lượng bootstrap cho lambda_low, lambda_high và lambda_diff.
        """
        n_samples = len(df)
        n_blocks = int(np.ceil(n_samples / block_size))
        block_starts = np.arange(0, n_samples - block_size + 1)
        offset = np.arange(block_size)

        lambda_low_boot = np.empty(n_iterations, dtype=np.float64)
        lambda_high_boot = np.empty(n_iterations, dtype=np.float64)

        x_all = df["ofi_centered"].values
        y_all = df["fwd_return_1m"].values
        regime_all = df["vol_regime"].values

        rng = np.random.default_rng(self.random_state)

        for i in range(n_iterations):
            selected_starts = rng.choice(block_starts, size=n_blocks, replace=True)
            indices = (selected_starts[:, None] + offset).ravel()[:n_samples]

            b_x = x_all[indices]
            b_y = y_all[indices]
            b_regime = regime_all[indices]

            # Fit cho Low Vol
            mask_low = (b_regime == 0)
            if np.sum(mask_low) > 10:
                l_l, _ = _fast_ols_slope(b_x[mask_low], b_y[mask_low])
                lambda_low_boot[i] = l_l
            else:
                lambda_low_boot[i] = np.nan

            # Fit cho High Vol
            mask_high = (b_regime == 1)
            if np.sum(mask_high) > 10:
                l_h, _ = _fast_ols_slope(b_x[mask_high], b_y[mask_high])
                lambda_high_boot[i] = l_h
            else:
                lambda_high_boot[i] = np.nan

        valid_mask = ~np.isnan(lambda_low_boot) & ~np.isnan(lambda_high_boot)
        lambda_low_valid = lambda_low_boot[valid_mask]
        lambda_high_valid = lambda_high_boot[valid_mask]
        lambda_diff = lambda_high_valid - lambda_low_valid

        return {
            "lambda_low": lambda_low_valid,
            "lambda_high": lambda_high_valid,
            "lambda_diff": lambda_diff,
        }

    def run_full_deep_dive(
        self,
        df: pd.DataFrame,
        n_iterations: int = 1000,
        block_size: int = 60,
        output_figure_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Thực hiện quy trình phân tích chuyên sâu toàn diện và kiểm định giả thuyết.

        Args:
            df: DataFrame dữ liệu thị trường nến 1 phút.
            n_iterations: Số lượt mô phỏng Block Bootstrap (mặc định: 1,000).
            block_size: Độ dài khối liên tục (mặc định: 60 nến).
            output_figure_path: Đường dẫn lưu trữ biểu đồ kết quả (tùy chọn).

        Returns:
            Dict chứa kết quả định lượng đầy đủ theo schema đặc tả deep_dive_SPEC.md.
        """
        df_prep = self.prepare_microstructure_data(df)
        l_low, a_low, l_high, a_high = self.estimate_kyles_lambda(df_prep)

        boot_res = self.run_block_bootstrap(df_prep, n_iterations=n_iterations, block_size=block_size)

        ci_low = [float(np.percentile(boot_res["lambda_low"], 2.5)), float(np.percentile(boot_res["lambda_low"], 97.5))]
        ci_high = [float(np.percentile(boot_res["lambda_high"], 2.5)), float(np.percentile(boot_res["lambda_high"], 97.5))]
        ci_diff = [float(np.percentile(boot_res["lambda_diff"], 2.5)), float(np.percentile(boot_res["lambda_diff"], 97.5))]

        p_value = float(np.mean(boot_res["lambda_diff"] <= 0))
        impact_ratio = float(l_high / (l_low + 1e-8))
        accepted_hyp = "H1" if p_value < 0.05 and l_high > l_low else "H0"

        if output_figure_path:
            output_figure_path.parent.mkdir(parents=True, exist_ok=True)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

            # Scatter & OLS fit
            sample_size = min(2000, len(df_prep))
            df_low_s = df_prep[df_prep["vol_regime"] == 0].sample(n=sample_size, random_state=self.random_state)
            df_high_s = df_prep[df_prep["vol_regime"] == 1].sample(n=sample_size, random_state=self.random_state)

            axes[0].scatter(df_low_s["ofi_centered"], df_low_s["fwd_return_1m"], alpha=0.15, color="#2563eb", s=8, label="Low Vol Scatter")
            axes[0].scatter(df_high_s["ofi_centered"], df_high_s["fwd_return_1m"], alpha=0.15, color="#dc2626", s=8, label="High Vol Scatter")

            x_line = np.linspace(-0.5, 0.5, 100)
            axes[0].plot(x_line, a_low + l_low * x_line, color="#1d4ed8", linewidth=2.2, label=f"Low Vol Regime (Lambda = {l_low:.5f})")
            axes[0].plot(x_line, a_high + l_high * x_line, color="#b91c1c", linewidth=2.2, label=f"High Vol Regime (Lambda = {l_high:.5f})")

            axes[0].set_title("Order Flow Price Impact (Kyle's Lambda) by Volatility Regime", fontsize=11, fontweight="bold")
            axes[0].set_xlabel("Order Flow Imbalance (Centered OFI)", fontsize=10)
            axes[0].set_ylabel("Forward 1-Minute Return", fontsize=10)
            axes[0].set_ylim(-0.008, 0.008)
            axes[0].grid(True, linestyle="--", alpha=0.5)
            axes[0].legend(loc="upper left", frameon=True)

            # Bootstrap KDE distribution
            sns.kdeplot(boot_res["lambda_low"], fill=True, color="#2563eb", label=f"Lambda Low Vol (95% CI: [{ci_low[0]:.5f}, {ci_low[1]:.5f}])", ax=axes[1])
            sns.kdeplot(boot_res["lambda_high"], fill=True, color="#dc2626", label=f"Lambda High Vol (95% CI: [{ci_high[0]:.5f}, {ci_high[1]:.5f}])", ax=axes[1])
            axes[1].axvline(l_low, color="#1d4ed8", linestyle="--", alpha=0.8)
            axes[1].axvline(l_high, color="#b91c1c", linestyle="--", alpha=0.8)
            axes[1].set_title(f"Block Bootstrap Distribution (M=1,000, B=60m, Impact Ratio={impact_ratio:.2f}x)", fontsize=11, fontweight="bold")
            axes[1].set_xlabel("Estimated Lambda Coefficient", fontsize=10)
            axes[1].set_ylabel("Density", fontsize=10)
            axes[1].grid(True, linestyle="--", alpha=0.5)
            axes[1].legend(loc="upper right", frameon=True)

            plt.tight_layout()
            plt.savefig(output_figure_path)
            plt.close()

        result_dict = {
            "investigated_phenomenon": "Order Flow Toxicity Asymmetry in High Volatility Regimes",
            "hypothesis": {
                "H0": "lambda_high_vol == lambda_low_vol",
                "H1": "lambda_high_vol > lambda_low_vol",
            },
            "quantitative_results": {
                "lambda_low_vol": float(l_low),
                "lambda_low_vol_95ci": ci_low,
                "lambda_high_vol": float(l_high),
                "lambda_high_vol_95ci": ci_high,
                "lambda_diff_95ci": ci_diff,
                "impact_ratio": float(impact_ratio),
                "p_value_bootstrap_difference": float(p_value),
                "hypothesis_accepted": accepted_hyp,
            },
            "sample_size": int(len(df_prep)),
            "bootstrap_iterations": int(len(boot_res["lambda_low"])),
            "block_size_bars": int(block_size),
        }
        return result_dict
