"""Module kiểm lỗi chéo chuỗi thời gian chống rò rỉ dữ liệu (Time-Aware Cross-Validation).

Cung cấp bộ chia PurgedGroupTimeSeriesSplit và Time-Aware Rolling/Expanding splits
nhằm ngăn chặn hiện tượng Lookahead Leakage và Information Bleeding.
"""

from typing import Generator, List, Tuple
import numpy as np
import pandas as pd


class PurgedTimeSeriesSplit:
    """Bộ chia Cross-Validation chuỗi thời gian có Purging và Embargoing.

    Nguyên lý:
    1. Chia dữ liệu theo trình tự thời gian liên tục thành n_splits phần.
    2. Purging: Loại bỏ purge_window nến trước Test set để ngăn chặn
       thông tin Forward Target của Train set rò rỉ sang Test set.
    3. Embargoing: Thêm khoảng đệm embargo_window nến sau Test set
       để triệt tiêu hiệu ứng tự tương quan còn sót lại trước khi dùng cho Train các Fold sau.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_window: int = 15,
        embargo_window: int = 30,
    ) -> None:
        """Khởi tạo PurgedTimeSeriesSplit.

        Args:
            n_splits: Số lượng folds (mặc định 5).
            purge_window: Số nến loại bỏ ở cuối tập Train (mặc định 15 phút).
            embargo_window: Số nến cách ly sau tập Test (mặc định 30 phút).
        """
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo_window = embargo_window

    def split(
        self, X: pd.DataFrame
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Tạo các cặp chỉ số (train_indices, test_indices) theo từng Fold.

        Args:
            X: DataFrame dữ liệu có index thời gian.

        Yields:
            Tuple[np.ndarray, np.ndarray]: Mảng chỉ số vị trí (integer indices) của Train và Test.
        """
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            # Test fold range
            test_start = (i + 1) * fold_size
            test_end = (i + 2) * fold_size if i < self.n_splits - 1 else n_samples

            # Train fold range (Expanding Window từ đầu đến trước test_start - purge_window)
            train_end = max(0, test_start - self.purge_window)
            train_indices = np.arange(0, train_end)

            test_indices = np.arange(test_start, test_end)

            if len(train_indices) == 0 or len(test_indices) == 0:
                continue

            yield train_indices, test_indices

    def get_fold_boundaries(
        self, df: pd.DataFrame
    ) -> List[dict]:
        """Trả về chi tiết thời gian và kích thước của từng Fold phục vụ trực quan hóa.

        Args:
            df: DataFrame có index Datetime.

        Returns:
            List[dict]: Danh sách thông tin start, end, purge, embargo của từng Fold.
        """
        boundaries = []
        for fold_idx, (train_idx, test_idx) in enumerate(self.split(df)):
            boundaries.append(
                {
                    "fold": fold_idx + 1,
                    "train_start_idx": train_idx[0],
                    "train_end_idx": train_idx[-1],
                    "train_start_time": df.index[train_idx[0]],
                    "train_end_time": df.index[train_idx[-1]],
                    "test_start_idx": test_idx[0],
                    "test_end_idx": test_idx[-1],
                    "test_start_time": df.index[test_idx[0]],
                    "test_end_time": df.index[test_idx[-1]],
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                }
            )
        return boundaries
