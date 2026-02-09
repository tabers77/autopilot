"""Temporal cross-validation strategies.

Time-series requires fundamentally different CV: no shuffling,
respect temporal order, optional gaps to prevent leakage.
"""

from dataclasses import dataclass
from typing import Generator, Optional, Tuple

import numpy as np


@dataclass
class TemporalSplitPolicy:
    """Cross-validation policy for time-series data.

    Supports expanding window, sliding window, and blocked strategies.
    """

    strategy: str = 'expanding_window'
    """Strategy: 'expanding_window', 'sliding_window', or 'blocked'."""

    n_splits: int = 5
    """Number of CV splits."""

    gap: int = 0
    """Number of samples to skip between train and test sets (prevents leakage)."""

    min_train_size: Optional[int] = None
    """Minimum number of training samples. None means use at least one split's worth."""

    max_train_size: Optional[int] = None
    """Maximum training set size (for sliding window). None means no limit."""

    test_size: Optional[int] = None
    """Fixed test set size. None means auto-calculate."""

    def split(self, n_samples: int) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/test index splits.

        :param n_samples: Total number of samples.
        :yields: (train_indices, test_indices) tuples.
        """
        if self.strategy == 'expanding_window':
            yield from self._expanding_window(n_samples)
        elif self.strategy == 'sliding_window':
            yield from self._sliding_window(n_samples)
        elif self.strategy == 'blocked':
            yield from self._blocked(n_samples)
        else:
            raise ValueError(f'Unknown strategy: {self.strategy}')

    def _expanding_window(self, n_samples: int):
        """Expanding window: train grows, test slides forward."""
        test_size = self.test_size or max(1, n_samples // (self.n_splits + 1))
        min_train = self.min_train_size or test_size

        for i in range(self.n_splits):
            test_start = min_train + i * test_size + self.gap
            test_end = test_start + test_size

            if test_end > n_samples:
                break

            train_end = test_start - self.gap
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)

            if len(train_indices) < 1 or len(test_indices) < 1:
                continue

            yield train_indices, test_indices

    def _sliding_window(self, n_samples: int):
        """Sliding window: fixed-size train window slides forward."""
        test_size = self.test_size or max(1, n_samples // (self.n_splits + 1))

        if self.max_train_size is not None:
            max_train = self.max_train_size
        else:
            max_train = max(test_size, n_samples - test_size * self.n_splits)

        # First test window starts after the first train window + gap
        initial_offset = max_train + self.gap

        for i in range(self.n_splits):
            test_start = initial_offset + i * test_size
            test_end = test_start + test_size

            if test_end > n_samples:
                break

            train_end = test_start - self.gap
            train_start = max(0, train_end - max_train)

            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)

            if len(train_indices) < 1 or len(test_indices) < 1:
                continue

            yield train_indices, test_indices

    def _blocked(self, n_samples: int):
        """Blocked: non-overlapping consecutive blocks with gap."""
        block_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = (i + 1) * block_size
            test_start = train_end + self.gap
            test_end = test_start + block_size

            if test_end > n_samples:
                test_end = n_samples

            if test_start >= n_samples:
                break

            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)

            if len(train_indices) < 1 or len(test_indices) < 1:
                continue

            yield train_indices, test_indices

    def get_n_splits(self) -> int:
        return self.n_splits
