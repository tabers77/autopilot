"""TimeSeriesDataset: extends Dataset with time-series aware operations.

Time-series requires fundamentally different handling: temporal ordering,
cutoff-aware slicing, and leakage prevention by construction.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Text, Union

import numpy as np
import pandas as pd

from taberspilotml.scoring_funcs.datasets import Dataset


@dataclass
class TimeSeriesDataset(Dataset):
    """A dataset with time-series awareness.

    Extends the base Dataset with temporal ordering, frequency info,
    and cutoff-aware slicing.
    """

    time_index: Optional[str] = None
    """Name of the datetime column used as the time index."""

    freq: Optional[str] = None
    """Frequency of the time series (e.g., 'D', 'H', 'M')."""

    covariates: Optional[List[str]] = None
    """Names of covariate columns (exogenous features)."""

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, target_columns: List[Text],
                       time_index: str, freq: Optional[str] = None,
                       covariates: Optional[List[str]] = None,
                       scaler=None) -> 'TimeSeriesDataset':
        """Create a TimeSeriesDataset from a DataFrame.

        Validates temporal ordering and sets up the time index.

        :param df: DataFrame with time column, features, and targets.
        :param target_columns: Target column names.
        :param time_index: Name of the datetime column.
        :param freq: Expected frequency.
        :param covariates: Names of covariate columns.
        :param scaler: Optional scaler for features.
        """
        # Validate temporal ordering
        if time_index not in df.columns:
            raise ValueError(f'Time index column "{time_index}" not found in DataFrame.')

        sorted_df = df.sort_values(time_index).reset_index(drop=True)

        # Validate ordering
        time_col = pd.to_datetime(sorted_df[time_index])
        if not time_col.is_monotonic_increasing:
            raise ValueError('Time index is not monotonically increasing after sorting.')

        # Build base dataset (drop time_index from inputs)
        feature_cols = [c for c in sorted_df.columns
                        if c not in target_columns and c != time_index]

        labels = sorted_df[target_columns[0]].copy() if len(target_columns) == 1 \
            else sorted_df[target_columns].copy()

        if scaler:
            inputs = pd.DataFrame(
                scaler.fit_transform(sorted_df[feature_cols]),
                columns=feature_cols,
            )
        else:
            inputs = sorted_df[feature_cols].copy()

        return cls(
            inputs=inputs,
            labels=labels,
            time_index=time_index,
            freq=freq,
            covariates=covariates,
        )

    def slice_by_cutoff(self, cutoff_idx: int) -> tuple:
        """Split dataset at a cutoff index (temporal split).

        :param cutoff_idx: Index position for the cutoff.
        :returns: (train_dataset, test_dataset) tuple.
        """
        if isinstance(self.inputs, pd.DataFrame):
            train_inputs = self.inputs.iloc[:cutoff_idx]
            test_inputs = self.inputs.iloc[cutoff_idx:]
        else:
            train_inputs = self.inputs[:cutoff_idx]
            test_inputs = self.inputs[cutoff_idx:]

        if isinstance(self.labels, (pd.DataFrame, pd.Series)):
            train_labels = self.labels.iloc[:cutoff_idx]
            test_labels = self.labels.iloc[cutoff_idx:]
        else:
            train_labels = self.labels[:cutoff_idx]
            test_labels = self.labels[cutoff_idx:]

        train_ds = Dataset(inputs=train_inputs, labels=train_labels)
        test_ds = Dataset(inputs=test_inputs, labels=test_labels)

        return train_ds, test_ds

    def validate_temporal_ordering(self, df: pd.DataFrame) -> bool:
        """Validate that the DataFrame is temporally ordered.

        :param df: DataFrame with time_index column.
        :returns: True if valid.
        """
        if self.time_index not in df.columns:
            return False
        time_col = pd.to_datetime(df[self.time_index])
        return time_col.is_monotonic_increasing
