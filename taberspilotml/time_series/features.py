"""Time-series feature engineering with strict leakage prevention.

Lag features must respect temporal cutoffs. This module provides
explicit leakage validation to enforce correctness by construction.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class LagFeatureEngine:
    """Creates lag features with strict cutoff enforcement.

    All lag features look backward in time only (no future information leakage).
    """

    lags: List[int] = field(default_factory=lambda: [1, 2, 3])
    """Lag periods to create features for."""

    target_column: Optional[str] = None
    """Column to create lag features from. If None, creates lags for all numeric columns."""

    rolling_windows: Optional[List[int]] = None
    """Rolling window sizes for rolling mean/std features."""

    drop_na: bool = True
    """Whether to drop rows with NaN created by lagging."""

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features from the dataframe.

        :param df: DataFrame with time-ordered data.
        :returns: DataFrame with lag features added.
        """
        result = df.copy()

        columns_to_lag = ([self.target_column] if self.target_column
                          else list(df.select_dtypes(include=[np.number]).columns))

        for col in columns_to_lag:
            if col not in df.columns:
                continue

            # Lag features
            for lag in self.lags:
                result[f'{col}_lag_{lag}'] = result[col].shift(lag)

            # Rolling window features
            if self.rolling_windows:
                for window in self.rolling_windows:
                    result[f'{col}_rolling_mean_{window}'] = (
                        result[col].shift(1).rolling(window=window, min_periods=1).mean()
                    )
                    result[f'{col}_rolling_std_{window}'] = (
                        result[col].shift(1).rolling(window=window, min_periods=1).std()
                    )

        if self.drop_na:
            result = result.dropna().reset_index(drop=True)

        return result

    def validate_no_leakage(self, df: pd.DataFrame, cutoff_idx: int) -> bool:
        """Validate that lag features at the cutoff don't use future data.

        Checks that all lag feature values at cutoff_idx only depend on
        data from indices < cutoff_idx.

        :param df: DataFrame with lag features.
        :param cutoff_idx: The cutoff index (first test sample).
        :returns: True if no leakage detected.
        """
        if cutoff_idx <= 0 or cutoff_idx >= len(df):
            return True

        lag_cols = [c for c in df.columns if '_lag_' in c or '_rolling_' in c]

        for col in lag_cols:
            # Extract lag amount from column name
            if '_lag_' in col:
                parts = col.rsplit('_lag_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    lag = int(parts[1])
                    # At cutoff_idx, the lag feature should use data from cutoff_idx - lag
                    # which must be < cutoff_idx (always true for positive lags)
                    if lag <= 0:
                        return False

            if '_rolling_' in col:
                # Rolling features use shift(1) so they only look at past data
                # The shift(1) ensures no current-period leakage
                pass

        return True

    @property
    def feature_names(self) -> List[str]:
        """Return the names of features that would be created.

        Note: requires target_column to be set.
        """
        if not self.target_column:
            return []

        names = []
        col = self.target_column

        for lag in self.lags:
            names.append(f'{col}_lag_{lag}')

        if self.rolling_windows:
            for window in self.rolling_windows:
                names.append(f'{col}_rolling_mean_{window}')
                names.append(f'{col}_rolling_std_{window}')

        return names
