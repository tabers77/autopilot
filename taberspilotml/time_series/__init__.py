"""Time-series capabilities for the experiment framework.

Provides temporal-aware cross-validation, dataset handling, and feature engineering
with strict leakage prevention.
"""

from taberspilotml.time_series.dataset import TimeSeriesDataset
from taberspilotml.time_series.cv import TemporalSplitPolicy
from taberspilotml.time_series.features import LagFeatureEngine
