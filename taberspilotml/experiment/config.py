"""Typed, serializable configuration objects that capture ALL pipeline decisions as one reproducible unit.

Borrows from: auto-sklearn (ConfigSpace), EvalML (typed Pipeline), PyCaret (setup() captures all decisions upfront).
Separating *specification* from *execution* is essential for pipeline search and reproducibility.
"""

import enum
import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

from taberspilotml import constants
from taberspilotml.scoring_funcs.evaluation_metrics import EvalMetrics


class TaskType(enum.Enum):
    CLASSIFICATION = 'classification'
    REGRESSION = 'regression'
    TIME_SERIES = 'time_series'


@dataclass
class PreprocessingSpec:
    """Captures all preprocessing decisions."""

    imputation_strategy: str = 'mean'
    """Strategy for handling missing values: 'mean', 'median', 'drop', 'knn', etc."""

    encoding_method: str = 'default'
    """Encoding method for categorical variables: 'default', 'label', 'onehot', etc."""

    outlier_strategy: Optional[str] = None
    """Outlier handling strategy: None, 'iqr', 'zscore', 'isolation_forest', etc."""

    oversampler: Optional[str] = None
    """Oversampling method for imbalanced data: None, 'smote', 'adasyn', etc."""


@dataclass
class FeatureSpec:
    """Captures feature engineering decisions."""

    feature_selection_method: Optional[str] = None
    """Feature selection method: None, 'rfe', 'importance', 'variance_threshold', etc."""

    scaler_name: Optional[str] = None
    """Scaler name from configs.scalers: None, 'MinMax', 'Standard'."""

    transformer_name: Optional[str] = None
    """Transformer name from configs.transformers: None, 'PCA', 'KBins', 'PowerTransformer', etc."""

    use_transformers: bool = False
    """Whether to apply a transformer before scaling."""


@dataclass
class ModelSpec:
    """Captures model selection and hyperparameter decisions."""

    model_name: str = 'RF'
    """Model name from configs.models: 'KNN', 'LR', 'CART', 'NB', 'SVC', 'RF', 'XGB', 'ADA', 'MLP'."""

    hyperparameters: Optional[Dict[str, Any]] = None
    """Explicit hyperparameters to set on the model. None means use defaults."""

    stacking: bool = False
    """Whether to include a stacking ensemble."""

    stacking_models: Optional[List[str]] = None
    """Models to include in the stacking ensemble. None means use all available."""

    ensemble: Optional[Any] = None
    """Advanced ensemble specification (EnsembleSpec from modelling.ensembles)."""


@dataclass
class CVSpec:
    """Captures cross-validation decisions."""

    policy_type: str = 'k_fold'
    """Type of CV: 'k_fold', 'stratified_k_fold', 'repeated_k_fold', 'repeated_stratified_k_fold'."""

    n_splits: int = 5
    """Number of CV splits."""

    n_repeats: int = 1
    """Number of CV repeats (for repeated strategies)."""

    shuffle: bool = True
    """Whether to shuffle before splitting."""

    random_state: int = 0
    """Random state for reproducibility."""


@dataclass
class TimeSeriesSpec:
    """Captures time-series specific configuration."""

    time_index: str = 'date'
    """Name of the datetime column."""

    freq: Optional[str] = None
    """Frequency of the time series (e.g., 'D', 'H', 'M')."""

    cv_strategy: str = 'expanding_window'
    """Temporal CV strategy: 'expanding_window', 'sliding_window', or 'blocked'."""

    gap: int = 0
    """Number of samples to skip between train and test (prevents leakage)."""

    lags: Optional[List[int]] = None
    """Lag periods for feature engineering."""

    rolling_windows: Optional[List[int]] = None
    """Rolling window sizes for feature engineering."""


@dataclass
class ExperimentConfig:
    """Complete specification of a pipeline experiment.

    Combines all pipeline decisions into one reproducible unit.
    This is the central object that the experiment framework works with.
    """

    name: str = 'experiment'
    """Human-readable name for this experiment configuration."""

    task_type: TaskType = TaskType.CLASSIFICATION
    """The ML task type."""

    target_label: str = 'target'
    """The target column name in the dataframe."""

    preprocessing: PreprocessingSpec = field(default_factory=PreprocessingSpec)
    """Preprocessing configuration."""

    features: FeatureSpec = field(default_factory=FeatureSpec)
    """Feature engineering configuration."""

    model: ModelSpec = field(default_factory=ModelSpec)
    """Model configuration."""

    cv: CVSpec = field(default_factory=CVSpec)
    """Cross-validation configuration."""

    evaluation_metric: str = 'accuracy'
    """Primary evaluation metric name."""

    evaluation_metrics: Optional[List[str]] = None
    """Additional evaluation metrics. None means use only the primary metric."""

    models_list: Optional[List[str]] = None
    """List of model names to evaluate. None means use only model.model_name."""

    n_jobs: int = -1
    """Number of parallel jobs for CV."""

    random_state: int = 0
    """Global random state."""

    time_series: Optional[Any] = None
    """Time-series specific configuration (Phase 8)."""

    @property
    def classification(self) -> bool:
        return self.task_type == TaskType.CLASSIFICATION

    @property
    def fingerprint(self) -> str:
        """Deterministic hash for deduplication.

        Two configs with the same decisions produce the same fingerprint.
        """
        canonical = self._to_canonical_dict()
        json_str = json.dumps(canonical, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def _to_canonical_dict(self) -> dict:
        """Convert to a canonical dict for fingerprinting (excludes name)."""
        d = self.to_dict()
        d.pop('name', None)
        return d

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        result = {
            'name': self.name,
            'task_type': self.task_type.value,
            'target_label': self.target_label,
            'preprocessing': {
                'imputation_strategy': self.preprocessing.imputation_strategy,
                'encoding_method': self.preprocessing.encoding_method,
                'outlier_strategy': self.preprocessing.outlier_strategy,
                'oversampler': self.preprocessing.oversampler,
            },
            'features': {
                'feature_selection_method': self.features.feature_selection_method,
                'scaler_name': self.features.scaler_name,
                'transformer_name': self.features.transformer_name,
                'use_transformers': self.features.use_transformers,
            },
            'model': {
                'model_name': self.model.model_name,
                'hyperparameters': self.model.hyperparameters,
                'stacking': self.model.stacking,
                'stacking_models': self.model.stacking_models,
            },
            'cv': {
                'policy_type': self.cv.policy_type,
                'n_splits': self.cv.n_splits,
                'n_repeats': self.cv.n_repeats,
                'shuffle': self.cv.shuffle,
                'random_state': self.cv.random_state,
            },
            'evaluation_metric': self.evaluation_metric,
            'evaluation_metrics': self.evaluation_metrics,
            'models_list': self.models_list,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
        }
        return result

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, d: dict) -> 'ExperimentConfig':
        """Deserialize from a plain dictionary."""
        return cls(
            name=d.get('name', 'experiment'),
            task_type=TaskType(d['task_type']),
            target_label=d.get('target_label', 'target'),
            preprocessing=PreprocessingSpec(**d.get('preprocessing', {})),
            features=FeatureSpec(**d.get('features', {})),
            model=ModelSpec(
                model_name=d.get('model', {}).get('model_name', 'RF'),
                hyperparameters=d.get('model', {}).get('hyperparameters'),
                stacking=d.get('model', {}).get('stacking', False),
                stacking_models=d.get('model', {}).get('stacking_models'),
            ),
            cv=CVSpec(**d.get('cv', {})),
            evaluation_metric=d.get('evaluation_metric', 'accuracy'),
            evaluation_metrics=d.get('evaluation_metrics'),
            models_list=d.get('models_list'),
            n_jobs=d.get('n_jobs', -1),
            random_state=d.get('random_state', 0),
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'ExperimentConfig':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def to_config_dict(self, df) -> dict:
        """Bridge method: convert to the config_dict format that execute_steps() expects.

        This enables backward compatibility with the existing pipeline system.
        """
        eval_metrics_objs = self._resolve_evaluation_metrics()

        config_dict = {
            'df': df,
            'base_encoded_df': df,
            'target_label': self.target_label,
            'run_id_number': hash(self.fingerprint) % 100000,
            'classification': self.classification,
            'evaluation_metric': self.evaluation_metric,
            'evaluation_metrics': eval_metrics_objs,
            'multiple_eval_scores': len(eval_metrics_objs) > 1,
            'k_fold_method': self.cv.policy_type,
            'n_folds': self.cv.n_splits,
            'n_repeats': self.cv.n_repeats,
            'random_state': self.cv.random_state,
            'n_jobs': self.n_jobs,
            'model_name': self.model.model_name,
            'models_list': self.models_list or constants.models_list_default,
            'scaled_df': self.features.scaler_name is not None,
            'stacking': self.model.stacking,
            'num_rows': df.shape[0],
            'scores_selected': {'cv'},
        }

        return config_dict

    def _resolve_evaluation_metrics(self) -> list:
        """Convert metric name strings to EvalMetrics enum instances."""
        if self.evaluation_metrics:
            return [EvalMetrics.from_str(m) for m in self.evaluation_metrics]
        return [EvalMetrics.from_str(self.evaluation_metric)]
