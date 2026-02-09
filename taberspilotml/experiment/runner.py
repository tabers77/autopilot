"""Pipeline-level comparison engine: run multiple ExperimentConfig instances and collect results.

Borrows from: EvalML (pipeline-level evaluation), PyCaret (compare_models() leaderboard).
"""

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from taberspilotml.experiment.config import ExperimentConfig


@dataclass
class ExperimentResult:
    """Result of running a single ExperimentConfig."""

    config: ExperimentConfig
    """The config that produced this result."""

    scores: Dict[str, float] = field(default_factory=dict)
    """Mapping of metric name -> score value."""

    score_stds: Dict[str, float] = field(default_factory=dict)
    """Mapping of metric name -> standard deviation."""

    artifacts: Dict[str, Any] = field(default_factory=dict)
    """Any artifacts produced (models, dataframes, etc.)."""

    execution_time: float = 0.0
    """Time in seconds to run this experiment."""

    status: str = 'pending'
    """Status: 'pending', 'running', 'completed', 'failed'."""

    error_message: Optional[str] = None
    """Error message if status is 'failed'."""

    @property
    def primary_score(self) -> Optional[float]:
        """Return the primary metric score."""
        metric = self.config.evaluation_metric
        return self.scores.get(metric)

    @property
    def primary_std(self) -> Optional[float]:
        """Return the primary metric standard deviation."""
        metric = self.config.evaluation_metric
        return self.score_stds.get(metric)

    @property
    def fingerprint(self) -> str:
        return self.config.fingerprint

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            'config_name': self.config.name,
            'fingerprint': self.fingerprint,
            'model': self.config.model.model_name,
            'scaler': self.config.features.scaler_name,
            'transformer': self.config.features.transformer_name,
            'imputation': self.config.preprocessing.imputation_strategy,
            'cv_policy': self.config.cv.policy_type,
            'n_splits': self.config.cv.n_splits,
            'stacking': self.config.model.stacking,
            'scores': self.scores,
            'score_stds': self.score_stds,
            'primary_score': self.primary_score,
            'primary_std': self.primary_std,
            'execution_time': self.execution_time,
            'status': self.status,
            'error_message': self.error_message,
        }


class ExperimentRunner:
    """Runs multiple ExperimentConfig instances and collects results.

    This is the core engine for pipeline-level comparison.
    """

    def __init__(self, df: pd.DataFrame, configs: List[ExperimentConfig]):
        self.df = df
        self.configs = configs
        self.results: List[ExperimentResult] = []

    def run_all(self, registry=None) -> List[ExperimentResult]:
        """Run all configs and collect results.

        :param registry: Optional ExperimentRegistry to auto-register results.
        """
        self.results = []
        for config in self.configs:
            result = run_single(config, self.df)
            self.results.append(result)
            if registry is not None:
                registry.register(result)
        return self.results

    def leaderboard(self, metric: Optional[str] = None) -> pd.DataFrame:
        """PyCaret-style ranked DataFrame of results.

        :param metric: Metric to sort by. Defaults to primary evaluation metric.
        """
        if not self.results:
            return pd.DataFrame()

        rows = []
        for r in self.results:
            if r.status != 'completed':
                continue
            row = {
                'Name': r.config.name,
                'Model': r.config.model.model_name,
                'Scaler': r.config.features.scaler_name or 'None',
                'Transformer': r.config.features.transformer_name or 'None',
                'Imputation': r.config.preprocessing.imputation_strategy,
                'Stacking': r.config.model.stacking,
                'CV Policy': r.config.cv.policy_type,
                'Splits': r.config.cv.n_splits,
            }
            # Add all scores
            for metric_name, score in r.scores.items():
                row[metric_name] = score
            for metric_name, std in r.score_stds.items():
                row[f'{metric_name}_std'] = std
            row['Time (s)'] = round(r.execution_time, 2)
            row['Fingerprint'] = r.fingerprint
            rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        sort_metric = metric or self.configs[0].evaluation_metric
        if sort_metric in df.columns:
            ascending = not self.configs[0].classification
            df = df.sort_values(sort_metric, ascending=ascending).reset_index(drop=True)

        return df

    def compare(self) -> pd.DataFrame:
        """Side-by-side comparison table of all completed results."""
        return self.leaderboard()


def run_single(config: ExperimentConfig, df: pd.DataFrame, timeout: Optional[float] = None) -> ExperimentResult:
    """Run a single ExperimentConfig and return an ExperimentResult.

    This is the main entry point that converts:
    ExperimentConfig -> config_dict -> execute pipeline -> ExperimentResult

    :param config: The experiment configuration.
    :param df: The dataframe to run the experiment on.
    :param timeout: Optional timeout in seconds. Uses threading.Timer for cross-platform support.
    """
    from taberspilotml.experiment.pipeline_spec import PipelineSpec
    from taberspilotml.scoring_funcs.cross_validation import SplitPolicy
    from taberspilotml.scoring_funcs import datasets as d, scorers
    from taberspilotml.scoring_funcs.evaluation_metrics import EvalMetrics
    from taberspilotml import constants
    import taberspilotml.conf.configs as model_configs

    result = ExperimentResult(config=config, status='running')
    start_time = time.time()

    try:
        # Resolve evaluation metrics
        if config.evaluation_metrics:
            eval_metrics = [EvalMetrics.from_str(m) for m in config.evaluation_metrics]
        else:
            eval_metrics = [EvalMetrics.from_str(config.evaluation_metric)]

        # Build split policy from CVSpec
        policy = SplitPolicy.from_cv_spec(config.cv)

        # Get the model
        model_key = 'clf' if config.classification else 'reg'
        model = model_configs.models[model_key][config.model.model_name]

        # Apply hyperparameters if specified
        if config.model.hyperparameters:
            model.set_params(**config.model.hyperparameters)

        # Build dataset with optional scaling
        scaler = None
        if config.features.scaler_name:
            scaler = model_configs.scalers[config.features.scaler_name]

        ds = d.Dataset.from_dataframe(df, [config.target_label], scaler=scaler)

        # Run cross-validation
        cv_results = scorers.get_cross_validation_score(
            dataset=ds,
            model=model,
            split_policy=policy,
            evaluation_metrics=eval_metrics,
            n_jobs=config.n_jobs,
            verbose=0,
        )

        # Extract scores
        for metric_name, (mean, std) in cv_results.items():
            result.scores[metric_name] = mean
            result.score_stds[metric_name] = std

        result.artifacts['model'] = model
        result.status = 'completed'

    except Exception as e:
        result.status = 'failed'
        result.error_message = f'{type(e).__name__}: {e}\n{traceback.format_exc()}'

    result.execution_time = time.time() - start_time
    return result
