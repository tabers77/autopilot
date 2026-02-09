"""Comparison and ablation analysis utilities for pipeline-level experimentation.

Provides tools to understand which pipeline decisions matter most.
"""

from typing import Dict, List, Optional

import pandas as pd

from taberspilotml.experiment.runner import ExperimentResult


def pipeline_comparison_table(results: List[ExperimentResult]) -> pd.DataFrame:
    """Flatten all decisions + scores into one comparison table.

    Each row is one experiment, columns are decisions and metric scores.
    """
    rows = []
    for r in results:
        if r.status != 'completed':
            continue
        row = {
            'name': r.config.name,
            'fingerprint': r.fingerprint,
            'model': r.config.model.model_name,
            'scaler': r.config.features.scaler_name or 'None',
            'transformer': r.config.features.transformer_name or 'None',
            'imputation': r.config.preprocessing.imputation_strategy,
            'outlier_strategy': r.config.preprocessing.outlier_strategy or 'None',
            'oversampler': r.config.preprocessing.oversampler or 'None',
            'feature_selection': r.config.features.feature_selection_method or 'None',
            'cv_policy': r.config.cv.policy_type,
            'n_splits': r.config.cv.n_splits,
            'stacking': r.config.model.stacking,
        }
        for metric_name, score in r.scores.items():
            row[metric_name] = score
        for metric_name, std in r.score_stds.items():
            row[f'{metric_name}_std'] = std
        row['execution_time'] = r.execution_time
        rows.append(row)

    return pd.DataFrame(rows)


def ablation_analysis(results: List[ExperimentResult],
                      baseline_fingerprint: str,
                      metric: Optional[str] = None) -> pd.DataFrame:
    """Delta from changing each single decision relative to a baseline.

    Identifies which single decision changes have the biggest impact.

    :param results: List of experiment results.
    :param baseline_fingerprint: Fingerprint of the baseline experiment.
    :param metric: Metric to compute deltas on. Defaults to primary metric.
    """
    # Find baseline
    baseline = None
    for r in results:
        if r.fingerprint == baseline_fingerprint:
            baseline = r
            break

    if baseline is None:
        raise ValueError(f'Baseline with fingerprint {baseline_fingerprint} not found')

    if metric is None:
        metric = baseline.config.evaluation_metric

    baseline_score = baseline.scores.get(metric, 0.0)

    # Compare each result to baseline
    decision_fields = [
        ('model', lambda r: r.config.model.model_name),
        ('scaler', lambda r: r.config.features.scaler_name or 'None'),
        ('transformer', lambda r: r.config.features.transformer_name or 'None'),
        ('imputation', lambda r: r.config.preprocessing.imputation_strategy),
        ('cv_policy', lambda r: r.config.cv.policy_type),
        ('stacking', lambda r: str(r.config.model.stacking)),
    ]

    baseline_decisions = {name: getter(baseline) for name, getter in decision_fields}

    rows = []
    for r in results:
        if r.status != 'completed' or r.fingerprint == baseline_fingerprint:
            continue

        current_decisions = {name: getter(r) for name, getter in decision_fields}
        current_score = r.scores.get(metric, 0.0)
        delta = current_score - baseline_score

        # Find which decisions differ
        changed = {name: current_decisions[name]
                   for name in baseline_decisions
                   if current_decisions[name] != baseline_decisions[name]}

        rows.append({
            'name': r.config.name,
            'fingerprint': r.fingerprint,
            metric: current_score,
            'delta': delta,
            'changed_decisions': changed,
            'n_changes': len(changed),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('delta', ascending=False).reset_index(drop=True)
    return df


def decision_impact_matrix(results: List[ExperimentResult],
                           metric: Optional[str] = None) -> pd.DataFrame:
    """Cross-tabulate decisions vs. metrics.

    Shows e.g. does PCA help more with RF or XGB?

    :param results: List of completed experiment results.
    :param metric: Metric to analyze. Defaults to primary metric.
    """
    completed = [r for r in results if r.status == 'completed']
    if not completed:
        return pd.DataFrame()

    if metric is None:
        metric = completed[0].config.evaluation_metric

    # Build a table of decision -> {value -> [scores]}
    decision_getters = {
        'model': lambda r: r.config.model.model_name,
        'scaler': lambda r: r.config.features.scaler_name or 'None',
        'transformer': lambda r: r.config.features.transformer_name or 'None',
        'imputation': lambda r: r.config.preprocessing.imputation_strategy,
    }

    rows = []
    for decision_name, getter in decision_getters.items():
        value_scores: Dict[str, list] = {}
        for r in completed:
            value = getter(r)
            score = r.scores.get(metric, 0.0)
            value_scores.setdefault(value, []).append(score)

        for value, scores in value_scores.items():
            import numpy as np
            rows.append({
                'decision': decision_name,
                'value': value,
                'mean_score': np.mean(scores),
                'std_score': np.std(scores) if len(scores) > 1 else 0.0,
                'n_experiments': len(scores),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['decision', 'mean_score'], ascending=[True, False]).reset_index(drop=True)
    return df
