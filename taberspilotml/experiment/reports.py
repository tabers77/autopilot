"""Automated experiment reports: Markdown experiment cards and comparison reports.

Borrows from: LightAutoML (automated reports), PyCaret (visualization + comparison tables).
Turns raw metrics into narrative insights -- the difference between data and information.
"""

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from taberspilotml.experiment.runner import ExperimentResult
from taberspilotml.experiment.comparison import (
    pipeline_comparison_table,
    ablation_analysis,
    decision_impact_matrix,
)


class ExperimentCard:
    """Markdown card for a single experiment result.

    Includes: config summary, results table, key decisions, reproducibility info.
    """

    def __init__(self, result: ExperimentResult):
        self.result = result

    def to_markdown(self) -> str:
        r = self.result
        c = r.config
        lines = []

        lines.append(f'# Experiment: {c.name}')
        lines.append('')
        lines.append(f'**Fingerprint**: `{r.fingerprint}`')
        lines.append(f'**Status**: {r.status}')
        lines.append(f'**Execution Time**: {r.execution_time:.2f}s')
        lines.append('')

        # Config summary
        lines.append('## Configuration')
        lines.append('')
        lines.append(f'| Decision | Value |')
        lines.append(f'|----------|-------|')
        lines.append(f'| Task Type | {c.task_type.value} |')
        lines.append(f'| Target | {c.target_label} |')
        lines.append(f'| Model | {c.model.model_name} |')
        lines.append(f'| Scaler | {c.features.scaler_name or "None"} |')
        lines.append(f'| Transformer | {c.features.transformer_name or "None"} |')
        lines.append(f'| Imputation | {c.preprocessing.imputation_strategy} |')
        lines.append(f'| CV Policy | {c.cv.policy_type} |')
        lines.append(f'| Splits | {c.cv.n_splits} |')
        lines.append(f'| Stacking | {c.model.stacking} |')
        lines.append(f'| Evaluation Metric | {c.evaluation_metric} |')
        lines.append('')

        # Results
        if r.scores:
            lines.append('## Results')
            lines.append('')
            lines.append('| Metric | Score | Std |')
            lines.append('|--------|-------|-----|')
            for metric in r.scores:
                score = r.scores[metric]
                std = r.score_stds.get(metric, float('nan'))
                lines.append(f'| {metric} | {score:.4f} | {std:.4f} |')
            lines.append('')

        # Error info
        if r.error_message:
            lines.append('## Error')
            lines.append('')
            lines.append(f'```\n{r.error_message}\n```')
            lines.append('')

        # Reproducibility
        lines.append('## Reproducibility')
        lines.append('')
        lines.append('```python')
        lines.append('from taberspilotml.experiment import ExperimentConfig, run_single')
        lines.append(f'config = ExperimentConfig.from_json(\'{c.to_json()}\')')
        lines.append('result = run_single(config, df)')
        lines.append('```')

        return '\n'.join(lines)


class ComparisonReport:
    """Comparison report across multiple experiment results.

    Includes: leaderboard, decision impact analysis, stability analysis, key insights.
    """

    def __init__(self, results: List[ExperimentResult]):
        self.results = [r for r in results if r.status == 'completed']

    def to_markdown(self) -> str:
        lines = []

        lines.append('# Pipeline Comparison Report')
        lines.append('')
        lines.append(f'**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        lines.append(f'**Total Experiments**: {len(self.results)}')
        lines.append('')

        if not self.results:
            lines.append('*No completed experiments to compare.*')
            return '\n'.join(lines)

        metric = self.results[0].config.evaluation_metric
        classification = self.results[0].config.classification

        # Leaderboard
        lines.append('## Leaderboard')
        lines.append('')
        lines.append(self._leaderboard_table(metric, classification))
        lines.append('')

        # Decision Impact
        lines.append('## Decision Impact Analysis')
        lines.append('')
        impact = decision_impact_matrix(self.results, metric)
        if not impact.empty:
            lines.append(self._impact_table(impact))
        else:
            lines.append('*Not enough data for impact analysis.*')
        lines.append('')

        # Stability Analysis
        lines.append('## Stability Analysis')
        lines.append('')
        lines.append(self._stability_analysis(metric))
        lines.append('')

        # Key Insights
        lines.append('## Key Insights')
        lines.append('')
        lines.append(self._key_insights(metric, classification))

        return '\n'.join(lines)

    def _leaderboard_table(self, metric: str, classification: bool) -> str:
        sorted_results = sorted(
            self.results,
            key=lambda r: r.scores.get(metric, float('-inf') if classification else float('inf')),
            reverse=classification,
        )

        lines = ['| Rank | Name | Model | Score | Std | Time (s) |',
                 '|------|------|-------|-------|-----|----------|']
        for i, r in enumerate(sorted_results, 1):
            score = r.scores.get(metric, float('nan'))
            std = r.score_stds.get(metric, float('nan'))
            lines.append(f'| {i} | {r.config.name} | {r.config.model.model_name} '
                         f'| {score:.4f} | {std:.4f} | {r.execution_time:.2f} |')
        return '\n'.join(lines)

    def _impact_table(self, impact: pd.DataFrame) -> str:
        lines = ['| Decision | Value | Mean Score | N Experiments |',
                 '|----------|-------|------------|---------------|']
        for _, row in impact.iterrows():
            lines.append(f'| {row["decision"]} | {row["value"]} '
                         f'| {row["mean_score"]:.4f} | {row["n_experiments"]} |')
        return '\n'.join(lines)

    def _stability_analysis(self, metric: str) -> str:
        stds = [r.score_stds.get(metric, 0.0) for r in self.results if metric in r.score_stds]
        if not stds:
            return '*No standard deviation data available.*'

        lines = []
        avg_std = np.mean(stds)
        max_std = np.max(stds)
        min_std = np.min(stds)
        lines.append(f'- Average CV std: {avg_std:.4f}')
        lines.append(f'- Min CV std: {min_std:.4f}')
        lines.append(f'- Max CV std: {max_std:.4f}')

        if avg_std > 0.05:
            lines.append('- **Warning**: High variance suggests unstable results. Consider more CV folds.')
        elif avg_std < 0.01:
            lines.append('- Results appear highly stable across CV folds.')

        return '\n'.join(lines)

    def _key_insights(self, metric: str, classification: bool) -> str:
        if not self.results:
            return '*No insights available.*'

        sorted_results = sorted(
            self.results,
            key=lambda r: r.scores.get(metric, float('-inf') if classification else float('inf')),
            reverse=classification,
        )

        best = sorted_results[0]
        worst = sorted_results[-1]
        best_score = best.scores.get(metric, 0)
        worst_score = worst.scores.get(metric, 0)

        lines = []
        lines.append(f'- **Best pipeline**: {best.config.name} ({best.config.model.model_name}) '
                      f'with {metric}={best_score:.4f}')
        lines.append(f'- **Worst pipeline**: {worst.config.name} ({worst.config.model.model_name}) '
                      f'with {metric}={worst_score:.4f}')
        lines.append(f'- **Score range**: {abs(best_score - worst_score):.4f}')

        # Fastest vs slowest
        fastest = min(self.results, key=lambda r: r.execution_time)
        slowest = max(self.results, key=lambda r: r.execution_time)
        lines.append(f'- **Fastest**: {fastest.config.name} ({fastest.execution_time:.2f}s)')
        lines.append(f'- **Slowest**: {slowest.config.name} ({slowest.execution_time:.2f}s)')

        return '\n'.join(lines)


class SearchReport:
    """Report for pipeline search results.

    Includes: search summary, budget used, configs explored, winner.
    """

    def __init__(self, results: List[ExperimentResult], budget_summary: Optional[Dict] = None):
        self.results = results
        self.budget_summary = budget_summary

    def to_markdown(self) -> str:
        lines = []

        lines.append('# Pipeline Search Report')
        lines.append('')
        lines.append(f'**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        lines.append('')

        # Search summary
        completed = [r for r in self.results if r.status == 'completed']
        failed = [r for r in self.results if r.status == 'failed']

        lines.append('## Search Summary')
        lines.append('')
        lines.append(f'- **Total configs evaluated**: {len(self.results)}')
        lines.append(f'- **Completed**: {len(completed)}')
        lines.append(f'- **Failed**: {len(failed)}')

        if completed:
            total_time = sum(r.execution_time for r in self.results)
            lines.append(f'- **Total execution time**: {total_time:.2f}s')
        lines.append('')

        # Budget summary
        if self.budget_summary:
            lines.append('## Budget')
            lines.append('')
            for k, v in self.budget_summary.items():
                lines.append(f'- **{k}**: {v}')
            lines.append('')

        # Winner
        if completed:
            metric = completed[0].config.evaluation_metric
            classification = completed[0].config.classification
            best = max(completed, key=lambda r: r.scores.get(metric, float('-inf'))
                       ) if classification else min(
                completed, key=lambda r: r.scores.get(metric, float('inf')))

            lines.append('## Winner')
            lines.append('')
            lines.append(f'- **Name**: {best.config.name}')
            lines.append(f'- **Model**: {best.config.model.model_name}')
            lines.append(f'- **{metric}**: {best.scores.get(metric, 0):.4f}')
            lines.append(f'- **Time**: {best.execution_time:.2f}s')
            lines.append('')

        # Comparison table
        if len(completed) > 1:
            comparison = ComparisonReport(completed)
            lines.append(comparison.to_markdown())

        return '\n'.join(lines)
