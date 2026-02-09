"""Pipeline search orchestrator: generates variants from a search space and evaluates them.

Borrows from: TPOT (pipeline structure search), FLAML (cost-aware trial ordering).
"""

from typing import List, Optional

import pandas as pd

from taberspilotml.experiment.config import ExperimentConfig
from taberspilotml.experiment.runner import ExperimentResult, ExperimentRunner, run_single
from taberspilotml.experiment.search_space import SearchSpace


class PipelineSearch:
    """Orchestrates pipeline variant generation + evaluation.

    :param df: The dataframe to run experiments on.
    :param search_space: The SearchSpace defining which dimensions to search.
    :param base_config: Base config for non-searched dimensions.
    :param strategy: Search strategy: 'factorial', 'random', or 'cost_aware'.
    :param n_samples: Number of samples for 'random' strategy.
    :param seed: Random seed for 'random' strategy.
    """

    def __init__(self, df: pd.DataFrame, search_space: SearchSpace,
                 base_config: ExperimentConfig,
                 strategy: str = 'factorial',
                 n_samples: int = 10,
                 seed: Optional[int] = None):
        self.df = df
        self.search_space = search_space
        self.base_config = base_config
        self.strategy = strategy
        self.n_samples = n_samples
        self.seed = seed
        self.results: List[ExperimentResult] = []
        self.configs: List[ExperimentConfig] = []

    def _generate_configs(self) -> List[ExperimentConfig]:
        """Generate configs based on the selected strategy."""
        if self.strategy == 'factorial':
            return self.search_space.enumerate_configs(self.base_config)
        elif self.strategy == 'random':
            return self.search_space.sample_configs(self.base_config, self.n_samples, self.seed)
        elif self.strategy == 'cost_aware':
            return self.search_space.cost_ordered_configs(self.base_config)
        else:
            raise ValueError(f'Unknown strategy: {self.strategy}')

    def run(self, budget=None, registry=None) -> List[ExperimentResult]:
        """Run the pipeline search.

        :param budget: Optional BudgetTracker for budget controls (Phase 4).
        :param registry: Optional ExperimentRegistry for auto-registration (Phase 5).
        :returns: List of ExperimentResult instances.
        """
        self.configs = self._generate_configs()
        self.results = []

        for config in self.configs:
            if budget is not None and budget.should_stop():
                print(f'Budget exhausted after {len(self.results)} trials.')
                break

            timeout = None
            if budget is not None:
                timeout = budget.config.max_time_per_trial_seconds

            result = run_single(config, self.df, timeout=timeout)
            self.results.append(result)

            if budget is not None:
                budget.record_trial(result)

            if registry is not None:
                registry.register(result)

        return self.results

    def leaderboard(self, metric: Optional[str] = None) -> pd.DataFrame:
        """Get ranked leaderboard from search results."""
        runner = ExperimentRunner(self.df, [r.config for r in self.results])
        runner.results = self.results
        return runner.leaderboard(metric)

    @property
    def best_result(self) -> Optional[ExperimentResult]:
        """Return the best result based on primary metric."""
        completed = [r for r in self.results if r.status == 'completed']
        if not completed:
            return None

        classification = self.base_config.classification
        metric = self.base_config.evaluation_metric

        return max(completed, key=lambda r: r.scores.get(metric, float('-inf'))
                   ) if classification else min(
            completed, key=lambda r: r.scores.get(metric, float('inf')))
