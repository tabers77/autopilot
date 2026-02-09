"""Budget controls: time caps, max trial limits, and early stopping.

Borrows from: FLAML (budget-aware search, efficient early stopping),
Optuna (trial pruning), H2O (max_runtime_secs).

FLAML's core insight: cheap exploration before expensive exploitation.
Budget-awareness as a first-class concern.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

from taberspilotml.experiment.runner import ExperimentResult


@dataclass
class BudgetConfig:
    """Configuration for budget controls."""

    max_time_seconds: Optional[float] = None
    """Total time budget in seconds. None means no limit."""

    max_trials: Optional[int] = None
    """Maximum number of trials. None means no limit."""

    max_time_per_trial_seconds: Optional[float] = None
    """Maximum time per individual trial in seconds. None means no limit."""

    early_stopping_patience: Optional[int] = None
    """Stop if no improvement after this many consecutive trials. None means no early stopping."""

    early_stopping_threshold: float = 0.001
    """Minimum improvement threshold to reset the patience counter."""


class BudgetTracker:
    """Tracks budget consumption and determines when to stop.

    Call `should_stop()` before each trial and `record_trial()` after.
    """

    def __init__(self, config: BudgetConfig):
        self.config = config
        self.start_time: float = time.time()
        self.n_trials: int = 0
        self.best_score: Optional[float] = None
        self.trials_without_improvement: int = 0
        self.classification: bool = True
        self._results: List[ExperimentResult] = []

    def set_classification(self, classification: bool):
        """Set whether higher scores are better (classification) or lower is better (regression)."""
        self.classification = classification

    @property
    def elapsed_time(self) -> float:
        """Total elapsed time in seconds since tracking started."""
        return time.time() - self.start_time

    @property
    def remaining_time(self) -> Optional[float]:
        """Remaining time in seconds. None if no time limit."""
        if self.config.max_time_seconds is None:
            return None
        return max(0, self.config.max_time_seconds - self.elapsed_time)

    def should_stop(self) -> bool:
        """Check if the budget is exhausted.

        Returns True if any budget limit has been reached.
        """
        # Check time limit
        if self.config.max_time_seconds is not None:
            if self.elapsed_time >= self.config.max_time_seconds:
                return True

        # Check trial limit
        if self.config.max_trials is not None:
            if self.n_trials >= self.config.max_trials:
                return True

        # Check early stopping
        if self.config.early_stopping_patience is not None:
            if self.trials_without_improvement >= self.config.early_stopping_patience:
                return True

        return False

    def record_trial(self, result: ExperimentResult):
        """Record the result of a trial and update budget tracking.

        :param result: The result of the completed trial.
        """
        self.n_trials += 1
        self._results.append(result)

        if result.status != 'completed':
            self.trials_without_improvement += 1
            return

        # Get primary score
        metric = result.config.evaluation_metric
        score = result.scores.get(metric)

        if score is None:
            self.trials_without_improvement += 1
            return

        # Check improvement
        if self.best_score is None:
            self.best_score = score
            self.trials_without_improvement = 0
        else:
            if self.classification:
                improvement = score - self.best_score
            else:
                improvement = self.best_score - score  # lower is better

            if improvement >= self.config.early_stopping_threshold:
                self.best_score = score
                self.trials_without_improvement = 0
            else:
                self.trials_without_improvement += 1

    def summary(self) -> dict:
        """Return a summary of budget consumption."""
        return {
            'total_trials': self.n_trials,
            'elapsed_time': round(self.elapsed_time, 2),
            'remaining_time': round(self.remaining_time, 2) if self.remaining_time is not None else None,
            'best_score': self.best_score,
            'trials_without_improvement': self.trials_without_improvement,
            'stopped_by': self._stopped_by(),
        }

    def _stopped_by(self) -> Optional[str]:
        """Return the reason for stopping, or None if not stopped."""
        if self.config.max_time_seconds is not None and self.elapsed_time >= self.config.max_time_seconds:
            return 'time_limit'
        if self.config.max_trials is not None and self.n_trials >= self.config.max_trials:
            return 'trial_limit'
        if (self.config.early_stopping_patience is not None
                and self.trials_without_improvement >= self.config.early_stopping_patience):
            return 'early_stopping'
        return None
