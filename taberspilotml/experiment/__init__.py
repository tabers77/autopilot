"""Experiment framework for pipeline-level comparison and search."""

from taberspilotml.experiment.config import (
    TaskType,
    PreprocessingSpec,
    FeatureSpec,
    ModelSpec,
    CVSpec,
    TimeSeriesSpec,
    ExperimentConfig,
)
from taberspilotml.experiment.pipeline_spec import PipelineStep, PipelineSpec
from taberspilotml.experiment.runner import ExperimentResult, ExperimentRunner, run_single
from taberspilotml.experiment.comparison import (
    pipeline_comparison_table,
    ablation_analysis,
    decision_impact_matrix,
)
from taberspilotml.experiment.search_space import SearchSpace, MODEL_COST_ESTIMATES
from taberspilotml.experiment.search import PipelineSearch
from taberspilotml.experiment.budget import BudgetConfig, BudgetTracker
from taberspilotml.experiment.registry import ExperimentRegistry
from taberspilotml.experiment.reports import ExperimentCard, ComparisonReport, SearchReport
