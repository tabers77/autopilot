"""Maps ExperimentConfig choices to actual functions from existing modules, producing executable pipeline steps.

PipelineStep and PipelineSpec bridge the gap between *specification* (ExperimentConfig)
and *execution* (the OrderedDict format that execute_steps() expects).
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from taberspilotml.experiment.config import ExperimentConfig


@dataclass
class PipelineStep:
    """A single step in the pipeline."""

    name: str
    """Step name (used as key in the step dict)."""

    function: Callable
    """The function to call during this step."""

    handler_type: str = 'initial_checkpoint'
    """Handler type: 'initial_checkpoint', 'scoring', 'support', 'mixed', 'hyper_p'."""

    enabled: bool = True
    """Whether this step is enabled."""


@dataclass
class PipelineSpec:
    """Complete pipeline specification that can be converted to the step dict format."""

    steps: List[PipelineStep] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> 'PipelineSpec':
        """Build a PipelineSpec from an ExperimentConfig.

        Maps config choices to actual functions from existing modules.
        """
        import taberspilotml.preprocessing.generals as dv
        import taberspilotml.pre_modelling.handle_nulls as handle_nulls
        import taberspilotml.pre_modelling.encoders as enc
        import taberspilotml.base_helpers as bh
        import taberspilotml.pre_modelling.outliers as outliers
        import taberspilotml.pre_modelling.imbalance as imbalance
        import taberspilotml.modelling.ml_models as ml_models
        import taberspilotml.pre_modelling.feature_importance as fi
        import taberspilotml.hyper_opti as hyper_p

        steps = []

        # Default preprocessing steps (always enabled)
        steps.append(PipelineStep(
            name='dataframe_transformation',
            function=dv.dataframe_transformation,
            handler_type='initial_checkpoint',
        ))

        steps.append(PipelineStep(
            name='handle_missing_values',
            function=handle_nulls.eval_imputation_method_wrapper,
            handler_type='initial_checkpoint',
        ))

        steps.append(PipelineStep(
            name='encoding',
            function=enc.default_encoding,
            handler_type='initial_checkpoint',
        ))

        steps.append(PipelineStep(
            name='baseline_score',
            function=bh.get_baseline_score,
            handler_type='support',
        ))

        # Outlier handling
        if config.preprocessing.outlier_strategy is not None:
            steps.append(PipelineStep(
                name='handle_outliers',
                function=outliers.handle_outliers,
                handler_type='mixed',
            ))

        # Oversampling
        if config.preprocessing.oversampler is not None:
            steps.append(PipelineStep(
                name='evaluate_oversamplers',
                function=imbalance.evaluate_oversamplers,
                handler_type='mixed',
            ))

        # Model evaluation
        steps.append(PipelineStep(
            name='evaluate_models',
            function=ml_models.evaluate_models_wrapper,
            handler_type='scoring',
        ))

        # Feature selection
        if config.features.feature_selection_method is not None:
            steps.append(PipelineStep(
                name='feature_selection',
                function=fi.get_reduced_features_cv_scores,
                handler_type='mixed',
            ))

        # Transformation methods (scaling/transforming)
        if config.features.scaler_name is not None:
            steps.append(PipelineStep(
                name='transformation_methods',
                function=ml_models.eval_model_scaler_wrapper,
                handler_type='mixed',
            ))

        return cls(steps=steps)

    def to_step_dict(self) -> OrderedDict:
        """Produce the OrderedDict format that execute_steps() expects.

        Each entry is: step_name -> (function, handler)
        """
        from taberspilotml.auto_mode import (
            initial_checkpoint_handler,
            scoring_handler,
            support_handler,
            mixed_handler,
            hyper_p_handler,
        )

        handler_map = {
            'initial_checkpoint': initial_checkpoint_handler,
            'scoring': scoring_handler,
            'support': support_handler,
            'mixed': mixed_handler,
            'hyper_p': hyper_p_handler,
        }

        result = OrderedDict()
        for step in self.steps:
            if step.enabled:
                handler = handler_map[step.handler_type]
                result[step.name] = (step.function, handler)

        return result
