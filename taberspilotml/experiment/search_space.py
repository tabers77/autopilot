"""Search space definition and pipeline variant generation.

Borrows from: TPOT (pipeline structure search), auto-sklearn (CASH search space),
FLAML (cost-aware trial ordering).
"""

import itertools
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional

from taberspilotml.experiment.config import (
    ExperimentConfig,
    ModelSpec,
    FeatureSpec,
    PreprocessingSpec,
)


# Relative cost estimates per model (1=cheapest, 10=most expensive)
# Used for FLAML-inspired cost-aware ordering: cheap models first
MODEL_COST_ESTIMATES = {
    'NB': 1,
    'LR': 1,
    'KNN': 2,
    'CART': 2,
    'ADA': 4,
    'SVC': 5,
    'SVR': 5,
    'RF': 6,
    'MLP': 7,
    'XGB': 8,
}


@dataclass
class SearchSpace:
    """Defines the search space for pipeline variant generation.

    Lists of options per dimension. The cross-product of all dimensions
    defines the full combinatorial search space.
    """

    model_names: List[str] = field(default_factory=lambda: ['RF'])
    """Model names to try."""

    scaler_names: List[Optional[str]] = field(default_factory=lambda: [None])
    """Scaler names to try. None means no scaling."""

    transformer_names: List[Optional[str]] = field(default_factory=lambda: [None])
    """Transformer names to try. None means no transformer."""

    imputation_strategies: List[str] = field(default_factory=lambda: ['mean'])
    """Imputation strategies to try."""

    cv_policy_types: List[str] = field(default_factory=lambda: ['k_fold'])
    """CV policy types to try."""

    stacking_options: List[bool] = field(default_factory=lambda: [False])
    """Whether to try stacking."""

    @property
    def total_combinations(self) -> int:
        """Total number of pipeline variants in the full factorial."""
        return (len(self.model_names)
                * len(self.scaler_names)
                * len(self.transformer_names)
                * len(self.imputation_strategies)
                * len(self.cv_policy_types)
                * len(self.stacking_options))

    def enumerate_configs(self, base_config: ExperimentConfig) -> List[ExperimentConfig]:
        """Full factorial enumeration via itertools.product.

        :param base_config: Base config to use for all non-searched dimensions.
        :returns: List of all possible ExperimentConfig variants.
        """
        configs = []
        combinations = itertools.product(
            self.model_names,
            self.scaler_names,
            self.transformer_names,
            self.imputation_strategies,
            self.cv_policy_types,
            self.stacking_options,
        )

        for i, (model, scaler, transformer, imputation, cv_policy, stacking) in enumerate(combinations):
            config = deepcopy(base_config)
            config.name = f'{base_config.name}_variant_{i}'
            config.model = ModelSpec(
                model_name=model,
                hyperparameters=base_config.model.hyperparameters,
                stacking=stacking,
                stacking_models=base_config.model.stacking_models,
            )
            config.features = FeatureSpec(
                feature_selection_method=base_config.features.feature_selection_method,
                scaler_name=scaler,
                transformer_name=transformer,
                use_transformers=transformer is not None,
            )
            config.preprocessing = PreprocessingSpec(
                imputation_strategy=imputation,
                encoding_method=base_config.preprocessing.encoding_method,
                outlier_strategy=base_config.preprocessing.outlier_strategy,
                oversampler=base_config.preprocessing.oversampler,
            )
            config.cv.policy_type = cv_policy
            configs.append(config)

        return configs

    def sample_configs(self, base_config: ExperimentConfig, n: int,
                       seed: Optional[int] = None) -> List[ExperimentConfig]:
        """Random sampling from the search space.

        :param base_config: Base config for non-searched dimensions.
        :param n: Number of configs to sample.
        :param seed: Random seed for reproducibility.
        :returns: List of sampled ExperimentConfig variants.
        """
        rng = random.Random(seed)
        all_configs = self.enumerate_configs(base_config)
        n = min(n, len(all_configs))
        return rng.sample(all_configs, n)

    def cost_ordered_configs(self, base_config: ExperimentConfig) -> List[ExperimentConfig]:
        """FLAML-inspired cost-aware ordering: cheap models first.

        Generates all configs but orders them so cheap-to-evaluate models
        come first, allowing early stopping to save expensive evaluations.

        :param base_config: Base config for non-searched dimensions.
        :returns: Configs sorted by estimated cost (cheapest first).
        """
        configs = self.enumerate_configs(base_config)

        def cost_key(config):
            model_cost = MODEL_COST_ESTIMATES.get(config.model.model_name, 5)
            stacking_cost = 3 if config.model.stacking else 0
            return model_cost + stacking_cost

        configs.sort(key=cost_key)
        return configs
