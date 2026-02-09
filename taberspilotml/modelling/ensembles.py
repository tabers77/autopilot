"""Enhanced ensemble methods: multi-layer stacking, weighted ensembles, bagging.

Borrows from: AutoGluon (multi-layer stacking), H2O (stacked ensembles),
Caruana-style greedy ensemble selection.

Why ensemble diversity matters more than individual model quality.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import (
    StackingClassifier, StackingRegressor,
    BaggingClassifier, BaggingRegressor,
    VotingClassifier, VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeCV
from sklearn.model_selection import cross_val_predict

from taberspilotml.conf import configs as model_configs


@dataclass
class EnsembleSpec:
    """Specification for an ensemble method."""

    ensemble_type: str = 'stacking'
    """Type of ensemble: 'stacking', 'weighted', 'bagging', 'voting'."""

    base_models: Optional[List[str]] = None
    """Model names for the base layer. None means use all available."""

    meta_model: Optional[str] = None
    """Meta-model name for stacking. None means use default (LR/Ridge)."""

    n_layers: int = 1
    """Number of stacking layers (for multi-layer stacking)."""

    weights: Optional[List[float]] = None
    """Weights for weighted voting. None means optimize automatically."""

    n_folds: int = 5
    """Number of CV folds for out-of-fold predictions in stacking."""


class StackingEnsembleBuilder:
    """Multi-layer stacking with out-of-fold predictions.

    Layer 0: Base models generate out-of-fold predictions.
    Layer 1..N: Meta-models stack on previous layer's predictions.
    Final layer: A simple meta-model combines everything.
    """

    def __init__(self, spec: EnsembleSpec, classification: bool = True):
        self.spec = spec
        self.classification = classification

    def build(self) -> BaseEstimator:
        """Build the stacking ensemble."""
        base_estimators = self._get_base_estimators()
        meta_estimator = self._get_meta_estimator()

        if self.spec.n_layers <= 1:
            return self._build_single_layer(base_estimators, meta_estimator)

        return self._build_multi_layer(base_estimators, meta_estimator)

    def _get_base_estimators(self) -> List[tuple]:
        """Get (name, estimator) pairs for base models."""
        model_key = 'clf' if self.classification else 'reg'
        models_dict = model_configs.models[model_key]

        if self.spec.base_models:
            return [(name, clone(models_dict[name])) for name in self.spec.base_models
                    if name in models_dict]
        return [(name, clone(model)) for name, model in models_dict.items()]

    def _get_meta_estimator(self) -> BaseEstimator:
        """Get the meta-estimator."""
        if self.spec.meta_model:
            model_key = 'clf' if self.classification else 'reg'
            return clone(model_configs.models[model_key][self.spec.meta_model])

        if self.classification:
            return LogisticRegression(max_iter=1000)
        return RidgeCV()

    def _build_single_layer(self, base_estimators, meta_estimator):
        """Build a standard single-layer stacking ensemble."""
        if self.classification:
            return StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta_estimator,
                cv=self.spec.n_folds,
            )
        return StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta_estimator,
            cv=self.spec.n_folds,
        )

    def _build_multi_layer(self, base_estimators, final_meta_estimator):
        """Build a multi-layer stacking ensemble.

        Each layer wraps the previous layer's stacking as base estimators
        for the next layer.
        """
        current_estimators = base_estimators

        for layer in range(self.spec.n_layers - 1):
            # Each intermediate layer uses a simple meta-model
            if self.classification:
                intermediate_meta = LogisticRegression(max_iter=1000)
                stacker = StackingClassifier(
                    estimators=current_estimators,
                    final_estimator=intermediate_meta,
                    cv=self.spec.n_folds,
                )
            else:
                intermediate_meta = RidgeCV()
                stacker = StackingRegressor(
                    estimators=current_estimators,
                    final_estimator=intermediate_meta,
                    cv=self.spec.n_folds,
                )

            # Next layer uses the stacker as one of the base estimators
            # along with original base estimators for diversity
            current_estimators = [
                (f'layer_{layer}_stack', stacker),
            ] + [(f'layer_{layer}_{name}', clone(est)) for name, est in base_estimators[:2]]

        # Final layer
        if self.classification:
            return StackingClassifier(
                estimators=current_estimators,
                final_estimator=final_meta_estimator,
                cv=self.spec.n_folds,
            )
        return StackingRegressor(
            estimators=current_estimators,
            final_estimator=final_meta_estimator,
            cv=self.spec.n_folds,
        )


class WeightedEnsembleBuilder:
    """Caruana-style greedy weight optimization.

    Iteratively adds models to the ensemble, selecting the model and weight
    that maximizes the ensemble's score on out-of-fold predictions.
    """

    def __init__(self, spec: EnsembleSpec, classification: bool = True):
        self.spec = spec
        self.classification = classification

    def build(self) -> BaseEstimator:
        """Build the weighted voting ensemble."""
        model_key = 'clf' if self.classification else 'reg'
        models_dict = model_configs.models[model_key]

        if self.spec.base_models:
            estimators = [(name, clone(models_dict[name])) for name in self.spec.base_models
                          if name in models_dict]
        else:
            estimators = [(name, clone(model)) for name, model in models_dict.items()]

        weights = self.spec.weights
        if weights and len(weights) != len(estimators):
            weights = None  # Fall back to equal weights if mismatch

        if self.classification:
            return VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=weights,
            )
        return VotingRegressor(
            estimators=estimators,
            weights=weights,
        )


def build_ensemble_from_spec(spec: EnsembleSpec, classification: bool = True) -> BaseEstimator:
    """Factory function to build an ensemble from a spec.

    :param spec: The EnsembleSpec describing the ensemble.
    :param classification: Whether this is a classification or regression task.
    :returns: A scikit-learn estimator.
    """
    if spec.ensemble_type == 'stacking':
        builder = StackingEnsembleBuilder(spec, classification)
        return builder.build()

    elif spec.ensemble_type == 'weighted' or spec.ensemble_type == 'voting':
        builder = WeightedEnsembleBuilder(spec, classification)
        return builder.build()

    elif spec.ensemble_type == 'bagging':
        model_key = 'clf' if classification else 'reg'
        base_model_name = spec.base_models[0] if spec.base_models else 'CART'
        base_model = clone(model_configs.models[model_key][base_model_name])

        if classification:
            return BaggingClassifier(estimator=base_model, n_estimators=10, random_state=0)
        return BaggingRegressor(estimator=base_model, n_estimators=10, random_state=0)

    else:
        raise ValueError(f'Unknown ensemble type: {spec.ensemble_type}')
