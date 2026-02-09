"""Tests for enhanced ensembles: multi-layer stacking, weighted voting, bagging."""

import random
import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from taberspilotml.modelling.ensembles import (
    EnsembleSpec,
    StackingEnsembleBuilder,
    WeightedEnsembleBuilder,
    build_ensemble_from_spec,
)


def _make_clf_data(n=200, seed=42):
    X, y = make_classification(n_samples=n, n_features=5, random_state=seed)
    return X, y


def _make_reg_data(n=200, seed=42):
    X, y = make_regression(n_samples=n, n_features=5, random_state=seed)
    return X, y


class TestStackingEnsembleBuilder(unittest.TestCase):

    def test_single_layer_classification(self):
        spec = EnsembleSpec(
            ensemble_type='stacking',
            base_models=['RF', 'KNN'],
            n_folds=3,
            n_layers=1,
        )
        builder = StackingEnsembleBuilder(spec, classification=True)
        model = builder.build()

        X, y = _make_clf_data()
        model.fit(X, y)
        predictions = model.predict(X)
        self.assertEqual(len(predictions), len(y))

    def test_single_layer_regression(self):
        spec = EnsembleSpec(
            ensemble_type='stacking',
            base_models=['RF', 'KNN'],
            n_folds=3,
            n_layers=1,
        )
        builder = StackingEnsembleBuilder(spec, classification=False)
        model = builder.build()

        X, y = _make_reg_data()
        model.fit(X, y)
        predictions = model.predict(X)
        self.assertEqual(len(predictions), len(y))

    def test_multi_layer_stacking(self):
        spec = EnsembleSpec(
            ensemble_type='stacking',
            base_models=['CART', 'KNN'],
            n_folds=3,
            n_layers=2,
        )
        builder = StackingEnsembleBuilder(spec, classification=True)
        model = builder.build()

        X, y = _make_clf_data()
        model.fit(X, y)
        predictions = model.predict(X)
        self.assertEqual(len(predictions), len(y))


class TestWeightedEnsembleBuilder(unittest.TestCase):

    def test_weighted_voting_classification(self):
        spec = EnsembleSpec(
            ensemble_type='weighted',
            base_models=['RF', 'KNN'],
            weights=[0.7, 0.3],
        )
        builder = WeightedEnsembleBuilder(spec, classification=True)
        model = builder.build()

        X, y = _make_clf_data()
        model.fit(X, y)
        predictions = model.predict(X)
        self.assertEqual(len(predictions), len(y))

    def test_weighted_voting_regression(self):
        spec = EnsembleSpec(
            ensemble_type='weighted',
            base_models=['RF', 'KNN'],
            weights=[0.6, 0.4],
        )
        builder = WeightedEnsembleBuilder(spec, classification=False)
        model = builder.build()

        X, y = _make_reg_data()
        model.fit(X, y)
        predictions = model.predict(X)
        self.assertEqual(len(predictions), len(y))

    def test_equal_weights_when_none(self):
        spec = EnsembleSpec(
            ensemble_type='weighted',
            base_models=['RF', 'KNN'],
            weights=None,
        )
        builder = WeightedEnsembleBuilder(spec, classification=True)
        model = builder.build()
        # Should not raise
        X, y = _make_clf_data()
        model.fit(X, y)


class TestBuildEnsembleFromSpec(unittest.TestCase):

    def test_factory_stacking(self):
        spec = EnsembleSpec(ensemble_type='stacking', base_models=['RF', 'KNN'], n_folds=3)
        model = build_ensemble_from_spec(spec, classification=True)
        X, y = _make_clf_data()
        model.fit(X, y)
        self.assertIsNotNone(model.predict(X))

    def test_factory_weighted(self):
        spec = EnsembleSpec(ensemble_type='weighted', base_models=['RF', 'KNN'])
        model = build_ensemble_from_spec(spec, classification=True)
        X, y = _make_clf_data()
        model.fit(X, y)
        self.assertIsNotNone(model.predict(X))

    def test_factory_bagging(self):
        spec = EnsembleSpec(ensemble_type='bagging', base_models=['CART'])
        model = build_ensemble_from_spec(spec, classification=True)
        X, y = _make_clf_data()
        model.fit(X, y)
        self.assertIsNotNone(model.predict(X))

    def test_factory_unknown_type(self):
        spec = EnsembleSpec(ensemble_type='unknown')
        with self.assertRaises(ValueError):
            build_ensemble_from_spec(spec)

    def test_factory_regression_bagging(self):
        spec = EnsembleSpec(ensemble_type='bagging', base_models=['CART'])
        model = build_ensemble_from_spec(spec, classification=False)
        X, y = _make_reg_data()
        model.fit(X, y)
        self.assertIsNotNone(model.predict(X))


class TestEnsembleSpec(unittest.TestCase):

    def test_defaults(self):
        spec = EnsembleSpec()
        self.assertEqual(spec.ensemble_type, 'stacking')
        self.assertIsNone(spec.base_models)
        self.assertEqual(spec.n_layers, 1)
        self.assertEqual(spec.n_folds, 5)


if __name__ == '__main__':
    unittest.main()
