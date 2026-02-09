"""Experiment Registry: structured storage for querying past experiments by pipeline decisions.

Borrows from: LightAutoML (experiment comparison), EvalML (pipeline metadata storage).

The distinction between experiment *tracking* (MLflow -- metrics/artifacts) and
experiment *management* (querying across experiments by pipeline structure).

Storage backend: SQLite with JSON columns.
"""

import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from taberspilotml.experiment.config import ExperimentConfig
from taberspilotml.experiment.runner import ExperimentResult


class ExperimentRegistry:
    """SQLite-backed registry for experiment results.

    Enables querying past experiments by pipeline decisions,
    not just flat MLflow params.

    :param storage_path: Path to SQLite database file.
    """

    def __init__(self, storage_path: str = 'experiments.db'):
        self.storage_path = storage_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with self._connect() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint TEXT NOT NULL,
                    name TEXT,
                    config_json TEXT NOT NULL,
                    scores_json TEXT,
                    score_stds_json TEXT,
                    primary_score REAL,
                    primary_std REAL,
                    model_name TEXT,
                    scaler_name TEXT,
                    transformer_name TEXT,
                    imputation_strategy TEXT,
                    cv_policy TEXT,
                    n_splits INTEGER,
                    stacking INTEGER,
                    evaluation_metric TEXT,
                    task_type TEXT,
                    execution_time REAL,
                    status TEXT,
                    error_message TEXT,
                    created_at REAL
                )
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_fingerprint ON experiments(fingerprint)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_model ON experiments(model_name)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_status ON experiments(status)
            ''')

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.storage_path)

    def register(self, result: ExperimentResult) -> int:
        """Register an experiment result.

        :param result: The ExperimentResult to store.
        :returns: The database ID of the registered experiment.
        """
        config = result.config
        with self._connect() as conn:
            cursor = conn.execute('''
                INSERT INTO experiments (
                    fingerprint, name, config_json, scores_json, score_stds_json,
                    primary_score, primary_std,
                    model_name, scaler_name, transformer_name, imputation_strategy,
                    cv_policy, n_splits, stacking, evaluation_metric, task_type,
                    execution_time, status, error_message, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                config.fingerprint,
                config.name,
                config.to_json(),
                json.dumps(result.scores),
                json.dumps(result.score_stds),
                result.primary_score,
                result.primary_std,
                config.model.model_name,
                config.features.scaler_name,
                config.features.transformer_name,
                config.preprocessing.imputation_strategy,
                config.cv.policy_type,
                config.cv.n_splits,
                int(config.model.stacking),
                config.evaluation_metric,
                config.task_type.value,
                result.execution_time,
                result.status,
                result.error_message,
                time.time(),
            ))
            return cursor.lastrowid

    def get(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Get an experiment by its database ID.

        :param experiment_id: The database ID.
        :returns: Dictionary of experiment data, or None if not found.
        """
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('SELECT * FROM experiments WHERE id = ?', (experiment_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            return self._row_to_dict(row)

    def query(self, **filters) -> pd.DataFrame:
        """Query experiments by pipeline decisions.

        :param filters: Keyword arguments to filter by (e.g., model_name='RF', status='completed').
        :returns: DataFrame of matching experiments.
        """
        conditions = []
        params = []
        for key, value in filters.items():
            conditions.append(f'{key} = ?')
            params.append(value)

        where_clause = ' AND '.join(conditions) if conditions else '1=1'
        sql = f'SELECT * FROM experiments WHERE {where_clause} ORDER BY created_at DESC'

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

        return pd.DataFrame([self._row_to_dict(row) for row in rows])

    def compare(self, ids: List[int]) -> pd.DataFrame:
        """Compare specific experiments side-by-side.

        :param ids: List of experiment database IDs to compare.
        :returns: DataFrame with one row per experiment.
        """
        placeholders = ','.join(['?'] * len(ids))
        sql = f'SELECT * FROM experiments WHERE id IN ({placeholders})'

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, ids)
            rows = cursor.fetchall()

        return pd.DataFrame([self._row_to_dict(row) for row in rows])

    def best(self, metric: Optional[str] = None, n: int = 5,
             classification: bool = True) -> pd.DataFrame:
        """Get the top N experiments by a metric.

        :param metric: Metric name. Defaults to primary_score.
        :param n: Number of results to return.
        :param classification: If True, higher is better; if False, lower is better.
        :returns: DataFrame of top experiments.
        """
        order = 'DESC' if classification else 'ASC'
        sql = f'''
            SELECT * FROM experiments
            WHERE status = 'completed' AND primary_score IS NOT NULL
            ORDER BY primary_score {order}
            LIMIT ?
        '''

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, (n,))
            rows = cursor.fetchall()

        return pd.DataFrame([self._row_to_dict(row) for row in rows])

    def history(self, metric: Optional[str] = None) -> pd.DataFrame:
        """Get the history of experiments ordered by creation time.

        :param metric: Not used currently but reserved for metric-specific history.
        :returns: DataFrame of all completed experiments ordered by creation time.
        """
        sql = '''
            SELECT * FROM experiments
            WHERE status = 'completed'
            ORDER BY created_at ASC
        '''

        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql)
            rows = cursor.fetchall()

        return pd.DataFrame([self._row_to_dict(row) for row in rows])

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a database row to a dictionary."""
        d = dict(row)
        d['scores'] = json.loads(d.pop('scores_json', '{}') or '{}')
        d['score_stds'] = json.loads(d.pop('score_stds_json', '{}') or '{}')
        d['stacking'] = bool(d.get('stacking', 0))
        return d
