# AutoPilotML

**An experimental, modular machine learning experimentation framework for rapid research and method comparison.**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-experimental-orange.svg)](https://github.com/yourusername/autopilotml)

---

## 🎯 Overview

AutoPilotML is a highly extensible machine learning experimentation framework designed for data scientists and researchers who need to:

- **Rapidly test and compare** multiple ML techniques, algorithms, and preprocessing methods
- **Easily extend** the framework with new methods as research evolves
- **Systematically evaluate** different approaches across the entire ML pipeline
- **Track experiments** with integrated MLflow logging
- **Adapt quickly** to new modeling paradigms (supervised learning, time series, statistical modeling, etc.)

### Design Philosophy

> **"A controlled experimentation framework for time-series forecasting that enables fast, reproducible comparison of modeling assumptions while preventing temporal leakage and exposing why methods succeed or fail."**

The core idea is **modularity and adaptability**. When you discover a new preprocessing technique, feature engineering method, or hyperparameter optimization algorithm, you simply add it to the appropriate module. The `autopilot_mode` automatically integrates it into your experimental pipeline, allowing systematic comparison against existing methods.

**Core Principles:**

1. **No hidden decisions** - Every modeling choice must be inspectable, logged, and reproducible
2. **Ablation-first, not optimization-first** - The system answers *why*, not only *what wins*
3. **Time-series correctness over convenience** - Leakage-safe CV, horizon handling, and covariate alignment are non-negotiable
4. **Methods > Models** - Pipelines, transformations, and assumptions are first-class citizens
5. **Agents must justify their cost** - Every LLM component has measurable value (accuracy delta, stability, insight quality)

This framework prioritizes **understanding, reproducibility, and controlled experimentation** over production deployment, making it ideal for research, prototyping, and method validation.

### ❌ What AutoPilotML Is NOT

To maintain focus and quality, the following are **explicitly out of scope**:

- ❌ **Not an AutoML tool** - We don't compete on raw leaderboard performance or "best model" automation
- ❌ **Not production-ready MLOps** - No deployment pipelines, model serving, or production monitoring
- ❌ **Not a general-purpose ML library** - Focus is on time-series forecasting and structured experimentation
- ❌ **Not a one-click solution** - Requires understanding of modeling assumptions and tradeoffs
- ❌ **Not using LLMs as forecasters** - LLMs assist with analysis and interpretation, not prediction
- ❌ **Not optimized for tabular classification** - While supported, the architecture now prioritizes temporal modeling

If a feature doesn't strengthen **understanding, reproducibility, or insight generation**, it's deprioritized.

### 🎯 Canonical Use Case

**Primary Focus:** Energy consumption forecasting with temporal patterns

The framework is anchored around realistic time-series forecasting scenarios where:
- Temporal ordering matters
- Multiple horizons need evaluation
- Feature engineering is critical
- Leakage prevention is non-negotiable
- Explainability is as important as accuracy

Example: Industrial energy consumption prediction with hourly/daily patterns, seasonal effects, and external covariates.

---

## 🏗️ Architecture

### Pipeline Structure

The framework organizes the ML workflow into three sequential stages:

```
┌─────────────────────────────────────────────────────────────┐
│                    DEFAULT STEPS                             │
│  (Always executed - data preparation & baseline)             │
├─────────────────────────────────────────────────────────────┤
│  • dataframe_transformation  → Format and clean data         │
│  • handle_missing_values     → Imputation strategies         │
│  • encoding                  → Categorical encoding          │
│  • baseline_score            → Initial performance benchmark │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    MODELLING STEPS                           │
│  (Core ML experimentation - configurable)                    │
├─────────────────────────────────────────────────────────────┤
│  • handle_outliers           → Outlier detection/treatment   │
│  • evaluate_oversamplers     → Imbalance handling (SMOTE)    │
│  • evaluate_models           → Model comparison              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 POST-MODELLING STEPS                         │
│  (Optimization & refinement)                                 │
├─────────────────────────────────────────────────────────────┤
│  • feature_selection         → Feature importance analysis   │
│  • transformation_methods    → Scaling/transformation eval   │
│  • hyper_param_opt          → Hyperopt tuning               │
│  • optuna                    → Optuna optimization           │
│  • grid_search               → Grid search tuning            │
└─────────────────────────────────────────────────────────────┘
```

### Module Organization

```
taberspilotml/
├── auto_mode.py              # Orchestration & pipeline execution
├── base_helpers.py           # Shared utilities & config management
├── constants.py              # Default values & configurations
├── mlflow_uploader.py        # Experiment tracking integration
├── hyper_opti.py            # Hyperparameter optimization engines
├── visualization.py          # Plotting & result visualization
│
├── preprocessing/            # Data transformation utilities
│   ├── generals.py          # General preprocessing functions
│   └── jsons.py             # JSON handling
│
├── pre_modelling/           # Pre-model data preparation
│   ├── encoders.py          # Categorical encoding methods
│   ├── feature_engineering.py
│   ├── feature_importance.py
│   ├── handle_nulls.py      # Missing value strategies
│   ├── imbalance.py         # Class imbalance handling
│   └── outliers.py          # Outlier detection/treatment
│
├── modelling/               # Model training & evaluation
│   ├── ml_models.py         # Classical ML algorithms
│   └── neural_nets.py       # Neural network implementations
│
├── scoring_funcs/           # Evaluation framework
│   ├── cross_validation.py  # CV strategies
│   ├── datasets.py          # Dataset handling
│   ├── evaluation_metrics.py
│   └── scorers.py           # Scoring utilities
│
├── analytics/               # Analysis tools
│   ├── eda.py              # Exploratory Data Analysis
│   ├── pca.py              # Dimensionality reduction
│   └── pareto.py           # Pareto analysis
│
└── conf/                    # Configuration
    ├── configs.py          # Model & hyperparameter definitions
    └── llm_configs.py      # LLM integration configs
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/autopilotml.git
cd autopilotml

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### Basic Usage

```python
import pandas as pd
from taberspilotml.auto_mode import autopilot_mode
from taberspilotml.base_helpers import initialize_config

# Load your dataset
df = pd.read_csv("your_data.csv")

# Initialize configuration
config = initialize_config(
    df=df,
    target_label='target_column',
    classification=True,
    evaluation_metric='accuracy',
    run_id_number='experiment_001'
)

# Define pipeline steps to execute
steps = [
    'handle_outliers',
    'evaluate_oversamplers',
    'evaluate_models',
    'feature_selection',
    'hyper_param_opt'
]

# Run autopilot mode
summary = autopilot_mode(steps=steps, config_dict=config)
```

### Example: Comparing Multiple Models

```python
# The framework automatically evaluates multiple algorithms
# Default models: KNN, Naive Bayes, SVC, Random Forest, XGBoost, AdaBoost, MLP

steps = ['evaluate_models']
summary = autopilot_mode(steps=steps, config_dict=config)

# Results are automatically logged to MLflow
# Check mlruns/ directory or start MLflow UI:
# mlflow ui
```

---

## 🔧 Extending the Framework

### Adding a New Preprocessing Method

The framework is designed for easy extensibility. Here's how to add a new technique:

#### Example: Adding a New Encoding Method

1. **Add your function** to `taberspilotml/pre_modelling/encoders.py`:

```python
def my_custom_encoding(df, target_label, **kwargs):
    """
    Your custom encoding implementation.
    
    Returns:
        pd.DataFrame: Encoded dataframe
    """
    # Your implementation here
    encoded_df = df.copy()
    # ... encoding logic ...
    return encoded_df
```

2. **Inject into pipeline** using `task_specs`:

```python
from taberspilotml.pre_modelling.encoders import my_custom_encoding

task_specs = [{
    'step_name': 'default_steps',
    'task_name': 'my_custom_encoding',
    'function': my_custom_encoding,
    'position': 2  # Insert position in pipeline
}]

# Run with custom step
autopilot_mode(steps=steps, config_dict=config, task_specs=task_specs)
```

3. **Test automatically**: The framework will now test your method alongside existing approaches!

### Adding a New Model

Add to `taberspilotml/conf/configs.py`:

```python
from sklearn.ensemble import GradientBoostingClassifier

models = {
    'clf': {
        'XGB': XGBClassifier(),
        'RF': RandomForestClassifier(),
        'GBM': GradientBoostingClassifier(),  # New model
        # ... other models
    }
}
```

### Adding Hyperparameter Spaces

Define search spaces in `taberspilotml/conf/configs.py`:

```python
from hyperopt import hp

hyper_params = {
    'clf': {
        'GBM': {
            'n_estimators': hp.choice('n_estimators', [100, 200, 500]),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
            'max_depth': hp.choice('max_depth', range(3, 10))
        }
    }
}
```

---

## 📊 Available Pipeline Steps

### Default Steps (Always Run)

| Step | Purpose | Module |
|------|---------|--------|
| `dataframe_transformation` | Format data types, handle basic cleaning | `preprocessing.generals` |
| `handle_missing_values` | Evaluate imputation methods | `pre_modelling.handle_nulls` |
| `encoding` | Categorical variable encoding | `pre_modelling.encoders` |
| `baseline_score` | Initial performance benchmark | `base_helpers` |

### Modelling Steps (Configurable)

| Step | Purpose | Module |
|------|---------|--------|
| `handle_outliers` | Outlier detection and treatment comparison | `pre_modelling.outliers` |
| `evaluate_oversamplers` | Imbalance handling (SMOTE, RandomOverSampler, etc.) | `pre_modelling.imbalance` |
| `evaluate_models` | Compare multiple ML algorithms | `modelling.ml_models` |

### Post-Modelling Steps (Optimization)

| Step | Purpose | Module |
|------|---------|--------|
| `feature_selection` | Feature importance and selection strategies | `pre_modelling.feature_importance` |
| `transformation_methods` | Scaling and transformation evaluation | `modelling.ml_models` |
| `hyper_param_opt` | Hyperopt-based tuning | `hyper_opti` |
| `optuna` | Optuna optimization framework | `hyper_opti` |
| `grid_search` | Grid search hyperparameter tuning | `hyper_opti` |

---

## 🔬 Supported Techniques

### Current Capabilities

**Models (Supervised Learning)**
- K-Nearest Neighbors (KNN)
- Naive Bayes (NB)
- Support Vector Classifier (SVC)
- Random Forest (RF)
- XGBoost (XGB)
- AdaBoost (ADA)
- Multi-Layer Perceptron (MLP)
- Stacking Ensembles

**Hyperparameter Optimization**
- Hyperopt (Tree-structured Parzen Estimator)
- Optuna (Bayesian optimization)
- Grid Search
- Cross-validation integrated

**Imbalance Handling**
- SMOTE (Synthetic Minority Over-sampling)
- Random Over-sampling
- Custom samplers

**Feature Engineering**
- Feature importance analysis
- Feature selection
- Dimensionality reduction (PCA)

**Preprocessing**
- Multiple imputation strategies
- Categorical encoding
- Outlier detection/treatment
- Data transformation & scaling

### Roadmap (Experimental Extensions)

- ⏳ **Time Series Modeling**: ARIMA, Prophet, LSTM sequences
- 📈 **Statistical Modeling**: GLM, Bayesian methods
- 🧪 **Additional Techniques**: As research and experimentation demands

---

## 📈 Experiment Tracking

The framework integrates **MLflow** for comprehensive experiment tracking:

```python
# Start MLflow UI to view results
# mlflow ui

# Access at http://localhost:5000
```

**What's Tracked:**
- Model parameters and hyperparameters
- Cross-validation scores
- Evaluation metrics (accuracy, F1, precision, recall, etc.)
- Model artifacts
- Dataset characteristics
- Preprocessing configurations

Results are stored in:
- `mlruns/` - MLflow tracking data
- `mlartifacts/` - Model artifacts and outputs

---

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run specific test module
pytest tests/test_auto_mode.py

# With coverage
pytest --cov=taberspilotml tests/
```

**Test Structure:**
```
tests/
├── test_auto_mode.py         # Pipeline orchestration tests
├── test_base_helpers.py      # Utility function tests
├── test_datasets.py          # Dataset handling tests
├── test_hyper_opti.py        # Hyperparameter tuning tests
├── test_mlflow_uploader.py   # MLflow integration tests
└── modelling/                # Model-specific tests
```

---

## 🛠️ Best Practices Implemented

This repository serves as a testing ground for production ML best practices:

✅ **Experiment Tracking**: MLflow integration for reproducibility  
✅ **Modular Architecture**: Easy to extend and maintain  
✅ **Configuration Management**: Centralized configs for consistency  
✅ **Cross-Validation**: Robust evaluation strategies  
✅ **Unit Testing**: Pytest framework with test coverage  
✅ **Version Control**: Git-based workflow  
✅ **Documentation**: Comprehensive docstrings  

**Future Best Practices to Integrate:**
- `pytest-monitor` for performance tracking
- CI/CD pipelines
- Data versioning (DVC)
- Model registry integration
- Containerization (Docker)

---

## 📝 Configuration

### Example Config Structure

```python
config_dict = {
    'df': dataframe,                    # Your dataset
    'target_label': 'target',           # Target column name
    'classification': True,             # Classification vs Regression
    'evaluation_metric': 'accuracy',    # Primary metric
    'run_id_number': 'exp_001',        # Unique run identifier
    'k_fold_method': 'k_fold',         # CV strategy
    'n_folds': 5,                      # Number of folds
    'n_repeats': 3,                    # CV repeats
    'n_jobs': -1,                      # Parallel jobs
    'random_state': 0                  # Reproducibility seed
}
```

---

## 🤝 Contributing

This is an experimental research framework. Contributions, ideas, and feedback are welcome!

**Ways to Contribute:**
- Add new ML algorithms or techniques
- Implement additional preprocessing methods
- Extend to new modeling paradigms (time series, NLP, etc.)
- Improve documentation
- Add test cases
- Share best practices

---

## 📄 License

[MIT License](LICENSE) - Feel free to use, modify, and distribute.

---

## 🙏 Acknowledgments

Built with:
- scikit-learn
- XGBoost
- MLflow
- Hyperopt / Optuna
- imbalanced-learn
- And many other excellent open-source libraries

---

## 📬 Contact & Support

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub.

**Status**: 🧪 Experimental - Active development and research

---

*AutoPilotML: Because every experiment deserves a systematic approach.*

