import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    'dateparser',
    'pandas',
    'matplotlib',
    'seaborn',
    'xgboost',
    'scikit-learn==0.24.2',
    'hyperopt',
    'optuna',
    'imblearn',
    'mlflow',
    'sweetviz',
    'google-cloud-bigquery',
    'tensorflow',
    'snowflake-connector-python',
    'py4j'
]


setuptools.setup(
    name="taberspilotml",
    version="0.0.1",
    author="",
    author_email="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://source.tui/cr/analytics-capabilities/shared/python_packages/tuiautopilotml",
    packages=setuptools.find_packages(exclude=["tests"]),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
)
