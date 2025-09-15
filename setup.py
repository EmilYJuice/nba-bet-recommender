from setuptools import setup, find_packages

setup(
    name="nba-bet-recommender",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "lightgbm",
        "matplotlib",
        "plotly",
        "streamlit",
        "joblib",
        "python-dateutil",
    ],
    entry_points={
        "console_scripts": [
            "nba-train=nba_reco.pipeline.train:main",
            "nba-validate=nba_reco.pipeline.validate:main",
            "nba-daily-run=nba_reco.pipeline.daily_run:main",
        ],
    },
)
