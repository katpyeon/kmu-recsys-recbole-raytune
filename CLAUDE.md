# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RecBole AutoML project for automated hyperparameter tuning and model selection for recommendation systems. RecBole is a unified, comprehensive recommendation library built on PyTorch.

## Project Structure

- `auto_ml.ipynb` - Main Jupyter notebook for AutoML experiments
- `datasets/` - Training and evaluation datasets (CSV format)
- `outputs/` - Model outputs, logs, and experiment results

## Development Environment

### Running Jupyter Notebooks

```bash
jupyter notebook auto_ml.ipynb
# or
jupyter lab
```

### Python Environment

This project likely requires:
- RecBole library (`pip install recbole`)
- AutoML libraries (e.g., Optuna, Ray Tune, or similar)
- Standard ML libraries: pandas, numpy, scikit-learn

Install dependencies as they are added to requirements files.

## RecBole Architecture Notes

RecBole uses a configuration-based approach where:
- Models are defined with specific hyperparameters
- Datasets follow RecBole's atomic file format or interaction format
- Training is controlled via YAML configuration files or Python dicts

When implementing AutoML:
- Hyperparameter search spaces should align with RecBole model parameters
- Evaluation metrics follow RecBole's metric definitions (e.g., Recall@K, NDCG@K)
- Dataset loading should use RecBole's dataset classes

## Data Format

The `apply_train.csv` in datasets/ contains training data. Ensure data preprocessing:
- Converts to RecBole's expected format (user_id, item_id, rating/timestamp)
- Handles missing values appropriately
- Splits into train/validation/test sets for proper AutoML evaluation
