# Decision Trees & Random Forest — Day 32 Assignments

**Week 6 · Day 32 | Machine Learning & AI**
IIT Gandhinagar | PG Diploma in AI-ML & Agentic AI Engineering

## Overview

This repository contains take-home assignments for Day 32 covering **Decision Trees**, **Random Forest**, hyperparameter tuning, feature importance, and model comparison.

## Repository Structure

### AM Session — Decision Trees & Random Forest (Theory + Implementation)

| File | Part | Description |
|------|------|-------------|
| `part-a-am.py` | Concept Application (40%) | Loan approval system — synthetic data, DT rules extraction, RF with RandomizedSearchCV, permutation importance, deployment recommendation |
| `part-b-am.py` | Stretch Problem (30%) | ExtraTrees vs RandomForest — splitting differences, speed benchmark, performance comparison |
| `part-c-am.py` | Interview Ready (20%) | Bias-variance tradeoff, `plot_overfitting_curve()` function, debugging identical train/test accuracy |
| `part-d-am.py` | AI-Augmented (10%) | Matplotlib infographic — DT vs RF vs Logistic Regression for non-technical audience |

### PM Session — Decision Trees & Random Forest: Applied

| File | Part | Description |
|------|------|-------------|
| `part-a-pm.py` | Concept Application (40%) | Insurance fraud detection — cost-sensitive evaluation (FN = 10× FP), DT fraud rules, RF tuned for recall |
| `part-b-pm.py` | Stretch Problem (30%) | Gradient Boosting preview — bagging vs boosting comparison, resource links |
| `part-c-pm.py` | Interview Ready (20%) | 1000 vs 100 trees tradeoff, `compare_models()` function, debugging unstable feature importances |
| `part-d-pm.py` | AI-Augmented (10%) | OOB error explanation with analogy, evaluation, critique, and code demo |

## Topics Covered

- Gini impurity, entropy, information gain
- Overfitting & pruning, Decision Tree hyperparameters
- Bagging & bootstrap, feature randomness
- Random Forest hyperparameters, feature importance (MDI vs permutation)
- GridSearchCV, RandomizedSearchCV, cross-validation
- Cost-sensitive evaluation, interpretability vs accuracy tradeoff
- ExtraTrees, Gradient Boosting (preview), OOB error estimation

## Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib
```

## How to Run

```bash
# AM Session
python part-a-am.py
python part-b-am.py
python part-c-am.py
python part-d-am.py

# PM Session
python part-a-pm.py
python part-b-pm.py
python part-c-pm.py
python part-d-pm.py
```

## Author

Avi Chattopadhyay
