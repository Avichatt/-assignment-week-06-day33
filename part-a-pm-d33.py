import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

algorithm_cards = {
    "Logistic Regression": {
        "when":   "Binary/multiclass, need probability, large dataset",
        "params": "C, max_iter, solver",
        "pros":   "Fast, interpretable, probabilistic output",
        "cons":   "Assumes linear boundary, sensitive to outliers",
        "model":  LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    },
    "Decision Tree": {
        "when":   "Need interpretability, mixed feature types",
        "params": "max_depth, min_samples_split, criterion",
        "pros":   "Visual rules, handles non-linearity, no scaling needed",
        "cons":   "Overfits easily, unstable",
        "model":  DecisionTreeClassifier(max_depth=5, random_state=42),
    },
    "Random Forest": {
        "when":   "Tabular data, need robustness, feature importance",
        "params": "n_estimators, max_depth, max_features",
        "pros":   "Reduces overfitting, good accuracy, feature importance",
        "cons":   "Slow on large data, less interpretable",
        "model":  RandomForestClassifier(n_estimators=100, random_state=42),
    },
    "Gradient Boosting": {
        "when":   "Tabular data, need best accuracy, structured/clean data",
        "params": "n_estimators, learning_rate, max_depth",
        "pros":   "High accuracy, handles mixed types, regularized",
        "cons":   "Slower training, more hyperparameters",
        "model":  GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    },
    "SVM (RBF)": {
        "when":   "Small-medium dataset, high-dimensional, non-linear boundary",
        "params": "C, gamma, kernel",
        "pros":   "Great in high dimensions, robust to outliers, kernel trick",
        "cons":   "Slow on large data, hard to tune, no probabilities by default",
        "model":  SVC(kernel='rbf', C=10, gamma=0.01, random_state=42),
    },
    "KNN": {
        "when":   "Simple baseline, low-dimensional, small dataset",
        "params": "n_neighbors, metric, weights",
        "pros":   "No training phase, intuitive, adapts to local structure",
        "cons":   "Slow prediction, sensitive to scale and irrelevant features",
        "model":  KNeighborsClassifier(n_neighbors=5),
    },
    "Naive Bayes": {
        "when":   "Text classification, very fast baseline, large feature space",
        "params": "var_smoothing (GaussianNB), alpha (MultinomialNB)",
        "pros":   "Extremely fast, works with little data, probabilistic",
        "cons":   "Assumes feature independence (rarely true)",
        "model":  GaussianNB(),
    },
    "MLP (Neural Net)": {
        "when":   "Complex patterns, medium-large data, enough compute",
        "params": "hidden_layer_sizes, activation, learning_rate",
        "pros":   "Learns complex non-linear patterns, flexible architecture",
        "cons":   "Needs scaling, many hyperparameters, slow, black box",
        "model":  MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    },
}

results = []

for name, card in algorithm_cards.items():
    model = card["model"]
    cv_scores = cross_val_score(
        Pipeline([("scaler", StandardScaler()), ("model", model)]),
        X, y, cv=5, scoring='accuracy'
    )
    model.fit(X_train_sc, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test_sc))

    results.append({
        "Algorithm":    name,
        "CV Mean":      round(cv_scores.mean(), 4),
        "CV Std":       round(cv_scores.std(), 4),
        "Test Acc":     round(test_acc, 4),
        "When to Use":  card["when"],
        "Key Params":   card["params"],
        "Pros":         card["pros"],
        "Cons":         card["cons"],
    })

results_df = pd.DataFrame(results).sort_values("CV Mean", ascending=False).reset_index(drop=True)

best_model_name = results_df.iloc[0]["Algorithm"]
best_cv         = results_df.iloc[0]["CV Mean"]
best_test       = results_df.iloc[0]["Test Acc"]

fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(results_df))]
bars = ax.barh(results_df["Algorithm"], results_df["CV Mean"], xerr=results_df["CV Std"],
               color=colors, edgecolor='white', capsize=4)
ax.set_xlabel("5-Fold CV Accuracy")
ax.set_title("Algorithm Comparison – Breast Cancer Dataset\n(Green = Best Model)")
ax.set_xlim(0.85, 1.0)
for i, (cv, std) in enumerate(zip(results_df["CV Mean"], results_df["CV Std"])):
    ax.text(cv + std + 0.001, i, f"{cv:.4f}±{std:.4f}", va='center', fontsize=8)
plt.tight_layout()
plt.savefig('d33_pm_algorithm_comparison.png', dpi=110)
plt.show()
