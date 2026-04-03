import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier



SVM (RBF/Linear) — works well in high-dim, maximises margin, few support vectors matter
Naive Bayes— assumes independence, very few parameters, works with small N
Logistic Reg (L2) — regularisation prevents overfitting, stable with many features
KNN — no training params, but test set must be small too
 
Deep Neural Net   — needs thousands of samples, will massively overfit
Gradient Boosting — many params, overfits easily on tiny data
Random Forest     — needs more samples to build unbiased trees (100 splits on 50 rows = bad)
Decision Tree     — 100 features / 50 rows = perfect memorisation of training set

More features than samples = p >> n problem. Use regularised or kernel methods.



def model_selection_report(X, y, models_dict):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_scores = {}
    summary_rows = []

    for name, model in models_dict.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  model),
        ])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
        fold_scores[name] = scores
        summary_rows.append({
            "Model":   name,
            "CV Mean": round(scores.mean(), 4),
            "CV Std":  round(scores.std(),  4),
            "Min":     round(scores.min(),  4),
            "Max":     round(scores.max(),  4),
        })

    report_df = pd.DataFrame(summary_rows).sort_values("CV Mean", ascending=False).reset_index(drop=True)

    best_name   = report_df.iloc[0]["Model"]
    second_name = report_df.iloc[1]["Model"]

    t_stat, p_value = stats.ttest_rel(
        fold_scores[best_name],
        fold_scores[second_name]
    )

    report_df["Rank"] = range(1, len(report_df) + 1)

    return report_df, best_name, p_value


data = load_breast_cancer()
X, y = data.data, data.target

models = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (RBF)":           SVC(kernel='rbf', C=10, gamma=0.01, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
}

report_df, best_model, p_val = model_selection_report(X, y, models)

significant = p_val < 0.05



problem": "SVM(RBF) — train acc=1.0, test acc=0.52 — SEVERE OVERFITTING
root_cause": "Model memorised all training points instead of learning general patterns

name": "Reduce C (increase regularisation)
explanation": "High C forces SVM to classify EVERY training point correctly. 
                       "Lower C (e.g. C=0.1) allows some misclassifications → wider margin → generalises better.
code": "SVC(kernel='rbf', C=0.1, gamma='scale')
    

name": "Tune gamma with GridSearchCV
explanation": "High gamma = each training point only influences its tiny neighbourhood → memorises data.
                       "Lower gamma = smoother, wider influence → better generalisation.",
code": "GridSearchCV(SVC(kernel='rbf'), {'C':[0.1,1,10], 'gamma':[1,0.1,0.01,0.001]}, cv=5)
    

name": "Get more training data or use cross-validation properly",
explanation": "With very little data, SVM can memorise. CV score gives a more honest estimate. "
                       "If data is genuinely small, try simpler models (Logistic Regression, Naive Bayes).
code": "cross_val_score(svm, X, y, cv=5, scoring='accuracy')
    
