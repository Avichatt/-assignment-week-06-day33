import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

categories = [
    'sci.space',
    'rec.sport.football',
    'comp.graphics',
    'talk.politics.guns',
]

train_data = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
test_data  = fetch_20newsgroups(subset='test',  categories=categories, remove=('headers', 'footers', 'quotes'))

X_train_raw = train_data.data
X_test_raw  = test_data.data
y_train      = train_data.target
y_test       = test_data.target

svm_pipeline = Pipeline([
    ('tfidf',  TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)),
    ('clf',    LinearSVC(C=1.0, max_iter=2000)),
])

lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)),
    ('clf',   LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', multi_class='auto')),
])

svm_pipeline.fit(X_train_raw, y_train)
lr_pipeline.fit(X_train_raw, y_train)

svm_pred = svm_pipeline.predict(X_test_raw)
lr_pred  = lr_pipeline.predict(X_test_raw)

svm_acc = accuracy_score(y_test, svm_pred)
lr_acc  = accuracy_score(y_test, lr_pred)

svm_report = classification_report(y_test, svm_pred, target_names=categories)
lr_report  = classification_report(y_test, lr_pred,  target_names=categories)

svm_cv = cross_val_score(svm_pipeline, X_train_raw, y_train, cv=5, scoring='accuracy')
lr_cv  = cross_val_score(lr_pipeline,  X_train_raw, y_train, cv=5, scoring='accuracy')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_svm = confusion_matrix(y_test, svm_pred)
ConfusionMatrixDisplay(cm_svm, display_labels=categories).plot(ax=axes[0], colorbar=False)
axes[0].set_title(f"LinearSVC – Acc={svm_acc:.4f}")
axes[0].tick_params(axis='x', rotation=30)

cm_lr = confusion_matrix(y_test, lr_pred)
ConfusionMatrixDisplay(cm_lr, display_labels=categories).plot(ax=axes[1], colorbar=False)
axes[1].set_title(f"Logistic Regression – Acc={lr_acc:.4f}")
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('d33_pm_text_classification.png', dpi=110)
plt.show()

comparison_df = pd.DataFrame({
    "Model":    ["LinearSVC", "Logistic Regression"],
    "CV Mean":  [round(svm_cv.mean(), 4), round(lr_cv.mean(), 4)],
    "CV Std":   [round(svm_cv.std(),  4), round(lr_cv.std(),  4)],
    "Test Acc": [round(svm_acc, 4),       round(lr_acc, 4)],
})
