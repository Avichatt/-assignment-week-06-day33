from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'C':     [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
}

svm_grid = GridSearchCV(
    SVC(kernel='rbf', random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
svm_grid.fit(X_train_scaled, y_train)

best_C = svm_grid.best_params_['C']
best_gamma = svm_grid.best_params_['gamma']
best_svm = svm_grid.best_estimator_

svm_test_acc = best_svm.score(X_test_scaled, y_test)

k_values = list(range(1, 16))
k_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    acc = knn.score(X_test_scaled, y_test)
    k_scores.append(acc)

best_k = k_values[k_scores.index(max(k_scores))]
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_scaled, y_train)

svm_pred = best_svm.predict(X_test_scaled)
knn_pred = best_knn.predict(X_test_scaled)

svm_acc = accuracy_score(y_test, svm_pred)
knn_acc = accuracy_score(y_test, knn_pred)

svm_report = classification_report(y_test, svm_pred, target_names=[str(d) for d in range(10)])
knn_report = classification_report(y_test, knn_pred, target_names=[str(d) for d in range(10)])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, pred, title in zip(
    axes,
    [svm_pred, knn_pred],
    [f"SVM RBF (C={best_C}, γ={best_gamma})\nAcc={svm_acc:.4f}",
     f"KNN (K={best_k})\nAcc={knn_acc:.4f}"]
):
    cm = confusion_matrix(y_test, pred)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Predicted Digit")
    ax.set_ylabel("True Digit")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for row in range(10):
        for col in range(10):
            ax.text(col, row, str(cm[row, col]),
                    ha='center', va='center',
                    color='white' if cm[row, col] > cm.max() / 2 else 'black',
                    fontsize=8)
    fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('d33_confusion_matrices.png', dpi=110)
plt.show()

cm_svm = confusion_matrix(y_test, svm_pred)

confused_pairs = []
for true_digit in range(10):
    for pred_digit in range(10):
        if true_digit != pred_digit and cm_svm[true_digit, pred_digit] > 0:
            confused_pairs.append(
                (cm_svm[true_digit, pred_digit], true_digit, pred_digit)
            )

confused_pairs.sort(reverse=True)
top_confused = confused_pairs[:5]

plt.figure(figsize=(8, 4))
plt.plot(k_values, k_scores, marker='o', color='teal', linewidth=2)
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Best K={best_k}')
plt.title("KNN Accuracy vs K value")
plt.xlabel("K (number of neighbors)")
plt.ylabel("Test Accuracy")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('d33_knn_k_vs_accuracy.png', dpi=110)
plt.show()
