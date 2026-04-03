import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def knn_from_scratch(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        differences = X_train - test_point
        distances = np.linalg.norm(differences, axis=1)
        sorted_indices = np.argsort(distances)
        k_nearest_indices = sorted_indices[:k]
        k_nearest_labels = y_train[k_nearest_indices]
        vote_counts = np.bincount(k_nearest_labels.astype(int))
        predicted_label = vote_counts.argmax()
        predictions.append(predicted_label)
    return np.array(predictions)


digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_te_sc = scaler.transform(X_test)

n_test_samples = 100
X_te_small = X_te_sc[:n_test_samples]
y_te_small = y_test[:n_test_samples]

scratch_pred = knn_from_scratch(X_tr_sc, y_train, X_te_small, k=3)
scratch_acc = accuracy_score(y_te_small, scratch_pred)

sk_knn = KNeighborsClassifier(n_neighbors=3)
sk_knn.fit(X_tr_sc, y_train)
sk_pred = sk_knn.predict(X_te_small)
sk_acc = accuracy_score(y_te_small, sk_pred)

results_match = np.array_equal(scratch_pred, sk_pred)

np.random.seed(42)
n = 300
salary = np.concatenate([
    np.random.uniform(50_000, 100_000, n // 2),
    np.random.uniform(120_000, 200_000, n // 2),
])
age = np.concatenate([
    np.random.uniform(20, 60, n // 2),
    np.random.uniform(20, 60, n // 2),
])
labels = np.array([0] * (n // 2) + [1] * (n // 2))

X_toy = np.column_stack([salary, age])

X_tr, X_te, y_tr, y_te = train_test_split(X_toy, labels, test_size=0.2, random_state=42)

broken_svm = SVC(kernel='rbf', C=1.0)
broken_svm.fit(X_tr, y_tr)
broken_acc = broken_svm.score(X_te, y_te)

sc = StandardScaler()
X_tr_sc2 = sc.fit_transform(X_tr)
X_te_sc2 = sc.transform(X_te)

fixed_svm = SVC(kernel='rbf', C=1.0)
fixed_svm.fit(X_tr_sc2, y_tr)
fixed_acc = fixed_svm.score(X_te_sc2, y_te)
