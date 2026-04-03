import numpy as np
import time
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

try:
    import faiss
except ImportError:
    raise ImportError("Run: pip install faiss-cpu")

digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train).astype('float32')
X_test_sc = scaler.transform(X_test).astype('float32')

d = X_train_sc.shape[1]
faiss_index = faiss.IndexFlatL2(d)
faiss_index.add(X_train_sc)

K = 3
distances, indices = faiss_index.search(X_test_sc, K)

def majority_vote(neighbor_indices, labels):
    predictions = []
    for neighbors in neighbor_indices:
        neighbor_labels = labels[neighbors]
        pred = np.bincount(neighbor_labels).argmax()
        predictions.append(pred)
    return np.array(predictions)

faiss_pred = majority_vote(indices, y_train)
faiss_acc = accuracy_score(y_test, faiss_pred)

n_queries = 1000
queries_sklearn = np.tile(X_test_sc, (n_queries // len(X_test_sc) + 1, 1))[:n_queries]
queries_faiss = queries_sklearn.astype('float32')

sklearn_knn = KNeighborsClassifier(n_neighbors=K)
sklearn_knn.fit(X_train_sc, y_train)

_ = sklearn_knn.predict(queries_sklearn[:5])
t0 = time.perf_counter()
sklearn_knn.predict(queries_sklearn)
sklearn_time = time.perf_counter() - t0

faiss_index.search(queries_faiss[:5], K)
t0 = time.perf_counter()
faiss_index.search(queries_faiss, K)
faiss_time = time.perf_counter() - t0

sklearn_ms = sklearn_time * 1000
faiss_ms = faiss_time * 1000

if faiss_time < sklearn_time:
    speedup = sklearn_time / faiss_time
else:
    speedup = faiss_time / sklearn_time
