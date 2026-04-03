import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles

np.random.seed(42)
X_plot, y_plot = make_moons(n_samples=200, noise=0.25, random_state=42)

scaler = StandardScaler()
X_plot = scaler.fit_transform(X_plot)

C_values = [0.01, 0.1, 1.0, 10, 100]

fig, axes = plt.subplots(1, len(C_values), figsize=(18, 4))
fig.suptitle(
    "SVM Decision Boundary as C Changes\n"
    "(Low C = smoother/more errors allowed  |  High C = tight fit/fewer errors)",
    fontsize=13, fontweight='bold', y=1.02
)

x_min, x_max = X_plot[:, 0].min() - 0.5, X_plot[:, 0].max() + 0.5
y_min, y_max = X_plot[:, 1].min() - 0.5, X_plot[:, 1].max() + 0.5
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

colors_map = {0: '#4C9BE8', 1: '#E85E4C'}

for ax, C in zip(axes, C_values):
    svm = SVC(kernel='rbf', C=C, gamma='scale')
    svm.fit(X_plot, y_plot)

    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm.predict(mesh_points).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.25, cmap='RdBu_r', levels=[-0.5, 0.5, 1.5])
    ax.contour(xx, yy, Z, colors='k', linewidths=1.0, levels=[0.5])

    for label in [0, 1]:
        mask = y_plot == label
        ax.scatter(
            X_plot[mask, 0], X_plot[mask, 1],
            color=colors_map[label],
            edgecolors='white', s=35, zorder=3, alpha=0.85
        )

    sv = svm.support_vectors_
    ax.scatter(
        sv[:, 0], sv[:, 1],
        s=100, facecolors='none', edgecolors='black',
        linewidths=1.5, zorder=4, label='Support Vectors'
    )

    acc = svm.score(X_plot, y_plot)
    n_sv = len(svm.support_vectors_)

    ax.set_title(
        f"C = {C}\nAcc={acc:.2f}  |  Support Vectors={n_sv}",
        fontsize=10
    )
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2" if C == C_values[0] else "")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig('d33_svm_C_boundary.png', dpi=110, bbox_inches='tight')
plt.show()

np.random.seed(42)
X_circ, y_circ = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Kernel Trick — Circles Example", fontsize=13, fontweight='bold')

ax = axes[0]
for label in [0, 1]:
    mask = y_circ == label
    ax.scatter(X_circ[mask, 0], X_circ[mask, 1],
               label=f'Class {label}', s=30, alpha=0.8)
ax.set_title("2D Data – No straight line can separate these")
ax.legend()
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")

ax = axes[1]
svm_circ = SVC(kernel='rbf', C=10, gamma='scale')
svm_circ.fit(X_circ, y_circ)

x0, x1 = X_circ[:, 0].min() - 0.3, X_circ[:, 0].max() + 0.3
y0, y1 = X_circ[:, 1].min() - 0.3, X_circ[:, 1].max() + 0.3
xx2, yy2 = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
Z2 = svm_circ.predict(np.c_[xx2.ravel(), yy2.ravel()]).reshape(xx2.shape)

ax.contourf(xx2, yy2, Z2, alpha=0.2, cmap='RdBu_r')
ax.contour(xx2, yy2, Z2, colors='black', linewidths=1)
for label in [0, 1]:
    mask = y_circ == label
    ax.scatter(X_circ[mask, 0], X_circ[mask, 1],
               label=f'Class {label}', s=30, alpha=0.8)
ax.set_title(f"SVM RBF Kernel – Acc={svm_circ.score(X_circ, y_circ):.2f}\nKernel lifts data to separate them!")
ax.legend()
ax.set_xlabel("Feature 1")

plt.tight_layout()
plt.savefig('d33_kernel_trick_demo.png', dpi=110, bbox_inches='tight')
plt.show()
