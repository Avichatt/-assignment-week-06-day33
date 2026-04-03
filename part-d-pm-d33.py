import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


very_small (<200 samples)": ["Naive Bayes", "SVM (RBF)", "Logistic Regression", "KNN"],
small (200-2000)":          ["SVM (RBF)", "Random Forest", "Logistic Regression", "KNN"],
medium (2K-100K)":          ["Random Forest", "Gradient Boosting", "SVM (Linear)", "Logistic Regression"],
large (>100K)":             ["Logistic Regression", "Gradient Boosting", "MLP (Neural Net)"],
    
  
all numeric:      Any algorithm works — scale first for SVM/KNN/
mixed (num+cat):  Gradient Boosting or Random Forest — handle without encod
text:             TF-IDF + LinearSVC or Logistic Regress
images:           CNN (MLP as baseline) — flatten pixels fi
high-dimensional: SVM (linear/RBF), Naive Bayes, Logistic Regression (
    

linearly separable:    Logistic Regression, SVM (Linear), Naive Bayes
non-linear:            SVM (RBF), Random Forest, Gradient Boosting
unknown (start simple):Logistic Regression first → if poor → try RBF SVM or Random Forest
 
yes (must explain): Logistic Regression, Decision Tree
partial: Random Forest (feature importance)
no (just accuracy):  Gradient Boosting, SVM (RBF), MLP
    

Logistic Regression, Naive Bayes, Random Forest, Gradient Boosting, MLP
SVM (SVC does not output probs by default — use probability=True if needed)
    

Imbalanced classes → use class_weight='balanced' in LR/SVM/RF, not just accuracy
Categorical features with high cardinality → tree methods better than one-hot + LR
Temporal data (time series) → standard CV leaks future; must use TimeSeriesSplit
Noisy labels → Naive Bayes or SVM more robust than deep trees",
Online/streaming data → SGDClassifier (LR/SVM with SGD); batch models fail
Multi-label (not multiclass) → standard sklearn classifiers need MultiOutputClassifier wrapper
Very sparse features → Naive Bayes (Bernoulli/Multinomial) or LinearSVC outperform RBF SVM



fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')

title = ax.text(8, 9.5, "Algorithm Selection Decision Guide",
                ha='center', va='center', fontsize=16, fontweight='bold', color='white')

boxes = [
    dict(x=0.3, y=7.5, w=3.5, h=1.6, label="Dataset Size?",
         options="<200 → SVM/NB/KNN\n200-2K → RF/SVM\n>100K → LR/GBM",
         color='#2980b9'),
    dict(x=4.3, y=7.5, w=3.5, h=1.6, label="Feature Type?",
         options="Text → TF-IDF+LinearSVC\nMixed → RF/GBM\nNumeric → Any (scale!)",
         color='#8e44ad'),
    dict(x=8.3, y=7.5, w=3.5, h=1.6, label="Need Interpretability?",
         options="Yes → LR or DTree\nPartial → RandomForest\nNo → GBM/SVM/MLP",
         color='#16a085'),
    dict(x=12.3, y=7.5, w=3.4, h=1.6, label="Linear Boundary?",
         options="Yes → LR/SVM-Linear\nNo → SVM-RBF/RF/GBM\nUnknown → Start LR",
         color='#d35400'),
]

for b in boxes:
    rect = FancyBboxPatch((b['x'], b['y']), b['w'], b['h'],
                          boxstyle="round,pad=0.05", linewidth=2,
                          edgecolor='white', facecolor=b['color'], alpha=0.85)
    ax.add_patch(rect)
    ax.text(b['x'] + b['w']/2, b['y'] + b['h'] - 0.25, b['label'],
            ha='center', va='top', fontsize=9, fontweight='bold', color='white')
    ax.text(b['x'] + b['w']/2, b['y'] + b['h']/2 - 0.25, b['options'],
            ha='center', va='center', fontsize=7.5, color='#ecf0f1',
            linespacing=1.5)

quick_ref = [
    Logistic Regression: Linear, prob output, interpretable, large data
    Decision Tree:       Visual rules, no scaling, overfits easily
    Random Forest:     Robust, feature importance, handles non-linearity
    Gradient Boosting:    Best accuracy on tabular, slower, more params
    SVM (RBF):           High-dim, small data, powerful, no probabilities
    KNN:                 Simple baseline, no training, slow prediction
    Naive Bayes:          Very fast, text/small data, feature independence
    MLP (Neural Net):    Complex patterns, needs scale+lots of data
]

row_y = 6.5
ax.text(8, row_y, "Quick Reference Cards", ha='center', va='center',
        fontsize=12, fontweight='bold', color='white')

for i, (name, color, desc) in enumerate(quick_ref):
    col = i % 4
    row = i // 4
    bx = 0.3 + col * 3.9
    by = 5.4 - row * 1.6
    rect = FancyBboxPatch((bx, by), 3.5, 1.3,
                          boxstyle="round,pad=0.05", linewidth=1.5,
                          edgecolor=color, facecolor='#16213e', alpha=0.9)
    ax.add_patch(rect)
    ax.text(bx + 1.75, by + 1.0, name, ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=color)
    ax.text(bx + 1.75, by + 0.45, desc, ha='center', va='center',
            fontsize=7, color='#bdc3c7', wrap=True)

edge_title_y = 1.85
ax.text(0.3, edge_title_y, "Edge Cases (often missed):", fontsize=9,
        fontweight='bold', color='#e74c3c', va='center')

edge_cases:

Imbalanced classes → use class_weight='balanced'
Time series → use TimeSeriesSplit (not random CV)
Sparse features → LinearSVC/NaiveBayes not RBF
Online/streaming → SGDClassifier only
]
for j, ec in enumerate(edge_cases):
    col = j % 2
    row = j // 2
    ax.text(0.3 + col * 8, edge_title_y - 0.5 - row * 0.55, f"⚠  {ec}",
            fontsize=7.5, color='#f39c12')

plt.tight_layout()
plt.savefig('d33_pm_algorithm_guide.png', dpi=120, bbox_inches='tight',
            facecolor='#1a1a2e')
plt.show()

personal_guide_df = pd.DataFrame(
    quick_ref, columns=["Algorithm", "Color", "When to Use"]
).drop(columns=["Color"])

edge_cases_df = pd.DataFrame(
    {"Edge Case": algorithm_selection_guide["edge_cases_ai_missed"]}
)

import pandas as pd
