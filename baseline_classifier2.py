import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score

from skmultilearn.model_selection import IterativeStratification  # multi-label stratified split

# =========================
# 1. Load Data
# =========================

df = pd.read_csv("malware_filtered.csv")
texts = df["rule_full_text"].fillna("")
labels = df["techniques"].fillna("").apply(lambda x: x.split("|"))

# =========================
# 2. Encode Labels
# =========================

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(labels)
label_names = mlb.classes_

# =========================
# 3. K-Fold Stratified CV
# =========================

k = 5  # 5-fold
iter_strat = IterativeStratification(n_splits=k, order=1)

micro_scores = []
macro_scores = []

fold_num = 1

for train_idx, val_idx in iter_strat.split(np.zeros(len(Y)), Y):
    X_train, X_val = texts.iloc[train_idx], texts.iloc[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]

    # =========================
    # 4. Build Pipeline
    # =========================

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,       # reduce vocab for small dataset
            ngram_range=(1, 2),      # 1-2 grams
            stop_words="english",
            min_df=2                 # ignore very rare words
        )),
        ("clf", OneVsRestClassifier(
            LogisticRegression(
                max_iter=2000,
                solver="liblinear",
                class_weight="balanced",
                C=0.5  # stronger regularization
            )
        ))
    ])

    # =========================
    # 5. Train
    # =========================

    model.fit(X_train, Y_train)

    # =========================
    # 6. Predict
    # =========================

    Y_probs = model.predict_proba(X_val)
    
    # Optional: tune threshold per fold
    threshold = 0.5  # slightly lower for rare labels
    Y_pred = (Y_probs >= threshold).astype(int)

    # =========================
    # 7. Evaluate
    # =========================

    micro = f1_score(Y_val, Y_pred, average="micro")
    macro = f1_score(Y_val, Y_pred, average="macro")
    print(f"Fold {fold_num} - Micro F1: {micro:.3f}, Macro F1: {macro:.3f}")
    micro_scores.append(micro)
    macro_scores.append(macro)
    fold_num += 1

# =========================
# 8. Overall CV Metrics
# =========================

print("\n=== Cross-Validation Summary ===")
print(f"Micro F1: {np.mean(micro_scores):.3f} ± {np.std(micro_scores):.3f}")
print(f"Macro F1: {np.mean(macro_scores):.3f} ± {np.std(macro_scores):.3f}")

# =========================
# 9. Train final model on full data
# =========================

final_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2
    )),
    ("clf", OneVsRestClassifier(
        LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            C=0.5
        )
    ))
])

final_model.fit(texts, Y)

# =========================
# 10. Optional: Per-Technique Analysis
# =========================

Y_pred = (final_model.predict_proba(texts) >= 0.5).astype(int)

report = classification_report(
    Y, Y_pred, target_names=label_names, output_dict=True, zero_division=0
)

tech_stats = []
for tech in label_names:
    precision = report[tech]["precision"]
    recall = report[tech]["recall"]
    f1 = report[tech]["f1-score"]
    support = report[tech]["support"]
    tech_stats.append((tech, precision, recall, f1, support))

tech_stats.sort(key=lambda x: x[4], reverse=True)

print("\n=== Top 15 Techniques (by samples) ===")
print("Technique   Support   Precision   Recall   F1")
for t, p, r, f1, sup in tech_stats[:15]:
    print(f"{t:10}  {int(sup):7}     {p:6.3f}     {r:6.3f}   {f1:6.3f}")


    '''Fold 1 - Micro F1: 0.667, Macro F1: 0.631
Fold 2 - Micro F1: 0.824, Macro F1: 0.783
Fold 3 - Micro F1: 0.533, Macro F1: 0.518
Fold 4 - Micro F1: 0.700, Macro F1: 0.692
Fold 5 - Micro F1: 0.588, Macro F1: 0.575

=== Cross-Validation Summary ===
Micro F1: 0.662 � 0.100
Macro F1: 0.640 � 0.092

=== Top 15 Techniques (by samples) ===
Technique   Support   Precision   Recall   F1
T1636.004        15      1.000      1.000    1.000
T1636.003        12      1.000      1.000    1.000
T1430            10      0.909      1.000    0.952
T1426             9      1.000      1.000    1.000'''
