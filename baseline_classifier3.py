import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score

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
# 3. K-Fold CV
# =========================

kf = KFold(n_splits=5, shuffle=True, random_state=42)

micro_scores = []
macro_scores = []

fold_num = 1

for train_idx, val_idx in kf.split(texts):
    X_train, X_val = texts.iloc[train_idx], texts.iloc[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]

    # =========================
    # 4. Build Pipeline
    # =========================

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,  # reduce vocab for small dataset
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

    # =========================
    # 5. Train
    # =========================

    model.fit(X_train, Y_train)

    # =========================
    # 6. Per-Label Threshold Tuning
    # =========================

    Y_probs_train = model.predict_proba(X_train)
    label_thresholds = []

    for i in range(Y_train.shape[1]):
        best_f1 = 0
        best_thresh = 0.5
        probs = Y_probs_train[:, i]
        for t in np.arange(0.1, 0.91, 0.05):
            pred = (probs >= t).astype(int)
            f1 = f1_score(Y_train[:, i], pred)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        label_thresholds.append(best_thresh)
    label_thresholds = np.array(label_thresholds)

    # =========================
    # 7. Predict with Per-Label Thresholds
    # =========================

    Y_probs_val = model.predict_proba(X_val)
    Y_pred_val = np.zeros(Y_probs_val.shape)
    for i, t in enumerate(label_thresholds):
        Y_pred_val[:, i] = (Y_probs_val[:, i] >= t).astype(int)

    # =========================
    # 8. Evaluate
    # =========================

    micro = f1_score(Y_val, Y_pred_val, average="micro")
    macro = f1_score(Y_val, Y_pred_val, average="macro")
    print(f"Fold {fold_num} - Micro F1: {micro:.3f}, Macro F1: {macro:.3f}")
    micro_scores.append(micro)
    macro_scores.append(macro)
    fold_num += 1

# =========================
# 9. Overall CV Metrics
# =========================

print("\n=== Cross-Validation Summary ===")
print(f"Micro F1: {np.mean(micro_scores):.3f} ± {np.std(micro_scores):.3f}")
print(f"Macro F1: {np.mean(macro_scores):.3f} ± {np.std(macro_scores):.3f}")

# =========================
# 10. Train Final Model on Full Data
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
# 11. Per-Label Threshold Tuning on Full Data
# =========================

Y_probs_full = final_model.predict_proba(texts)
label_thresholds_final = []

for i in range(Y.shape[1]):
    best_f1 = 0
    best_thresh = 0.5
    probs = Y_probs_full[:, i]
    for t in np.arange(0.1, 0.91, 0.05):
        pred = (probs >= t).astype(int)
        f1 = f1_score(Y[:, i], pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    label_thresholds_final.append(best_thresh)

label_thresholds_final = np.array(label_thresholds_final)
print("\nPer-label thresholds for final model:", label_thresholds_final)

# =========================
# 12. Predictions on Full Data (Optional)
# =========================

Y_pred_full = np.zeros(Y_probs_full.shape)
for i, t in enumerate(label_thresholds_final):
    Y_pred_full[:, i] = (Y_probs_full[:, i] >= t).astype(int)

# Per-label evaluation
report = classification_report(
    Y, Y_pred_full, target_names=label_names, output_dict=True, zero_division=0
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

print("Total samples:", len(df))
