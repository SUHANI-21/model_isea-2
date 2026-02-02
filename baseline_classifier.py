import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
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
# 3. Train / Test Split
# =========================

X_train, X_test, Y_train, Y_test = train_test_split(
    texts,
    Y,
    test_size=0.2,
    random_state=42
)


# =========================
# 4. Build Baseline Model
# =========================

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 3),
        stop_words="english",
        min_df=1
    )),

    ("clf", OneVsRestClassifier(
        LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced"
        )
    ))
])


# =========================
# 5. Train
# =========================

print("Training model...")

model.fit(X_train, Y_train)


# =========================
# 6. Predict
# =========================

Y_probs = model.predict_proba(X_test)

threshold = 0.5   # try 0.2–0.4
Y_pred = (Y_probs >= threshold).astype(int)


# =========================
# 7. Overall Metrics
# =========================

micro_f1 = f1_score(Y_test, Y_pred, average="micro")
macro_f1 = f1_score(Y_test, Y_pred, average="macro")

print("\n=== Overall Performance ===")
print(f"Micro F1 : {micro_f1:.3f}")
print(f"Macro F1 : {macro_f1:.3f}")


# =========================
# 8. Per-Technique (Top Only)
# =========================

report = classification_report(
    Y_test,
    Y_pred,
    target_names=label_names,
    output_dict=True,
    zero_division=0
)

# Sort by support
tech_stats = []


for tech in label_names:
    precision = report[tech]["precision"]
    recall = report[tech]["recall"]
    f1 = report[tech]["f1-score"]
    support = report[tech]["support"]
    tech_stats.append((tech, precision, recall, f1, support))
# Sort by support (most common first)
tech_stats.sort(key=lambda x: x[4], reverse=True)

# Print
print("\n=== Top 15 Techniques (by samples) ===")
print("Technique   Support   Precision   Recall   F1")

for t, p, r, f1, sup in tech_stats[:15]:
    print(f"{t:10}  {int(sup):7}     {p:6.3f}     {r:6.3f}   {f1:6.3f}")
    
    
'''df_test = pd.DataFrame({
    "text": X_test,
    "true_labels": [ "|".join(l) for l in mlb.inverse_transform(Y_test) ],
    "pred_labels": [ "|".join(l) for l in mlb.inverse_transform(Y_pred) ]
})

df_test.to_csv("prediction.csv", index=False)'''
print("Total samples:", len(df))
print("Train samples:", len(X_train))
print("Test samples :", len(X_test))
print("Predictions  :", Y_pred.shape[0])

'''Training model...

=== Overall Performance ===
Micro F1 : 0.762
Macro F1 : 0.728

=== Top 15 Techniques (by samples) ===
Technique   Support   Precision   Recall   F1
T1071.001        72      0.851      0.875    0.863
T1105            72      0.800      0.833    0.816
T1082            58      0.786      0.948    0.859
T1059.003        57      0.753      0.965    0.846
T1070.004        52      0.836      0.885    0.860
T1083            52      0.726      0.865    0.789
T1140            52      0.681      0.942    0.790
T1057            47      0.755      0.851    0.800
T1016            40      0.673      0.875    0.761
T1106            35      0.674      0.886    0.765
T1033            34      0.609      0.824    0.700
T1547.001        33      0.711      0.970    0.821
T1027.013        32      0.595      0.688    0.638
T1573.001        32      0.698      0.938    0.800
T1041            31      0.806      0.806    0.806
Total samples: 759
Train samples: 607
Test samples : 152
Predictions  : 152'''
