import pandas as pd
from collections import Counter


INPUT_FILE = "malware_with_rule_text.csv"
OUTPUT_FILE = "malware_filtered.csv"

MIN_COUNT = 50   # You can change to 5 if dataset is small


# Load
df = pd.read_csv(INPUT_FILE)


# -------------------------------
# STEP 1: Count technique freq
# -------------------------------

counter = Counter()

for techs in df["techniques"].dropna():

    for t in str(techs).split("|"):
        if t.strip():
            counter[t.strip()] += 1


print("Top techniques:")
for t, c in counter.most_common(20):
    print(t, "->", c)


# -------------------------------
# STEP 2: Select valid labels
# -------------------------------

valid_techs = {
    t for t, c in counter.items()
    if c >= MIN_COUNT
}

print("\nKept techniques:", len(valid_techs))


# -------------------------------
# STEP 3: Filter dataset
# -------------------------------

def filter_labels(label_str):

    if pd.isna(label_str):
        return ""

    kept = []

    for t in str(label_str).split("|"):
        if t in valid_techs:
            kept.append(t)

    return "|".join(kept)


df["filtered_techniques"] = df["techniques"].apply(filter_labels)


# Remove rows with no labels left
df = df[df["filtered_techniques"] != ""]


# Drop old column
df = df.drop(columns=["techniques"])


# Rename
df = df.rename(columns={
    "filtered_techniques": "techniques"
})


# Save
df.to_csv(OUTPUT_FILE, index=False)


print("\nSaved cleaned dataset:", OUTPUT_FILE)
print("Final rows:", len(df))
