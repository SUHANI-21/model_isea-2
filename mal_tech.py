import json
import csv


# Load attack-pattern → technique mapping
ap_to_tech = {}

with open("attack_pattern_technique.csv", "r", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:
        ap_to_tech[row["attack_pattern_id"]] = row["technique_id"]


# Load malware + relationships
with open("malware_all.json", "r", encoding="utf-8") as f:
    data = json.load(f)


rows = []


for obj in data.get("objects", []):

    if obj.get("type") != "relationship":
        continue

    src = obj.get("source_ref")
    tgt = obj.get("target_ref")

    # malware → attack-pattern
    if tgt in ap_to_tech:

        tech = ap_to_tech[tgt]

        rows.append([src, tech])


# Save CSV
with open("malware_technique_labels.csv", "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow(["malware_id", "technique_id"])
    writer.writerows(rows)


print(f"[+] Saved {len(rows)} rows to malware_technique_labels.csv")
