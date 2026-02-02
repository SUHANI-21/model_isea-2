import json
import csv

# Load attack patterns
with open("attack_pattern.json", "r", encoding="utf-8") as f:
    data = json.load(f)

VALID_SOURCES = ["attack", "mitre", "mitre-attack", "mitre attack"]

rows = []

for obj in data.get("objects", []):

    if obj.get("type") != "attack-pattern":
        continue

    ap_id = obj.get("id")
    external_refs = obj.get("external_references", [])

    for ref in external_refs:

        source = ref.get("source_name", "").lower()
        ext_id = ref.get("external_id", "")

        if any(s in source for s in VALID_SOURCES) and ext_id.startswith("T"):

            rows.append([ap_id, ext_id])
            break


# Save CSV
with open("attack_pattern_technique.csv", "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow(["attack_pattern_id", "technique_id"])
    writer.writerows(rows)


print(f"[+] Saved {len(rows)} rows to attack_pattern_technique.csv")
