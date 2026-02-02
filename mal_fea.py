import json
import csv
from collections import defaultdict


# Load malware bundle
with open("malware_all.json", "r", encoding="utf-8") as f:
    data = json.load(f)


malware_objs = {}
relationship_counts = defaultdict(int)
attack_counts = defaultdict(int)
tool_counts = defaultdict(int)
campaign_counts = defaultdict(int)
intrusion_counts = defaultdict(int)


# First pass: collect malware objects
for obj in data.get("objects", []):

    if obj.get("type") == "malware":

        malware_objs[obj["id"]] = obj


# Second pass: count relationships
for obj in data.get("objects", []):

    if obj.get("type") != "relationship":
        continue

    src = obj.get("source_ref")
    tgt = obj.get("target_ref")

    relationship_counts[src] += 1

    # Rough categorization by target type
    if tgt.startswith("attack-pattern"):
        attack_counts[src] += 1

    elif tgt.startswith("tool"):
        tool_counts[src] += 1

    elif tgt.startswith("campaign"):
        campaign_counts[src] += 1

    elif tgt.startswith("intrusion-set"):
        intrusion_counts[src] += 1


rows = []


for mid, obj in malware_objs.items():

    name = obj.get("name", "")
    desc = obj.get("description", "")
    is_family = obj.get("is_family", False)
    confidence = obj.get("confidence", 0)

    num_rels = relationship_counts[mid]
    num_attack = attack_counts[mid]
    num_tools = tool_counts[mid]
    num_campaign = campaign_counts[mid]
    num_intrusion = intrusion_counts[mid]

    full_text = f"{name} {desc}".lower()

    rows.append([
        mid,
        name,
        desc,
        int(is_family),
        confidence,
        num_rels,
        num_attack,
        num_tools,
        num_campaign,
        num_intrusion,
        full_text
    ])


# Save CSV
with open("malware_features.csv", "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "malware_id",
        "name",
        "description",
        "is_family",
        "confidence",
        "num_relationships",
        "num_attack_patterns",
        "num_tools",
        "num_campaigns",
        "num_intrusion_sets",
        "full_text"
    ])

    writer.writerows(rows)


print(f"[+] Saved {len(rows)} rows to malware_features.csv")
