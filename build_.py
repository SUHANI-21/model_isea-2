import json
import pandas as pd
from collections import defaultdict


INPUT_FILE = "malware_all.json"
OUTPUT_FILE = "malware_rule_features.csv"


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

objects = data.get("objects", [])


# Store objects
malware_objs = {}
attack_patterns = {}
relationships = []


for obj in objects:

    t = obj.get("type")

    if t == "malware":
        malware_objs[obj["id"]] = obj

    elif t == "attack-pattern":
        attack_patterns[obj["id"]] = obj

    elif t == "relationship":
        relationships.append(obj)


# Build relationship maps
malware_to_attack = defaultdict(list)
malware_to_campaign = defaultdict(list)
malware_to_intrusion = defaultdict(list)
malware_to_tool = defaultdict(list)


for r in relationships:

    src = r.get("source_ref")
    tgt = r.get("target_ref")
    rtype = r.get("relationship_type")

    if not src or not tgt:
        continue

    if src.startswith("malware--"):

        if tgt.startswith("attack-pattern--"):
            malware_to_attack[src].append(tgt)

        elif tgt.startswith("campaign--"):
            malware_to_campaign[src].append(tgt)

        elif tgt.startswith("intrusion-set--"):
            malware_to_intrusion[src].append(tgt)

        elif tgt.startswith("tool--"):
            malware_to_tool[src].append(tgt)


rows = []


for mid, m in malware_objs.items():

    name = m.get("name", "")
    desc = m.get("description", "")
    aliases = "|".join(m.get("aliases", []))
    platforms = "|".join(m.get("x_mitre_platforms", []))
    is_family = m.get("is_family", False)

    # External references
    refs = []
    confidence = ""

    for r in m.get("external_references", []):
        src = r.get("source_name", "")
        eid = r.get("external_id", "")

        if src and eid:
            refs.append(f"{src}:{eid}")

        if "confidence" in r:
            confidence = r["confidence"]

    refs = "|".join(refs)

    # Techniques
    tech_ids = []

    for ap_id in malware_to_attack.get(mid, []):

        ap = attack_patterns.get(ap_id)

        if not ap:
            continue

        for r in ap.get("external_references", []):

            src = r.get("source_name", "").lower()
            eid = r.get("external_id", "")

            if "attack" in src and eid.startswith("T"):
                tech_ids.append(eid)

    techniques = "|".join(sorted(set(tech_ids)))

    row = {
        "malware_id": mid,
        "malware_name": name,
        "description": desc,
        "aliases": aliases,
        "platforms": platforms,
        "is_family": is_family,
        "techniques": techniques,
        "num_campaigns": len(malware_to_campaign[mid]),
        "num_intrusion_sets": len(malware_to_intrusion[mid]),
        "num_tools": len(malware_to_tool[mid]),
        "num_relationships": sum([
            len(malware_to_attack[mid]),
            len(malware_to_campaign[mid]),
            len(malware_to_intrusion[mid]),
            len(malware_to_tool[mid])
        ]),
        "external_references": refs,
        "confidence": confidence
    }

    rows.append(row)


df = pd.DataFrame(rows)

df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved: {OUTPUT_FILE}")
