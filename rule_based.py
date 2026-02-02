import pandas as pd


INPUT_FILE = "malware_rule_features.csv"
OUTPUT_FILE = "malware_with_rule_text.csv"


df = pd.read_csv(INPUT_FILE)


def clean(val):
    if pd.isna(val) or val == "":
        return "unknown"
    return str(val)


def make_rule_text(row):

    name = clean(row["malware_name"])
    desc = clean(row["description"])
    aliases = clean(row["aliases"])
    platforms = clean(row["platforms"])
    techniques = clean(row["techniques"])
    refs = clean(row["external_references"])
    confidence = clean(row["confidence"])

    is_family = row["is_family"]

    campaigns = row["num_campaigns"]
    intrusions = row["num_intrusion_sets"]
    tools = row["num_tools"]
    rels = row["num_relationships"]

    mtype = "malware family" if is_family else "malware sample"

    text = f"""
{name} is a {mtype} that primarily targets {platforms} systems.

Description: {desc}.

It is associated with the following ATT&CK techniques: {techniques}.

The malware has documented links to {campaigns} campaigns,
{intrusions} intrusion sets, and {tools} related tools.

It participates in approximately {rels} known relationships
within threat intelligence datasets.

Known aliases include: {aliases}.

External references: {refs}.

Reported confidence level: {confidence}.
"""

    return " ".join(text.split())


df["rule_full_text"] = df.apply(make_rule_text, axis=1)


df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved: {OUTPUT_FILE}")
