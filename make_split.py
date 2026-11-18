import random
import re
from pathlib import Path

import pandas as pd

root = Path("data/chest_xray")
rows = []

# collect every image from the train and val folders
for split in ["train", "val"]:
    for cls in ["NORMAL", "PNEUMONIA"]:
        for f in (root / split / cls).glob("*.jpeg"): #using pathlib and only selecting jpegs
            rows.append({"path": str(f), "label": cls, "name": f.name})
df = pd.DataFrame(rows)


def patient_id(row):
    name = row["name"]
    if row["label"] == "PNEUMONIA":
        patient = name.split("_")[0]
    else:
        parts = name.split("-")
        patient = parts[parts.index("IM") + 1] if "IM" in parts else name
    return row["label"] + "_" + patient

df["patient"] = df.apply(patient_id, axis=1)

# choosing 15% of the patients in each class for validation
random.seed(42)
val_patients = set()
for label, group in df.groupby("label"):
    patients = sorted(group["patient"].unique())
    random.shuffle(patients)
    n_val = int(len(patients) * 0.15)
    val_patients.update(patients[:n_val])

df["split"] = df["patient"].apply(lambda p: "val" if p in val_patients else "train")

# addig the official test set unchanged
test_rows = []
for cls in ["NORMAL", "PNEUMONIA"]:
    for f in (root / "test" / cls).glob("*.jpeg"):
        test_rows.append(
            {"path": str(f), "label": cls, "name": f.name,
             "patient": "test_" + f.name, "split": "test"}
        )
df = pd.concat([df, pd.DataFrame(test_rows)], ignore_index=True)

# no patient is in both train and val, check
train_p = set(df[df.split == "train"].patient)
val_p = set(df[df.split == "val"].patient)
assert train_p.isdisjoint(val_p), "Leakage: a patient is in train and val"

df.to_csv("splits.csv", index=False)
print(pd.crosstab(df["split"], df["label"]))
print("Patients: train", len(train_p), "val", len(val_p))