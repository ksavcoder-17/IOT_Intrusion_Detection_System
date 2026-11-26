import glob
from collections import Counter
import pandas as pd

files = sorted(glob.glob("merged_csv/*.csv"))
OUT = "ciciot2023_all_attacks.csv"
CAP = 100_000  # max rows kept per label; rare attacks keep all their rows

# Pass 1: count rows per label across all files
counts = Counter()
for f in files:
    counts.update(pd.read_csv(f, usecols=["Label"])["Label"])

frac = {lab: min(1.0, CAP / n) for lab, n in counts.items()}
print("Total rows in dataset:", sum(counts.values()))

# Pass 2: write a sample from each file to the output, one file at a time
first = True
for f in files:
    df = pd.read_csv(f)
    keep = df.groupby("Label", group_keys=False).apply(
        lambda g: g.sample(frac=frac[g.name], random_state=42)
    )
    feats = keep.select_dtypes("float64").columns
    keep[feats] = keep[feats].astype("float32")
    keep.to_csv(OUT, mode="w" if first else "a", header=first, index=False)
    first = False
    print("Done:", f, len(keep), "rows kept")

final = pd.read_csv(OUT)
print(final["Label"].value_counts())