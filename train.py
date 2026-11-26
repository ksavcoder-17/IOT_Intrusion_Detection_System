import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("ciciot2023_all_attacks.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df.drop(columns=["Label"])
y_attack = df["Label"]
y_bin = (df["Label"] != "BENIGN").astype(int)

X_tr, X_te, yb_tr, yb_te, ya_tr, ya_te = train_test_split(
    X, y_bin, y_attack, test_size=0.2, stratify=y_attack, random_state=42)

# Layer 1: normal vs malicious
print("Training Layer 1...")
l1 = RandomForestClassifier(n_estimators=50, max_depth=15,
                            class_weight="balanced", n_jobs=-1, random_state=42)
l1.fit(X_tr, yb_tr)
print(classification_report(yb_te, l1.predict(X_te),
                            target_names=["Benign", "Malicious"]))

# Layer 2: which attack (trained on malicious rows only)
print("Training Layer 2...")
mal_tr = yb_tr == 1
mal_te = yb_te == 1
l2 = RandomForestClassifier(n_estimators=50, max_depth=15,
                            class_weight="balanced", n_jobs=-1, random_state=42)
l2.fit(X_tr[mal_tr], ya_tr[mal_tr])
print(classification_report(ya_te[mal_te], l2.predict(X_te[mal_te]), zero_division=0))

joblib.dump(l1, "layer1.joblib")
joblib.dump(l2, "layer2.joblib")

# Save a demo stream from the test set (features + true label)
demo = pd.concat([X_te, ya_te], axis=1).sample(5000, random_state=1)
demo.to_csv("demo_stream.csv", index=False)
print("Done. Saved layer1.joblib, layer2.joblib, demo_stream.csv")