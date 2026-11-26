import time
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="IoT IDS Demo", layout="wide")
st.title("AI-Based IoT Intrusion Detection: Live Demo")

@st.cache_resource
def load_models():
    return joblib.load("layer1.joblib"), joblib.load("layer2.joblib")

@st.cache_data
def load_stream():
    return pd.read_csv("demo_stream.csv")

l1, l2 = load_models()
stream = load_stream()
feat_cols = [c for c in stream.columns if c != "Label"]

delay = st.sidebar.slider("Delay per batch (seconds)", 0.0, 2.0, 0.3)
batch_size = st.sidebar.slider("Flows per batch", 10, 200, 50)

m1, m2, m3, m4 = st.columns(4)
left, right = st.columns(2)
chart_ph = left.empty()
table_ph = right.empty()
status = st.empty()

if st.button("▶ Start live feed"):
    total = mal_n = n_both = 0
    correct1 = correct2 = 0
    attack_counts = {}
    alerts = []

    for s in range(0, len(stream), batch_size):
        batch = stream.iloc[s:s + batch_size]
        X = batch[feat_cols]
        true_attack = batch["Label"].values
        true_bin = (true_attack != "BENIGN").astype(int)

        # Layer 1
        pred_bin = l1.predict(X)
        mal = pred_bin == 1

        # Layer 2 (only on flows flagged malicious)
        pred_attack = np.array(["BENIGN"] * len(batch), dtype=object)
        if mal.any():
            pred_attack[mal] = l2.predict(X[mal])

        # Update counters
        total += len(batch)
        mal_n += int(mal.sum())
        correct1 += int((pred_bin == true_bin).sum())
        both = (pred_bin == 1) & (true_bin == 1)
        n_both += int(both.sum())
        correct2 += int((pred_attack[both] == true_attack[both]).sum())

        for name in pred_attack[mal]:
            attack_counts[name] = attack_counts.get(name, 0) + 1

        for i in np.where(mal)[0]:
            alerts.append({
                "Flow #": total - len(batch) + i + 1,
                "Predicted attack": pred_attack[i],
                "Actual label": true_attack[i],
            })

        # Refresh the display
        m1.metric("Flows processed", f"{total:,}")
        m2.metric("Malicious detected", f"{mal_n:,}")
        m3.metric("Layer 1 accuracy", f"{correct1 / total:.1%}")
        m4.metric("Layer 2 accuracy", f"{correct2 / max(n_both, 1):.1%}")

        if attack_counts:
            chart_ph.bar_chart(pd.Series(attack_counts).sort_values(ascending=False))
        if alerts:
            table_ph.dataframe(pd.DataFrame(alerts[-20:]).iloc[::-1])

        status.info(f"Streaming... {total:,} of {len(stream):,} flows")
        time.sleep(delay)

    status.success("Demo finished")