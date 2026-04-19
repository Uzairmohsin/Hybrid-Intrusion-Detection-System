import pandas as pd
import numpy as np
import joblib
import time
import os

# ==========================================
# CONFIG
# ==========================================
ZEEK_LOG = "/home/kali/hybrid_ids_capturing_system/zeek_logs/conn.log"

XGB_MODEL_PATH = "/home/kali/Documents/Intrusion-Detection-System-IDS-using-Neural-Networks/new_ids/models/xgboost_model.pkl"
ISO_MODEL_PATH = "/home/kali/Documents/Intrusion-Detection-System-IDS-using-Neural-Networks/new_ids/models/isolation_forest_model.pkl"

CHECK_INTERVAL = 5   # seconds

# ==========================================
# LOAD MODELS
# ==========================================
xgb_data = joblib.load(XGB_MODEL_PATH)
xgb_model = xgb_data["model"]
scaler = xgb_data["scaler"]

iso_data = joblib.load(ISO_MODEL_PATH)
iso_model = iso_data["model"]

# ==========================================
# PROCESS ZEEK LOG
# ==========================================
def process_zeek_log(file_path):
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        # Extract header from #fields
        for line in lines:
            if line.startswith("#fields"):
                columns = line.strip().split("\t")[1:]
                break

        # Read log
        df = pd.read_csv(file_path, sep="\t", comment="#", names=columns)

        # Select required features
        df = df[[
            "duration",
            "orig_bytes",
            "resp_bytes",
            "orig_pkts",
            "resp_pkts",
            "id.resp_p"
        ]]

        # Clean data
        df = df.fillna(0)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Feature Engineering
        df["total_bytes"] = df["orig_bytes"] + df["resp_bytes"]
        df["total_packets"] = df["orig_pkts"] + df["resp_pkts"]

        return df

    except Exception as e:
        print("Error processing Zeek log:", e)
        return None

# ==========================================
# DETECTION ENGINE
# ==========================================
def detect_intrusion(df):
    if df is None or df.empty:
        return

    # Scale features
    X_scaled = scaler.transform(df)

    # -------------------------------
    # Predictions
    # -------------------------------
    xgb_pred = xgb_model.predict(X_scaled)

    iso_scores = iso_model.decision_function(X_scaled)
    iso_pred = (iso_scores < -0.05).astype(int)

    df["XGB_Pred"] = xgb_pred
    df["ISO_Pred"] = iso_pred

    # -------------------------------
    # ANALYSIS + OUTPUT
    # -------------------------------
    for i, row in df.iterrows():

        # 🚨 ATTACK
        if row["XGB_Pred"] == 1:
            print("🚨 ATTACK DETECTED (XGBoost)")

            reasons = []

            # Heuristic reasons (based on your features)
            if row["duration"] > 60:
                reasons.append("Long connection duration")

            if row["orig_bytes"] == 0 and row["resp_bytes"] > 10000:
                reasons.append("Unusual one-way traffic (server heavy)")

            if row["total_packets"] > 100:
                reasons.append("High packet count")

            if row["id.resp_p"] not in [80, 443, 53]:
                reasons.append(f"Uncommon destination port ({row['id.resp_p']})")

            if row["total_bytes"] > 50000:
                reasons.append("High data transfer volume")

            # Print reasons
            if reasons:
                print("🔍 Reason(s):")
                for r in reasons:
                    print(f" - {r}")
            else:
                print("🔍 Reason: Pattern matched learned attack behavior")

            print(row)
            print("-" * 50)

        # ⚠️ SUSPICIOUS
        elif row["ISO_Pred"] == 1:
            print("⚠️ SUSPICIOUS TRAFFIC (Isolation Forest)")
            print(f"🔍 Anomaly Score: {iso_scores[i]:.4f}")

        # ✅ NORMAL
        else:
            print("✅ NORMAL TRAFFIC")

    # ==========================================
    # ALERT SYSTEM
    # ==========================================
    for i, row in df.iterrows():

        if row["XGB_Pred"] == 1:
            print("🚨 ATTACK DETECTED (XGBoost)")
            print(row)
            print("-" * 50)

        elif row["ISO_Pred"] == 1:
            print("⚠️ Suspicious Traffic (Isolation Forest)")

    return df

# ==========================================
# REAL-TIME MONITOR LOOP
# ==========================================
def monitor():
    print("🚀 Real-Time Hybrid IDS Started...")

    last_position = 0

    try:
        while True:

            with open(ZEEK_LOG, "r") as f:
                f.seek(last_position)   # go to last read point
                new_lines = f.readlines()
                last_position = f.tell()

            if new_lines:
                print(f"📡 New records: {len(new_lines)}")

                # Extract header
                columns = None
                with open(ZEEK_LOG, "r") as f:
                    for line in f:
                        if line.startswith("#fields"):
                            columns = line.strip().split("\t")[1:]
                            break

                # Convert only new lines to DataFrame
                df = pd.DataFrame(
                    [line.strip().split("\t") for line in new_lines if not line.startswith("#")],
                    columns=columns
                )

                if not df.empty:
                    df = process_zeek_log_from_df(df)   # NEW function
                    detect_intrusion(df)

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n🛑 IDS stopped")

def process_zeek_log_from_df(df):
    try:
        df = df[[
            "duration",
            "orig_bytes",
            "resp_bytes",
            "orig_pkts",
            "resp_pkts",
            "id.resp_p"
        ]]

        df = df.fillna(0)
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

        df["total_bytes"] = df["orig_bytes"] + df["resp_bytes"]
        df["total_packets"] = df["orig_pkts"] + df["resp_pkts"]

        return df

    except Exception as e:
        print("Processing error:", e)
        return None
# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    monitor()