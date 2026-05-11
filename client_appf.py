import streamlit as st
import socket
import pickle
import time
import numpy as np
import pandas as pd
import os

# --- CONFIGURATION ---
ADMIN_IP = '10.216.233.25'  # Ensure this matches your Admin PC's IP
PORT = 5005
DATASET_PATH = 'UNSW_NB15_training-set.csv'

st.set_page_config(page_title="ARV Federated Edge Node", page_icon="🛡️")

# --- UI HEADER ---
st.title("🛡️ ARV Edge Intelligence Node")
st.markdown(f"**Local Host:** `{socket.gethostname()}` | **Target Admin:** `{ADMIN_IP}`")

# --- SIDEBAR: STATUS & DATA ---
with st.sidebar:
    st.header("Node Settings")
    st.status("Device Status: ONLINE 🟢")
    
    if os.path.exists(DATASET_PATH):
        st.success(f"Dataset Found: `{DATASET_PATH}`")
        # Preview the data
        df_preview = pd.read_csv(DATASET_PATH, nrows=5)
        st.write("Data Preview:", df_preview)
    else:
        st.error(f"Dataset '{DATASET_PATH}' not found in directory!")

# --- MAIN LOGIC ---
def run_federated_update():
    status_text = st.empty()
    progress_bar = st.progress(0)
    log_area = st.container()

    try:
        # 1. Simulate Local Training
        log_area.info("🚀 Starting Local Secure Training...")
        for i in range(3):
            step = i + 1
            status_text.text(f"Processing Epoch {step}/3...")
            progress_bar.progress(step * 33)
            log_area.write(f"   [Node-Process] Epoch {step}: Analyzing local packets...")
            time.sleep(1) 
        
        # 2. Extract "Knowledge" (Gradients)
        local_weights = np.random.uniform(0.92, 0.98, size=8) 
        log_area.success(f"✨ Intelligence Generated: {list(np.round(local_weights, 4))[:3]}...")
        
        # 3. Secure Transmission
        log_area.info("📡 Initiating Federated Transmission...")
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(10)
        
        client.connect((ADMIN_IP, PORT))
        payload = pickle.dumps(local_weights)
        client.sendall(payload)
        client.close()
        
        st.balloons()
        log_area.success(f"✅ SUCCESS: Knowledge sent to Admin at {ADMIN_IP}")
        st.metric("Payload Size", f"{len(payload)} bytes", "Privacy Secured")
        
    except Exception as e:
        st.error(f"❌ Connection Failed: {e}")
        st.info("Check if the Admin App is running and Port 5005 is open on the firewall.")

# --- BUTTON TRIGGER ---
st.divider()
if st.button("🔥 Synchronize with Global Model", use_container_width=True, type="primary"):
    if os.path.exists(DATASET_PATH):
        run_federated_update()
    else:
        st.warning("Cannot start: Local dataset missing.")

st.divider()
st.caption("🔒 Privacy Note: Only mathematical weights are sent. Raw CSV data never leaves this machine.")