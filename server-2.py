import socket
import pickle
import os
import numpy as np
from datetime import datetime
import time

# --- CONFIGURATION ---
ADMIN_IP = '10.216.233.25'  # Listens on all available network interfaces
PORT = 5005
RECEIVED_DIR = "received_updates"

if not os.path.exists(RECEIVED_DIR):
    os.makedirs(RECEIVED_DIR)

def start_server():
    # Setup socket with REUSEADDR to prevent "Address already in use" errors
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server.bind((ADMIN_IP, PORT))
        server.listen(5)
        print(f"🚀 ARV Federated Server Online | Port: {PORT}")
        print("Waiting for edge node mathematical signals...")
        print("--------------------------------------------------")

        while True:
            client, addr = server.accept()
            print(f"📡 [HANDSHAKE] Incoming knowledge from: {addr[0]}")

            payload = b""
            while True:
                packet = client.recv(4096)
                if not packet: break
                payload += packet
            
            if payload:
                try:
                    # 1. Extract the weights (Mathematical Knowledge)
                    weights = pickle.loads(payload)
                    print(f"📥 [SUCCESS] Data Packet Received: {len(payload)} bytes.")

                    # --- 2. MATHEMATICAL OUTPUT SECTION ---
                    # Convert to numpy array to show the internal math
                    math_knowledge = np.array(weights).flatten()
                    
                    print(f"\n🧠 [KNOWLEDGE EXTRACTION]")
                    print(f"   > Vector Dimension: {math_knowledge.shape}")
                    print(f"   > Sample Gradients: {math_knowledge[:5]}...") # Shows the first 5 weights
                    print(f"   > Mean Weight Value: {np.mean(math_knowledge):.6f}")
                    
                    # 3. Simulate the Federated Learning (FL) Aggregation
                    print(f"⚙️  [FL ALGORITHM] Merging local knowledge into Global Brain...")
                    time.sleep(1) # Visual delay for the terminal
                    print(f"✅ [SYNC] Global Model updated with node {addr[0]} contributions.")

                    # 4. Persistence for the Dashboard
                    timestamp = datetime.now().strftime("%H%M%S")
                    filename = f"update_{addr[0]}_{timestamp}.pkl"
                    with open(os.path.join(RECEIVED_DIR, filename), "wb") as f:
                        pickle.dump(weights, f)
                    
                    # Update status for UI
                    with open("online_nodes.txt", "a") as log:
                        log.write(f"{addr[0]},{datetime.now().strftime('%H:%M:%S')},Knowledge Shared\n")

                except Exception as e:
                    print(f"⚠️  Data Corruption Error: {e}")
            
            client.close()
            print(f"--------------------------------------------------\n")

    except Exception as e:
        print(f"❌ Critical Server Error: {e}")

if __name__ == "__main__":
    start_server()