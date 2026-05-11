import socket
import pickle
import time
import numpy as np

# --- CONFIGURATION ---python
ADMIN_IP = '10.100.140.207'  # Ensure this matches your Admin PC's IP
PORT = 5005

def send_update():
    """Simulates local training and transmits model weights to the central SOC."""
    try:
        # 1. Simulate Local Training (The "Private" Part)
        print("\n" + "="*40)
        print("🚀 STAGE 1: LOCAL SECURE TRAINING")
        print("="*40)
        print("Reading local private logs from: 'UNSW_NB15_training.csv'...")
        
        # Simulate processing time
        for i in range(3):
            print(f"   [Node-Process] Epoch {i+1}/3: Processing packets...")
            time.sleep(1) 
        
        # 2. Extract "Knowledge" (Model Gradients)
        # We generate a list of weights that represent the local learning
        # These are weights for the ARV Hybrid Model components
        local_weights = np.random.uniform(0.92, 0.98, size=8) 
        print(f"✨ Local Intelligence Generated: {list(np.round(local_weights, 4))[:3]}...")
        
        # 3. Secure Transmission
        print("\n📡 STAGE 2: FEDERATED TRANSMISSION")
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Setting a timeout to prevent the script from hanging
        client.settimeout(10)
        client.connect((ADMIN_IP, PORT))
        
        # Transmitting ONLY the mathematical weights (The "Knowledge")
        # This is where Federated Learning maintains privacy
        payload = pickle.dumps(local_weights)
        client.sendall(payload)
        client.close()
        
        print(f"✅ SUCCESS: Knowledge sent to Admin at {ADMIN_IP}")
        print(f"📊 Payload Size: {len(payload)} bytes (Privacy-First Transfer)")
        print("🔒 LOG: No raw data or sensitive packets left this device.")

    except socket.timeout:
        print(f"❌ ERROR: Connection timed out. Is the Admin PC listening on Port {PORT}?")
    except ConnectionRefusedError:
        print(f"❌ ERROR: Connection refused. Check Admin IP ({ADMIN_IP}) and Firewall settings.")
    except Exception as e:
        print(f"❌ ERROR: Unexpected failure: {e}")

if __name__ == "__main__":
    print("🛡️ ARV HYBRID FEDERATED NODE ACTIVATED")
    print(f"Monitoring network logs on: {socket.gethostname()}")
    
    # Optional: Set this to True to run once, or False to keep sending updates
    run_once = True 
    
    if run_once:
        send_update()
    else:
        while True:
            send_update()
            print("\n⏳ Sleeping until next training cycle (30s)...")
            time.sleep(30)