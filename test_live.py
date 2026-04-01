import requests

# Pointing to your local server instead of Hugging Face!
BASE_URL = "http://127.0.0.1:7860"

print(f"🚀 Testing Local Server: {BASE_URL}\n")

# 1. Test the Root Ping (Must be 200 OK)
try:
    print("📡 Pinging Root (/) ...")
    res = requests.get(f"{BASE_URL}/")
    print(f"Status: {res.status_code} | Response: {res.json()}\n")
except Exception as e:
    print(f"❌ Root Ping Failed: {e}\n")

# 2. Test the Reset Endpoint
try:
    print("🔄 Pinging Reset (/reset) ...")
    res = requests.post(f"{BASE_URL}/reset")
    print(f"Status: {res.status_code} | Has Observation: {'observation' in res.json()}\n")
except Exception as e:
    print(f"❌ Reset Ping Failed: {e}\n")

# 3. Test the Step Endpoint
try:
    print("👟 Pinging Step (/step) ...")
    payload = {"actions": [{"op": "set_contrast", "node_id": "test", "value": 5.0}]}
    res = requests.post(f"{BASE_URL}/step", json=payload)
    data = res.json()
    print(f"Status: {res.status_code} | Reward: {data.get('reward')} | Terminated: {data.get('terminated')}\n")
except Exception as e:
    print(f"❌ Step Ping Failed: {e}\n")

print("🏁 Testing Complete!")