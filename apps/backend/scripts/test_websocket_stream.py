import asyncio
import json
import websockets
import sys
import urllib.request
import urllib.parse
from pathlib import Path

# Add project root to path to resolve local imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from scripts.generate_mock_data import generate_synthetic_heartbeat

def register_device(client_id: str, email: str):
    url = "http://127.0.0.1:8000/api/clients/register"
    print(f"Registering device '{client_id}' with emergency email '{email}'...")
    data = urllib.parse.urlencode({
        "device_id": client_id,
        "device_model": "WearableSensorBand",
        "emergency_email": email
    }).encode("utf-8")
    
    req = urllib.request.Request(url, data=data)
    try:
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            print(f"Registration response: {res}")
    except Exception as e:
        print(f"Device registration failed: {e}")

async def stream_ecg():
    client_id = "device_watch_test"
    register_device(client_id, "guardian_test@example.com")
    
    url = f"ws://127.0.0.1:8000/api/signals/stream/{client_id}"
    print(f"Connecting to WebSocket endpoint: {url}")
    try:
        async with websockets.connect(url) as websocket:
            print("Connected! Streaming simulated ECG heartbeat data...")
            
            # Generate normal rhythm (0) and arrhythmia (1) waveforms
            normal_wave = generate_synthetic_heartbeat(sequence_length=1000, label=0)
            arrhythmia_wave = generate_synthetic_heartbeat(sequence_length=1000, label=1)
            
            # 1. Stream normal rhythm while STATIONARY
            print("\n--- 1. Streaming Sinus Rhythm (STATIONARY) ---")
            for i in range(0, 1000, 100):
                block = normal_wave[i:i+100].tolist()
                await websocket.send(json.dumps({
                    "values": block,
                    "activity_state": "STATIONARY"
                }))
                response = await websocket.recv()
                data = json.loads(response)
                conf = data.get('confidence')
                conf_str = f"{conf * 100:.1f}%" if conf is not None else "N/A"
                print(f"Server Response: {data.get('message')} | HR: {data.get('heart_rate')} BPM | Conf: {conf_str}")
                await asyncio.sleep(0.1)
                
            # 2. Stream arrhythmia while EXERCISING (suppression check)
            print("\n--- 2. Streaming Arrhythmia Anomaly during EXERCISING (Suppression Check) ---")
            for i in range(0, 1000, 100):
                block = arrhythmia_wave[i:i+100].tolist()
                await websocket.send(json.dumps({
                    "values": block,
                    "activity_state": "EXERCISING"
                }))
                response = await websocket.recv()
                data = json.loads(response)
                conf = data.get('confidence')
                conf_str = f"{conf * 100:.1f}%" if conf is not None else "N/A"
                print(f"Server Response: {data.get('message')} | HR: {data.get('heart_rate')} BPM | Conf: {conf_str}")
                await asyncio.sleep(0.1)

            # 3. Stream arrhythmia while STATIONARY (trigger check)
            print("\n--- 3. Streaming Arrhythmia Anomaly during STATIONARY (Trigger Check) ---")
            for i in range(0, 1000, 100):
                block = arrhythmia_wave[i:i+100].tolist()
                await websocket.send(json.dumps({
                    "values": block,
                    "activity_state": "STATIONARY"
                }))
                response = await websocket.recv()
                data = json.loads(response)
                msg_type = data.get("type")
                conf = data.get('confidence')
                conf_str = f"{conf * 100:.1f}%" if conf is not None else "N/A"
                if msg_type == "ALERT":
                    print(f"🚨 Server ALERT: {data.get('message')} | HR: {data.get('heart_rate')} BPM | CONFIRMATION ID: {data.get('alert_id')}")
                else:
                    print(f"Server Response: {data.get('message')} | HR: {data.get('heart_rate')} BPM | Conf: {conf_str}")
                await asyncio.sleep(0.1)
                
            print("\nStreaming simulation finished. Waiting a few seconds for Gemma-4 report and email log completion...")
            await asyncio.sleep(5)
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(stream_ecg())
