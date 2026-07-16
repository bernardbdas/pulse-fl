import os
import sys
import json
import uuid
import argparse
import urllib.request
import urllib.error
from pathlib import Path

# Add project root to path to resolve local imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import save_file, load_file

from better_pulse.models.ecg_net import ECGNet
from scripts.generate_mock_data import create_synthetic_dataset

def get_json_response(url: str, data: bytes = None) -> dict:
    """Helper to perform GET/POST requests and decode JSON output."""
    req = urllib.request.Request(url, data=data)
    if data:
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"HTTP error: {e.code} - {e.read().decode('utf-8')}")
        raise e

def send_multipart_upload(url: str, fields: dict, file_path: Path) -> dict:
    """Zero-dependency multipart/form-data upload using standard library urllib."""
    boundary = uuid.uuid4().hex
    body = bytearray()
    
    # Write text fields
    for key, val in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        body.extend(f"{val}\r\n".encode("utf-8"))
        
    # Write file field
    file_name = file_path.name
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{file_name}"\r\n'.encode("utf-8"))
    body.extend(b"Content-Type: application/octet-stream\r\n\r\n")
    
    with open(file_path, "rb") as f:
        body.extend(f.read())
        
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    
    req = urllib.request.Request(url, data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", str(len(body)))
    
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"Upload failed: {e.code} - {e.read().decode('utf-8')}")
        raise e

def simulate_client(client_id: str, server_url: str, samples_count: int, epochs: int):
    print(f"\n=== Simulating Client: {client_id} ===")
    
    # 1. Register Client with the backend
    register_url = f"{server_url}/api/clients/register"
    print(f"Registering client device at {register_url}...")
    register_data = urllib.parse.urlencode({
        "device_id": client_id,
        "device_model": "AppleWatchS9"
    }).encode("utf-8")
    
    reg_resp = get_json_response(register_url, data=register_data)
    print(f"Registration response: {reg_resp}")
    
    # 2. Query for the active federated learning round
    active_url = f"{server_url}/api/rounds/active"
    print(f"Fetching active FL round from {active_url}...")
    round_info = get_json_response(active_url)
    round_number = round_info["round_number"]
    print(f"Current active round: {round_number} (status: {round_info['status']})")
    
    # 3. Download the global weights file
    download_url = f"{server_url}/api/rounds/download?format=safetensors"
    temp_weights_path = Path(f"/tmp/global_model_r{round_number}_{client_id}.safetensors")
    print(f"Downloading global weights from {download_url} to {temp_weights_path}...")
    
    urllib.request.urlretrieve(download_url, str(temp_weights_path))
    print("Download completed successfully.")
    
    # 4. Load the weights into a local ECGNet model
    print("Loading model and weights...")
    model = ECGNet(input_channels=1, num_classes=2)
    global_weights = load_file(str(temp_weights_path))
    model.load_state_dict(global_weights)
    
    # 5. Generate local synthetic ECG traces for this patient
    print(f"Generating {samples_count} synthetic ECG readings...")
    X, Y = create_synthetic_dataset(num_samples=samples_count)
    
    # 6. Simulate local training / on-device fine-tuning
    print(f"Fine-tuning model on-device for {epochs} epochs...")
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    final_loss = 0.0
    final_accuracy = 0.0
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
        
        # Calculate training accuracy
        _, preds = torch.max(outputs, 1)
        corrects = torch.sum(preds == Y).item()
        final_accuracy = corrects / samples_count
        
        print(f"  Epoch [{epoch+1}/{epochs}] - Loss: {final_loss:.6f} - Accuracy: {final_accuracy * 100:.2f}%")
        
    # 7. Save updated weights to a temporary file
    client_update_path = Path(f"/tmp/client_update_r{round_number}_{client_id}.safetensors")
    save_file(model.state_dict(), str(client_update_path))
    print(f"Local updates saved locally at: {client_update_path}")
    
    # 8. Upload the updated weights to the backend
    upload_url = f"{server_url}/api/rounds/upload"
    print(f"Uploading updates to {upload_url}...")
    upload_fields = {
        "client_id": client_id,
        "round_number": str(round_number),
        "local_loss": f"{final_loss:.6f}",
        "local_accuracy": f"{final_accuracy:.6f}",
        "sample_count": str(samples_count)
    }
    
    upload_resp = send_multipart_upload(upload_url, upload_fields, client_update_path)
    print(f"Server upload response: {upload_resp}")
    
    # Cleanup temp files
    if temp_weights_path.exists():
        temp_weights_path.unlink()
    if client_update_path.exists():
        client_update_path.unlink()
    print("Done simulation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate client participating in Federated Learning.")
    parser.add_argument("--client-id", type=str, default=None, help="Unique identifier for the client device.")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8000", help="URL of the Pulse-FL backend server.")
    parser.add_argument("--samples", type=int, default=50, help="Number of ECG samples in the local dataset.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of local fine-tuning epochs.")
    
    args = parser.parse_args()
    
    cid = args.client_id or f"watch_{uuid.uuid4().hex[:8]}"
    simulate_client(cid, args.server_url, args.samples, args.epochs)
