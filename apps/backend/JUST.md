# Developer Tasks Documentation (justfile)

This project uses `just` as a command runner. The `justfile` at the root of the project defines shortcuts for local development, container orchestration, and signal stream simulation.

---

## Prerequisites

Ensure you have the following installed:
- `just` (command runner)
- `uv` (modern Python package manager)
- `podman` and `podman-compose` (for container execution)

---

## Available Commands

### 1. Local Development
Runs the coordinator server locally on `http://127.0.0.1:8000` using a local SQLite database file.
```bash
just run-local
```

### 2. Container Orchestration (Podman)
Brings up the orchestrated container services (PostgreSQL database and the FastAPI backend) in detached mode, automatically compiling the local Python package.
```bash
just up
```

To stop and tear down the running containers:
```bash
just down
```

To view live aggregated logs from all running containers:
```bash
just logs
```

### 3. Real-Time Signal Stream Testing
Executes a simulated client that connects to the server's WebSocket endpoint (`ws://127.0.0.1:8000/api/signals/stream/device_watch_test`). It streams a series of normal sinus heartbeats followed by arrhythmia anomalies to test real-time classification and Gemma-4 clinical report trigger logic.
```bash
just test-signals
```

### 4. Federated Client Simulation
Runs a single federated learning client simulator that downloads the current global model weights, fine-tunes the network on synthetic samples, and uploads the local updates.
```bash
# Uses default parameters (client_id=device_watch_01, samples=40, epochs=3)
just simulate-client

# Override parameters
just simulate-client device_watch_02 60 5
```

### 5. Cleaning Workspace
Cleans up temporary state files, federated round weights, and clears the local SQLite database.
```bash
just clean
```
