# 🏥 Pulse-FL Backend: FastAPI Coordinator & Federated Aggregator

The Pulse-FL backend is an asynchronous Python service that manages federated learning aggregation, monitors real-time wearable telemetry streams, filters false alerts using activity-aware suppression logic, and requests generative triage advice from a local Ollama model.

---

## 🛠️ Technology Stack & Architecture

* **Framework**: FastAPI (Asynchronous REST & WebSocket server)
* **ML & Compilation**: PyTorch 2.x (with custom `ECGNet` architecture, EXIR export tracing, and `.pte` fallback compiling)
* **Database**: SQLModel ORM (backed by a local SQLite instance at `apps/storage/pulse_fl.db`)
* **Dependency Engine**: Managed using **uv** for reproducible, ultra-fast virtual environments.

### Design Patterns Implemented
1. **Singleton Pattern**:
   * `DatabaseConnectionManager` (manages connections pool/session creation)
   * `websocket_manager` (tracks and broadcasts JSON packets to active client web sockets)
2. **Factory Pattern**:
   * `ModelFactory` (centralizes instantiation and initialization of the PyTorch `ECGNet` model structures)
3. **Strategy Pattern**:
   * `AggregationStrategy` (defines abstract interface for mathematical federated aggregation, concretely implemented by `FedAvgStrategy`)
4. **Repository Pattern**:
   * Separate repo classes (`ClientRepository`, `RoundRepository`, `ContributionRepository`, `AlertRepository`) encapsulate database transaction queries.

---

## 🚀 Installation & Setup

### 1. Synchronize Python Environment
Navigate to the backend directory and synchronize packages using `uv`:
```bash
cd apps/backend
uv sync
```
This automatically establishes a localized virtual environment at `.venv` containing all dependencies (`fastapi`, `torch`, `sqlmodel`, `websockets`, `httpx`, etc.).

### 2. Configure Environment (`.env`)
Create a local config file from the template:
```bash
cp example.env .env
```
Update configuration parameters as required:
```ini
# Database Path (SQLite default)
DATABASE_URL=sqlite:///../storage/pulse_fl.db

# Federated Learning Orchestration Parameters
MIN_PARTICIPANTS_REQUIRED=3
GLOBAL_MODELS_DIR=../storage/global_models
CLIENT_UPDATES_DIR=../storage/client_updates

# Ollama AI Triage Configuration
OLLAMA_API_URL=http://localhost:11434

# Emergency SMTP Mailer Setup
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_username@gmail.com
SMTP_PASSWORD=your_app_password
SENDER_EMAIL=triage_coordinator@pulsefl.com
```

---

## ⚡ Task Commands & Execution

The project provides direct task runners inside `apps/backend/justfile`:

### 1. Launch FastAPI Coordinator Server
Runs the web dashboard, database seeding logic, and contribution endpoints:
```bash
just run-local
```
* **Dashboard Portal**: http://127.0.0.1:8000/
* **Interactive API Reference Docs**: http://127.0.0.1:8000/docs

### 2. Run WebSocket Telemetry Simulation
Simulates continuous patient cardiac telemetry stream frames over a live socket connection to verify anomaly thresholds and activity state alert suppressions:
```bash
just test-signals
```

### 3. Run Federated Learning Client Trainer
Simulates a client participating in the federated averaging round (downloads model weights, runs 3 epochs of training on synthetic ECG signals, and uploads parameters):
```bash
# Simulates a unique device (automatically registers the ID)
uv run scripts/simulate_client.py --client-id device_watch_01 --samples 45 --epochs 3
```

---

## 📡 Core API Reference

### REST Endpoints
* **`POST /api/clients/register`**: Registers wearable devices (takes `device_id`, `device_model`, and `emergency_email`).
* **`GET /api/rounds/active`**: Returns metadata for the currently open federated learning round.
* **`GET /api/rounds/download?format={safetensors|pte}`**: Downloads the active global weight program.
* **`POST /api/rounds/upload`**: Receives client contribution parameters (receives multipart form metadata and the weight binary file).

### WebSocket Telemetry Protocol
* **`WS /api/signals/stream/{client_id}`**: Accepts a continuous stream of JSON telemetry frames.
  * **Input Payload Format**:
    ```json
    {
      "values": [0.1, -0.05, 0.22, ...],  // Array of 1D ECG lead signal readings
      "activity_state": "STATIONARY"       // Patient state: STATIONARY, WALKING, RUNNING, EXERCISING
    }
    ```
  * **Suppression Logic**:
    * If `activity_state` is active (e.g. `EXERCISING`), any detected cardiac anomaly (arrhythmia) is logged but **suppressed** to avoid false triggers from motion artifacts or high heart rate.
    * If `activity_state` is `STATIONARY`, a confirmed anomaly triggers an immediate database `AnomalyAlert` entry, spawning an asynchronous task to fetch clinician triage.

---

## 🧠 Local Ollama & Gemma-4 Setup

To enable the generative clinician report summaries:

1. Install Ollama on your system ([ollama.com](https://ollama.com)).
2. Pull the latest Gemma model locally:
   ```bash
   ollama pull gemma4:latest
   ```
3. Boot the Ollama background service (typically runs on `http://localhost:11434`).
4. If Ollama is unavailable, the server gracefully logs emergency alerts and outputs warnings to `sent_emails.log` with a placeholder description.
