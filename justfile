# Root justfile - Pulse-FL Orchestrator
set dotenv-load := true

# List all available task recipes
default:
    @just --list

# Initialize virtual environments and install all dependencies
setup:
    uv sync --project apps/backend
    npm install

# Launch both local development servers concurrently (FastAPI + Expo Metro)
dev:
    npm run dev

# Spin up orchestrated Docker containers (PostgreDB, FastAPI Server, Nginx Web Client)
docker-up:
    docker compose up --build -d

# Shut down all orchestrated Docker containers and clean volume database caches
docker-down:
    docker compose down -v

# Run the WebSocket continuous lead-1 ECG stream telemetry simulator
test-signals:
    cd apps/backend && uv run scripts/test_websocket_stream.py

# Run on-device federated learning round simulated gradient descent client
simulate-client client_id="device_sim" samples="50" epochs="3":
    cd apps/backend && uv run scripts/simulate_client.py --client-id {{client_id}} --samples {{samples}} --epochs {{epochs}}
