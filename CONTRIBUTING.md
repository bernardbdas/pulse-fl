# Contributing to Pulse-FL

Thank you for your interest in contributing to Pulse-FL! We welcome contributions to enhance edge signal processing, federated learning aggregations, and local clinical triage integrations.

---

## 🛠️ Local Development Setup

We manage our workspace dependencies and virtual environments with `uv`. Follow these steps to prepare your local machine:

1. **Install uv**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and Synchronize**:
   ```bash
   git clone <repository-url>
   cd pulse-fl
   uv sync
   ```

3. **Configure Environment Variables**:
   ```bash
   cp example.env .env
   ```
   Edit `.env` as required (e.g. enabling local Ollama API gateway or custom database connection strings).

---

## 📐 Extending Code Capabilities

Our codebase relies on clear, documented design patterns. When adding functionality, follow these structures:

### Adding a New Model
1. Declare your PyTorch network architecture inside `src/pulse_fl/models/`.
2. Register your new model type inside the `ModelFactory` in `src/pulse_fl/models/factory.py`.
3. Update `export_executorch.py` to ensure correct EXIR tracing and `.pte` compilation.

### Adding an Aggregation Strategy
1. Create a concrete class implementing the `AggregationStrategy` interface in `src/pulse_fl/aggregation/strategy.py`.
2. Code your mathematical aggregation updates (e.g., Federated Proximal `FedProx`, Federated Adam).
3. Inject the new strategy inside the API contribution endpoints.

---

## 🧪 Testing Your Changes

Before submitting a Pull Request, ensure that all telemetry pipelines function correctly:

1. **Lint and Format**:
   Validate code styles:
   ```bash
   uv run ruff check src/
   ```

2. **Verify Signal Telemetry**:
   Run the coordinator server and execute the simulation script:
   ```bash
   # Terminal 1
   just run-local
   
   # Terminal 2
   just test-signals
   ```
   Confirm that Phase 1 (Sinus), Phase 2 (Suppressed Arrhythmia), and Phase 3 (Active Arrhythmia Alert) all evaluate correctly.

---

## 📥 Pull Request Guidelines

1. Fork the repository and create a descriptive feature branch (e.g., `feature/fed-prox-strategy`).
2. Write unit tests for any new algorithms or repositories.
3. Commit with concise, descriptive message headers.
4. Open a Pull Request detailing the changes, the tests run, and the expected performance impact.
