---
name: pulse-fl-skill
description: Use this skill when managing the Pulse-FL monorepo, coordinating federated learning rounds, running telemetry simulators, adjusting safetensors serializers, or deploying docker containers.
---

# Pulse-FL Monorepo Developer Skill Guide

This skill provides workspace-specific guidelines, patterns, and commands to maintain, verify, and extend the Pulse-FL clinical wearable intelligence framework.

---

## 📂 Codebase Workspaces

* **Backend Server (`apps/backend/`)**: FastAPI, SQLModel (SQLite), PyTorch EXIR tracing.
* **Mobile Client (`apps/mobile/`)**: React Native, Expo, custom ExecuTorch native modules.

---

## 📐 Core Architecture & Design Patterns

Ensure all new features conform to the following core object designs:
1. **Singleton Database & Sockets**:
   * `DatabaseConnectionManager` (manages connections pool/session creation)
   * `websocket_manager` (tracks and broadcasts JSON packets to active client web sockets)
2. **Factory Neural Net Instantiations**:
   * Use `ModelFactory` in `models/factory.py` to instantiate and initialize the PyTorch `ECGNet` model structure. Do not instantiate `ECGNet` directly in server APIs.
3. **Strategy Federated Averaging**:
   * Mathematically isolate weight updates. Inherit from `AggregationStrategy` inside `apps/backend/src/pulse_fl/aggregation/strategy.py`.
4. **Repository Data Transactions**:
   * Direct database CRUD operations must be encapsulated in their corresponding repository classes (`ClientRepository`, `RoundRepository`, `ContributionRepository`, `AlertRepository`). Do not execute session statements directly inside API routers.

---

## ⚡ Task Commands

Always execute these tasks from the root folder [pulse-fl](file:///home/galahad/.gemini/antigravity/scratch/pulse-fl):

### 1. Project Setup
Synchronizes the backend python packages via `uv` and installs client dependencies:
```bash
npm run setup
```

### 2. Concurrent Dev Servers
Starts both Uvicorn and Expo Metro bundler concurrently:
```bash
npm run dev
```

### 3. Verify Telemetry Streams
Tests the WebSocket ECG stream window buffers, activity-aware suppressions, and Ollama Gemma triage advice triggers:
```bash
cd apps/backend
uv run scripts/test_websocket_stream.py
```

### 4. Verify Federated Rounds
Simulates client registrations, weights downloads, training, and contribution uploads:
```bash
cd apps/backend
uv run scripts/simulate_client.py --client-id device_01 --samples 50 --epochs 3
```

---

## 🔒 Security & Safety Guidelines

* **Pickle-Free Weights Serialization**: Never use Python `pickle` (e.g. `torch.load` / `torch.save` without safety checks) for remote weight sharing. Always parse, validate, and serialize model variables in `.safetensors` formats.
* **Activity-Aware Suppression**: Continuous patient cardiac alerts must be suppressed if `activity_state` indicates movement (e.g. `WALKING`, `RUNNING`, `EXERCISING`) to filter signal noise.
* **Generative Triage Fallback**: Maintain clean, non-blocking error handling around the local Ollama service (`http://localhost:11434` requesting `gemma4:latest`). Fall back to writing plain-text emergency summaries to `apps/storage/sent_emails.log` if the model or SMTP network is offline.
* **Authorship**: All authorship and copyright references across source files, podspecs, and licenses must be assigned to `@bernardbdas` on GitHub.
