# Pulse-FL Project Customization Rules

These rules apply project-scoped instructions and styling guidelines to all agents working in the Pulse-FL workspace.

---

## 🔒 Security & Safe Deserialization
1. **No Pickle Deserialization**: All models shared between server and client must use the `.safetensors` format (using the raw binary parsers in JS and safetensors library in Python). Python `pickle` deserialization is strictly prohibited.
2. **Activity-Aware Suppressors**: Arrhythmia alerts generated via continuous Wearable WebSocket streams must be suppressed if the activity state indicates movement (e.g. `WALKING`, `RUNNING`, `EXERCISING`).

---

## ⚖️ Code Authorship & Project Naming
1. **GitHub Authorship**: All authorship, copyright lines, and code credits across package configurations (e.g., CocoaPods `.podspec` files, licenses, README files) must be assigned to `@bernardbdas` on GitHub.
2. **No Legacy Names**: Do not reference the standalone legacy directory names (such as `pulse-fl-client`) in documentation or configurations. The workspace structure is divided into `apps/backend/` and `apps/mobile/`.

---

## 🛠️ Workspaces & Script Orchestration
1. **Monorepo Workspaces**: The codebase is a Node monorepo. Launch tasks concurrently using root scripts (`npm run dev`) rather than booting servers separately.
2. **Virtualenvs**: Python dependencies are managed via `uv` in `apps/backend/.venv`. Never run global `pip install` commands. Use `npm run setup` to synchronize all projects concurrently.
