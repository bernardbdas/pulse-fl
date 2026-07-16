# Security Policy

## Supported Versions

We actively maintain and issue security patches for the following versions of Pulse-FL:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

---

## 🔒 Safe Weights Deserialization (Safetensors)

In Federated Learning architectures, clients submit updated weights to the server. Under standard PyTorch workflows, weights are saved via `torch.save` which relies on Python's native `pickle` library. This introduces a major vulnerability where a compromised client can upload weights containing malicious bytecode execution.

To guarantee server security:
* **Strict Safetensors Execution**: Pulse-FL **only** accepts client updates in the `.safetensors` format. Safetensors prevents execution of arbitrary Python objects by strictly parsing only raw tensor buffer arrays.
* **Hash Verification**: In production networks, updates should be signed with private client keys to ensure data provenance and prevent tampering.

---

## 🐛 Reporting a Vulnerability

If you discover a security vulnerability in the Pulse-FL platform, please report it following these guidelines:

1. **Do Not Open a Public Issue**: To prevent exposing users to risks, do not open public GitHub issues or discuss vulnerabilities in public forums.
2. **Email Disclosure**: Send a detailed description of the vulnerability, including step-by-step reproduction instructions or a proof-of-concept script, to the maintainers' security email.
3. **Response Timeline**: We will acknowledge receipt of your report within 48 hours and work with you to analyze and patch the vulnerability.
4. **Coordinated Disclosure**: Once a patch is released, we will publish a security advisory and credit you for the discovery.
