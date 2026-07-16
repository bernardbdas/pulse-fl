# 📱 Pulse-FL Mobile Node: Expo React Native & ExecuTorch Edge Client

This is the mobile client codebase for Pulse-FL, a cross-platform React Native Expo application designed to execute on-device federated learning (FL) fine-tuning and real-time arrhythmia classification using ExecuTorch.

---

## ✨ Features & Edge Innovations

1. **Pickle-Free Safe Weight Serialization**:
   * Traditional PyTorch models use python serialization (`pickle`), which presents critical remote code execution vulnerabilities on both server and client layers.
   * This client reads, parses, and serializes model weights using the raw binary **Safetensors** file structure directly in memory (zero pickle execution).
2. **On-Device Backpropagation Fallback Engine**:
   * Provides a fully functional, pure TypeScript backpropagation execution path (`src/services/executorch.ts`) that runs gradient descent on the final fully connected layers (`fc1`, `fc2`).
   * Extracts feature representations dynamically through Conv1D and Adaptive Average Pooling, calculates Cross-Entropy Loss, runs SGD weight revisions, and writes out a binary safetensors array.
3. **ExecuTorch Native C++ Bridge Skeleton**:
   * Pre-scaffolded Expo Native Module (`modules/pulse-fl-executorch`) providing Swift/Kotlin wrapper boundaries to link C++ ExecuTorch libraries (`.pte`) on physical device builds.
4. **Dark Mode Clinical Dashboard**:
   * A premium dark-mode interface designed with clear status cards for Coordinator Connections, Active Round Synchronization, Local Wearable Training, and weight updates upload logs.

---

## 🚀 Setup & Installation

### Prerequisites
* **Node.js**: Version 18+ (with `npm` package manager)
* **CocoaPods** (for iOS emulator builds)
* **Android SDK** (for Android emulator builds)

### 1. Install Dependencies
Synchronize local Node packages:
```bash
cd apps/mobile
npm install
```

### 2. Check TypeScript Types
Validate compile-time type-safety across all components:
```bash
npm run ts:check
```

---

## ⚡ Running the Client

Start the Metro Bundler:
```bash
npm run start
```

* Press **`i`** to boot the application on the iOS Simulator.
* Press **`a`** to boot the application on the Android Emulator.
* Scan the **QR Code** displayed in the terminal using the Expo Go application to test on a physical iOS or Android device.

---

## 📐 On-Device Training & Serialization Details

### 1. Safetensors Binary Parsing (Simulator Fallback)
When running in simulator/fallback mode, the app interacts with downloaded weights directly:
* Reads model files using `expo-file-system` in **Base64** format.
* Extracts the 64-bit Little-Endian header size, parses the JSON tensor shape/metadata offsets, and reads raw byte values into `Float32Array` buffers.
* Overwrites updated float data dynamically in the buffer and serializes the array back to Base64 to construct a valid `.safetensors` file.

### 2. JS Backpropagation Math
On-device fine-tuning executes real gradients on patient data:
* **Feature Extraction**: Passes 1D patient ECG arrays (shape `[1, 1000]`) through the CNN layers (Conv1D + BatchNorm1D + ReLU + MaxPool1D) to output a 64-dimensional feature array.
* **Backpropagation**:
  $$\text{Loss} = -\log(P_{\text{target}})$$
  Runs standard feedforward and backpropagation on fully connected layers:
  * FC1 Layer: $32 \times 64$ weights
  * FC2 Layer: $2 \times 32$ weights
  * Learns local patient signal traits dynamically via SGD (`learningRate = 0.05`) across 3 epochs.

---

## 🔌 C++ Native Module Integration (Physical Build)

To build the client with the actual ExecuTorch C++ runtime on devices:

1. **iOS (`modules/pulse-fl-executorch/ios/`)**:
   * The custom native podspec [PulseFlExecuTorch.podspec](file:///home/galahad/.gemini/antigravity/scratch/pulse-fl/apps/mobile/modules/pulse-fl-executorch/ios/PulseFlExecuTorch.podspec) handles module configurations.
   * Link your compiled C++ ExecuTorch static libraries or add dependencies like `s.dependency 'executorch-cpp-framework'` directly in the spec.
2. **Android (`modules/pulse-fl-executorch/android/`)**:
   * Configure `build.gradle` to compile the standard C++ runtime JNI wrapper.
3. Run `npx expo prebuild` to eject from Expo Go and compile the native binary wrappers for target devices.
