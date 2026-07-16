import { requireNativeModule } from 'expo-modules-core';

// Loads the native library using the Expo Modules framework.
// Falls back to undefined if the app is running in standard Expo Go.
let PulseFlExecuTorch;
try {
  PulseFlExecuTorch = requireNativeModule('PulseFlExecuTorch');
} catch (e) {
  PulseFlExecuTorch = null;
}

export default PulseFlExecuTorch;
