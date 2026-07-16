import React, { useState, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TextInput,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
  SafeAreaView,
  StatusBar,
} from 'react-native';
import * as SecureStore from 'expo-secure-store';

import { PulseFLAPI, ActiveRoundInfo } from './src/services/api';
import { ExecuTorchService } from './src/services/executorch';
import { generateECGDataset } from './src/utils/ecgGenerator';

const api = new PulseFLAPI();
const executorch = new ExecuTorchService();

export default function App() {
  const [deviceId, setDeviceId] = useState<string>('');
  const [serverUrl, setServerUrl] = useState<string>('http://127.0.0.1:8000');
  const [connectionStatus, setConnectionStatus] = useState<'DISCONNECTED' | 'CONNECTED' | 'ERROR'>('DISCONNECTED');
  
  // FL Round states
  const [activeRound, setActiveRound] = useState<ActiveRoundInfo | null>(null);
  const [localModelUri, setLocalModelUri] = useState<string | null>(null);
  
  // Training states
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainingLoss, setTrainingLoss] = useState<number | null>(null);
  const [trainingAccuracy, setTrainingAccuracy] = useState<number | null>(null);
  const [localWeightsUri, setLocalWeightsUri] = useState<string | null>(null);
  
  // Operation loading states
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [statusMessage, setStatusMessage] = useState<string>('App ready.');

  // Initialize Client Identity
  useEffect(() => {
    async function initIdentity() {
      try {
        let storedId = await SecureStore.getItemAsync('PULSE_FL_CLIENT_ID');
        if (!storedId) {
          storedId = `watch_${Math.random().toString(36).substring(2, 10)}`;
          await SecureStore.setItemAsync('PULSE_FL_CLIENT_ID', storedId);
        }
        setDeviceId(storedId);
      } catch (e) {
        // Fallback for environment running in pure web or simulator without secure store
        const fallbackId = `watch_sim_${Math.random().toString(36).substring(2, 8)}`;
        setDeviceId(fallbackId);
      }
    }
    initIdentity();
  }, []);

  // Update server endpoint URL dynamically
  useEffect(() => {
    api.setServerUrl(serverUrl);
  }, [serverUrl]);

  // Test Server Connection & Fetch active round info
  const handleConnect = async () => {
    setIsLoading(true);
    setStatusMessage('Connecting to coordinator...');
    try {
      // Test register
      await api.registerClient(deviceId, 'ExpoWearableNode');
      // Fetch round
      const roundInfo = await api.getActiveRound();
      setActiveRound(roundInfo);
      setConnectionStatus('CONNECTED');
      setStatusMessage('Connected successfully. Round retrieved.');
    } catch (e: any) {
      setConnectionStatus('ERROR');
      setStatusMessage(`Connection failed: ${e.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Download Global Model
  const handleDownloadModel = async () => {
    if (!activeRound) return;
    setIsLoading(true);
    setStatusMessage(`Downloading round ${activeRound.round_number} weights...`);
    try {
      const uri = await api.downloadModelPTE(activeRound.round_number);
      setLocalModelUri(uri);
      
      // Load model into ExecuTorch binary engine
      await executorch.loadModel(uri);
      
      setStatusMessage(`Model successfully loaded at: ${uri.substring(uri.lastIndexOf('/') + 1)}`);
    } catch (e: any) {
      setStatusMessage(`Download error: ${e.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Run On-Device Local Training / Fine-tuning
  const handleTrainOnDevice = async () => {
    if (!localModelUri || !activeRound) {
      setStatusMessage('Please download the global model first.');
      return;
    }
    
    setIsTraining(true);
    setTrainingLogs(['Preparing local ECG signals...']);
    
    try {
      // Generate synthetic patient ECG readings locally (simulating sensor collection)
      const dataSize = 50;
      const dataset = generateECGDataset(dataSize);
      
      setTrainingLogs(prev => [...prev, `Generated ${dataSize} training signals (shape [1, 1000]).`]);
      setTrainingLogs(prev => [...prev, 'Starting backpropagation iterations...']);

      // Execute on-device training
      // (This will run native training in C++, or modify safetensors via fallback Javascript simulator)
      const trainResult = await executorch.trainModel(
        localModelUri,
        activeRound.round_number,
        dataset.x,
        dataset.y,
        (epoch, loss, accuracy) => {
          setTrainingLogs(prev => [...prev, `Epoch [${epoch}/3] - Loss: ${loss.toFixed(6)} | Acc: ${(accuracy * 100).toFixed(2)}%`]);
        }
      );

      setTrainingLoss(trainResult.localLoss);
      setTrainingAccuracy(trainResult.localAccuracy);
      setLocalWeightsUri(trainResult.weightsFileUri);
      setStatusMessage('Local training completed.');
    } catch (e: any) {
      setStatusMessage(`Training failed: ${e.message}`);
    } finally {
      setIsTraining(false);
    }
  };

  // Upload Local Model Weights Update to Server
  const handleUploadUpdate = async () => {
    if (!localWeightsUri || !activeRound || trainingLoss === null || trainingAccuracy === null) {
      setStatusMessage('No training updates available to upload.');
      return;
    }
    
    setIsLoading(true);
    setStatusMessage('Uploading weights update file to server...');
    try {
      const response = await api.uploadLocalUpdate(
        deviceId,
        activeRound.round_number,
        trainingLoss,
        trainingAccuracy,
        50, // sample count
        localWeightsUri
      );
      
      setStatusMessage(`Upload success: ${response.message}`);
      // Clear updates to prevent double submission
      setLocalWeightsUri(null);
      setTrainingLoss(null);
      setTrainingAccuracy(null);
    } catch (e: any) {
      setStatusMessage(`Upload failed: ${e.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar barStyle="light-content" />
      <ScrollView contentContainerStyle={styles.container}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Pulse-FL Node</Text>
          <Text style={styles.subtitle}>On-Device Wearable ECG Trainer</Text>
        </View>

        {/* Identity & Configuration */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>Client Node Configuration</Text>
          
          <Text style={styles.label}>Node ID (Stored Device Unique Token)</Text>
          <Text style={styles.deviceIdText}>{deviceId || 'Generating...'}</Text>
          
          <Text style={styles.label}>Coordinator Server URL</Text>
          <View style={styles.row}>
            <TextInput
              style={styles.input}
              value={serverUrl}
              onChangeText={setServerUrl}
              placeholder="http://10.0.2.2:8000"
              placeholderTextColor="#64748b"
              autoCapitalize="none"
              autoCorrect={false}
            />
            <TouchableOpacity style={styles.buttonConnect} onPress={handleConnect} disabled={isLoading}>
              <Text style={styles.buttonText}>Connect</Text>
            </TouchableOpacity>
          </View>
          
          <View style={styles.statusBox}>
            <Text style={styles.statusLabel}>Status:</Text>
            <Text style={[
              styles.statusValue,
              connectionStatus === 'CONNECTED' ? styles.statusConnected : 
              connectionStatus === 'ERROR' ? styles.statusError : styles.statusDisconnected
            ]}>
              {connectionStatus}
            </Text>
          </View>
        </View>

        {/* Round Synchronization */}
        {connectionStatus === 'CONNECTED' && activeRound && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Active Federated Round: {activeRound.round_number}</Text>
            
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Round Status:</Text>
              <Text style={styles.detailValue}>{activeRound.status}</Text>
            </View>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Active Participants:</Text>
              <Text style={styles.detailValue}>
                {activeRound.participants_count} / {activeRound.min_participants_required} (req)
              </Text>
            </View>
            
            <TouchableOpacity 
              style={[styles.buttonPrimary, styles.buttonSync]} 
              onPress={handleDownloadModel} 
              disabled={isLoading}
            >
              <Text style={styles.buttonText}>Download Global Model</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* On-Device Local Training */}
        {localModelUri && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>ExecuTorch Local Trainer</Text>
            <Text style={styles.modelUriText}>Active Binary: {localModelUri.substring(localModelUri.lastIndexOf('/') + 1)}</Text>
            
            <TouchableOpacity 
              style={[styles.buttonPrimary, styles.buttonTrain]} 
              onPress={handleTrainOnDevice} 
              disabled={isTraining || isLoading}
            >
              {isTraining ? (
                <ActivityIndicator color="#ffffff" size="small" />
              ) : (
                <Text style={styles.buttonText}>Run Fine-Tuning On-Device</Text>
              )}
            </TouchableOpacity>

            {trainingLogs.length > 0 && (
              <View style={styles.logContainer}>
                <Text style={styles.logTitle}>Training Console Logs</Text>
                {trainingLogs.map((log, index) => (
                  <Text key={index} style={styles.logText}>➔ {log}</Text>
                ))}
              </View>
            )}
          </View>
        )}

        {/* Upload Updates */}
        {localWeightsUri && trainingLoss !== null && trainingAccuracy !== null && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Model Parameters Upload</Text>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Final Local Loss:</Text>
              <Text style={[styles.detailValue, {color: '#06b6d4', fontWeight: 'bold'}]}>
                {trainingLoss.toFixed(6)}
              </Text>
            </View>
            <View style={styles.detailRow}>
              <Text style={styles.detailLabel}>Final Local Accuracy:</Text>
              <Text style={[styles.detailValue, {color: '#10b981', fontWeight: 'bold'}]}>
                {(trainingAccuracy * 100).toFixed(2)}%
              </Text>
            </View>
            
            <TouchableOpacity 
              style={[styles.buttonPrimary, styles.buttonUpload]} 
              onPress={handleUploadUpdate} 
              disabled={isLoading}
            >
              <Text style={styles.buttonText}>Submit Local Weights Update</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* System Message Log */}
        <View style={styles.consoleCard}>
          <Text style={styles.consoleText}>{statusMessage}</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#0f172a',
  },
  container: {
    padding: 20,
    paddingBottom: 40,
  },
  header: {
    marginBottom: 25,
  },
  title: {
    fontSize: 28,
    fontWeight: '800',
    color: '#f8fafc',
    letterSpacing: 0.5,
  },
  subtitle: {
    fontSize: 14,
    color: '#94a3b8',
    marginTop: 4,
  },
  card: {
    backgroundColor: '#1e293b',
    borderRadius: 16,
    padding: 20,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 5,
    elevation: 3,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#f8fafc',
    marginBottom: 15,
  },
  label: {
    fontSize: 12,
    fontWeight: '600',
    color: '#94a3b8',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 6,
  },
  deviceIdText: {
    fontSize: 14,
    fontFamily: 'monospace',
    color: '#6366f1',
    backgroundColor: 'rgba(99, 102, 241, 0.1)',
    padding: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: 'rgba(99, 102, 241, 0.2)',
    marginBottom: 15,
  },
  modelUriText: {
    fontSize: 12,
    fontFamily: 'monospace',
    color: '#06b6d4',
    backgroundColor: 'rgba(6, 182, 212, 0.1)',
    padding: 8,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: 'rgba(6, 182, 212, 0.15)',
    marginBottom: 15,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
  },
  input: {
    flex: 1,
    backgroundColor: '#0f172a',
    borderRadius: 10,
    padding: 12,
    color: '#f8fafc',
    fontSize: 14,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.08)',
    marginRight: 10,
  },
  buttonConnect: {
    backgroundColor: '#6366f1',
    borderRadius: 10,
    paddingVertical: 12,
    paddingHorizontal: 20,
    justifyContent: 'center',
    alignItems: 'center',
  },
  buttonPrimary: {
    borderRadius: 12,
    paddingVertical: 15,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 3,
  },
  buttonSync: {
    backgroundColor: '#06b6d4',
  },
  buttonTrain: {
    backgroundColor: '#8b5cf6',
  },
  buttonUpload: {
    backgroundColor: '#10b981',
  },
  buttonText: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '700',
  },
  statusBox: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#0f172a',
    padding: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.04)',
  },
  statusLabel: {
    fontSize: 13,
    color: '#94a3b8',
    marginRight: 8,
  },
  statusValue: {
    fontSize: 13,
    fontWeight: '700',
  },
  statusConnected: {
    color: '#10b981',
  },
  statusDisconnected: {
    color: '#94a3b8',
  },
  statusError: {
    color: '#ef4444',
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  detailLabel: {
    color: '#94a3b8',
    fontSize: 14,
  },
  detailValue: {
    color: '#f8fafc',
    fontSize: 14,
    fontWeight: '600',
  },
  logContainer: {
    marginTop: 15,
    backgroundColor: '#0f172a',
    borderRadius: 10,
    padding: 15,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
  },
  logTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: '#94a3b8',
    marginBottom: 8,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.05)',
    paddingBottom: 4,
  },
  logText: {
    fontSize: 12,
    fontFamily: 'monospace',
    color: '#cbd5e1',
    lineHeight: 18,
    marginBottom: 4,
  },
  consoleCard: {
    backgroundColor: '#020617',
    padding: 12,
    borderRadius: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#06b6d4',
  },
  consoleText: {
    color: '#38bdf8',
    fontFamily: 'monospace',
    fontSize: 11,
  },
});
