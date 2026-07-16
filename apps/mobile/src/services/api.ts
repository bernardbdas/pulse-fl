import * as FileSystem from 'expo-file-system';

export interface ActiveRoundInfo {
  round_number: number;
  status: string;
  start_time: string;
  participants_count: number;
  min_participants_required: number;
  available_formats: {
    safetensors: boolean;
    executorch_pte: boolean;
  };
}

export class PulseFLAPI {
  private serverUrl: string;

  constructor(serverUrl: string = 'http://127.0.0.1:8000') {
    this.serverUrl = serverUrl;
  }

  setServerUrl(url: string) {
    this.serverUrl = url;
  }

  /**
   * Registers client device with the backend coordinator.
   */
  async registerClient(deviceId: string, deviceModel: string): Promise<{ status: string; device_id: string; message: string }> {
    const params = new URLSearchParams();
    params.append('device_id', deviceId);
    params.append('device_model', deviceModel);

    const response = await fetch(`${this.serverUrl}/api/clients/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: params.toString(),
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`Registration failed: ${response.status} - ${errText}`);
    }
    return response.json();
  }

  /**
   * Queries metadata for the active federated learning round.
   */
  async getActiveRound(): Promise<ActiveRoundInfo> {
    const response = await fetch(`${this.serverUrl}/api/rounds/active`);
    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`Failed to retrieve active round: ${response.status} - ${errText}`);
    }
    return response.json();
  }

  /**
   * Downloads the compiled ExecuTorch .pte file directly to the device file system.
   * Returns the local file system URI path.
   */
  async downloadModelPTE(roundNumber: number): Promise<string> {
    const downloadUrl = `${this.serverUrl}/api/rounds/download?format=pte`;
    const localUri = `${FileSystem.documentDirectory}global_model_round_${roundNumber}.pte`;

    console.log(`Downloading global model from ${downloadUrl} to ${localUri}...`);
    const downloadResult = await FileSystem.downloadAsync(downloadUrl, localUri);
    
    if (downloadResult.status !== 200) {
      throw new Error(`Model download failed. Server responded with status: ${downloadResult.status}`);
    }
    return downloadResult.uri;
  }

  /**
   * Uploads client's updated weights to the backend after local training completes.
   */
  async uploadLocalUpdate(
    clientId: string,
    roundNumber: number,
    localLoss: number,
    localAccuracy: number,
    sampleCount: number,
    weightsFileUri: string
  ): Promise<{ status: string; message: string }> {
    const uploadUrl = `${this.serverUrl}/api/rounds/upload`;

    console.log(`Uploading local updates to ${uploadUrl}...`);
    const uploadResult = await FileSystem.uploadAsync(uploadUrl, weightsFileUri, {
      fieldName: 'file',
      httpMethod: 'POST',
      uploadType: FileSystem.FileSystemUploadType.MULTIPART,
      parameters: {
        client_id: clientId,
        round_number: roundNumber.toString(),
        local_loss: localLoss.toFixed(6),
        local_accuracy: localAccuracy.toFixed(6),
        sample_count: sampleCount.toString(),
      },
    });

    if (uploadResult.status !== 200 && uploadResult.status !== 201) {
      throw new Error(`Model update upload failed with status ${uploadResult.status}: ${uploadResult.body}`);
    }

    return JSON.parse(uploadResult.body);
  }
}
