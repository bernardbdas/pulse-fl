import numpy as np
import torch
from typing import Tuple

def generate_synthetic_heartbeat(
    sequence_length: int = 1000, 
    label: int = 0, 
    sample_rate: float = 500.0
) -> np.ndarray:
    """
    Generates a synthetic single-lead ECG waveform.
    
    Parameters:
        sequence_length: Number of time steps.
        label: 0 for Normal Sinus Rhythm (regular R-peaks), 
               1 for Arrhythmia (irregular, erratic spacing, missing P-waves).
        sample_rate: Frequency in Hz.
    """
    t = np.arange(sequence_length) / sample_rate
    ecg = np.zeros(sequence_length)
    
    # Base heart rate (beats per minute)
    bpm = 70 if label == 0 else 115
    heart_rate_freq = bpm / 60.0
    
    # Signal generation parameters
    if label == 0:
        # Normal sinus rhythm: regular intervals
        r_peaks = np.arange(0.2, t[-1], 1.0 / heart_rate_freq)
    else:
        # Arrhythmia: highly irregular intervals (atrial fibrillation simulation)
        r_peaks = []
        curr_time = 0.15
        while curr_time < t[-1]:
            r_peaks.append(curr_time)
            # Add random heart rate variability
            curr_time += np.random.uniform(0.3, 0.7)
        r_peaks = np.array(r_peaks)

    # Render waves: P, Q, R, S, T components using Gaussian pulses
    for peak in r_peaks:
        # R wave: Sharp, tall positive deflection
        r_width = 0.015
        ecg += 1.0 * np.exp(-((t - peak) ** 2) / (2 * r_width ** 2))
        
        # S wave: Small negative deflection right after R
        s_peak = peak + 0.03
        s_width = 0.015
        ecg -= 0.25 * np.exp(-((t - s_peak) ** 2) / (2 * s_width ** 2))
        
        # Q wave: Small negative deflection right before R
        q_peak = peak - 0.02
        q_width = 0.01
        ecg -= 0.15 * np.exp(-((t - q_peak) ** 2) / (2 * q_width ** 2))
        
        if label == 0:
            # Normal: add P-wave (atrial depolarization)
            p_peak = peak - 0.15
            p_width = 0.035
            ecg += 0.12 * np.exp(-((t - p_peak) ** 2) / (2 * p_width ** 2))
            
            # Normal: add T-wave (ventricular repolarization)
            t_peak = peak + 0.22
            t_width = 0.065
            ecg += 0.25 * np.exp(-((t - t_peak) ** 2) / (2 * t_width ** 2))
        else:
            # Arrhythmia (AFib): No P-wave, instead continuous high-frequency f-waves (fibrillation noise)
            f_noise = 0.08 * np.sin(2 * np.pi * 18 * t)
            ecg += f_noise
            
            # Erratic T-waves
            t_peak = peak + 0.18
            t_width = 0.075
            ecg += np.random.uniform(0.1, 0.3) * np.exp(-((t - t_peak) ** 2) / (2 * t_width ** 2))

    # Add baseline drift and white noise to represent sensor movement/electrode artifacts
    baseline_drift = 0.05 * np.sin(2 * np.pi * 0.15 * t)
    sensor_noise = np.random.normal(0, 0.03, sequence_length)
    
    ecg += baseline_drift + sensor_noise
    
    # Normalize signal between -1 and 1
    max_val = np.max(np.abs(ecg))
    if max_val > 0:
        ecg = ecg / max_val
        
    return ecg

def create_synthetic_dataset(num_samples: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a mock local database of patient ECG signal readings.
    Returns PyTorch tensors for signals (X) and labels (Y).
    """
    X = []
    Y = []
    for _ in range(num_samples):
        label = np.random.choice([0, 1])
        signal = generate_synthetic_heartbeat(sequence_length=1000, label=label)
        X.append(signal)
        Y.append(label)
        
    # Format: [num_samples, channels (1), sequence_length (1000)]
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(1)
    Y_tensor = torch.tensor(np.array(Y), dtype=torch.long)
    
    return X_tensor, Y_tensor
