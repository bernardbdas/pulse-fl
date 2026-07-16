export function generateECGTrace(label: number, sequenceLength: number = 1000, sampleRate: number = 500.0): number[] {
  const t: number[] = [];
  for (let i = 0; i < sequenceLength; i++) {
    t.push(i / sampleRate);
  }

  let ecg = new Array(sequenceLength).fill(0);
  
  // Heart rate in beats per minute
  const bpm = label === 0 ? 70 : 115;
  const heartRateFreq = bpm / 60.0;
  
  const rPeaks: number[] = [];
  if (label === 0) {
    // Normal sinus rhythm: regular R peaks
    let currTime = 0.2;
    while (currTime < t[t.length - 1]) {
      rPeaks.push(currTime);
      currTime += 1.0 / heartRateFreq;
    }
  } else {
    // Arrhythmia (Atrial Fibrillation): irregular spacing
    let currTime = 0.15;
    while (currTime < t[t.length - 1]) {
      rPeaks.push(currTime);
      // Random heart rate intervals
      currTime += 0.3 + Math.random() * 0.4;
    }
  }

  // Draw ECG wave components (P, Q, R, S, T)
  for (const peak of rPeaks) {
    for (let i = 0; i < sequenceLength; i++) {
      const time = t[i];
      
      // R-wave: sharp positive pulse
      const rWidth = 0.015;
      ecg[i] += 1.0 * Math.exp(-Math.pow(time - peak, 2) / (2 * Math.pow(rWidth, 2)));
      
      // S-wave: small negative deflection after R
      const sPeak = peak + 0.03;
      const sWidth = 0.015;
      ecg[i] -= 0.25 * Math.exp(-Math.pow(time - sPeak, 2) / (2 * Math.pow(sWidth, 2)));
      
      // Q-wave: small negative deflection before R
      const qPeak = peak - 0.02;
      const qWidth = 0.01;
      ecg[i] -= 0.15 * Math.exp(-Math.pow(time - qPeak, 2) / (2 * Math.pow(qWidth, 2)));
      
      if (label === 0) {
        // P-wave: small round pulse before QRS
        const pPeak = peak - 0.15;
        const pWidth = 0.035;
        ecg[i] += 0.12 * Math.exp(-Math.pow(time - pPeak, 2) / (2 * Math.pow(pWidth, 2)));
        
        // T-wave: wider round pulse after QRS
        const tPeak = peak + 0.22;
        const tWidth = 0.065;
        ecg[i] += 0.25 * Math.exp(-Math.pow(time - tPeak, 2) / (2 * Math.pow(tWidth, 2)));
      } else {
        // Fibrillation f-waves: high frequency baseline oscillations
        const fNoise = 0.08 * Math.sin(2 * Math.PI * 18 * time);
        ecg[i] += fNoise;
        
        // Erratic T-waves
        const tPeak = peak + 0.18;
        const tWidth = 0.075;
        const tAmp = 0.1 + Math.random() * 0.2;
        ecg[i] += tAmp * Math.exp(-Math.pow(time - tPeak, 2) / (2 * Math.pow(tWidth, 2)));
      }
    }
  }

  // Add random drift and sensor noise
  for (let i = 0; i < sequenceLength; i++) {
    const drift = 0.05 * Math.sin(2 * Math.PI * 0.15 * t[i]);
    const noise = (Math.random() - 0.5) * 0.06;
    ecg[i] += drift + noise;
  }

  // Normalize between -1 and 1
  let maxVal = 0;
  for (let i = 0; i < sequenceLength; i++) {
    if (Math.abs(ecg[i]) > maxVal) {
      maxVal = Math.abs(ecg[i]);
    }
  }
  
  if (maxVal > 0) {
    for (let i = 0; i < sequenceLength; i++) {
      ecg[i] = ecg[i] / maxVal;
    }
  }

  return ecg;
}

export function generateECGDataset(size: number): { x: number[][]; y: number[] } {
  const x: number[][] = [];
  const y: number[] = [];
  
  for (let i = 0; i < size; i++) {
    const label = Math.random() > 0.5 ? 1 : 0;
    x.push(generateECGTrace(label));
    y.push(label);
  }
  
  return { x, y };
}
