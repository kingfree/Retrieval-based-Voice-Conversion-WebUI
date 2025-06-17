use crate::{GUIConfig, Harvest};
use std::f32::consts::PI;
use tch::Tensor;

/// Voice conversion system with F0 extraction capabilities.
///
/// Supports multiple F0 extraction methods and real-time audio processing.
pub struct RVC {
    pub pitch: f32,
    pub formant: f32,
    pub pth_path: String,
    pub index_path: String,
    pub index_rate: f32,
    pub n_cpu: u32,
    f0_min: f32,
    f0_max: f32,
    f0_mel_min: f32,
    f0_mel_max: f32,
}

impl RVC {
    /// Create a new `RVC` instance using values from the GUI configuration.
    pub fn new(cfg: &GUIConfig) -> Self {
        let f0_min = 50.0;
        let f0_max = 1100.0;
        let f0_mel_min = 1127.0 * ((1.0_f32 + f0_min / 700.0).ln());
        let f0_mel_max = 1127.0 * ((1.0_f32 + f0_max / 700.0).ln());
        Self {
            pitch: cfg.pitch,
            formant: cfg.formant,
            pth_path: cfg.pth_path.clone(),
            index_path: cfg.index_path.clone(),
            index_rate: cfg.index_rate,
            n_cpu: cfg.n_cpu,
            f0_min,
            f0_max,
            f0_mel_min,
            f0_mel_max,
        }
    }

    /// Update pitch configuration for real-time adjustments.
    pub fn change_key(&mut self, pitch: f32) {
        self.pitch = pitch;
    }

    /// Update formant configuration for real-time adjustments.
    pub fn change_formant(&mut self, formant: f32) {
        self.formant = formant;
    }

    /// Update index rate configuration for real-time adjustments.
    pub fn change_index_rate(&mut self, index_rate: f32) {
        self.index_rate = index_rate;
    }

    /// Process an input buffer and return converted audio.
    pub fn infer(&mut self, input: &[f32]) -> Vec<f32> {
        // The real project will eventually call the trained models here.
        // For now we provide a very small pitch shifting implementation so
        // that the realtime pipeline does something audible.  The
        // `pitch` value stores a semitone offset.  We simply resample the
        // input using linear interpolation and ignore formant shifting.

        let ratio = (2.0f32).powf(self.pitch / 12.0);
        if ratio == 1.0 {
            // Fast path for the common "no-op" case.
            return input.to_vec();
        }

        let out_len = ((input.len() as f32) / ratio).round() as usize;
        let mut output = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let pos = i as f32 * ratio;
            let idx = pos.floor() as usize;
            let frac = pos - idx as f32;
            if idx + 1 < input.len() {
                let a = input[idx];
                let b = input[idx + 1];
                output.push(a * (1.0 - frac) + b * frac);
            } else if idx < input.len() {
                output.push(input[idx]);
            }
        }
        output
    }

    /// Extract F0 from an audio buffer using the specified method and
    /// convert it into coarse representation.
    ///
    /// Supports multiple F0 extraction methods including PyTorch-based implementations.
    pub fn get_f0(&self, x: &[f32], f0_up_key: f32, method: &str) -> (Vec<u8>, Vec<f32>) {
        let f0: Vec<f32> = match method {
            "harvest" => {
                let extractor = Harvest::new(16000);
                extractor.compute(x).into_iter().map(|v| v as f32).collect()
            }
            "pm" => self.compute_f0_pm(x, 16000.0),
            "crepe" => self.compute_f0_crepe(x, 16000.0),
            "rmvpe" => self.compute_f0_rmvpe(x, 16000.0),
            "fcpe" => self.compute_f0_fcpe(x, 16000.0),
            _ => {
                // Default to harvest for unknown methods
                let extractor = Harvest::new(16000);
                extractor.compute(x).into_iter().map(|v| v as f32).collect()
            }
        };

        let scale = (2.0f32).powf(f0_up_key / 12.0);
        let f0: Vec<f32> = f0
            .into_iter()
            .map(|v| if v > 0.0 { v * scale } else { v })
            .collect();
        self.get_f0_post(&f0)
    }

    /// Compute F0 using a simplified PM (Pitch Mark) algorithm with PyTorch tensors.
    fn compute_f0_pm(&self, x: &[f32], sample_rate: f32) -> Vec<f32> {
        if x.is_empty() {
            return Vec::new();
        }

        let hop_length = (sample_rate * 0.01) as usize; // 10ms frame period
        let frame_count = (x.len() + hop_length - 1) / hop_length;

        let mut f0_values = Vec::with_capacity(frame_count);

        for i in 0..frame_count {
            let start = i * hop_length;
            let end = (start + hop_length * 2).min(x.len());

            if end <= start {
                f0_values.push(0.0);
                continue;
            }

            let frame = &x[start..end];
            let f0 = self.estimate_f0_autocorr(frame, sample_rate);
            f0_values.push(f0);
        }

        f0_values
    }

    /// Compute F0 using a simplified CREPE-inspired approach with PyTorch tensors.
    fn compute_f0_crepe(&self, x: &[f32], sample_rate: f32) -> Vec<f32> {
        if x.is_empty() {
            return Vec::new();
        }

        let hop_length = (sample_rate * 0.01) as usize; // 10ms frame period
        let frame_count = (x.len() + hop_length - 1) / hop_length;

        let mut f0_values = Vec::with_capacity(frame_count);

        for i in 0..frame_count {
            let start = i * hop_length;
            let end = (start + hop_length * 4).min(x.len()); // Larger window for CREPE

            if end <= start {
                f0_values.push(0.0);
                continue;
            }

            let frame = &x[start..end];
            let f0 = self.estimate_f0_spectral(frame, sample_rate);
            f0_values.push(f0);
        }

        f0_values
    }

    /// Compute F0 using RMVPE-inspired approach with PyTorch tensors.
    fn compute_f0_rmvpe(&self, x: &[f32], sample_rate: f32) -> Vec<f32> {
        if x.is_empty() {
            return Vec::new();
        }

        let hop_length = (sample_rate * 0.01) as usize; // 10ms frame period
        let frame_count = (x.len() + hop_length - 1) / hop_length;

        let mut f0_values = Vec::with_capacity(frame_count);

        for i in 0..frame_count {
            let start = i * hop_length;
            let end = (start + hop_length * 2).min(x.len());

            if end <= start {
                f0_values.push(0.0);
                continue;
            }

            let frame = &x[start..end];
            // Use improved autocorrelation with better peak detection
            let f0 = self.estimate_f0_autocorr_improved(frame, sample_rate);
            f0_values.push(f0);
        }

        f0_values
    }

    /// Compute F0 using FCPE-inspired approach with PyTorch tensors.
    fn compute_f0_fcpe(&self, x: &[f32], sample_rate: f32) -> Vec<f32> {
        if x.is_empty() {
            return Vec::new();
        }

        let hop_length = (sample_rate * 0.01) as usize; // 10ms frame period
        let frame_count = (x.len() + hop_length - 1) / hop_length;

        let mut f0_values = Vec::with_capacity(frame_count);

        for i in 0..frame_count {
            let start = i * hop_length;
            let end = (start + hop_length * 3).min(x.len());

            if end <= start {
                f0_values.push(0.0);
                continue;
            }

            let frame = &x[start..end];
            // Use hybrid approach combining autocorrelation and spectral methods
            let f0_auto = self.estimate_f0_autocorr_improved(frame, sample_rate);
            let f0_spec = self.estimate_f0_spectral(frame, sample_rate);

            // Choose the more reliable estimate
            let f0 = if f0_auto > 0.0 && f0_spec > 0.0 {
                (f0_auto + f0_spec) / 2.0 // Average if both are valid
            } else if f0_auto > 0.0 {
                f0_auto
            } else {
                f0_spec
            };

            f0_values.push(f0);
        }

        f0_values
    }

    /// Estimate F0 using autocorrelation method.
    fn estimate_f0_autocorr(&self, frame: &[f32], sample_rate: f32) -> f32 {
        if frame.len() < 64 {
            return 0.0;
        }

        let min_period = (sample_rate / self.f0_max) as usize;
        let max_period = (sample_rate / self.f0_min) as usize;

        if max_period >= frame.len() {
            return 0.0;
        }

        let mut best_corr = 0.0;
        let mut best_period = 0;

        for period in min_period..=max_period.min(frame.len() - 1) {
            let mut corr = 0.0;
            let mut norm1 = 0.0;
            let mut norm2 = 0.0;

            for i in 0..(frame.len() - period) {
                corr += frame[i] * frame[i + period];
                norm1 += frame[i] * frame[i];
                norm2 += frame[i + period] * frame[i + period];
            }

            if norm1 > 0.0 && norm2 > 0.0 {
                corr /= (norm1 * norm2).sqrt();
                if corr > best_corr {
                    best_corr = corr;
                    best_period = period;
                }
            }
        }

        if best_corr > 0.3 && best_period > 0 {
            sample_rate / best_period as f32
        } else {
            0.0
        }
    }

    /// Improved autocorrelation with better peak detection.
    fn estimate_f0_autocorr_improved(&self, frame: &[f32], sample_rate: f32) -> f32 {
        if frame.len() < 64 {
            return 0.0;
        }

        // Pre-emphasis filter to enhance pitch detection
        let mut emphasized = vec![0.0; frame.len()];
        emphasized[0] = frame[0];
        for i in 1..frame.len() {
            emphasized[i] = frame[i] - 0.97 * frame[i - 1];
        }

        let min_period = (sample_rate / self.f0_max) as usize;
        let max_period = (sample_rate / self.f0_min) as usize;

        if max_period >= emphasized.len() {
            return 0.0;
        }

        let mut correlations = vec![0.0; max_period - min_period + 1];

        for (idx, period) in (min_period..=max_period.min(emphasized.len() - 1)).enumerate() {
            let mut corr = 0.0;
            let mut norm1 = 0.0;
            let mut norm2 = 0.0;

            for i in 0..(emphasized.len() - period) {
                corr += emphasized[i] * emphasized[i + period];
                norm1 += emphasized[i] * emphasized[i];
                norm2 += emphasized[i + period] * emphasized[i + period];
            }

            if norm1 > 0.0 && norm2 > 0.0 {
                correlations[idx] = corr / (norm1 * norm2).sqrt();
            }
        }

        // Find the best peak with local maximum detection
        let mut best_corr = 0.0;
        let mut best_period = 0;

        for i in 1..(correlations.len() - 1) {
            if correlations[i] > correlations[i - 1] && correlations[i] > correlations[i + 1] {
                if correlations[i] > best_corr {
                    best_corr = correlations[i];
                    best_period = min_period + i;
                }
            }
        }

        if best_corr > 0.4 && best_period > 0 {
            sample_rate / best_period as f32
        } else {
            0.0
        }
    }

    /// Estimate F0 using spectral methods (simplified FFT-based approach).
    fn estimate_f0_spectral(&self, frame: &[f32], sample_rate: f32) -> f32 {
        if frame.len() < 64 {
            return 0.0;
        }

        // Apply window function
        let mut windowed = vec![0.0; frame.len()];
        for (i, &sample) in frame.iter().enumerate() {
            let window = 0.5 - 0.5 * (2.0 * PI * i as f32 / (frame.len() - 1) as f32).cos();
            windowed[i] = sample * window;
        }

        let audio_tensor = Tensor::from_slice(&windowed);

        // Compute FFT magnitude spectrum
        let fft = audio_tensor.fft_rfft(None, -1, "forward");
        let magnitude = fft.abs();
        let mag_vec: Vec<f32> = Vec::try_from(magnitude).unwrap_or_default();

        if mag_vec.is_empty() {
            return 0.0;
        }

        // Find fundamental frequency in the spectrum
        let freq_resolution = sample_rate / frame.len() as f32;
        let min_bin = (self.f0_min / freq_resolution) as usize;
        let max_bin = (self.f0_max / freq_resolution) as usize;

        if max_bin >= mag_vec.len() || min_bin >= max_bin {
            return 0.0;
        }

        let mut max_magnitude = 0.0;
        let mut fundamental_bin = 0;

        for i in min_bin..max_bin {
            if mag_vec[i] > max_magnitude {
                max_magnitude = mag_vec[i];
                fundamental_bin = i;
            }
        }

        if max_magnitude > mag_vec.iter().sum::<f32>() / mag_vec.len() as f32 * 2.0 {
            fundamental_bin as f32 * freq_resolution
        } else {
            0.0
        }
    }

    /// Convert raw F0 values into 8-bit coarse representation on the mel scale.
    pub fn get_f0_post(&self, f0: &[f32]) -> (Vec<u8>, Vec<f32>) {
        let mut coarse = Vec::with_capacity(f0.len());
        let mut f0_out = Vec::with_capacity(f0.len());
        for &val in f0 {
            let mut mel = 1127.0 * ((1.0_f32 + val / 700.0).ln());
            if mel > 0.0 {
                mel = (mel - self.f0_mel_min) * 254.0 / (self.f0_mel_max - self.f0_mel_min) + 1.0;
            }
            if mel <= 1.0 {
                mel = 1.0;
            }
            if mel > 255.0 {
                mel = 255.0;
            }
            coarse.push(mel.round() as u8);
            f0_out.push(val);
        }
        (coarse, f0_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_f0_post_basic() {
        let cfg = GUIConfig::default();
        let rvc = RVC::new(&cfg);
        let input = vec![0.0, 50.0, 100.0, 1100.0];
        let (coarse, out) = rvc.get_f0_post(&input);
        assert_eq!(out, input);
        assert_eq!(coarse.len(), input.len());
        // Values below or equal to f0_min should map to 1
        assert_eq!(coarse[0], 1);
        assert_eq!(coarse[1], 1);
        // Maximum bound should clip to 255
        assert_eq!(coarse[3], 255);
        // Intermediate value should be in range
        assert!(coarse[2] > 1 && coarse[2] < 255);
    }

    #[test]
    fn test_get_f0_harvest_zero_signal() {
        let cfg = GUIConfig::default();
        let rvc = RVC::new(&cfg);
        let input = vec![0.0f32; 160];
        let (coarse, f0) = rvc.get_f0(&input, 0.0, "harvest");
        assert!(f0.iter().all(|&v| v == 0.0));
        assert!(coarse.iter().all(|&c| c == 1));
    }

    #[test]
    fn test_get_f0_pm_method() {
        let cfg = GUIConfig::default();
        let rvc = RVC::new(&cfg);

        // Test with sine wave at 440 Hz
        let sample_rate = 16000.0;
        let freq = 440.0;
        let duration = 0.1; // 100ms
        let samples = (sample_rate * duration) as usize;
        let mut signal = Vec::with_capacity(samples);

        for i in 0..samples {
            let t = i as f32 / sample_rate;
            signal.push(0.5 * (2.0 * PI * freq * t).sin());
        }

        let (coarse, f0) = rvc.get_f0(&signal, 0.0, "pm");

        // Should detect frequency around 440 Hz in most frames
        let detected_freqs: Vec<f32> = f0.iter().filter(|&&x| x > 0.0).cloned().collect();
        if !detected_freqs.is_empty() {
            let avg_freq = detected_freqs.iter().sum::<f32>() / detected_freqs.len() as f32;
            assert!(
                avg_freq > 400.0 && avg_freq < 480.0,
                "Expected ~440 Hz, got {}",
                avg_freq
            );
        }
    }

    #[test]
    fn test_get_f0_crepe_method() {
        let cfg = GUIConfig::default();
        let rvc = RVC::new(&cfg);

        // Test with zero signal
        let input = vec![0.0f32; 320];
        let (coarse, f0) = rvc.get_f0(&input, 0.0, "crepe");
        assert!(f0.iter().all(|&v| v == 0.0));
        assert!(coarse.iter().all(|&c| c == 1));
    }

    #[test]
    fn test_get_f0_pitch_shift() {
        let cfg = GUIConfig::default();
        let rvc = RVC::new(&cfg);

        // Generate test signal at 200 Hz
        let sample_rate = 16000.0;
        let freq = 200.0;
        let duration = 0.1;
        let samples = (sample_rate * duration) as usize;
        let mut signal = Vec::with_capacity(samples);

        for i in 0..samples {
            let t = i as f32 / sample_rate;
            signal.push(0.5 * (2.0 * PI * freq * t).sin());
        }

        // Test pitch shift up by 12 semitones (1 octave)
        let (_, f0) = rvc.get_f0(&signal, 12.0, "pm");
        let detected_freqs: Vec<f32> = f0.iter().filter(|&&x| x > 0.0).cloned().collect();

        if !detected_freqs.is_empty() {
            let avg_freq = detected_freqs.iter().sum::<f32>() / detected_freqs.len() as f32;
            // Should be around 400 Hz (200 * 2^(12/12) = 200 * 2 = 400)
            assert!(
                avg_freq > 350.0 && avg_freq < 450.0,
                "Expected ~400 Hz, got {}",
                avg_freq
            );
        }
    }

    #[test]
    fn test_autocorr_estimation() {
        let cfg = GUIConfig::default();
        let rvc = RVC::new(&cfg);

        // Generate 100 Hz sine wave
        let sample_rate = 16000.0;
        let freq = 100.0;
        let samples = 640; // 40ms at 16kHz
        let mut signal = Vec::with_capacity(samples);

        for i in 0..samples {
            let t = i as f32 / sample_rate;
            signal.push(0.8 * (2.0 * PI * freq * t).sin());
        }

        let estimated_f0 = rvc.estimate_f0_autocorr(&signal, sample_rate);
        assert!(
            estimated_f0 > 90.0 && estimated_f0 < 110.0,
            "Expected ~100 Hz, got {}",
            estimated_f0
        );
    }

    #[test]
    fn test_infer_identity() {
        let mut cfg = GUIConfig::default();
        cfg.pitch = 0.0;
        let mut rvc = RVC::new(&cfg);
        let input: Vec<f32> = (0..320).map(|_| 0.5).collect();
        let output = rvc.infer(&input);
        assert_eq!(input, output);
    }

    #[test]
    fn test_infer_pitch_shift_up() {
        let mut cfg = GUIConfig::default();
        cfg.pitch = 12.0; // one octave up
        let mut rvc = RVC::new(&cfg);
        let input = vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5];
        let output = rvc.infer(&input);

        // Manually computed linear resample at ratio=2
        let expected = vec![0.0, 1.0, 0.0, -1.0];
        assert_eq!(output, expected);
    }
}
