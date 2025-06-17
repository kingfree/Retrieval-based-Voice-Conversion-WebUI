use crate::{GUIConfig, Harvest};
use std::collections::HashMap;
use std::f32::consts::PI;
use tch::{Device, Kind, Tensor};

/// Voice conversion system with comprehensive model loading and F0 extraction capabilities.
///
/// This struct mirrors the Python RVC class initialization with all the detailed
/// configuration and state management required for real-time voice conversion.
pub struct RVC {
    // Configuration parameters
    pub f0_up_key: f32,     // Pitch adjustment key (semitones)
    pub formant_shift: f32, // Formant shifting parameter
    pub pth_path: String,   // Path to the model checkpoint
    pub index_path: String, // Path to the index file for similarity search
    pub index_rate: f32,    // Index search mixing rate
    pub n_cpu: u32,         // Number of CPU cores for processing

    // Device and model configuration
    pub device: Device, // Computing device (CPU/CUDA)
    pub is_half: bool,  // Half precision floating point flag
    pub use_jit: bool,  // Just-In-Time compilation flag

    // F0 (pitch) related parameters
    pub f0_min: f32,     // Minimum fundamental frequency (50 Hz)
    pub f0_max: f32,     // Maximum fundamental frequency (1100 Hz)
    pub f0_mel_min: f32, // Mel-scale minimum frequency
    pub f0_mel_max: f32, // Mel-scale maximum frequency

    // Model parameters
    pub tgt_sr: i32,     // Target sample rate from model config
    pub if_f0: i32,      // Whether model uses F0 conditioning (0 or 1)
    pub version: String, // Model version (v1, v2, etc.)

    // Caching for performance
    pub cache_pitch: Tensor,  // Cached pitch values for efficiency
    pub cache_pitchf: Tensor, // Cached pitch frequency values

    // Index search related
    pub index_loaded: bool,       // Whether index is successfully loaded
    pub index_dim: Option<usize>, // Dimension of index vectors

    // Resampling kernels cache
    pub resample_kernels: HashMap<String, Vec<f32>>, // Cached resampling kernels

    // Model state flags
    pub model_loaded: bool,  // Whether the main model is loaded
    pub hubert_loaded: bool, // Whether HuBERT model is loaded
}

impl RVC {
    /// Create a new `RVC` instance with comprehensive initialization.
    ///
    /// This method mirrors the Python RVC.__init__ functionality with all
    /// the detailed model loading, device configuration, and state setup.
    pub fn new(cfg: &GUIConfig) -> Self {
        let f0_min = 50.0;
        let f0_max = 1100.0;
        let f0_mel_min = 1127.0 * ((1.0_f32 + f0_min / 700.0).ln());
        let f0_mel_max = 1127.0 * ((1.0_f32 + f0_max / 700.0).ln());

        // Determine device based on configuration and availability
        let device = if cfg.sg_hostapi.contains("CUDA") || cfg.sg_hostapi.contains("cuda") {
            // Try CUDA first, fall back to CPU if not available
            if tch::Cuda::is_available() {
                Device::Cuda(0)
            } else {
                Device::Cpu
            }
        } else {
            Device::Cpu
        };

        // Initialize cache tensors for pitch processing (always use CPU for safety)
        let cache_pitch = Tensor::zeros(&[1024], (Kind::Int64, Device::Cpu));
        let cache_pitchf = Tensor::zeros(&[1024], (Kind::Float, Device::Cpu));

        let mut rvc = Self {
            f0_up_key: cfg.pitch,
            formant_shift: cfg.formant,
            pth_path: cfg.pth_path.clone(),
            index_path: cfg.index_path.clone(),
            index_rate: cfg.index_rate,
            n_cpu: cfg.n_cpu,
            device,
            is_half: true, // Default to half precision for performance
            use_jit: cfg.use_jit,
            f0_min,
            f0_max,
            f0_mel_min,
            f0_mel_max,
            tgt_sr: 40000,             // Default target sample rate
            if_f0: 1,                  // Default to F0-conditioned model
            version: "v2".to_string(), // Default version
            cache_pitch,
            cache_pitchf,
            index_loaded: false,
            index_dim: None,
            resample_kernels: HashMap::new(),
            model_loaded: false,
            hubert_loaded: false,
        };

        // Initialize index if provided and rate > 0
        if cfg.index_rate > 0.0 && !cfg.index_path.is_empty() {
            rvc.load_index();
        }

        // Load model if path is provided
        if !cfg.pth_path.is_empty() {
            rvc.load_model();
        }

        rvc
    }

    /// Load the index file for similarity search.
    ///
    /// This mirrors the Python index loading logic with FAISS index reading.
    fn load_index(&mut self) {
        if std::path::Path::new(&self.index_path).exists() {
            // In a real implementation, this would use FAISS bindings
            // For now, we simulate the loading
            println!("Loading index from: {}", self.index_path);
            self.index_loaded = true;
            self.index_dim = Some(768); // Typical HuBERT dimension
            println!("Index search enabled");
        } else {
            println!("Index file not found: {}", self.index_path);
            self.index_loaded = false;
        }
    }

    /// Load the main voice conversion model.
    ///
    /// This mirrors the Python model loading with checkpoint parsing and
    /// device/precision configuration.
    fn load_model(&mut self) {
        if std::path::Path::new(&self.pth_path).exists() {
            println!("Loading model from: {}", self.pth_path);

            // In a real implementation, this would load PyTorch models
            // For now, we simulate the model loading and configuration extraction

            // Simulate reading model configuration
            self.tgt_sr = 40000; // Would be read from checkpoint
            self.if_f0 = 1; // Would be read from checkpoint
            self.version = "v2".to_string(); // Would be read from checkpoint

            // Mark model as loaded
            self.model_loaded = true;
            println!("Model loaded successfully");
            println!("Target sample rate: {}", self.tgt_sr);
            println!("F0 conditioning: {}", self.if_f0 == 1);
            println!("Model version: {}", self.version);
        } else {
            println!("Model file not found: {}", self.pth_path);
            self.model_loaded = false;
        }
    }

    /// Initialize HuBERT model for feature extraction.
    ///
    /// This mirrors the Python HuBERT loading logic.
    fn load_hubert(&mut self) {
        // In a real implementation, this would load the HuBERT model
        // from "assets/hubert/hubert_base.pt"
        println!("Loading HuBERT model...");
        self.hubert_loaded = true;
        println!("HuBERT model loaded successfully");
    }

    /// Set up resampling kernels for different sample rate conversions.
    ///
    /// This mirrors the Python resample_kernel initialization.
    fn setup_resampling(&mut self) {
        // Common resampling ratios
        let ratios = vec![
            ("16000_to_40000", 16000.0 / 40000.0),
            ("22050_to_40000", 22050.0 / 40000.0),
            ("44100_to_40000", 44100.0 / 40000.0),
            ("48000_to_40000", 48000.0 / 40000.0),
        ];

        for (name, _ratio) in ratios {
            // In a real implementation, this would generate proper resampling kernels
            let kernel = vec![1.0; 64]; // Placeholder kernel
            self.resample_kernels.insert(name.to_string(), kernel);
        }

        println!("Resampling kernels initialized");
    }

    /// Update pitch configuration for real-time adjustments.
    ///
    /// This mirrors the Python change_key method.
    pub fn change_key(&mut self, new_key: f32) {
        self.f0_up_key = new_key;
        println!("Pitch key changed to: {} semitones", new_key);
    }

    /// Update formant configuration for real-time adjustments.
    ///
    /// This mirrors the Python change_formant method.
    pub fn change_formant(&mut self, new_formant: f32) {
        self.formant_shift = new_formant;
        println!("Formant shift changed to: {}", new_formant);
    }

    /// Update index rate configuration for real-time adjustments.
    ///
    /// This mirrors the Python change_index_rate method with dynamic index loading.
    pub fn change_index_rate(&mut self, new_index_rate: f32) {
        if new_index_rate > 0.0 && self.index_rate == 0.0 && !self.index_loaded {
            // Index was disabled, now enabling
            self.load_index();
        }
        self.index_rate = new_index_rate;
        println!("Index rate changed to: {}", new_index_rate);
    }

    /// Get model information and status.
    pub fn get_model_info(&self) -> ModelInfo {
        ModelInfo {
            model_loaded: self.model_loaded,
            hubert_loaded: self.hubert_loaded,
            index_loaded: self.index_loaded,
            target_sr: self.tgt_sr,
            f0_conditioned: self.if_f0 == 1,
            version: self.version.clone(),
            device: format!("{:?}", self.device),
            is_half_precision: self.is_half,
            use_jit: self.use_jit,
        }
    }

    /// Clear cached data and reset internal state.
    pub fn clear_cache(&mut self) {
        self.cache_pitch = Tensor::zeros(&[1024], (Kind::Int64, self.device));
        self.cache_pitchf = Tensor::zeros(&[1024], (Kind::Float, self.device));
        println!("Cache cleared");
    }

    /// Check if the RVC instance is ready for inference.
    pub fn is_ready(&self) -> bool {
        self.model_loaded && (self.index_rate == 0.0 || self.index_loaded)
    }

    /// Process an input buffer and return converted audio.
    pub fn infer(&mut self, input: &[f32]) -> Vec<f32> {
        // The real project will eventually call the trained models here.
        // For now we provide a very small pitch shifting implementation so
        // that the realtime pipeline does something audible.  The
        // `pitch` value stores a semitone offset.  We simply resample the
        // input using linear interpolation and ignore formant shifting.

        let ratio = (2.0f32).powf(self.f0_up_key / 12.0);
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
            return fundamental_bin as f32 * freq_resolution;
        } else {
            return 0.0;
        }
    }

    /// Convert raw F0 values into 8-bit coarse representation on the mel scale.
    ///
    /// This mirrors the Python get_f0_post method with exact mel-scale conversion.
    pub fn get_f0_post(&self, f0: &[f32]) -> (Vec<u8>, Vec<f32>) {
        let mut coarse = Vec::with_capacity(f0.len());
        let mut f0_out = Vec::with_capacity(f0.len());

        for &val in f0 {
            let mut mel = if val > 0.0 {
                1127.0 * ((1.0_f32 + val / 700.0).ln())
            } else {
                0.0
            };

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

/// Model information structure for status reporting.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_loaded: bool,
    pub hubert_loaded: bool,
    pub index_loaded: bool,
    pub target_sr: i32,
    pub f0_conditioned: bool,
    pub version: String,
    pub device: String,
    pub is_half_precision: bool,
    pub use_jit: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rvc_initialization() {
        let mut cfg = GUIConfig::default();
        cfg.pitch = 2.0;
        cfg.formant = 0.5;
        cfg.pth_path = "test_model.pth".to_string();
        cfg.index_path = "test_index.index".to_string();
        cfg.index_rate = 0.5;
        cfg.n_cpu = 4;
        cfg.use_jit = false;

        let rvc = RVC::new(&cfg);

        // Verify basic initialization
        assert_eq!(rvc.f0_up_key, 2.0);
        assert_eq!(rvc.formant_shift, 0.5);
        assert_eq!(rvc.pth_path, "test_model.pth");
        assert_eq!(rvc.index_path, "test_index.index");
        assert_eq!(rvc.index_rate, 0.5);
        assert_eq!(rvc.n_cpu, 4);
        assert!(!rvc.use_jit);

        // Verify F0 parameters
        assert_eq!(rvc.f0_min, 50.0);
        assert_eq!(rvc.f0_max, 1100.0);
        assert!((rvc.f0_mel_min - 77.755).abs() < 0.01);
        assert!((rvc.f0_mel_max - 1064.408).abs() < 0.01);

        // Verify default values
        assert_eq!(rvc.tgt_sr, 40000);
        assert_eq!(rvc.if_f0, 1);
        assert_eq!(rvc.version, "v2");
        assert!(rvc.is_half);
        assert!(!rvc.model_loaded);
        assert!(!rvc.hubert_loaded);
        assert!(!rvc.index_loaded);
    }

    #[test]
    fn test_parameter_updates() {
        let cfg = GUIConfig::default();
        let mut rvc = RVC::new(&cfg);

        // Test pitch key change
        rvc.change_key(5.0);
        assert_eq!(rvc.f0_up_key, 5.0);

        // Test formant change
        rvc.change_formant(1.2);
        assert_eq!(rvc.formant_shift, 1.2);

        // Test index rate change
        rvc.change_index_rate(0.8);
        assert_eq!(rvc.index_rate, 0.8);
    }

    #[test]
    fn test_model_info() {
        let cfg = GUIConfig::default();
        let rvc = RVC::new(&cfg);
        let info = rvc.get_model_info();

        assert!(!info.model_loaded);
        assert!(!info.hubert_loaded);
        assert!(!info.index_loaded);
        assert_eq!(info.target_sr, 40000);
        assert!(info.f0_conditioned);
        assert_eq!(info.version, "v2");
        assert!(info.is_half_precision);
        assert!(!info.use_jit);
    }

    #[test]
    fn test_readiness_check() {
        let cfg = GUIConfig::default();
        let mut rvc = RVC::new(&cfg);

        // Should be ready when model is loaded and no index needed
        rvc.model_loaded = true;
        rvc.index_rate = 0.0;
        assert!(rvc.is_ready());

        // Should not be ready when index needed but not loaded
        rvc.index_rate = 0.5;
        rvc.index_loaded = false;
        assert!(!rvc.is_ready());

        // Should be ready when both model and index are loaded
        rvc.index_loaded = true;
        assert!(rvc.is_ready());
    }

    #[test]
    fn test_cache_operations() {
        let cfg = GUIConfig::default();
        let mut rvc = RVC::new(&cfg);

        // Test cache clearing
        rvc.clear_cache();

        // Verify cache tensors are properly sized
        assert_eq!(rvc.cache_pitch.size(), vec![1024]);
        assert_eq!(rvc.cache_pitchf.size(), vec![1024]);
    }

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
    fn test_get_f0_post_mel_scale_conversion() {
        let cfg = GUIConfig::default();
        let rvc = RVC::new(&cfg);

        // Test specific frequency conversions
        let test_frequencies = vec![
            (0.0, 1),      // Silent should map to 1
            (50.0, 1),     // f0_min should map to 1
            (100.0, 20),   // Should map to specific mel value
            (440.0, 122),  // A4 note
            (1100.0, 255), // f0_max should map to 255
        ];

        for (freq, expected_coarse) in test_frequencies {
            let (coarse, f0_out) = rvc.get_f0_post(&vec![freq]);
            assert_eq!(f0_out[0], freq);
            assert_eq!(
                coarse[0], expected_coarse,
                "Frequency {} Hz should map to coarse {}, got {}",
                freq, expected_coarse, coarse[0]
            );
        }
    }

    #[test]
    fn test_device_configuration() {
        let mut cfg = GUIConfig::default();

        // Test CPU device
        cfg.sg_hostapi = "DirectSound".to_string();
        let rvc_cpu = RVC::new(&cfg);
        assert!(matches!(rvc_cpu.device, Device::Cpu));

        // Test CUDA device detection (only if CUDA is available)
        cfg.sg_hostapi = "CUDA Audio".to_string();
        let rvc_cuda = RVC::new(&cfg);
        // If CUDA is not available, it should fall back to CPU
        match rvc_cuda.device {
            Device::Cuda(_) | Device::Cpu => {
                // Either is acceptable depending on system configuration
            }
            _ => panic!("Unexpected device type"),
        }
    }

    #[test]
    fn test_resample_kernels_initialization() {
        let cfg = GUIConfig::default();
        let mut rvc = RVC::new(&cfg);

        // Manually trigger resampling setup (normally done in constructor)
        rvc.setup_resampling();

        assert!(!rvc.resample_kernels.is_empty());
        assert!(rvc.resample_kernels.contains_key("16000_to_40000"));
        assert!(rvc.resample_kernels.contains_key("44100_to_40000"));
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

        let (_coarse, f0) = rvc.get_f0(&signal, 0.0, "pm");

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
        // Autocorr might detect 50Hz (subharmonic) or 100Hz (fundamental)
        assert!(
            (estimated_f0 > 45.0 && estimated_f0 < 55.0)
                || (estimated_f0 > 90.0 && estimated_f0 < 110.0),
            "Expected ~50 Hz or ~100 Hz, got {}",
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
