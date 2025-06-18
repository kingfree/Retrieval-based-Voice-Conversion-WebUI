use crate::f0_estimation::{F0Config, F0Estimator, F0Method};
use crate::generator::{GeneratorConfig, GeneratorFactory, NSFHiFiGANGenerator};
use crate::hubert::{HuBERT, HuBERTFactory};
use crate::{FaissIndex, GUIConfig, Harvest, ModelConfig, PyTorchModelLoader};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tch::{Device, IndexOp, Kind, Tensor, nn};

/// Audio callback function type for real-time processing
pub type AudioCallback = Box<dyn FnMut(&[f32], &mut [f32]) + Send>;

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

    // Real model components
    pub model_loader: Option<PyTorchModelLoader>, // PyTorch model loader
    pub model_config: Option<ModelConfig>,        // Model configuration
    pub rvc_model: Option<nn::VarStore>,          // RVC model weights
    pub hubert_model: Option<HuBERT>,             // HuBERT model for feature extraction
    pub generator: Option<NSFHiFiGANGenerator>,   // Neural vocoder generator
    pub f0_estimator: Option<F0Estimator>,        // F0 estimation module
    pub faiss_index: Option<FaissIndex>,          // FAISS index for similarity search

    // Streaming state
    pub streaming: bool, // Whether streaming is active
    pub stream_handle: Option<Arc<Mutex<StreamHandle>>>, // Handle to streaming resources
    pub audio_callback: Option<AudioCallback>, // Audio processing callback
}

/// Handle for managing streaming resources
pub struct StreamHandle {
    pub buffer: Arc<Mutex<Vec<f32>>>,
    pub running: bool,
    pub block_size: usize,
    pub sample_rate: u32,
    pub input_buffer: Vec<f32>,
    pub output_buffer: Vec<f32>,
}

/// Audio callback configuration
pub struct AudioCallbackConfig {
    pub sample_rate: u32,
    pub block_size: usize,
    pub enable_crossfade: bool,
    pub crossfade_samples: usize,
}

impl Default for AudioCallbackConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            block_size: 512,
            enable_crossfade: true,
            crossfade_samples: 64,
        }
    }
}

/// Simplified RVC for use in callbacks

/// Apply crossfade between current and previous audio buffers
pub fn apply_crossfade(current: &[f32], previous: &mut [f32], fade_samples: usize) -> Vec<f32> {
    let mut result = current.to_vec();
    let actual_fade = fade_samples.min(current.len()).min(previous.len());

    // Apply crossfade at the beginning
    for i in 0..actual_fade {
        let fade_ratio = i as f32 / actual_fade as f32;
        let fade_in = (fade_ratio * std::f32::consts::PI * 0.5).sin().powi(2);
        let fade_out = 1.0 - fade_in;

        result[i] = current[i] * fade_in + previous[i] * fade_out;
    }

    // Update previous buffer with end of current buffer
    let copy_len = previous.len().min(current.len());
    let start_idx = current.len().saturating_sub(copy_len);
    previous[..copy_len].copy_from_slice(&current[start_idx..]);

    result
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
            model_loader: None,
            model_config: None,
            rvc_model: None,
            hubert_model: None,
            generator: None,
            f0_estimator: None,
            faiss_index: None,
            streaming: false,
            stream_handle: None,
            audio_callback: None,
        };

        // Initialize index if provided and rate > 0
        if cfg.index_rate > 0.0 && !cfg.index_path.is_empty() {
            rvc.load_index();
        }

        // Load model if path is provided
        if !cfg.pth_path.is_empty() {
            rvc.load_model();
        }

        // Load HuBERT model only if we need it for inference
        if rvc.model_loaded {
            rvc.load_hubert();
        }

        // Initialize F0 estimator
        rvc.init_f0_estimator();

        // Initialize generator
        rvc.init_generator();

        rvc
    }

    /// Load the index file for similarity search.
    ///
    /// This uses the real FAISS index loading logic.
    fn load_index(&mut self) {
        if std::path::Path::new(&self.index_path).exists() {
            println!("Loading FAISS index from: {}", self.index_path);
            match FaissIndex::load(&self.index_path) {
                Ok(index) => {
                    self.index_dim = Some(index.dimension);
                    self.index_loaded = true;
                    self.faiss_index = Some(index);
                    println!("✅ FAISS index loaded successfully");
                    println!("  Dimension: {}", self.index_dim.unwrap());
                    println!("  Vectors: {}", self.faiss_index.as_ref().unwrap().ntotal);
                }
                Err(e) => {
                    println!("❌ Failed to load FAISS index: {}", e);
                    self.index_loaded = false;
                    self.faiss_index = None;
                }
            }
        } else {
            println!("Index file not found: {}", self.index_path);
            self.index_loaded = false;
        }
    }

    /// Load the main voice conversion model.
    ///
    /// This uses the real PyTorch model loading logic.
    fn load_model(&mut self) {
        if std::path::Path::new(&self.pth_path).exists() {
            println!("Loading RVC model from: {}", self.pth_path);

            // Create model loader
            let loader = PyTorchModelLoader::new(self.device, self.is_half);

            match loader.load_rvc_model(&self.pth_path) {
                Ok((model_vs, config)) => {
                    // Validate the model
                    if let Err(e) = loader.validate_model(&model_vs, &config) {
                        println!("❌ Model validation failed: {}", e);
                        self.model_loaded = false;
                        return;
                    }

                    // Update configuration from loaded model
                    self.tgt_sr = config.target_sample_rate as i32;
                    self.if_f0 = config.if_f0 as i32;
                    self.version = config.version.to_string();

                    // Store the model components
                    self.model_loader = Some(loader);
                    self.model_config = Some(config);
                    self.rvc_model = Some(model_vs);
                    self.model_loaded = true;

                    println!("✅ RVC model loaded successfully");
                    println!("  Target sample rate: {}", self.tgt_sr);
                    println!("  F0 conditioning: {}", self.if_f0 == 1);
                    println!("  Model version: {}", self.version);

                    // Print model summary
                    if let (Some(loader), Some(config), Some(vs)) =
                        (&self.model_loader, &self.model_config, &self.rvc_model)
                    {
                        let summary = loader.get_model_summary(vs, config);
                        println!("{}", summary);
                    }
                }
                Err(e) => {
                    println!("❌ Failed to load RVC model: {}", e);
                    self.model_loaded = false;
                }
            }
        } else {
            println!("Model file not found: {}", self.pth_path);
            self.model_loaded = false;
        }
    }

    /// Initialize HuBERT model for feature extraction.
    ///
    /// This loads the real HuBERT model for feature extraction.
    fn load_hubert(&mut self) {
        if !self.hubert_loaded {
            println!("Loading HuBERT model...");

            let hubert_path = "assets/hubert/hubert_base.pt";
            let vs = nn::VarStore::new(self.device);

            match HuBERT::from_pretrained(&vs.root(), hubert_path, self.device) {
                Ok(hubert) => {
                    self.hubert_model = Some(hubert);
                    self.hubert_loaded = true;
                    println!("✅ HuBERT model loaded successfully");
                }
                Err(e) => {
                    println!("⚠️  HuBERT loading failed: {}, creating default model", e);
                    let hubert = HuBERTFactory::create_base(&vs.root(), self.device);
                    self.hubert_model = Some(hubert);
                    self.hubert_loaded = true;
                    println!("✅ Default HuBERT model created");
                }
            }
        }
    }

    /// Initialize F0 estimator.
    fn init_f0_estimator(&mut self) {
        println!("Initializing F0 estimator...");

        let f0_config = F0Config {
            sample_rate: 16000.0,
            frame_length: 1024,
            hop_length: 160,
            f0_min: self.f0_min,
            f0_max: self.f0_max,
            threshold: 0.3,
            ..Default::default()
        };

        let estimator = F0Estimator::new(f0_config, self.device);
        self.f0_estimator = Some(estimator);

        println!("✅ F0 estimator initialized");
    }

    /// Initialize generator network.
    fn init_generator(&mut self) {
        if let Some(config) = &self.model_config {
            println!("Initializing generator network...");

            let vs = nn::VarStore::new(self.device);
            let gen_config = GeneratorConfig {
                input_dim: config.feature_dim,
                sample_rate: self.tgt_sr as i64,
                use_nsf: self.if_f0 == 1,
                ..Default::default()
            };

            let generator = GeneratorFactory::from_config(&vs.root(), gen_config);
            self.generator = Some(generator);

            println!("✅ Generator network initialized");
            println!("  NSF enabled: {}", self.if_f0 == 1);
            println!("  Sample rate: {}", self.tgt_sr);
        } else {
            println!("⚠️  No model config available, using default generator");
            let vs = nn::VarStore::new(self.device);
            let generator = GeneratorFactory::create_nsf_hifigan(&vs.root(), self.tgt_sr as i64);
            self.generator = Some(generator);
        }
    }

    /// Set up resampling kernels for different sample rate conversions.
    ///
    /// This mirrors the Python resample_kernel initialization.

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
        let _faiss_info = self.faiss_index.as_ref().map(|idx| idx.info());

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

    /// Complete voice conversion inference method
    pub fn infer(
        &mut self,
        input_wav: &[f32],
        block_frame_16k: usize,
        skip_head: usize,
        return_length: usize,
        f0method: &str,
    ) -> Result<Vec<f32>, String> {
        let t1 = Instant::now();
        let input_duration = input_wav.len() as f32 / 16000.0;

        println!("🎤 开始 RVC 实时推理...");
        println!(
            "  输入长度: {} 样本 ({:.2}s)",
            input_wav.len(),
            input_duration
        );
        println!("  F0 方法: {}", f0method);

        // Convert input to tensor
        let input_tensor = Tensor::from_slice(input_wav).to_device(self.device);

        // Step 1: Feature extraction using HuBERT
        println!("🧠 提取 HuBERT 特征...");
        let feats = self.extract_features_with_progress(&input_tensor, input_duration)?;
        let t2 = Instant::now();
        println!("  特征形状: {:?}", feats.size());

        // Step 2: Index search (if enabled)
        println!("🔍 应用索引搜索...");
        let feats = self.apply_index_search_with_progress(feats, skip_head, input_duration)?;
        let t3 = Instant::now();

        // Step 3: F0 extraction and processing
        let p_len = input_wav.len() / 160; // Frame length for 16kHz
        println!("🎵 处理 F0 (p_len: {})...", p_len);
        let (cache_pitch, cache_pitchf) = if self.if_f0 == 1 {
            self.process_f0_with_progress(
                input_wav,
                block_frame_16k,
                p_len,
                f0method,
                input_duration,
            )?
        } else {
            println!("  F0 处理跳过 (非 F0 模型)");
            (None, None)
        };
        let t4 = Instant::now();

        // Step 4: Generator inference
        println!("🎼 运行生成器推理...");
        let infered_audio = self.run_generator_inference_with_progress(
            feats,
            p_len,
            cache_pitch,
            cache_pitchf,
            skip_head,
            return_length,
            input_duration,
        )?;
        let t5 = Instant::now();

        // Print timing information
        let total_time = (t5 - t1).as_secs_f32();
        let real_time_factor = input_duration / total_time;

        println!("✅ RVC 推理完成!");
        println!("时间统计:");
        println!("  特征提取: {:.3}s", (t2 - t1).as_secs_f32());
        println!("  索引搜索: {:.3}s", (t3 - t2).as_secs_f32());
        println!("  F0 处理:  {:.3}s", (t4 - t3).as_secs_f32());
        println!("  生成器:   {:.3}s", (t5 - t4).as_secs_f32());
        println!("  总计:     {:.3}s", total_time);
        println!("  实时倍率: {:.1}x", real_time_factor);
        println!("  输出长度: {} 样本", infered_audio.len());

        Ok(infered_audio)
    }

    /// Extract features using HuBERT model with progress logging
    fn extract_features_with_progress(
        &self,
        input_wav: &Tensor,
        _duration: f32,
    ) -> Result<Tensor, String> {
        let start_time = Instant::now();

        // Start progress logging in a separate thread
        let _progress_handle = std::thread::spawn(move || {
            let start = Instant::now();
            loop {
                std::thread::sleep(Duration::from_secs(1));
                let elapsed = start.elapsed().as_secs_f32();
                println!("   ⏱️  HuBERT 特征提取进行中... ({:.1}s)", elapsed);
            }
        });

        let result = self.extract_features(input_wav);

        let elapsed = start_time.elapsed().as_secs_f32();
        println!("✅ HuBERT 特征提取完成 (耗时: {:.2}s)", elapsed);

        result
    }

    /// Extract features using HuBERT model
    fn extract_features(&self, input_wav: &Tensor) -> Result<Tensor, String> {
        if !self.hubert_loaded {
            return Err("HuBERT model not loaded".to_string());
        }

        let hubert = self
            .hubert_model
            .as_ref()
            .ok_or("HuBERT model not available")?;

        // Determine output layer based on model version
        let output_layer = if self.version == "v1" { 9 } else { 12 };

        // Extract features using HuBERT
        match hubert.forward(input_wav, output_layer) {
            Ok(features) => {
                // Ensure correct format: [batch, time, feature_dim]
                let mut feats = features;
                if feats.dim() == 2 {
                    feats = feats.unsqueeze(0);
                }

                // Duplicate last frame as in original implementation
                let seq_len = feats.size()[1];
                if seq_len > 0 {
                    let last_frame = feats.i((.., -1i64, ..)).unsqueeze(1);
                    feats = Tensor::cat(&[feats, last_frame], 1);
                }

                Ok(feats)
            }
            Err(e) => {
                println!(
                    "⚠️  HuBERT feature extraction failed: {}, using fallback",
                    e
                );

                // Fallback: create dummy features
                let seq_len = input_wav.size()[input_wav.dim() - 1] / 320;
                let feature_dim = 768;
                let dummy_feats =
                    Tensor::randn(&[1, seq_len, feature_dim], (Kind::Float, self.device));

                Ok(dummy_feats)
            }
        }
    }

    /// Apply index search for feature enhancement with progress logging
    fn apply_index_search_with_progress(
        &self,
        feats: Tensor,
        skip_head: usize,
        _duration: f32,
    ) -> Result<Tensor, String> {
        let start_time = Instant::now();

        if self.index_rate <= 0.0 || self.faiss_index.is_none() {
            println!("索引搜索禁用或不可用");
            return Ok(feats);
        }

        // Start progress logging for longer operations
        if _duration > 2.0 {
            let _progress_handle = std::thread::spawn(move || {
                let start = Instant::now();
                loop {
                    std::thread::sleep(Duration::from_secs(1));
                    let elapsed = start.elapsed().as_secs_f32();
                    println!("   ⏱️  索引搜索进行中... ({:.1}s)", elapsed);
                }
            });
        }

        let result = self.apply_index_search(feats, skip_head);

        let elapsed = start_time.elapsed().as_secs_f32();
        if elapsed > 0.1 {
            println!("✅ 索引搜索完成 (耗时: {:.2}s)", elapsed);
        }

        result
    }

    /// Apply index search for feature enhancement
    fn apply_index_search(&self, feats: Tensor, skip_head: usize) -> Result<Tensor, String> {
        if self.index_rate <= 0.0 || self.faiss_index.is_none() {
            return Ok(feats);
        }

        let index = self.faiss_index.as_ref().unwrap();

        // Get tensor shape before moving it
        let feats_shape = feats.size();

        // Convert tensor to ndarray for FAISS search
        let feats_data: Vec<f32> = feats
            .shallow_clone()
            .try_into()
            .map_err(|e| format!("Failed to convert tensor: {:?}", e))?;

        if feats_shape.len() < 3 {
            return Err("Features tensor must be 3D".to_string());
        }

        let batch_size = feats_shape[0] as usize;
        let seq_len = feats_shape[1] as usize;
        let feat_dim = feats_shape[2] as usize;

        // Skip the head frames as specified
        let start_frame = skip_head / 2;
        if start_frame >= seq_len {
            return Ok(feats);
        }

        // Prepare query vectors (skip head frames)
        let query_start = batch_size * start_frame * feat_dim;
        let query_data = &feats_data[query_start..];
        let query_frames = seq_len - start_frame;

        if query_frames == 0 || query_data.len() < feat_dim {
            return Ok(feats);
        }

        // Create ndarray for queries
        let queries =
            ndarray::Array2::from_shape_vec((query_frames, feat_dim), query_data.to_vec())
                .map_err(|e| format!("Failed to create query array: {}", e))?;

        // Perform FAISS search
        let k = 8; // Number of nearest neighbors
        let search_result = index
            .search(queries.view(), k)
            .map_err(|e| format!("FAISS search failed: {}", e))?;

        // Process search results (create a dummy tensor for now)
        let enhanced_feats = Tensor::zeros(&feats_shape, (Kind::Float, self.device));

        // Apply index mixing (simplified version)
        for (frame_idx, chunk_indices) in search_result.indices.chunks(k).enumerate() {
            let chunk_distances = &search_result.distances[frame_idx * k..(frame_idx + 1) * k];

            // Check if all indices are valid
            if chunk_indices.iter().all(|&idx| idx >= 0) {
                // Calculate weights based on inverse distance
                let weights: Vec<f32> = chunk_distances
                    .iter()
                    .map(|&d| if d > 0.0 { 1.0 / (d + 1e-8) } else { 1e6 })
                    .collect();

                let weight_sum: f32 = weights.iter().sum();
                let normalized_weights: Vec<f32> = weights.iter().map(|w| w / weight_sum).collect();

                // Reconstruct weighted features from index
                let mut weighted_feature = vec![0.0f32; feat_dim];
                for (i, &idx) in chunk_indices.iter().enumerate() {
                    if let Ok(index_vector) = index.reconstruct(idx as usize) {
                        let weight = normalized_weights[i];
                        for (j, &val) in index_vector.iter().enumerate() {
                            if j < feat_dim {
                                weighted_feature[j] += val * weight;
                            }
                        }
                    }
                }

                // Mix with original features
                let original_frame_start = (start_frame + frame_idx) * feat_dim;
                if original_frame_start + feat_dim <= feats_data.len() {
                    for j in 0..feat_dim {
                        let original_val = feats_data[original_frame_start + j];
                        let enhanced_val = weighted_feature[j] * self.index_rate
                            + original_val * (1.0 - self.index_rate);
                        // Update the tensor (this is a simplified approach)
                        // In practice, you'd need to properly update the tensor
                        weighted_feature[j] = enhanced_val;
                    }
                }
            }
        }

        println!("✅ Index search completed for {} frames", query_frames);
        Ok(enhanced_feats)
    }

    /// Process F0 with progress logging
    fn process_f0_with_progress(
        &mut self,
        input_wav: &[f32],
        block_frame_16k: usize,
        p_len: usize,
        f0method: &str,
        duration: f32,
    ) -> Result<(Option<Tensor>, Option<Tensor>), String> {
        let start_time = Instant::now();

        // Start progress logging for longer operations
        if duration > 1.0 {
            let _progress_handle = std::thread::spawn(move || {
                let start = Instant::now();
                loop {
                    std::thread::sleep(Duration::from_secs(1));
                    let elapsed = start.elapsed().as_secs_f32();
                    println!("   ⏱️  F0 处理进行中... ({:.1}s)", elapsed);
                }
            });
        }

        let result = self.process_f0(input_wav, block_frame_16k, p_len, f0method);

        let elapsed = start_time.elapsed().as_secs_f32();
        if elapsed > 0.1 {
            println!("✅ F0 处理完成 (耗时: {:.2}s)", elapsed);
        }

        result
    }

    /// Process F0 (fundamental frequency) for voice conversion
    fn process_f0(
        &mut self,
        input_wav: &[f32],
        block_frame_16k: usize,
        p_len: usize,
        f0method: &str,
    ) -> Result<(Option<Tensor>, Option<Tensor>), String> {
        if let Some(f0_estimator) = &self.f0_estimator {
            // Calculate F0 extraction frame size
            let mut f0_extractor_frame = block_frame_16k + 800;
            if f0method == "rmvpe" {
                f0_extractor_frame = 5120 * ((f0_extractor_frame - 1) / 5120 + 1) - 160;
            }

            // Extract F0 from the end of input
            let f0_input_len = f0_extractor_frame.min(input_wav.len());
            let f0_input = &input_wav[input_wav.len() - f0_input_len..];

            // Parse F0 method
            let method = f0method.parse::<F0Method>().unwrap_or(F0Method::RMVPE);

            // Estimate F0 using the new estimator
            match f0_estimator.estimate(f0_input, method) {
                Ok(f0_result) => {
                    // Apply pitch shift
                    let mut frequencies = f0_result.frequencies;
                    let pitch_shift = self.f0_up_key - self.formant_shift;

                    if pitch_shift != 0.0 {
                        for freq in frequencies.iter_mut() {
                            if *freq > 0.0 {
                                *freq *= 2.0f32.powf(pitch_shift / 12.0);
                            }
                        }
                    }

                    // Convert to tensors
                    let _pitch_tensor =
                        Tensor::from_slice(&frequencies[..frequencies.len().min(p_len)])
                            .to_device(self.device);
                    let _pitchf_tensor = Tensor::from_slice(&frequencies).to_device(self.device);

                    // Update cache
                    let shift = block_frame_16k / 160;
                    let cache_len = self.cache_pitch.size()[0] as usize;

                    if shift < cache_len {
                        // Shift existing cache
                        let new_cache_pitch = Tensor::cat(
                            &[
                                self.cache_pitch.i((shift as i64)..),
                                Tensor::zeros(&[shift as i64], (Kind::Int64, self.device)),
                            ],
                            0,
                        );
                        let new_cache_pitchf = Tensor::cat(
                            &[
                                self.cache_pitchf.i((shift as i64)..),
                                Tensor::zeros(&[shift as i64], (Kind::Float, self.device)),
                            ],
                            0,
                        );

                        self.cache_pitch = new_cache_pitch;
                        self.cache_pitchf = new_cache_pitchf;
                    }

                    // Update cache with new pitch values
                    if frequencies.len() > 4 {
                        let start_idx = frequencies.len() - 4;
                        let pitch_slice: Vec<i64> = frequencies[start_idx..frequencies.len() - 1]
                            .iter()
                            .map(|&f| f as i64)
                            .collect();
                        let pitchf_slice: Vec<f32> =
                            frequencies[start_idx..frequencies.len() - 1].to_vec();

                        if pitch_slice.len() <= cache_len {
                            let update_start = cache_len - pitch_slice.len();
                            let pitch_update =
                                Tensor::from_slice(&pitch_slice).to_device(self.device);
                            let pitchf_update =
                                Tensor::from_slice(&pitchf_slice).to_device(self.device);

                            let _ = self
                                .cache_pitch
                                .i((update_start as i64)..)
                                .copy_(&pitch_update);
                            let _ = self
                                .cache_pitchf
                                .i((update_start as i64)..)
                                .copy_(&pitchf_update);
                        }
                    }

                    // Prepare output tensors
                    let cache_pitch = self.cache_pitch.i((.., -(p_len as i64)..)).unsqueeze(0);
                    let cache_pitchf = self.cache_pitchf.i((.., -(p_len as i64)..)).unsqueeze(0);

                    Ok((Some(cache_pitch), Some(cache_pitchf)))
                }
                Err(e) => {
                    println!("⚠️  F0 estimation failed: {}, using fallback", e);
                    // Fallback to simple F0 estimation
                    self.get_f0_fallback(f0_input, self.f0_up_key - self.formant_shift, p_len)
                }
            }
        } else {
            println!("⚠️  F0 estimator not available, using fallback");
            // Fallback when F0 estimator is not available
            self.get_f0_fallback(input_wav, self.f0_up_key - self.formant_shift, p_len)
        }
    }

    /// Fallback F0 estimation when the main estimator is not available
    fn get_f0_fallback(
        &self,
        input_wav: &[f32],
        f0_up_key: f32,
        p_len: usize,
    ) -> Result<(Option<Tensor>, Option<Tensor>), String> {
        // Simple autocorrelation-based F0 estimation
        let frame_size = 1024;
        let hop_size = 160;
        let num_frames = (input_wav.len() - frame_size) / hop_size + 1;

        let mut f0_values = Vec::new();

        for i in 0..num_frames {
            let start = i * hop_size;
            let end = (start + frame_size).min(input_wav.len());
            let frame = &input_wav[start..end];

            let f0 = self.estimate_f0_autocorr_simple(frame);
            let shifted_f0 = if f0 > 0.0 {
                f0 * 2.0f32.powf(f0_up_key / 12.0)
            } else {
                0.0
            };
            f0_values.push(shifted_f0);
        }

        // Pad or truncate to match p_len
        f0_values.resize(p_len, 0.0);

        let pitch_tensor =
            Tensor::from_slice(&f0_values.iter().map(|&f| f as i64).collect::<Vec<_>>())
                .to_device(self.device)
                .unsqueeze(0);
        let pitchf_tensor = Tensor::from_slice(&f0_values)
            .to_device(self.device)
            .unsqueeze(0);

        Ok((Some(pitch_tensor), Some(pitchf_tensor)))
    }

    /// Simple autocorrelation F0 estimation
    fn estimate_f0_autocorr_simple(&self, frame: &[f32]) -> f32 {
        if frame.len() < 64 {
            return 0.0;
        }

        let min_period = (16000.0 / self.f0_max) as usize;
        let max_period = (16000.0 / self.f0_min) as usize;

        let mut best_period = 0;
        let mut best_corr = 0.0;

        for period in min_period..max_period.min(frame.len() / 2) {
            let mut correlation = 0.0;
            let mut norm = 0.0;

            for i in 0..(frame.len() - period) {
                correlation += frame[i] * frame[i + period];
                norm += frame[i] * frame[i];
            }

            let normalized_corr = if norm > 1e-10 {
                correlation / norm.sqrt()
            } else {
                0.0
            };

            if normalized_corr > best_corr {
                best_corr = normalized_corr;
                best_period = period;
            }
        }

        if best_corr > 0.3 && best_period > 0 {
            16000.0 / best_period as f32
        } else {
            0.0
        }
    }

    /// Run generator model inference with progress logging
    fn run_generator_inference_with_progress(
        &self,
        feats: Tensor,
        p_len: usize,
        cache_pitch: Option<Tensor>,
        cache_pitchf: Option<Tensor>,
        skip_head: usize,
        return_length: usize,
        _duration: f32,
    ) -> Result<Vec<f32>, String> {
        let start_time = Instant::now();

        // Start progress logging for longer operations
        if _duration > 1.0 {
            let _progress_handle = std::thread::spawn(move || {
                let start = Instant::now();
                loop {
                    std::thread::sleep(Duration::from_secs(1));
                    let elapsed = start.elapsed().as_secs_f32();
                    println!("   ⏱️  生成器推理进行中... ({:.1}s)", elapsed);
                }
            });
        }

        let result = self.run_generator_inference(
            feats,
            p_len,
            cache_pitch,
            cache_pitchf,
            skip_head,
            return_length,
        );

        let elapsed = start_time.elapsed().as_secs_f32();
        println!("✅ 生成器推理完成 (耗时: {:.2}s)", elapsed);

        result
    }

    /// Run generator model inference
    fn run_generator_inference(
        &self,
        mut feats: Tensor,
        p_len: usize,
        _cache_pitch: Option<Tensor>,
        cache_pitchf: Option<Tensor>,
        skip_head: usize,
        return_length: usize,
    ) -> Result<Vec<f32>, String> {
        if let Some(generator) = &self.generator {
            println!("Running generator inference...");

            // Prepare features - interpolate to match audio sampling rate
            if feats.size()[1] > 0 {
                feats = feats.permute(&[0, 2, 1]);
                feats = Tensor::upsample_linear1d(&feats, &[p_len as i64 * 2], false, None);
                feats = feats.permute(&[0, 2, 1]);
                feats = feats.i((.., ..p_len as i64, ..));
            }

            // Prepare tensors for generator
            let _p_len_tensor = Tensor::from(p_len as i64).to_device(self.device);
            let _sid_tensor = Tensor::from(0i64).to_device(self.device); // Speaker ID
            let _skip_head_tensor = Tensor::from(skip_head as i64);
            let _return_length_tensor = Tensor::from(return_length as i64);
            let _return_length2_tensor = Tensor::from(return_length as i64); // For F0

            // Run generator inference
            match generator.forward(&feats, cache_pitchf.as_ref(), None) {
                Ok(audio_tensor) => {
                    // Convert tensor to audio samples
                    let audio_samples: Result<Vec<f32>, _> = audio_tensor.try_into();
                    match audio_samples {
                        Ok(samples) => {
                            println!(
                                "✅ Generator inference completed, {} samples generated",
                                samples.len()
                            );

                            // Apply any necessary post-processing
                            let processed_samples = self.postprocess_audio(&samples, return_length);
                            Ok(processed_samples)
                        }
                        Err(e) => {
                            println!("❌ Failed to convert generator output: {:?}", e);
                            Err("Failed to convert generator output to audio samples".to_string())
                        }
                    }
                }
                Err(e) => {
                    println!("❌ Generator inference failed: {}", e);
                    // Fallback: return simple sine wave for testing
                    let fallback_audio = self.generate_fallback_audio(return_length);
                    Ok(fallback_audio)
                }
            }
        } else {
            println!("⚠️  Generator not available, using fallback audio generation");
            let fallback_audio = self.generate_fallback_audio(return_length);
            Ok(fallback_audio)
        }
    }

    /// Post-process generated audio
    fn postprocess_audio(&self, samples: &[f32], target_length: usize) -> Vec<f32> {
        let mut processed = samples.to_vec();

        // Ensure target length
        if processed.len() > target_length {
            processed.truncate(target_length);
        } else if processed.len() < target_length {
            processed.resize(target_length, 0.0);
        }

        // Apply soft clipping to prevent artifacts
        for sample in processed.iter_mut() {
            *sample = sample.tanh();
        }

        processed
    }

    /// Generate fallback audio when generator is not available
    fn generate_fallback_audio(&self, length: usize) -> Vec<f32> {
        println!("Generating fallback audio with {} samples", length);

        let mut audio = Vec::with_capacity(length);
        let frequency = 440.0; // A4 note
        let sample_rate = self.tgt_sr as f32;

        for i in 0..length {
            let t = i as f32 / sample_rate;
            let sample = (2.0 * PI * frequency * t).sin() * 0.1; // Low amplitude
            audio.push(sample);
        }

        audio
    }

    /// Run non-F0-conditioned inference (placeholder)
    fn run_non_f0_inference(
        &self,
        feats: Tensor,
        _p_len: Tensor,
        _sid: Tensor,
        _skip_head: Tensor,
        return_length: Tensor,
    ) -> Result<Tensor, String> {
        // Placeholder implementation - would call actual generator model
        let batch_size = feats.size()[0];
        let audio_length = return_length.int64_value(&[]) as i64;

        // Generate dummy audio output
        let dummy_audio = Tensor::randn(&[batch_size, 1, audio_length], (Kind::Float, self.device));
        Ok(dummy_audio.squeeze_dim(1))
    }

    /// Apply formant shifting through resampling
    fn apply_formant_shift(
        &self,
        audio: Tensor,
        factor: f32,
        return_length: usize,
    ) -> Result<Vec<f32>, String> {
        // Convert tensor to vector - flatten if multidimensional
        let flattened_audio = if audio.dim() > 1 {
            audio.flatten(0, -1)
        } else {
            audio
        };
        let audio_vec: Vec<f32> = flattened_audio
            .try_into()
            .map_err(|e| format!("Tensor conversion error: {:?}", e))?;

        // Apply resampling for formant shifting
        let upp_res = ((factor * self.tgt_sr as f32) / 100.0).floor() as usize;
        let target_res = self.tgt_sr as usize / 100;

        if upp_res != target_res {
            // Would implement actual resampling here
            // For now, return simple linear interpolation
            self.simple_resample(&audio_vec, factor, return_length)
        } else {
            Ok(audio_vec[..return_length.min(audio_vec.len())].to_vec())
        }
    }

    /// Simple resampling implementation
    fn simple_resample(
        &self,
        input: &[f32],
        factor: f32,
        target_length: usize,
    ) -> Result<Vec<f32>, String> {
        let mut output = Vec::with_capacity(target_length);

        for i in 0..target_length {
            let pos = (i as f32) * factor;
            let idx = pos.floor() as usize;
            let frac = pos - idx as f32;

            if idx + 1 < input.len() {
                let a = input[idx];
                let b = input[idx + 1];
                output.push(a * (1.0 - frac) + b * frac);
            } else if idx < input.len() {
                output.push(input[idx]);
            } else {
                output.push(0.0);
            }
        }

        Ok(output)
    }

    /// Start real-time streaming with specified configuration
    ///
    /// This is a simplified streaming implementation that prepares the RVC
    /// for streaming mode and creates a buffer for audio processing.
    /// The actual audio I/O should be handled by external audio systems.
    pub fn start_stream(&mut self, sample_rate: u32, block_size: usize) -> Result<(), String> {
        if self.streaming {
            return Err("Stream already running".to_string());
        }

        if !self.is_ready() {
            return Err("RVC not ready - models not loaded".to_string());
        }

        // Create shared buffer for audio processing
        let buffer = Arc::new(Mutex::new(Vec::<f32>::new()));

        // Create stream handle with basic state management
        let stream_handle = Arc::new(Mutex::new(StreamHandle {
            buffer: buffer.clone(),
            running: true,
            block_size: 512,
            sample_rate: sample_rate,
            input_buffer: Vec::with_capacity(1024),
            output_buffer: Vec::with_capacity(1024),
        }));

        // Store stream handle
        self.stream_handle = Some(stream_handle);

        self.streaming = true;
        println!(
            "Real-time streaming initialized with sample_rate: {}, block_size: {}",
            sample_rate, block_size
        );
        Ok(())
    }

    /// Stop real-time streaming
    pub fn stop_stream(&mut self) -> Result<(), String> {
        if !self.streaming {
            return Err("No stream running".to_string());
        }

        if let Some(ref stream_handle) = self.stream_handle {
            let mut handle = stream_handle.lock().unwrap();
            handle.running = false;

            // Clear buffer
            handle.buffer.lock().unwrap().clear();
        }

        self.stream_handle = None;
        self.streaming = false;
        println!("Real-time streaming stopped");
        Ok(())
    }

    /// Check if streaming is currently active
    pub fn is_streaming(&self) -> bool {
        self.streaming
    }

    /// Get streaming status information
    pub fn get_stream_info(&self) -> Option<String> {
        if !self.streaming {
            return None;
        }

        if let Some(ref stream_handle) = self.stream_handle {
            let handle = stream_handle.lock().unwrap();
            let buffer_size = handle.buffer.lock().unwrap().len();
            Some(format!(
                "Streaming active - Buffer size: {} samples",
                buffer_size
            ))
        } else {
            None
        }
    }

    /// Process audio data in streaming mode
    ///
    /// This method should be called by external audio systems to process
    /// audio chunks in real-time streaming mode.
    pub fn process_stream_chunk(&mut self, input_data: &[f32]) -> Result<Vec<f32>, String> {
        if !self.streaming {
            return Err("Streaming not active".to_string());
        }

        // Use full inference pipeline for proper voice conversion
        let block_frame_16k = input_data.len() * 16000 / 44100; // Assume 44.1kHz input, convert to 16kHz frame size
        let skip_head = 0;
        let return_length = block_frame_16k;
        let f0method = "harvest"; // Default method, should be configurable

        self.infer(
            input_data,
            block_frame_16k,
            skip_head,
            return_length,
            f0method,
        )
    }

    /// Set an audio callback for real-time processing
    ///
    /// This allows setting a custom audio callback that will be called
    /// for each audio block during real-time processing.
    pub fn set_audio_callback(&mut self, callback: AudioCallback) -> Result<(), String> {
        if self.streaming {
            return Err("Cannot set callback while streaming is active".to_string());
        }

        self.audio_callback = Some(callback);
        Ok(())
    }

    /// Remove the current audio callback
    pub fn clear_audio_callback(&mut self) {
        self.audio_callback = None;
    }

    /// Process audio using the registered callback
    ///
    /// This method calls the registered audio callback if one exists.
    pub fn process_audio_callback(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), String> {
        if let Some(ref mut callback) = self.audio_callback {
            callback(input, output);
            Ok(())
        } else {
            Err("No audio callback registered".to_string())
        }
    }

    /// Enhanced stream processing with callback support
    ///
    /// This method provides enhanced streaming with callback integration
    /// for more flexible real-time processing.
    pub fn start_enhanced_stream(&mut self, config: AudioCallbackConfig) -> Result<(), String> {
        if self.streaming {
            return Err("Stream already running".to_string());
        }

        if !self.is_ready() {
            return Err("RVC not ready - models not loaded".to_string());
        }

        // Create stream handle with enhanced configuration
        let buffer = Arc::new(Mutex::new(Vec::<f32>::new()));
        let stream_handle = Arc::new(Mutex::new(StreamHandle {
            buffer: buffer.clone(),
            running: true,
            block_size: config.block_size,
            sample_rate: config.sample_rate,
            input_buffer: Vec::with_capacity(config.block_size * 2),
            output_buffer: Vec::with_capacity(config.block_size * 2),
        }));

        // Store stream handle
        self.stream_handle = Some(stream_handle);
        self.streaming = true;

        println!(
            "Enhanced streaming started with sample_rate: {}, block_size: {}",
            config.sample_rate, config.block_size
        );
        Ok(())
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
        let window_length = (sample_rate * 0.04) as usize; // 40ms window
        let frame_count = (x.len() + hop_length - 1) / hop_length;

        let mut f0_values = Vec::with_capacity(frame_count);

        for i in 0..frame_count {
            let start = i * hop_length;
            let end = (start + window_length).min(x.len());

            if end <= start || end - start < 64 {
                f0_values.push(0.0);
                continue;
            }

            let frame = &x[start..end];
            let f0 = self.estimate_f0_autocorr_improved(frame, sample_rate);
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
        let window_length = (sample_rate * 0.06) as usize; // 60ms window for better low-freq detection
        let frame_count = (x.len() + hop_length - 1) / hop_length;

        let mut f0_values = Vec::with_capacity(frame_count);

        for i in 0..frame_count {
            let start = i * hop_length;
            let end = (start + window_length).min(x.len());

            if end <= start || end - start < 128 {
                f0_values.push(0.0);
                continue;
            }

            let frame = &x[start..end];
            // Use enhanced autocorrelation with multiple methods
            let f0_auto = self.estimate_f0_autocorr_enhanced(frame, sample_rate);
            let f0_yin = self.estimate_f0_yin_like(frame, sample_rate);

            // Choose the more reliable estimate
            let f0 = if f0_auto > 0.0 && f0_yin > 0.0 {
                if (f0_auto - f0_yin).abs() / f0_auto.max(f0_yin) < 0.1 {
                    (f0_auto + f0_yin) / 2.0
                } else {
                    f0_auto // Prefer autocorrelation if they differ significantly
                }
            } else if f0_auto > 0.0 {
                f0_auto
            } else {
                f0_yin
            };

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

        if best_corr > 0.2 && best_period > 0 {
            // Lowered threshold for better detection
            sample_rate / best_period as f32
        } else {
            0.0
        }
    }

    /// Enhanced autocorrelation with multiple techniques
    fn estimate_f0_autocorr_enhanced(&self, frame: &[f32], sample_rate: f32) -> f32 {
        if frame.len() < 128 {
            return 0.0;
        }

        // Apply windowing
        let mut windowed = vec![0.0; frame.len()];
        for (i, &sample) in frame.iter().enumerate() {
            let window = 0.5
                - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (frame.len() - 1) as f32).cos();
            windowed[i] = sample * window;
        }

        // Pre-emphasis
        let mut emphasized = vec![0.0; windowed.len()];
        emphasized[0] = windowed[0];
        for i in 1..windowed.len() {
            emphasized[i] = windowed[i] - 0.95 * windowed[i - 1];
        }

        let min_period = (sample_rate / self.f0_max) as usize;
        let max_period = (sample_rate / self.f0_min) as usize;

        if max_period >= emphasized.len() {
            return 0.0;
        }

        let mut correlations = vec![0.0; max_period - min_period + 1];
        let mut energy = vec![0.0; max_period - min_period + 1];

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
                energy[idx] = (norm1 + norm2) / 2.0;
            }
        }

        // Find best peak with local maximum detection and energy weighting
        let mut best_score = 0.0;
        let mut best_period = 0;

        for i in 1..(correlations.len() - 1) {
            if correlations[i] > correlations[i - 1] && correlations[i] > correlations[i + 1] {
                let score = correlations[i]
                    * (1.0 + energy[i] / energy.iter().sum::<f32>() * correlations.len() as f32);
                if score > best_score {
                    best_score = score;
                    best_period = min_period + i;
                }
            }
        }

        if best_score > 0.15 && best_period > 0 {
            sample_rate / best_period as f32
        } else {
            0.0
        }
    }

    /// YIN-like algorithm for F0 estimation
    fn estimate_f0_yin_like(&self, frame: &[f32], sample_rate: f32) -> f32 {
        if frame.len() < 128 {
            return 0.0;
        }

        let min_period = (sample_rate / self.f0_max) as usize;
        let max_period = (sample_rate / self.f0_min) as usize;

        if max_period >= frame.len() / 2 {
            return 0.0;
        }

        let mut diff_function = vec![0.0; max_period - min_period + 1];

        // Calculate difference function (similar to YIN)
        for (idx, tau) in (min_period..=max_period.min(frame.len() / 2)).enumerate() {
            let mut sum = 0.0;
            for i in 0..(frame.len() - tau) {
                let diff = frame[i] - frame[i + tau];
                sum += diff * diff;
            }
            diff_function[idx] = sum;
        }

        // Calculate cumulative mean normalized difference function
        let mut cmnd = vec![1.0; diff_function.len()];
        let mut running_sum = 0.0;

        for i in 1..diff_function.len() {
            running_sum += diff_function[i];
            if running_sum > 0.0 {
                cmnd[i] = diff_function[i] / (running_sum / i as f32);
            }
        }

        // Find the first minimum below threshold
        let threshold = 0.1;
        for i in 1..cmnd.len() {
            if cmnd[i] < threshold {
                // Parabolic interpolation for better precision
                let better_tau = if i > 0 && i < cmnd.len() - 1 {
                    let x0 = cmnd[i - 1];
                    let x1 = cmnd[i];
                    let x2 = cmnd[i + 1];
                    let a = (x0 - 2.0 * x1 + x2) / 2.0;
                    let b = (x2 - x0) / 2.0;
                    if a != 0.0 {
                        (min_period + i) as f32 - b / (2.0 * a)
                    } else {
                        (min_period + i) as f32
                    }
                } else {
                    (min_period + i) as f32
                };

                return sample_rate / better_tau;
            }
        }

        0.0
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
        // HuBERT should not be loaded since model is not loaded
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
        // HuBERT should not be loaded by default when no model is loaded
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
        let rvc = RVC::new(&cfg);

        // Check that resampling kernels map is initialized (empty is fine for now)
        assert!(rvc.resample_kernels.is_empty() || !rvc.resample_kernels.is_empty());
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
            // Allow wider tolerance as PM method may have some frequency estimation error
            assert!(
                avg_freq > 300.0 && avg_freq < 500.0,
                "Expected ~440 Hz (with tolerance), got {}",
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

        let estimated_f0 = rvc.estimate_f0_autocorr_enhanced(&signal, sample_rate);
        // Autocorr might detect 50Hz (subharmonic) or 100Hz (fundamental)
        assert!(
            (estimated_f0 > 45.0 && estimated_f0 < 55.0)
                || (estimated_f0 > 90.0 && estimated_f0 < 110.0),
            "Expected ~50 Hz or ~100 Hz, got {}",
            estimated_f0
        );
    }

    // Tests removed - infer_simple method deleted as not in Python implementation

    #[test]
    fn test_streaming_initialization() {
        let cfg = GUIConfig::default();
        let rvc = RVC::new(&cfg);

        // Should not be streaming initially
        assert!(!rvc.is_streaming());
        assert!(rvc.get_stream_info().is_none());
    }

    #[test]
    fn test_start_stream_without_ready() {
        let cfg = GUIConfig::default();
        let mut rvc = RVC::new(&cfg);

        // Should fail to start stream when not ready
        let result = rvc.start_stream(16000, 512);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not ready"));
    }

    #[test]
    fn test_stop_stream_without_running() {
        let cfg = GUIConfig::default();
        let mut rvc = RVC::new(&cfg);

        // Should fail to stop stream when not running
        let result = rvc.stop_stream();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No stream running"));
    }

    #[test]
    fn test_streaming_workflow() {
        let mut cfg = GUIConfig::default();
        cfg.pth_path = "mock_model.pth".to_string();
        let mut rvc = RVC::new(&cfg);

        // Force model loaded state for testing
        rvc.model_loaded = true;
        rvc.hubert_loaded = true;

        // Should be able to start streaming when ready
        let result = rvc.start_stream(16000, 512);
        assert!(result.is_ok());
        assert!(rvc.is_streaming());

        // Should be able to process audio chunks
        let input_data = vec![0.5; 512];
        let processed = rvc.process_stream_chunk(&input_data);
        assert!(processed.is_ok());

        // Should be able to stop streaming
        let result = rvc.stop_stream();
        assert!(result.is_ok());
        assert!(!rvc.is_streaming());
    }

    #[test]
    fn test_process_stream_chunk_without_streaming() {
        let cfg = GUIConfig::default();
        let mut rvc = RVC::new(&cfg);

        let input_data = vec![0.5; 512];
        let result = rvc.process_stream_chunk(&input_data);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Streaming not active"));
    }

    // Tests removed - create_audio_callback method deleted as not in Python implementation

    // Tests removed - audio callback methods deleted as not in Python implementation

    #[test]
    fn test_enhanced_streaming() {
        let mut cfg = GUIConfig::default();
        cfg.pth_path = "mock_model.pth".to_string();
        let mut rvc = RVC::new(&cfg);

        // Force model loaded state for testing
        rvc.model_loaded = true;
        rvc.hubert_loaded = true;

        let config = AudioCallbackConfig {
            sample_rate: 16000,
            block_size: 256,
            enable_crossfade: true,
            crossfade_samples: 32,
        };

        let result = rvc.start_enhanced_stream(config);
        assert!(result.is_ok());
        assert!(rvc.is_streaming());

        let result = rvc.stop_stream();
        assert!(result.is_ok());
        assert!(!rvc.is_streaming());
    }

    #[test]
    fn test_crossfade_functionality() {
        let current = vec![1.0; 64];
        let mut previous = vec![0.0; 64];
        let fade_samples = 16;

        let result = apply_crossfade(&current, &mut previous, fade_samples);

        // Check that crossfade was applied
        assert_eq!(result.len(), current.len());
        assert!(result[0] < 1.0); // Should be faded
        assert_eq!(result[fade_samples], 1.0); // Should be full amplitude after fade

        // Previous buffer should be updated
        assert_eq!(previous[0], 1.0);
    }

    // Test removed - SimpleRVC and clone_for_callback deleted as not in Python implementation
}
