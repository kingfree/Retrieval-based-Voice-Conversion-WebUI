//! 完整的音频处理管道
//!
//! 该模块实现了端到端的音频处理流程，包括预处理、特征提取、
//! 语音转换和后处理等步骤。提供了高级 API 来简化音频转换操作。

use crate::{
    audio_utils::{AudioData, load_wav_simple, save_wav_simple},
    f0_estimation::{F0Config, F0Estimator},
    faiss_index::FaissIndex,
    generator::NSFHiFiGANGenerator,
    hubert::HuBERT,
    inference::InferenceConfig,
    model_loader::{ModelConfig as ModelLoaderConfig, ModelLoader},
};

use anyhow::{Result, anyhow};
use std::path::Path;
use std::time::Instant;
use tch::{Device, Tensor, nn};

/// 音频处理管道配置
#[derive(Debug, Clone)]
pub struct AudioPipelineConfig {
    /// 输入音频路径
    pub input_path: String,
    /// 输出音频路径
    pub output_path: String,
    /// 模型文件路径
    pub model_path: String,
    /// 索引文件路径（可选）
    pub index_path: Option<String>,
    /// 推理配置
    pub inference_config: InferenceConfig,
    /// 音频预处理配置
    pub preprocessing: AudioPreprocessingConfig,
    /// 音频后处理配置
    pub postprocessing: AudioPostprocessingConfig,
}

/// 音频预处理配置
#[derive(Debug, Clone)]
pub struct AudioPreprocessingConfig {
    /// 是否进行音频标准化
    pub normalize: bool,
    /// 是否移除静音
    pub remove_silence: bool,
    /// 静音阈值
    pub silence_threshold: f32,
    /// 是否应用预加重
    pub preemphasis: bool,
    /// 预加重系数
    pub preemphasis_coefficient: f32,
    /// 目标响度 (LUFS)
    pub target_lufs: Option<f32>,
}

/// 音频后处理配置
#[derive(Debug, Clone)]
pub struct AudioPostprocessingConfig {
    /// 是否应用去加重
    pub deemphasis: bool,
    /// 去加重系数
    pub deemphasis_coefficient: f32,
    /// 是否应用软限幅
    pub apply_soft_clipping: bool,
    /// 软限幅阈值
    pub soft_clip_threshold: f32,
    /// 是否应用噪声门限
    pub apply_noise_gate: bool,
    /// 噪声门限阈值
    pub noise_gate_threshold: f32,
    /// 输出增益 (dB)
    pub output_gain_db: f32,
}

/// 处理阶段
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProcessingStage {
    Loading,
    Preprocessing,
    FeatureExtraction,
    F0Estimation,
    VoiceConversion,
    Postprocessing,
    Saving,
    Complete,
}

/// 处理进度信息
#[derive(Debug, Clone)]
pub struct ProcessingProgress {
    /// 当前阶段
    pub stage: ProcessingStage,
    /// 进度百分比 (0-100)
    pub progress: f32,
    /// 阶段描述
    pub description: String,
    /// 耗时 (毫秒)
    pub elapsed_ms: u64,
    /// 是否有错误
    pub has_error: bool,
    /// 错误信息
    pub error_message: Option<String>,
}

/// 进度回调函数类型
pub type ProgressCallback = Box<dyn Fn(ProcessingProgress) + Send + Sync>;

/// 音频处理管道
pub struct AudioPipeline {
    config: AudioPipelineConfig,
    _model_config: ModelLoaderConfig,
    _vs: nn::VarStore,
    hubert: HuBERT,
    generator: NSFHiFiGANGenerator,
    f0_estimator: F0Estimator,
    index: Option<FaissIndex>,
    progress_callback: Option<ProgressCallback>,
    start_time: Instant,
}

impl AudioPipeline {
    /// 创建新的音频处理管道
    pub fn new(config: AudioPipelineConfig) -> Result<Self> {
        let start_time = Instant::now();

        // 初始化设备和变量存储
        let mut vs = nn::VarStore::new(config.inference_config.device);

        // 创建模型加载器并加载配置
        let model_loader = ModelLoader::new(config.inference_config.device);
        let model_config = Self::load_model_config(&config.model_path)?;

        // 加载模型权重
        if Path::new(&config.model_path).exists() {
            println!("📁 加载模型权重: {}", config.model_path);
            match model_loader.load_pytorch_model(&config.model_path, &mut vs) {
                Ok(stats) => {
                    println!(
                        "✅ 模型加载成功: {:.1}M 参数",
                        stats.total_params as f64 / 1_000_000.0
                    );
                }
                Err(e) => {
                    println!("⚠️  模型加载失败，使用随机权重: {}", e);
                }
            }
        }

        // 初始化组件
        let hubert = Self::create_hubert(&vs, &model_config, config.inference_config.device)?;
        let generator = Self::create_generator(&vs, &model_config)?;
        let f0_estimator =
            Self::create_f0_estimator(&model_config, config.inference_config.device)?;

        // 加载 FAISS 索引
        let index = if let Some(ref index_path) = config.index_path {
            match FaissIndex::load(index_path) {
                Ok(idx) => {
                    println!("✅ FAISS 索引加载成功");
                    Some(idx)
                }
                Err(e) => {
                    println!("⚠️  FAISS 索引加载失败: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            config,
            _model_config: model_config,
            _vs: vs,
            hubert,
            generator,
            f0_estimator,
            index,
            progress_callback: None,
            start_time,
        })
    }

    /// 设置进度回调
    pub fn with_progress_callback(mut self, callback: ProgressCallback) -> Self {
        self.progress_callback = Some(callback);
        self
    }

    /// 执行完整的音频处理流程
    pub async fn process(&mut self) -> Result<AudioData> {
        self.update_progress(ProcessingStage::Loading, 0.0, "开始处理音频");

        // 1. 加载音频
        let mut audio_data = self.load_audio().await?;
        self.update_progress(ProcessingStage::Loading, 100.0, "音频加载完成");

        // 2. 预处理
        self.update_progress(ProcessingStage::Preprocessing, 0.0, "开始音频预处理");
        audio_data = self.preprocess_audio(audio_data).await?;
        self.update_progress(ProcessingStage::Preprocessing, 100.0, "音频预处理完成");

        // 3. 特征提取
        self.update_progress(ProcessingStage::FeatureExtraction, 0.0, "提取 HuBERT 特征");
        let features = self.extract_features(&audio_data).await?;
        self.update_progress(ProcessingStage::FeatureExtraction, 100.0, "特征提取完成");

        // 4. F0 估计
        self.update_progress(ProcessingStage::F0Estimation, 0.0, "估计基频 (F0)");
        let f0_data = self.estimate_f0(&audio_data).await?;
        self.update_progress(ProcessingStage::F0Estimation, 100.0, "F0 估计完成");

        // 5. 语音转换
        self.update_progress(ProcessingStage::VoiceConversion, 0.0, "执行语音转换");
        let converted_audio = self.perform_voice_conversion(&features, &f0_data).await?;
        self.update_progress(ProcessingStage::VoiceConversion, 100.0, "语音转换完成");

        // 6. 后处理
        self.update_progress(ProcessingStage::Postprocessing, 0.0, "开始音频后处理");
        let processed_audio = self.postprocess_audio(converted_audio).await?;
        self.update_progress(ProcessingStage::Postprocessing, 100.0, "音频后处理完成");

        // 7. 保存结果
        self.update_progress(ProcessingStage::Saving, 0.0, "保存输出音频");
        self.save_audio(&processed_audio).await?;
        self.update_progress(ProcessingStage::Saving, 100.0, "音频保存完成");

        self.update_progress(ProcessingStage::Complete, 100.0, "处理完成");

        Ok(processed_audio)
    }

    /// 加载音频文件
    async fn load_audio(&self) -> Result<AudioData> {
        let audio_data =
            load_wav_simple(&self.config.input_path).map_err(|e| anyhow!("加载音频失败: {}", e))?;

        println!("📊 输入音频信息:");
        println!("   - 文件: {}", self.config.input_path);
        println!("   - 采样率: {}Hz", audio_data.sample_rate);
        println!(
            "   - 时长: {:.2}s",
            audio_data.samples.len() as f32 / audio_data.sample_rate as f32
        );
        println!("   - 声道数: {}", audio_data.channels);

        Ok(audio_data)
    }

    /// 音频预处理
    async fn preprocess_audio(&self, mut audio: AudioData) -> Result<AudioData> {
        let config = &self.config.preprocessing;

        // 应用预加重
        if config.preemphasis {
            audio = self.apply_preemphasis(audio, config.preemphasis_coefficient)?;
        }

        // 移除静音
        if config.remove_silence {
            audio = self.remove_silence(audio, config.silence_threshold)?;
        }

        // 音频标准化
        if config.normalize {
            audio = self.normalize_audio(audio)?;
        }

        // 响度标准化
        if let Some(target_lufs) = config.target_lufs {
            audio = self.normalize_loudness(audio, target_lufs)?;
        }

        // 重采样到目标采样率
        if audio.sample_rate != self.config.inference_config.target_sample_rate as u32 {
            audio = self.resample_audio(audio)?;
        }

        Ok(audio)
    }

    /// 提取 HuBERT 特征
    async fn extract_features(&self, audio: &AudioData) -> Result<Tensor> {
        let audio_tensor = Tensor::from_slice(&audio.samples)
            .to_device(self.config.inference_config.device)
            .unsqueeze(0); // 添加批次维度

        let output = self.hubert.extract_features(&audio_tensor, None, false)?;
        Ok(output.last_hidden_state)
    }

    /// 估计 F0
    async fn estimate_f0(&self, audio: &AudioData) -> Result<Tensor> {
        let f0_result = self
            .f0_estimator
            .estimate(&audio.samples, self.config.inference_config.f0_method)?;

        // 应用音调调整
        let mut f0_values: Vec<f64> = f0_result
            .frequencies
            .into_iter()
            .map(|f| {
                if f > 0.0 {
                    f as f64 * self.config.inference_config.pitch_shift
                } else {
                    0.0
                }
            })
            .collect();

        // 应用 F0 滤波
        f0_values = self.apply_f0_filtering(f0_values)?;

        Ok(Tensor::from_slice(&f0_values).to_device(self.config.inference_config.device))
    }

    /// 执行语音转换
    async fn perform_voice_conversion(&self, features: &Tensor, f0: &Tensor) -> Result<AudioData> {
        // 如果有 FAISS 索引，进行特征检索
        let enhanced_features = if let Some(ref index) = self.index {
            self.enhance_features_with_retrieval(features, index)?
        } else {
            features.shallow_clone()
        };

        // 生成音频
        let output_tensor = self.generator.forward(&enhanced_features, Some(f0), None)?;

        // 转换为音频数据
        let output_samples: Vec<f32> = output_tensor
            .try_into()
            .map_err(|e| anyhow!("张量转换失败: {:?}", e))?;

        Ok(AudioData {
            samples: output_samples,
            sample_rate: self.config.inference_config.target_sample_rate as u32,
            channels: 1,
        })
    }

    /// 音频后处理
    async fn postprocess_audio(&self, mut audio: AudioData) -> Result<AudioData> {
        let config = &self.config.postprocessing;

        // 应用去加重
        if config.deemphasis {
            audio = self.apply_deemphasis(audio, config.deemphasis_coefficient)?;
        }

        // 应用软限幅
        if config.apply_soft_clipping {
            audio = self.apply_soft_clipping(audio, config.soft_clip_threshold)?;
        }

        // 应用噪声门限
        if config.apply_noise_gate {
            audio = self.apply_noise_gate(audio, config.noise_gate_threshold)?;
        }

        // 应用输出增益
        if config.output_gain_db != 0.0 {
            audio = self.apply_gain(audio, config.output_gain_db)?;
        }

        Ok(audio)
    }

    /// 保存音频文件
    async fn save_audio(&self, audio: &AudioData) -> Result<()> {
        save_wav_simple(&self.config.output_path, audio)
            .map_err(|e| anyhow!("保存音频失败: {}", e))?;

        println!("💾 输出音频已保存到: {}", self.config.output_path);
        Ok(())
    }

    /// 应用预加重
    fn apply_preemphasis(&self, mut audio: AudioData, coefficient: f32) -> Result<AudioData> {
        if !audio.samples.is_empty() {
            for i in (1..audio.samples.len()).rev() {
                audio.samples[i] -= coefficient * audio.samples[i - 1];
            }
        }
        Ok(audio)
    }

    /// 应用去加重
    fn apply_deemphasis(&self, mut audio: AudioData, coefficient: f32) -> Result<AudioData> {
        if !audio.samples.is_empty() {
            for i in 1..audio.samples.len() {
                audio.samples[i] += coefficient * audio.samples[i - 1];
            }
        }
        Ok(audio)
    }

    /// 移除静音
    fn remove_silence(&self, audio: AudioData, threshold: f32) -> Result<AudioData> {
        let mut result_samples = Vec::new();
        let window_size = 1024;
        let hop_size = 512;

        for start in (0..audio.samples.len()).step_by(hop_size) {
            let end = (start + window_size).min(audio.samples.len());
            let window = &audio.samples[start..end];

            // 计算窗口的 RMS
            let rms = (window.iter().map(|x| x * x).sum::<f32>() / window.len() as f32).sqrt();

            // 如果 RMS 超过阈值，保留该窗口
            if rms > threshold {
                result_samples.extend_from_slice(window);
            }
        }

        Ok(AudioData {
            samples: result_samples,
            ..audio
        })
    }

    /// 音频标准化
    fn normalize_audio(&self, mut audio: AudioData) -> Result<AudioData> {
        if let Some(max_val) = audio.samples.iter().map(|x| x.abs()).fold(None, |acc, x| {
            Some(match acc {
                Some(y) => x.max(y),
                None => x,
            })
        }) {
            if max_val > 0.0 {
                let scale = 0.95 / max_val; // 留一些余量避免削波
                for sample in &mut audio.samples {
                    *sample *= scale;
                }
            }
        }
        Ok(audio)
    }

    /// 响度标准化
    fn normalize_loudness(&self, audio: AudioData, target_lufs: f32) -> Result<AudioData> {
        // 简化的响度计算（实际应该使用 ITU-R BS.1770 标准）
        let rms =
            (audio.samples.iter().map(|x| x * x).sum::<f32>() / audio.samples.len() as f32).sqrt();
        let current_lufs = 20.0 * rms.log10(); // 简化计算

        let gain_db = target_lufs - current_lufs;
        self.apply_gain(audio, gain_db)
    }

    /// 应用增益
    fn apply_gain(&self, mut audio: AudioData, gain_db: f32) -> Result<AudioData> {
        let gain_linear = 10.0_f32.powf(gain_db / 20.0);
        for sample in &mut audio.samples {
            *sample *= gain_linear;
        }
        Ok(audio)
    }

    /// 应用软限幅
    fn apply_soft_clipping(&self, mut audio: AudioData, threshold: f32) -> Result<AudioData> {
        for sample in &mut audio.samples {
            let abs_val = sample.abs();
            if abs_val > threshold {
                *sample =
                    sample.signum() * threshold * (1.0 - (-((abs_val - threshold) / 0.1)).exp());
            }
        }
        Ok(audio)
    }

    /// 应用噪声门限
    fn apply_noise_gate(&self, mut audio: AudioData, threshold: f32) -> Result<AudioData> {
        let window_size = 1024;
        for chunk in audio.samples.chunks_mut(window_size) {
            let rms = (chunk.iter().map(|x| x * x).sum::<f32>() / chunk.len() as f32).sqrt();
            if rms < threshold {
                for sample in chunk {
                    *sample *= rms / threshold; // 衰减而不是完全静音
                }
            }
        }
        Ok(audio)
    }

    /// 重采样音频
    fn resample_audio(&self, audio: AudioData) -> Result<AudioData> {
        let target_rate = self.config.inference_config.target_sample_rate as f32;
        let ratio = target_rate / audio.sample_rate as f32;

        let new_length = (audio.samples.len() as f32 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_length);

        // 简单线性插值重采样
        for i in 0..new_length {
            let src_index = i as f32 / ratio;
            let idx = src_index as usize;

            if idx + 1 < audio.samples.len() {
                let frac = src_index - idx as f32;
                let sample = audio.samples[idx] * (1.0 - frac) + audio.samples[idx + 1] * frac;
                resampled.push(sample);
            } else if idx < audio.samples.len() {
                resampled.push(audio.samples[idx]);
            }
        }

        Ok(AudioData {
            samples: resampled,
            sample_rate: target_rate as u32,
            channels: audio.channels,
        })
    }

    /// 应用 F0 滤波
    fn apply_f0_filtering(&self, mut f0_values: Vec<f64>) -> Result<Vec<f64>> {
        let config = &self.config.inference_config.f0_filter;

        // 中值滤波
        if config.median_filter_radius > 0 {
            f0_values = self.median_filter(f0_values, config.median_filter_radius);
        }

        // 平滑滤波
        if config.enable_smoothing {
            f0_values = self.smooth_f0(f0_values, config.smoothing_factor);
        }

        Ok(f0_values)
    }

    /// 中值滤波
    fn median_filter(&self, data: Vec<f64>, radius: usize) -> Vec<f64> {
        let mut filtered = data.clone();
        let len = data.len();

        for i in 0..len {
            let start = i.saturating_sub(radius);
            let end = (i + radius + 1).min(len);

            let mut window: Vec<f64> = data[start..end].to_vec();
            window.sort_by(|a, b| a.partial_cmp(b).unwrap());

            filtered[i] = window[window.len() / 2];
        }

        filtered
    }

    /// F0 平滑
    fn smooth_f0(&self, data: Vec<f64>, factor: f64) -> Vec<f64> {
        let mut smoothed = Vec::with_capacity(data.len());

        if !data.is_empty() {
            smoothed.push(data[0]);

            for i in 1..data.len() {
                let prev = smoothed[i - 1];
                let curr = data[i];

                let smooth_val = if curr > 0.0 && prev > 0.0 {
                    prev * factor + curr * (1.0 - factor)
                } else {
                    curr
                };

                smoothed.push(smooth_val);
            }
        }

        smoothed
    }

    /// 使用检索增强特征
    fn enhance_features_with_retrieval(
        &self,
        features: &Tensor,
        _index: &FaissIndex,
    ) -> Result<Tensor> {
        // 简化的特征检索实现
        // 实际实现需要更复杂的特征混合逻辑
        let mix_rate = self.config.inference_config.index_rate;
        let enhanced = features * (1.0 - mix_rate);
        Ok(enhanced)
    }

    /// 更新进度
    fn update_progress(&self, stage: ProcessingStage, progress: f32, description: &str) {
        if let Some(ref callback) = self.progress_callback {
            let progress_info = ProcessingProgress {
                stage,
                progress,
                description: description.to_string(),
                elapsed_ms: self.start_time.elapsed().as_millis() as u64,
                has_error: false,
                error_message: None,
            };
            callback(progress_info);
        }
    }

    /// 加载模型配置
    fn load_model_config(model_path: &str) -> Result<ModelLoaderConfig> {
        let config_path = Path::new(model_path).with_extension("json");

        if config_path.exists() {
            ModelLoader::load_config(&config_path)
        } else {
            Ok(ModelLoaderConfig::default())
        }
    }

    /// 创建 HuBERT 实例
    fn create_hubert(
        vs: &nn::VarStore,
        model_config: &ModelLoaderConfig,
        device: Device,
    ) -> Result<HuBERT> {
        let hubert_config = crate::hubert::HuBERTConfig {
            feature_dim: model_config.feature_dim,
            encoder_layers: model_config.hubert.encoder_layers,
            encoder_attention_heads: model_config.hubert.attention_heads,
            encoder_ffn_embed_dim: model_config.hubert.ffn_dim,
            dropout: model_config.hubert.dropout,
            ..Default::default()
        };

        Ok(HuBERT::new(&vs.root(), hubert_config, device))
    }

    /// 创建生成器实例
    fn create_generator(
        vs: &nn::VarStore,
        model_config: &ModelLoaderConfig,
    ) -> Result<NSFHiFiGANGenerator> {
        let generator_config = crate::generator::GeneratorConfig {
            input_dim: model_config.generator.input_dim,
            upsample_rates: model_config.generator.upsample_rates.clone(),
            upsample_kernel_sizes: model_config.generator.upsample_kernel_sizes.clone(),
            resblock_kernel_sizes: model_config.generator.resblock_kernel_sizes.clone(),
            resblock_dilation_sizes: model_config.generator.resblock_dilation_sizes.clone(),
            leaky_relu_slope: model_config.generator.leaky_relu_slope,
            use_nsf: model_config.generator.use_nsf,
            ..Default::default()
        };

        Ok(NSFHiFiGANGenerator::new(&vs.root(), generator_config))
    }

    /// 创建 F0 估计器实例
    fn create_f0_estimator(
        model_config: &ModelLoaderConfig,
        device: Device,
    ) -> Result<F0Estimator> {
        let f0_config = F0Config {
            f0_min: model_config.f0_config.f0_min,
            f0_max: model_config.f0_config.f0_max,
            ..Default::default()
        };

        Ok(F0Estimator::new(f0_config, device))
    }
}

impl Default for AudioPreprocessingConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            remove_silence: false,
            silence_threshold: 0.01,
            preemphasis: true,
            preemphasis_coefficient: 0.97,
            target_lufs: Some(-23.0), // EBU R128 标准
        }
    }
}

impl Default for AudioPostprocessingConfig {
    fn default() -> Self {
        Self {
            deemphasis: true,
            deemphasis_coefficient: 0.97,
            apply_soft_clipping: true,
            soft_clip_threshold: 0.95,
            apply_noise_gate: false,
            noise_gate_threshold: 0.001,
            output_gain_db: 0.0,
        }
    }
}

/// 便捷的音频处理函数
pub mod utils {
    use super::*;

    /// 简单的音频转换
    pub async fn convert_audio_simple(
        input_path: &str,
        output_path: &str,
        model_path: &str,
    ) -> Result<()> {
        let config = AudioPipelineConfig {
            input_path: input_path.to_string(),
            output_path: output_path.to_string(),
            model_path: model_path.to_string(),
            index_path: None,
            inference_config: InferenceConfig::default(),
            preprocessing: AudioPreprocessingConfig::default(),
            postprocessing: AudioPostprocessingConfig::default(),
        };

        let mut pipeline = AudioPipeline::new(config)?;
        pipeline.process().await?;

        Ok(())
    }

    /// 带进度回调的音频转换
    pub async fn convert_audio_with_progress(
        input_path: &str,
        output_path: &str,
        model_path: &str,
        progress_callback: ProgressCallback,
    ) -> Result<()> {
        let config = AudioPipelineConfig {
            input_path: input_path.to_string(),
            output_path: output_path.to_string(),
            model_path: model_path.to_string(),
            index_path: None,
            inference_config: InferenceConfig::default(),
            preprocessing: AudioPreprocessingConfig::default(),
            postprocessing: AudioPostprocessingConfig::default(),
        };

        let mut pipeline = AudioPipeline::new(config)?.with_progress_callback(progress_callback);
        pipeline.process().await?;

        Ok(())
    }

    /// 批量音频转换
    pub async fn batch_convert_audio(
        input_files: &[String],
        output_dir: &str,
        model_path: &str,
        _progress_callback: Option<ProgressCallback>,
    ) -> Result<Vec<String>> {
        let mut output_files = Vec::new();

        for (i, input_file) in input_files.iter().enumerate() {
            let input_path = Path::new(input_file);
            let file_stem = input_path.file_stem().unwrap().to_string_lossy();
            let output_file = format!("{}/{}_converted.wav", output_dir, file_stem);

            println!("处理文件 {}/{}: {}", i + 1, input_files.len(), input_file);

            let config = AudioPipelineConfig {
                input_path: input_file.clone(),
                output_path: output_file.clone(),
                model_path: model_path.to_string(),
                index_path: None,
                inference_config: InferenceConfig::default(),
                preprocessing: AudioPreprocessingConfig::default(),
                postprocessing: AudioPostprocessingConfig::default(),
            };

            let mut pipeline = AudioPipeline::new(config)?;

            pipeline.process().await?;
            output_files.push(output_file);
        }

        Ok(output_files)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_audio_pipeline_config_default() {
        let preprocessing = AudioPreprocessingConfig::default();
        assert!(preprocessing.normalize);
        assert!(preprocessing.preemphasis);
        assert_eq!(preprocessing.preemphasis_coefficient, 0.97);

        let postprocessing = AudioPostprocessingConfig::default();
        assert!(postprocessing.deemphasis);
        assert!(postprocessing.apply_soft_clipping);
        assert_eq!(postprocessing.output_gain_db, 0.0);
    }

    #[test]
    fn test_processing_stage() {
        let stage = ProcessingStage::Loading;
        assert_eq!(stage, ProcessingStage::Loading);
        assert_ne!(stage, ProcessingStage::Complete);
    }

    #[tokio::test]
    async fn test_audio_pipeline_creation() {
        let config = AudioPipelineConfig {
            input_path: "test.wav".to_string(),
            output_path: "output.wav".to_string(),
            model_path: "model.pth".to_string(),
            index_path: None,
            inference_config: InferenceConfig {
                device: Device::Cpu,
                ..Default::default()
            },
            preprocessing: AudioPreprocessingConfig::default(),
            postprocessing: AudioPostprocessingConfig::default(),
        };

        // 注意：这个测试会失败，因为文件不存在
        // 但它验证了配置结构的正确性
        let result = AudioPipeline::new(config);
        assert!(result.is_err()); // 预期失败，因为模型文件不存在
    }
}
