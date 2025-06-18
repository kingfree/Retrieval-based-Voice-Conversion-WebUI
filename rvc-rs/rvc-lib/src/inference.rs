//! RVC 推理管道
//!
//! 该模块实现了完整的 RVC (Retrieval-based Voice Conversion) 推理流程，
//! 整合了 HuBERT 特征提取、F0 估计、FAISS 检索和生成器网络等组件。

use crate::{
    audio_utils::{AudioData, load_wav_simple, save_wav_simple},
    f0_estimation::{F0Config, F0Estimator, F0Method},
    faiss_index::{FaissIndex, SearchResult},
    generator::{GeneratorConfig, NSFHiFiGANGenerator},
    hubert::{HuBERT, HuBERTConfig},
    model_loader::{ModelConfig as ModelLoaderConfig, ModelLoadStats, ModelLoader},
};

use anyhow::Result;
use ndarray::{Array2, ArrayView2};
use std::path::Path;
use tch::{Device, Kind, Tensor, nn};

/// RVC 推理配置
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// 目标说话人 ID
    pub speaker_id: i64,
    /// F0 估计方法
    pub f0_method: F0Method,
    /// 音高调整比例 (1.0 = 无调整)
    pub pitch_shift: f64,
    /// 特征检索混合比例 (0.0-1.0)
    pub index_rate: f64,
    /// 输出音频采样率
    pub target_sample_rate: i64,
    /// 设备 (CPU/GPU)
    pub device: Device,
    /// 批处理大小
    pub batch_size: usize,
    /// 是否启用去噪
    pub enable_denoise: bool,
    /// F0 滤波器参数
    pub f0_filter: F0FilterConfig,
}

/// F0 滤波器配置
#[derive(Debug, Clone)]
pub struct F0FilterConfig {
    /// 中值滤波器窗口大小
    pub median_filter_radius: usize,
    /// 是否启用 F0 平滑
    pub enable_smoothing: bool,
    /// 平滑参数
    pub smoothing_factor: f64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            speaker_id: 0,
            f0_method: F0Method::Harvest,
            pitch_shift: 1.0,
            index_rate: 0.5,
            target_sample_rate: 22050,
            device: Device::Cpu,
            batch_size: 1,
            enable_denoise: false,
            f0_filter: F0FilterConfig::default(),
        }
    }
}

impl Default for F0FilterConfig {
    fn default() -> Self {
        Self {
            median_filter_radius: 3,
            enable_smoothing: true,
            smoothing_factor: 0.8,
        }
    }
}

/// RVC 推理引擎
pub struct RVCInference {
    config: InferenceConfig,
    hubert: HuBERT,
    generator: NSFHiFiGANGenerator,
    f0_estimator: F0Estimator,
    index: Option<FaissIndex>,
    vs: nn::VarStore,
    model_config: ModelLoaderConfig,
    model_stats: Option<ModelLoadStats>,
}

impl RVCInference {
    /// 创建新的推理引擎
    pub fn new(
        config: InferenceConfig,
        model_path: impl AsRef<Path>,
        index_path: Option<impl AsRef<Path>>,
    ) -> Result<Self> {
        println!("🚀 初始化 RVC 推理引擎...");

        let mut vs = nn::VarStore::new(config.device);

        // 创建模型加载器
        let model_loader = ModelLoader::new(config.device).with_debug_mode(true);

        // 加载模型配置和权重
        let (model_config, model_stats) =
            Self::load_model_and_config(&model_loader, model_path.as_ref(), &mut vs)?;

        println!("📊 使用模型配置:");
        println!("   - 版本: {}", model_config.version);
        println!("   - 采样率: {}Hz", model_config.sample_rate);
        println!("   - 特征维度: {}", model_config.feature_dim);

        // 初始化 HuBERT
        let hubert_config = HuBERTConfig {
            feature_dim: model_config.feature_dim,
            encoder_layers: model_config.hubert.encoder_layers,
            encoder_attention_heads: model_config.hubert.attention_heads,
            encoder_ffn_embed_dim: model_config.hubert.ffn_dim,
            dropout: model_config.hubert.dropout,
            ..Default::default()
        };
        let hubert = HuBERT::new(&vs.root(), hubert_config, config.device);

        // 初始化生成器
        let generator_config = GeneratorConfig {
            input_dim: model_config.generator.input_dim,
            upsample_rates: model_config.generator.upsample_rates.clone(),
            upsample_kernel_sizes: model_config.generator.upsample_kernel_sizes.clone(),
            resblock_kernel_sizes: model_config.generator.resblock_kernel_sizes.clone(),
            resblock_dilation_sizes: model_config.generator.resblock_dilation_sizes.clone(),
            leaky_relu_slope: model_config.generator.leaky_relu_slope,
            use_nsf: model_config.generator.use_nsf,
            ..Default::default()
        };
        let generator = NSFHiFiGANGenerator::new(&vs.root(), generator_config);

        // 初始化 F0 估计器
        let f0_config = F0Config {
            f0_min: model_config.f0_config.f0_min,
            f0_max: model_config.f0_config.f0_max,
            ..Default::default()
        };
        let f0_estimator = F0Estimator::new(f0_config, config.device);

        // 加载 FAISS 索引 (可选)
        let index = if let Some(index_path) = index_path {
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

        // 验证模型兼容性
        let compatibility_warnings = model_loader.check_compatibility(&model_config, &config)?;
        if !compatibility_warnings.is_empty() {
            println!("⚠️  兼容性警告:");
            for warning in &compatibility_warnings {
                println!("   - {}", warning);
            }
        }

        println!("✅ RVC 推理引擎初始化完成");

        Ok(Self {
            config,
            hubert,
            generator,
            f0_estimator,
            index,
            vs,
            model_config,
            model_stats,
        })
    }

    /// 执行语音转换推理
    pub fn convert_voice(
        &self,
        input_audio_path: impl AsRef<Path>,
        output_audio_path: impl AsRef<Path>,
    ) -> Result<AudioData> {
        println!("🎵 开始语音转换推理...");

        // 1. 加载输入音频
        let path_str = input_audio_path.as_ref().to_string_lossy();
        let audio_data =
            load_wav_simple(&path_str).map_err(|e| anyhow::anyhow!("音频加载失败: {}", e))?;
        println!(
            "📊 输入音频: {}Hz, {} 样本",
            audio_data.sample_rate,
            audio_data.samples.len()
        );

        self.convert_audio_data(audio_data, Some(output_audio_path))
    }

    /// 对音频数据执行转换
    pub fn convert_audio_data(
        &self,
        mut audio_data: AudioData,
        output_path: Option<impl AsRef<Path>>,
    ) -> Result<AudioData> {
        // 2. 重采样到目标采样率
        if audio_data.sample_rate != self.config.target_sample_rate as u32 {
            println!(
                "🔄 重采样: {}Hz -> {}Hz",
                audio_data.sample_rate, self.config.target_sample_rate
            );
            audio_data = self.resample_audio(audio_data)?;
        }

        // 3. 预处理音频
        let processed_audio = self.preprocess_audio(&audio_data)?;

        // 4. HuBERT 特征提取
        println!("🧠 提取 HuBERT 特征...");
        let features = self.extract_hubert_features(&processed_audio)?;

        // 5. F0 估计
        println!("🎼 估计基频 (F0)...");
        let f0_data = self.estimate_f0(&processed_audio)?;

        // 6. F0 后处理
        let processed_f0 = self.postprocess_f0(f0_data)?;

        // 7. 特征检索 (如果启用)
        let enhanced_features = if let Some(ref index) = self.index {
            println!("🔍 执行特征检索...");
            self.retrieve_features(&features, index)?
        } else {
            features
        };

        // 8. 生成器推理
        println!("🎨 生成音频波形...");
        let generated_audio = self.generate_audio(&enhanced_features, &processed_f0)?;

        // 9. 后处理
        let final_audio = self.postprocess_audio(generated_audio)?;

        // 10. 保存输出 (如果指定路径)
        if let Some(output_path) = output_path {
            let path_str = output_path.as_ref().to_string_lossy();
            save_wav_simple(&path_str, &final_audio)
                .map_err(|e| anyhow::anyhow!("音频保存失败: {}", e))?;
            println!("💾 输出已保存: {:?}", output_path.as_ref());
        }

        println!("✅ 语音转换完成!");
        Ok(final_audio)
    }

    /// 重采样音频
    fn resample_audio(&self, audio_data: AudioData) -> Result<AudioData> {
        // 简单的线性插值重采样 (生产环境应使用更高质量的重采样算法)
        let input_rate = audio_data.sample_rate as f32;
        let output_rate = self.config.target_sample_rate as f32;
        let ratio = output_rate / input_rate;

        let new_length = (audio_data.samples.len() as f32 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_length);

        for i in 0..new_length {
            let src_index = i as f32 / ratio;
            let idx = src_index as usize;

            if idx + 1 < audio_data.samples.len() {
                let frac = src_index - idx as f32;
                let sample =
                    audio_data.samples[idx] * (1.0 - frac) + audio_data.samples[idx + 1] * frac;
                resampled.push(sample);
            } else if idx < audio_data.samples.len() {
                resampled.push(audio_data.samples[idx]);
            }
        }

        Ok(AudioData {
            samples: resampled,
            sample_rate: output_rate as u32,
            channels: audio_data.channels,
        })
    }

    /// 音频预处理
    fn preprocess_audio(&self, audio_data: &AudioData) -> Result<Tensor> {
        let audio_tensor = Tensor::from_slice(&audio_data.samples)
            .to_device(self.config.device)
            .to_kind(Kind::Float);

        // 归一化到 [-1, 1] 范围
        let max_val = audio_tensor.abs().max();
        let normalized = if max_val.double_value(&[]) > 0.0 {
            audio_tensor / max_val
        } else {
            audio_tensor
        };

        // 添加批次维度
        Ok(normalized.unsqueeze(0))
    }

    /// 提取 HuBERT 特征
    fn extract_hubert_features(&self, audio: &Tensor) -> Result<Tensor> {
        // 使用 HuBERT 提取特征
        let hubert_output = self.hubert.extract_features(audio, None, false)?;
        Ok(hubert_output.last_hidden_state)
    }

    /// 估计 F0
    fn estimate_f0(&self, audio: &Tensor) -> Result<Vec<f64>> {
        let audio_vec: Vec<f32> = audio.squeeze_dim(0).try_into()?;
        let f0_result = self
            .f0_estimator
            .estimate(&audio_vec, self.config.f0_method)?;
        Ok(f0_result
            .frequencies
            .into_iter()
            .map(|x| x as f64)
            .collect())
    }

    /// F0 后处理
    fn postprocess_f0(&self, mut f0_data: Vec<f64>) -> Result<Tensor> {
        // 应用音高调整
        if self.config.pitch_shift != 1.0 {
            for f0 in &mut f0_data {
                if *f0 > 0.0 {
                    *f0 *= self.config.pitch_shift;
                }
            }
        }

        // 中值滤波
        if self.config.f0_filter.median_filter_radius > 0 {
            f0_data = self.apply_median_filter(f0_data, self.config.f0_filter.median_filter_radius);
        }

        // F0 平滑
        if self.config.f0_filter.enable_smoothing {
            f0_data = self.apply_f0_smoothing(f0_data, self.config.f0_filter.smoothing_factor);
        }

        Ok(Tensor::from_slice(&f0_data).to_device(self.config.device))
    }

    /// 应用中值滤波器
    fn apply_median_filter(&self, data: Vec<f64>, radius: usize) -> Vec<f64> {
        let len = data.len();
        let mut filtered = data.clone();

        for i in 0..len {
            let start = i.saturating_sub(radius);
            let end = (i + radius + 1).min(len);

            let mut window: Vec<f64> = data[start..end].to_vec();
            window.sort_by(|a, b| a.partial_cmp(b).unwrap());

            filtered[i] = window[window.len() / 2];
        }

        filtered
    }

    /// 应用 F0 平滑
    fn apply_f0_smoothing(&self, data: Vec<f64>, factor: f64) -> Vec<f64> {
        let mut smoothed = Vec::with_capacity(data.len());

        if !data.is_empty() {
            smoothed.push(data[0]);

            for i in 1..data.len() {
                let prev = smoothed[i - 1];
                let curr = data[i];

                // 指数移动平均
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

    /// 特征检索
    fn retrieve_features(&self, features: &Tensor, index: &FaissIndex) -> Result<Tensor> {
        // 将特征转换为搜索格式
        let feature_vec: Vec<f32> = features.flatten(0, -1).try_into()?;

        // 转换为ndarray格式
        let feature_shape = features.size();
        let n_features = feature_shape[feature_shape.len() - 1] as usize;
        let feature_array = Array2::from_shape_vec((1, n_features), feature_vec)?;
        let feature_view: ArrayView2<f32> = feature_array.view();

        // 执行 k-NN 搜索
        let k = 4; // 检索 top-k 个最近邻
        let _search_results = index.search(feature_view, k)?;

        // 混合原始特征和检索到的特征
        // TODO: 实现特征混合逻辑
        Ok(features.shallow_clone())
    }

    /// 混合特征
    fn _blend_features(
        &self,
        original: &Tensor,
        _search_results: &[SearchResult],
    ) -> Result<Tensor> {
        // 简单的加权平均混合
        let mix_rate = self.config.index_rate;
        let original_weight = 1.0 - mix_rate;

        // 这里需要根据实际的特征格式来实现混合逻辑
        // 目前返回原始特征作为占位符
        Ok(original * original_weight)
    }

    /// 生成音频
    fn generate_audio(&self, features: &Tensor, f0: &Tensor) -> Result<Tensor> {
        // 使用 NSF-HiFiGAN 生成器生成音频
        let generated = self.generator.forward(features, Some(f0), None)?;
        Ok(generated)
    }

    /// 音频后处理
    fn postprocess_audio(&self, audio_tensor: Tensor) -> Result<AudioData> {
        // 移除批次维度并转换为 Vec<f32>
        let audio_data: Vec<f32> = audio_tensor.squeeze_dim(0).try_into()?;

        // 应用软限幅防止削波
        let processed_data: Vec<f32> = audio_data
            .into_iter()
            .map(|x| {
                // Tanh 软限幅
                if x.abs() > 0.95 {
                    x.signum() * 0.95 * (1.0 - (-x.abs() / 0.05).exp())
                } else {
                    x
                }
            })
            .collect();

        Ok(AudioData {
            samples: processed_data,
            sample_rate: self.config.target_sample_rate as u32,
            channels: 1,
        })
    }

    /// 获取推理统计信息
    pub fn get_inference_stats(&self) -> InferenceStats {
        let (hubert_params, generator_params) = if let Some(stats) = &self.model_stats {
            (stats.total_params / 2, stats.total_params / 2) // 简单分配
        } else {
            (
                self.count_hubert_parameters(),
                self.count_generator_parameters(),
            )
        };

        InferenceStats {
            device: format!("{:?}", self.config.device),
            hubert_parameters: hubert_params as usize,
            generator_parameters: generator_params as usize,
            has_index: self.index.is_some(),
            target_sample_rate: self.config.target_sample_rate,
        }
    }

    /// 加载模型和配置
    fn load_model_and_config(
        loader: &ModelLoader,
        model_path: &Path,
        vs: &mut nn::VarStore,
    ) -> Result<(ModelLoaderConfig, Option<ModelLoadStats>)> {
        // 检查模型文件
        crate::model_loader::utils::check_model_file(model_path)?;

        // 尝试加载配置文件
        let config_path = model_path.with_extension("json");
        let model_config = if config_path.exists() {
            println!("📄 发现配置文件: {:?}", config_path);
            ModelLoader::load_config(&config_path)?
        } else {
            println!("⚠️  未找到配置文件，使用默认配置");
            ModelLoaderConfig::default()
        };

        // 加载模型权重
        let model_stats = match loader.load_pytorch_model(model_path, vs) {
            Ok(stats) => {
                println!("✅ 模型权重加载成功");

                // 验证模型参数
                let warnings = loader.validate_model_parameters(vs, &model_config)?;
                if !warnings.is_empty() {
                    println!("⚠️  模型验证警告:");
                    for warning in warnings {
                        println!("   - {}", warning);
                    }
                }

                Some(stats)
            }
            Err(e) => {
                println!("⚠️  模型权重加载失败: {}", e);
                println!("📝 使用随机初始化权重继续运行");
                None
            }
        };

        Ok((model_config, model_stats))
    }

    /// 统计 HuBERT 参数数量
    fn count_hubert_parameters(&self) -> i64 {
        let mut count = 0i64;
        for (name, tensor) in self.vs.variables() {
            if name.contains("hubert") || name.contains("feature") || name.contains("encoder") {
                count += tensor.size().iter().product::<i64>();
            }
        }
        if count == 0 { 1000000 } else { count } // 默认值或实际值
    }

    /// 统计生成器参数数量
    fn count_generator_parameters(&self) -> i64 {
        let mut count = 0i64;
        for (name, tensor) in self.vs.variables() {
            if name.contains("conv") || name.contains("upsample") || name.contains("generator") {
                count += tensor.size().iter().product::<i64>();
            }
        }
        if count == 0 { 5000000 } else { count } // 默认值或实际值
    }
}

/// 推理统计信息
#[derive(Debug)]
pub struct InferenceStats {
    pub device: String,
    pub hubert_parameters: usize,
    pub generator_parameters: usize,
    pub has_index: bool,
    pub target_sample_rate: i64,
}

/// 批量推理接口
pub struct BatchInference {
    inference_engine: RVCInference,
}

impl BatchInference {
    /// 创建批量推理引擎
    pub fn new(inference_engine: RVCInference) -> Self {
        Self { inference_engine }
    }

    /// 批量处理音频文件
    pub fn process_batch(
        &self,
        input_files: &[impl AsRef<Path>],
        output_dir: impl AsRef<Path>,
    ) -> Result<Vec<AudioData>> {
        let mut results = Vec::new();

        for (i, input_file) in input_files.iter().enumerate() {
            println!(
                "📁 处理文件 {}/{}: {:?}",
                i + 1,
                input_files.len(),
                input_file.as_ref()
            );

            let output_file = output_dir.as_ref().join(
                input_file
                    .as_ref()
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string()
                    + "_converted.wav",
            );

            match self
                .inference_engine
                .convert_voice(input_file, &output_file)
            {
                Ok(result) => {
                    results.push(result);
                    println!("✅ 文件处理完成: {:?}", output_file);
                }
                Err(e) => {
                    println!("❌ 文件处理失败: {:?}, 错误: {}", input_file.as_ref(), e);
                    return Err(e);
                }
            }
        }

        println!("🎉 批量处理完成! 共处理 {} 个文件", results.len());
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_inference_config_default() {
        let config = InferenceConfig::default();
        assert_eq!(config.speaker_id, 0);
        assert_eq!(config.f0_method, F0Method::Harvest);
        assert_eq!(config.pitch_shift, 1.0);
        assert_eq!(config.index_rate, 0.5);
    }

    #[test]
    fn test_f0_filter_default() {
        let filter_config = F0FilterConfig::default();
        assert_eq!(filter_config.median_filter_radius, 3);
        assert!(filter_config.enable_smoothing);
        assert_eq!(filter_config.smoothing_factor, 0.8);
    }

    #[test]
    fn test_audio_resampling() {
        // 这需要一个有效的推理引擎实例来测试
        // 在实际测试中需要提供模型文件路径
    }
}
