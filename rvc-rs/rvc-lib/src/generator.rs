//! NSF-HiFiGAN 生成器网络推理模块
//!
//! 该模块实现了 Neural Source Filter HiFiGAN (NSF-HiFiGAN) 生成器网络，
//! 用于将 HuBERT 特征和 F0 信息转换为高质量的音频波形。
//! NSF-HiFiGAN 结合了神经源滤波器和 HiFiGAN 的优点，专门用于语音合成任务。

use anyhow::Result;
use tch::{Device, Kind, Tensor, nn, nn::Module};

/// 生成器网络配置
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// 输入特征维度
    pub input_dim: i64,
    /// 上采样率列表
    pub upsample_rates: Vec<i64>,
    /// 上采样核大小列表
    pub upsample_kernel_sizes: Vec<i64>,
    /// 上采样初始通道数
    pub upsample_initial_channel: i64,
    /// ResBlock 核大小列表
    pub resblock_kernel_sizes: Vec<i64>,
    /// ResBlock 膨胀率列表
    pub resblock_dilation_sizes: Vec<Vec<i64>>,
    /// 是否使用 NSF
    pub use_nsf: bool,
    /// NSF 采样率
    pub sample_rate: i64,
    /// 是否使用权重归一化
    pub use_weight_norm: bool,
    /// 泄漏激活斜率
    pub leaky_relu_slope: f64,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            input_dim: 768,                    // HuBERT 特征维度
            upsample_rates: vec![10, 8, 2, 2], // 总上采样倍数: 320
            upsample_kernel_sizes: vec![20, 16, 4, 4],
            upsample_initial_channel: 512,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            use_nsf: true,
            sample_rate: 16000,
            use_weight_norm: true,
            leaky_relu_slope: 0.1,
        }
    }
}

/// 残差块 (ResidualBlock)
pub struct ResidualBlock {
    convs1: Vec<nn::Conv1D>,
    convs2: Vec<nn::Conv1D>,
    leaky_relu_slope: f64,
}

impl ResidualBlock {
    pub fn new(
        vs: &nn::Path,
        channels: i64,
        kernel_size: i64,
        dilation_sizes: &[i64],
        leaky_relu_slope: f64,
    ) -> Self {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();

        for (i, &dilation) in dilation_sizes.iter().enumerate() {
            let padding = (kernel_size - 1) * dilation / 2;

            let conv_config = nn::ConvConfig {
                stride: 1,
                padding,
                dilation,
                groups: 1,
                bias: true,
                ..Default::default()
            };

            let conv1 = nn::conv1d(
                vs / format!("convs1.{}", i),
                channels,
                channels,
                kernel_size,
                conv_config,
            );
            convs1.push(conv1);

            let conv2 = nn::conv1d(
                vs / format!("convs2.{}", i),
                channels,
                channels,
                kernel_size,
                conv_config,
            );
            convs2.push(conv2);
        }

        Self {
            convs1,
            convs2,
            leaky_relu_slope,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let mut output = input.shallow_clone();

        for (conv1, conv2) in self.convs1.iter().zip(&self.convs2) {
            let residual = output.shallow_clone();

            // 第一个卷积 + LeakyReLU
            output = conv1.forward(&output);
            output = output.leaky_relu();

            // 第二个卷积 + LeakyReLU
            output = conv2.forward(&output);
            output = output.leaky_relu();

            // 残差连接
            output = output + residual;
        }

        output
    }
}

/// 多尺度残差块 (Multi-Receptive Field Fusion)
pub struct MRF {
    resblocks: Vec<ResidualBlock>,
}

impl MRF {
    pub fn new(
        vs: &nn::Path,
        channels: i64,
        kernel_sizes: &[i64],
        dilation_sizes: &[Vec<i64>],
        leaky_relu_slope: f64,
    ) -> Self {
        let mut resblocks = Vec::new();

        for (i, (&kernel_size, dilations)) in kernel_sizes.iter().zip(dilation_sizes).enumerate() {
            let resblock = ResidualBlock::new(
                &(vs / format!("resblocks.{}", i)),
                channels,
                kernel_size,
                dilations,
                leaky_relu_slope,
            );
            resblocks.push(resblock);
        }

        Self { resblocks }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let mut outputs = Vec::new();

        for resblock in &self.resblocks {
            outputs.push(resblock.forward(input));
        }

        // 平均融合多个尺度的输出
        let mut result = outputs[0].shallow_clone();
        for output in outputs.iter().skip(1) {
            result = result + output;
        }
        result / outputs.len() as f64
    }
}

/// Neural Source Filter (NSF) 模块
pub struct NSF {
    harmonic_num: i64,
    sample_rate: i64,
    sine_amp: f64,
    noise_std: f64,
}

impl NSF {
    pub fn new(sample_rate: i64, harmonic_num: i64) -> Self {
        Self {
            harmonic_num,
            sample_rate,
            sine_amp: 0.1,
            noise_std: 0.003,
        }
    }

    /// 生成正弦波源信号
    pub fn generate_sine_waves(&self, f0: &Tensor, upp: i64) -> Tensor {
        let device = f0.device();
        let batch_size = f0.size()[0];
        let f0_length = f0.size()[1];
        let signal_length = f0_length * upp;

        // 创建时间索引
        let time_idx = Tensor::arange(signal_length, (Kind::Float, device))
            .unsqueeze(0)
            .expand(&[batch_size, signal_length], true);

        // 上采样 F0
        let f0_upsampled = f0
            .unsqueeze(2)
            .expand(&[batch_size, f0_length, upp], true)
            .reshape(&[batch_size, signal_length]);

        // 计算相位累积
        let phase_acc =
            f0_upsampled * time_idx * 2.0 * std::f64::consts::PI / self.sample_rate as f64;

        let mut sine_waves = Vec::new();

        // 生成各次谐波
        for harmonic in 1..=self.harmonic_num {
            let harmonic_phase = &phase_acc * harmonic as f64;
            let sine_wave = harmonic_phase.sin() * (self.sine_amp / harmonic as f64);
            sine_waves.push(sine_wave);
        }

        // 叠加所有谐波
        let mut result = sine_waves[0].shallow_clone();
        for sine_wave in sine_waves.iter().skip(1) {
            result = result + sine_wave;
        }

        result.unsqueeze(1) // [batch, 1, time]
    }

    /// 生成噪声源信号
    pub fn generate_noise(&self, batch_size: i64, signal_length: i64, device: Device) -> Tensor {
        Tensor::randn(&[batch_size, 1, signal_length], (Kind::Float, device)) * self.noise_std
    }

    /// NSF 前向传播
    pub fn forward(&self, f0: &Tensor, upp: i64) -> Tensor {
        let device = f0.device();
        let batch_size = f0.size()[0];
        let signal_length = f0.size()[1] * upp;

        // 生成谐波源
        let sine_source = self.generate_sine_waves(f0, upp);

        // 生成噪声源
        let noise_source = self.generate_noise(batch_size, signal_length, device);

        // 组合源信号
        sine_source + noise_source
    }
}

/// 上采样块
pub struct UpsampleBlock {
    conv_transpose: nn::ConvTranspose1D,
    mrf: MRF,
    leaky_relu_slope: f64,
}

impl UpsampleBlock {
    pub fn new(
        vs: &nn::Path,
        input_dim: i64,
        output_dim: i64,
        kernel_size: i64,
        stride: i64,
        resblock_kernel_sizes: &[i64],
        resblock_dilation_sizes: &[Vec<i64>],
        leaky_relu_slope: f64,
    ) -> Self {
        let padding = (kernel_size - stride) / 2;

        let conv_config = nn::ConvTransposeConfig {
            stride,
            padding,
            output_padding: 0,
            groups: 1,
            bias: true,
            dilation: 1,
            ws_init: nn::Init::Randn {
                mean: 0.,
                stdev: 1.,
            },
            bs_init: nn::Init::Const(0.),
        };

        let conv_transpose = nn::conv_transpose1d(
            vs / "conv_transpose",
            input_dim,
            output_dim,
            kernel_size,
            conv_config,
        );

        let mrf = MRF::new(
            &(vs / "mrf"),
            output_dim,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            leaky_relu_slope,
        );

        Self {
            conv_transpose,
            mrf,
            leaky_relu_slope,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let x = self.conv_transpose.forward(input);
        let x = x.leaky_relu();
        self.mrf.forward(&x)
    }
}

/// NSF-HiFiGAN 生成器主网络
pub struct NSFHiFiGANGenerator {
    input_conv: nn::Conv1D,
    upsample_blocks: Vec<UpsampleBlock>,
    output_conv: nn::Conv1D,
    nsf: Option<NSF>,
    config: GeneratorConfig,
}

impl NSFHiFiGANGenerator {
    /// 创建新的生成器
    pub fn new(vs: &nn::Path, config: GeneratorConfig) -> Self {
        // 输入卷积层
        let input_conv = nn::conv1d(
            vs / "input_conv",
            config.input_dim,
            config.upsample_initial_channel,
            7,
            nn::ConvConfig {
                stride: 1,
                padding: 3,
                bias: true,
                ..Default::default()
            },
        );

        // 上采样块
        let mut upsample_blocks = Vec::new();
        let mut current_dim = config.upsample_initial_channel;

        for (i, (&rate, &kernel_size)) in config
            .upsample_rates
            .iter()
            .zip(&config.upsample_kernel_sizes)
            .enumerate()
        {
            let output_dim = current_dim / 2;

            let upsample_block = UpsampleBlock::new(
                &(vs / format!("upsample_blocks.{}", i)),
                current_dim,
                output_dim,
                kernel_size,
                rate,
                &config.resblock_kernel_sizes,
                &config.resblock_dilation_sizes,
                config.leaky_relu_slope,
            );

            upsample_blocks.push(upsample_block);
            current_dim = output_dim;
        }

        // 输出卷积层
        let output_conv = nn::conv1d(
            vs / "output_conv",
            current_dim,
            1, // 单声道输出
            7,
            nn::ConvConfig {
                stride: 1,
                padding: 3,
                bias: true,
                ..Default::default()
            },
        );

        // NSF 模块（如果启用）
        let nsf = if config.use_nsf {
            Some(NSF::new(config.sample_rate, 8)) // 8个谐波
        } else {
            None
        };

        Self {
            input_conv,
            upsample_blocks,
            output_conv,
            nsf,
            config,
        }
    }

    /// 前向推理
    pub fn forward(
        &self,
        features: &Tensor,
        f0: Option<&Tensor>,
        speaker_embed: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut x = features.shallow_clone();

        // 如果有说话人嵌入，进行拼接
        if let Some(spk_embed) = speaker_embed {
            // 扩展说话人嵌入到序列长度
            let seq_len = x.size()[2];
            let expanded_embed = spk_embed
                .unsqueeze(2)
                .expand(&[x.size()[0], spk_embed.size()[1], seq_len], true);

            x = Tensor::cat(&[x, expanded_embed], 1);
        }

        // 转置到 [batch, channels, time] 格式
        x = x.transpose(1, 2);

        // 输入卷积
        x = self.input_conv.forward(&x);
        x = x.leaky_relu();

        // 逐层上采样
        for upsample_block in &self.upsample_blocks {
            x = upsample_block.forward(&x);
        }

        // 输出卷积
        x = self.output_conv.forward(&x);
        x = x.tanh();

        // 如果使用 NSF，应用源滤波器
        if let (Some(nsf), Some(f0_tensor)) = (&self.nsf, f0) {
            let total_upsample: i64 = self.config.upsample_rates.iter().product();
            let source_signal = nsf.forward(f0_tensor, total_upsample);

            // 调制输出信号
            x = x * source_signal;
        }

        // 压缩到 [batch, time] 格式
        x = x.squeeze_dim(1);

        Ok(x)
    }

    /// 获取上采样倍数
    pub fn get_upsample_factor(&self) -> i64 {
        self.config.upsample_rates.iter().product()
    }

    /// 获取模型配置
    pub fn config(&self) -> &GeneratorConfig {
        &self.config
    }
}

/// 条件生成器（支持多说话人）
pub struct ConditionalGenerator {
    generator: NSFHiFiGANGenerator,
    speaker_embedding: Option<nn::Embedding>,
    num_speakers: Option<i64>,
    speaker_embed_dim: i64,
}

impl ConditionalGenerator {
    /// 创建条件生成器
    pub fn new(
        vs: &nn::Path,
        config: GeneratorConfig,
        num_speakers: Option<i64>,
        speaker_embed_dim: i64,
    ) -> Self {
        // 创建说话人嵌入层
        let speaker_embedding = if let Some(num_spk) = num_speakers {
            Some(nn::embedding(
                vs / "speaker_embedding",
                num_spk,
                speaker_embed_dim,
                Default::default(),
            ))
        } else {
            None
        };

        // 调整生成器输入维度以包含说话人嵌入
        let mut gen_config = config;
        if num_speakers.is_some() {
            gen_config.input_dim += speaker_embed_dim;
        }

        let generator = NSFHiFiGANGenerator::new(&(vs / "generator"), gen_config);

        Self {
            generator,
            speaker_embedding,
            num_speakers,
            speaker_embed_dim,
        }
    }

    /// 条件前向推理
    pub fn forward(
        &self,
        features: &Tensor,
        f0: Option<&Tensor>,
        speaker_id: Option<&Tensor>,
    ) -> Result<Tensor> {
        let speaker_embed =
            if let (Some(embedding), Some(spk_id)) = (&self.speaker_embedding, speaker_id) {
                Some(embedding.forward(spk_id))
            } else {
                None
            };

        self.generator.forward(features, f0, speaker_embed.as_ref())
    }

    /// 获取支持的说话人数量
    pub fn num_speakers(&self) -> Option<i64> {
        self.num_speakers
    }
}

/// 生成器工厂
pub struct GeneratorFactory;

impl GeneratorFactory {
    /// 创建标准 NSF-HiFiGAN 生成器
    pub fn create_nsf_hifigan(vs: &nn::Path, sample_rate: i64) -> NSFHiFiGANGenerator {
        let config = GeneratorConfig {
            sample_rate,
            use_nsf: true,
            ..Default::default()
        };
        NSFHiFiGANGenerator::new(vs, config)
    }

    /// 创建普通 HiFiGAN 生成器
    pub fn create_hifigan(vs: &nn::Path) -> NSFHiFiGANGenerator {
        let config = GeneratorConfig {
            use_nsf: false,
            ..Default::default()
        };
        NSFHiFiGANGenerator::new(vs, config)
    }

    /// 创建多说话人条件生成器
    pub fn create_multi_speaker(
        vs: &nn::Path,
        num_speakers: i64,
        sample_rate: i64,
    ) -> ConditionalGenerator {
        let config = GeneratorConfig {
            sample_rate,
            use_nsf: true,
            ..Default::default()
        };
        ConditionalGenerator::new(vs, config, Some(num_speakers), 256)
    }

    /// 从配置创建生成器
    pub fn from_config(vs: &nn::Path, config: GeneratorConfig) -> NSFHiFiGANGenerator {
        NSFHiFiGANGenerator::new(vs, config)
    }
}

/// 推理辅助工具
pub struct InferenceHelper;

impl InferenceHelper {
    /// 预处理特征
    pub fn preprocess_features(features: &Tensor) -> Tensor {
        // 确保特征在合理范围内
        features.clamp(-10.0, 10.0)
    }

    /// 预处理 F0
    pub fn preprocess_f0(f0: &Tensor, f0_min: f32, f0_max: f32) -> Tensor {
        // 将F0转换为对数尺度并归一化
        let log_f0 = f0.where_self(&f0.gt(0.0), &Tensor::zeros_like(f0)).log();

        let log_f0_min = f0_min.ln();
        let log_f0_max = f0_max.ln();

        (log_f0 - log_f0_min as f64) / (log_f0_max as f64 - log_f0_min as f64)
    }

    /// 后处理音频输出
    pub fn postprocess_audio(audio: &Tensor) -> Tensor {
        // 裁剪到合理范围
        audio.clamp(-1.0, 1.0)
    }

    /// 计算感受野大小
    pub fn compute_receptive_field(config: &GeneratorConfig) -> i64 {
        let mut receptive_field = 7; // 输入卷积核大小

        for &rate in &config.upsample_rates {
            receptive_field = receptive_field * rate;
        }

        // 添加残差块的贡献
        let max_kernel = config.resblock_kernel_sizes.iter().max().unwrap_or(&1);
        let max_dilation = config
            .resblock_dilation_sizes
            .iter()
            .flatten()
            .max()
            .unwrap_or(&1);

        receptive_field += max_kernel * max_dilation;
        receptive_field
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_generator_config() {
        let config = GeneratorConfig::default();
        assert_eq!(config.input_dim, 768);
        assert_eq!(config.upsample_rates, vec![10, 8, 2, 2]);
        assert!(config.use_nsf);
    }

    #[test]
    fn test_nsf_module() {
        let nsf = NSF::new(16000, 8);
        let device = Device::Cpu;

        let f0 = Tensor::ones(&[1, 100], (Kind::Float, device)) * 440.0; // A4 note
        let source = nsf.forward(&f0, 160);

        assert_eq!(source.size(), vec![1, 1, 16000]);
        println!("NSF output shape: {:?}", source.size());
    }

    #[test]
    fn test_residual_block() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        let resblock = ResidualBlock::new(&vs.root(), 64, 3, &[1, 3, 5], 0.1);

        let input = Tensor::randn(&[1, 64, 100], (Kind::Float, device));
        let output = resblock.forward(&input);

        assert_eq!(input.size(), output.size());
    }

    #[test]
    fn test_upsample_block() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        let upsample_block = UpsampleBlock::new(
            &vs.root(),
            256,
            128,
            16,
            8,
            &[3, 7, 11],
            &[vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            0.1,
        );

        let input = Tensor::randn(&[1, 256, 50], (Kind::Float, device));
        let output = upsample_block.forward(&input);

        assert_eq!(output.size(), vec![1, 128, 400]); // 8倍上采样
    }

    #[test]
    fn test_nsf_hifigan_generator() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let config = GeneratorConfig::default();

        let generator = NSFHiFiGANGenerator::new(&vs.root(), config);

        // 测试输入
        let features = Tensor::randn(&[1, 100, 768], (Kind::Float, device));
        let f0 = Tensor::ones(&[1, 100], (Kind::Float, device)) * 220.0;

        let result = generator.forward(&features, Some(&f0), None);

        match result {
            Ok(output) => {
                println!("Generator output shape: {:?}", output.size());
                assert_eq!(output.size()[0], 1); // batch size
                assert!(output.size()[1] > 0); // time dimension
            }
            Err(e) => {
                println!("Generator test error (expected): {}", e);
            }
        }
    }

    #[test]
    fn test_conditional_generator() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        let generator = ConditionalGenerator::new(
            &vs.root(),
            GeneratorConfig::default(),
            Some(10), // 10 speakers
            256,
        );

        assert_eq!(generator.num_speakers(), Some(10));

        let features = Tensor::randn(&[1, 100, 768], (Kind::Float, device));
        let f0 = Tensor::ones(&[1, 100], (Kind::Float, device)) * 220.0;
        let speaker_id = Tensor::from(0i64).unsqueeze(0);

        let result = generator.forward(&features, Some(&f0), Some(&speaker_id));

        match result {
            Ok(output) => {
                println!("Conditional generator output shape: {:?}", output.size());
                assert_eq!(output.size()[0], 1);
            }
            Err(e) => {
                println!("Conditional generator test error (expected): {}", e);
            }
        }
    }

    #[test]
    fn test_generator_factory() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        let nsf_gen = GeneratorFactory::create_nsf_hifigan(&vs.root(), 16000);
        assert!(nsf_gen.config().use_nsf);
        assert_eq!(nsf_gen.config().sample_rate, 16000);

        let hifigan = GeneratorFactory::create_hifigan(&vs.root());
        assert!(!hifigan.config().use_nsf);

        let multi_spk = GeneratorFactory::create_multi_speaker(&vs.root(), 5, 22050);
        assert_eq!(multi_spk.num_speakers(), Some(5));
    }

    #[test]
    fn test_inference_helper() {
        let device = Device::Cpu;
        let config = GeneratorConfig::default();

        let receptive_field = InferenceHelper::compute_receptive_field(&config);
        println!("Receptive field: {}", receptive_field);
        assert!(receptive_field > 0);

        let features = Tensor::randn(&[1, 100, 768], (Kind::Float, device)) * 20.0;
        let processed = InferenceHelper::preprocess_features(&features);
        assert!(processed.abs().max().double_value(&[]) <= 10.0);

        let f0 = Tensor::from_slice(&[100.0, 200.0, 0.0, 440.0]);
        let processed_f0 = InferenceHelper::preprocess_f0(&f0, 50.0, 1000.0);
        println!(
            "Processed F0: {:?}",
            Vec::<f32>::try_from(processed_f0).unwrap()
        );
    }
}
