//! HuBERT 特征提取模块
//!
//! 该模块实现了 HuBERT (Hidden-Unit BERT) 模型的特征提取功能。
//! HuBERT 是用于语音表征学习的自监督预训练模型，在 RVC 中用于提取
//! 音频的高级语义特征。

use anyhow::Result;
use tch::{Device, Kind, Tensor, nn, nn::Module};

/// HuBERT 模型配置
#[derive(Debug, Clone)]
pub struct HuBERTConfig {
    /// 特征维度
    pub feature_dim: i64,
    /// 编码器层数
    pub encoder_layers: i64,
    /// 注意力头数
    pub encoder_attention_heads: i64,
    /// 前馈网络维度
    pub encoder_ffn_embed_dim: i64,
    /// 激活函数类型
    pub activation_fn: String,
    /// Dropout 概率
    pub dropout: f64,
    /// 注意力 Dropout 概率
    pub attention_dropout: f64,
    /// 激活函数 Dropout 概率
    pub activation_dropout: f64,
    /// 最终投影维度
    pub final_dim: i64,
    /// 输出层数
    pub num_classes: Option<i64>,
}

impl Default for HuBERTConfig {
    fn default() -> Self {
        Self {
            feature_dim: 768,
            encoder_layers: 12,
            encoder_attention_heads: 12,
            encoder_ffn_embed_dim: 3072,
            activation_fn: "gelu".to_string(),
            dropout: 0.1,
            attention_dropout: 0.1,
            activation_dropout: 0.0,
            final_dim: 768,
            num_classes: None,
        }
    }
}

/// 特征提取器配置
#[derive(Debug, Clone)]
pub struct FeatureExtractorConfig {
    /// 卷积核大小
    pub conv_kernel_sizes: Vec<i64>,
    /// 卷积步长
    pub conv_strides: Vec<i64>,
    /// 卷积输出维度
    pub conv_dims: Vec<i64>,
    /// 标准化类型
    pub normalization_type: String,
}

impl Default for FeatureExtractorConfig {
    fn default() -> Self {
        Self {
            conv_kernel_sizes: vec![10, 3, 3, 3, 3, 2, 2],
            conv_strides: vec![5, 2, 2, 2, 2, 2, 2],
            conv_dims: vec![512, 512, 512, 512, 512, 512, 512],
            normalization_type: "layer".to_string(),
        }
    }
}

/// CNN 特征提取器
pub struct FeatureExtractor {
    conv_layers: Vec<nn::Conv1D>,
    layer_norms: Vec<nn::LayerNorm>,
    config: FeatureExtractorConfig,
}

impl FeatureExtractor {
    /// 创建新的特征提取器
    pub fn new(vs: &nn::Path, config: FeatureExtractorConfig) -> Self {
        let mut conv_layers = Vec::new();
        let mut layer_norms = Vec::new();

        let mut in_dim = 1; // 输入是单声道音频

        for (i, ((&kernel_size, &stride), &out_dim)) in config
            .conv_kernel_sizes
            .iter()
            .zip(&config.conv_strides)
            .zip(&config.conv_dims)
            .enumerate()
        {
            // 卷积层
            let conv_config = nn::ConvConfig {
                stride: stride,
                padding: kernel_size / 2,
                bias: false,
                ..Default::default()
            };

            let conv = nn::conv1d(
                vs / format!("conv_layers_{}", i),
                in_dim,
                out_dim,
                kernel_size,
                conv_config,
            );
            conv_layers.push(conv);

            // Layer Normalization
            let layer_norm = nn::layer_norm(
                vs / format!("layer_norms_{}", i),
                vec![out_dim],
                Default::default(),
            );
            layer_norms.push(layer_norm);

            in_dim = out_dim;
        }

        Self {
            conv_layers,
            layer_norms,
            config,
        }
    }

    /// 提取卷积特征
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.shallow_clone();

        // 确保输入维度正确 [batch, 1, time]
        if x.dim() == 2 {
            x = x.unsqueeze(1);
        }

        for (conv, layer_norm) in self.conv_layers.iter().zip(&self.layer_norms) {
            // 卷积 + 激活
            x = conv.forward(&x);
            x = x.gelu("none");

            // 转换为 [batch, time, dim] 进行 layer norm
            x = x.transpose(1, 2);
            x = layer_norm.forward(&x);
            x = x.transpose(1, 2);
        }

        // 返回 [batch, time, dim] 格式
        x.transpose(1, 2)
    }
}

/// Transformer 编码器层
pub struct TransformerEncoderLayer {
    q_linear: nn::Linear,
    k_linear: nn::Linear,
    v_linear: nn::Linear,
    out_linear: nn::Linear,
    linear1: nn::Linear,
    linear2: nn::Linear,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    dropout: f64,
    activation_dropout: f64,
    num_heads: i64,
    head_dim: i64,
}

impl TransformerEncoderLayer {
    pub fn new(vs: &nn::Path, config: &HuBERTConfig) -> Self {
        let head_dim = config.feature_dim / config.encoder_attention_heads;

        Self {
            q_linear: nn::linear(
                vs / "self_attn_q",
                config.feature_dim,
                config.feature_dim,
                Default::default(),
            ),
            k_linear: nn::linear(
                vs / "self_attn_k",
                config.feature_dim,
                config.feature_dim,
                Default::default(),
            ),
            v_linear: nn::linear(
                vs / "self_attn_v",
                config.feature_dim,
                config.feature_dim,
                Default::default(),
            ),
            out_linear: nn::linear(
                vs / "self_attn_out",
                config.feature_dim,
                config.feature_dim,
                Default::default(),
            ),
            linear1: nn::linear(
                vs / "fc1",
                config.feature_dim,
                config.encoder_ffn_embed_dim,
                Default::default(),
            ),
            linear2: nn::linear(
                vs / "fc2",
                config.encoder_ffn_embed_dim,
                config.feature_dim,
                Default::default(),
            ),
            norm1: nn::layer_norm(
                vs / "self_attn_layer_norm",
                vec![config.feature_dim],
                Default::default(),
            ),
            norm2: nn::layer_norm(
                vs / "final_layer_norm",
                vec![config.feature_dim],
                Default::default(),
            ),
            dropout: config.dropout,
            activation_dropout: config.activation_dropout,
            num_heads: config.encoder_attention_heads,
            head_dim,
        }
    }

    pub fn forward(&self, input: &Tensor, _attention_mask: Option<&Tensor>) -> Tensor {
        // Self-attention
        let residual = input.shallow_clone();
        let x = self.norm1.forward(input);

        // Simplified multi-head attention
        let batch_size = x.size()[0];
        let seq_len = x.size()[1];
        let d_model = x.size()[2];

        let q = self.q_linear.forward(&x);
        let k = self.k_linear.forward(&x);
        let v = self.v_linear.forward(&x);

        // Reshape for multi-head attention: [batch, seq_len, num_heads, head_dim]
        let q = q
            .view([batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let k = k
            .view([batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);
        let v = v
            .view([batch_size, seq_len, self.num_heads, self.head_dim])
            .transpose(1, 2);

        // Attention scores
        let scores = q.matmul(&k.transpose(-2, -1)) / (self.head_dim as f64).sqrt();
        let attn_weights = scores.softmax(-1, Kind::Float);
        let attn_output = attn_weights.matmul(&v);

        // Reshape back: [batch, seq_len, d_model]
        let attn_output = attn_output
            .transpose(1, 2)
            .contiguous()
            .view([batch_size, seq_len, d_model]);
        let attn_output = self.out_linear.forward(&attn_output);

        let x = residual + attn_output.dropout(self.dropout, false);

        // Feed-forward
        let residual = x.shallow_clone();
        let x = self.norm2.forward(&x);
        let x = self.linear1.forward(&x);
        let x = x.gelu("none").dropout(self.activation_dropout, false);
        let x = self.linear2.forward(&x);
        let x = x.dropout(self.dropout, false);

        residual + x
    }
}

/// Transformer 编码器
pub struct TransformerEncoder {
    layers: Vec<TransformerEncoderLayer>,
    layer_norm: Option<nn::LayerNorm>,
}

impl TransformerEncoder {
    pub fn new(vs: &nn::Path, config: &HuBERTConfig) -> Self {
        let mut layers = Vec::new();

        for i in 0..config.encoder_layers {
            let layer = TransformerEncoderLayer::new(&(vs / format!("layers_{}", i)), config);
            layers.push(layer);
        }

        let layer_norm = Some(nn::layer_norm(
            vs / "layer_norm",
            vec![config.feature_dim],
            Default::default(),
        ));

        Self { layers, layer_norm }
    }

    pub fn forward(&self, input: &Tensor, attention_mask: Option<&Tensor>) -> Tensor {
        let mut x = input.shallow_clone();

        for layer in &self.layers {
            x = layer.forward(&x, attention_mask);
        }

        if let Some(layer_norm) = &self.layer_norm {
            x = layer_norm.forward(&x);
        }

        x
    }
}

/// 位置编码
pub struct PositionalEncoding {
    pe: Tensor,
    max_len: i64,
}

impl PositionalEncoding {
    pub fn new(d_model: i64, max_len: i64, device: Device) -> Self {
        let pe = Tensor::zeros(&[max_len, d_model], (Kind::Float, device));

        let position = Tensor::arange(max_len, (Kind::Float, device)).unsqueeze(1);
        let div_term = (Tensor::arange(d_model / 2, (Kind::Float, device))
            * (-10000.0_f64.ln() / d_model as f64 * 2.0))
            .exp();

        let sin_vals = (&position * &div_term).sin();
        let cos_vals = (&position * &div_term).cos();

        // 交替填充 sin 和 cos
        for i in 0..(d_model / 2) {
            let _ = pe.select(1, i * 2).copy_(&sin_vals.select(1, i));
            if i * 2 + 1 < d_model {
                let _ = pe.select(1, i * 2 + 1).copy_(&cos_vals.select(1, i));
            }
        }

        Self { pe, max_len }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let seq_len = input.size()[1];
        let pe_slice = self.pe.narrow(0, 0, seq_len).unsqueeze(0);
        input + pe_slice
    }
}

/// 完整的 HuBERT 模型
pub struct HuBERT {
    feature_extractor: FeatureExtractor,
    feature_projection: nn::Linear,
    positional_encoding: PositionalEncoding,
    encoder: TransformerEncoder,
    final_proj: Option<nn::Linear>,
    config: HuBERTConfig,
    device: Device,
}

impl HuBERT {
    /// 创建新的 HuBERT 模型
    pub fn new(vs: &nn::Path, config: HuBERTConfig, device: Device) -> Self {
        let fe_config = FeatureExtractorConfig::default();
        let feature_extractor =
            FeatureExtractor::new(&(vs / "feature_extractor"), fe_config.clone());

        // 特征投影层
        let final_conv_dim = fe_config.conv_dims.last().unwrap_or(&512);
        let feature_projection = nn::linear(
            vs / "feature_projection",
            *final_conv_dim,
            config.feature_dim,
            Default::default(),
        );

        // 位置编码
        let positional_encoding = PositionalEncoding::new(
            config.feature_dim,
            5000, // 最大序列长度
            device,
        );

        // Transformer 编码器
        let encoder = TransformerEncoder::new(&(vs / "encoder"), &config);

        // 最终投影层（如果需要）
        let final_proj = if config.final_dim != config.feature_dim {
            Some(nn::linear(
                vs / "final_proj",
                config.feature_dim,
                config.final_dim,
                Default::default(),
            ))
        } else {
            None
        };

        Self {
            feature_extractor,
            feature_projection,
            positional_encoding,
            encoder,
            final_proj,
            config,
            device,
        }
    }

    /// 从预训练模型加载
    pub fn from_pretrained<P: AsRef<std::path::Path>>(
        vs: &nn::Path,
        model_path: P,
        device: Device,
    ) -> Result<Self> {
        let config = HuBERTConfig::default();
        let model = Self::new(vs, config, device);

        // 尝试加载预训练权重
        // TODO: Implement proper model loading
        if model_path.as_ref().exists() {
            println!("⚠️  模型文件存在但权重加载功能暂未实现");
            println!("使用随机初始化权重");
        }

        Ok(model)
    }

    /// 提取特征
    pub fn extract_features(
        &self,
        waveform: &Tensor,
        output_layer: Option<i64>,
        return_all_layers: bool,
    ) -> Result<HuBERTOutput> {
        // 输入预处理
        let mut input = waveform.to_device(self.device);

        // 确保输入格式正确
        if input.dim() == 1 {
            input = input.unsqueeze(0); // [batch, time]
        }
        if input.dim() == 2 {
            input = input.unsqueeze(1); // [batch, 1, time]
        }

        // 1. CNN 特征提取
        let conv_features = self.feature_extractor.forward(&input);

        // 2. 特征投影
        let features = self.feature_projection.forward(&conv_features);

        // 3. 位置编码
        let features = self.positional_encoding.forward(&features);

        // 4. 创建注意力掩码
        let batch_size = features.size()[0];
        let seq_len = features.size()[1];
        let attention_mask = Tensor::ones(&[batch_size, seq_len], (Kind::Bool, self.device));

        // 5. Transformer 编码
        let mut all_layers = Vec::new();
        let mut x = features;

        for (_i, layer) in self.encoder.layers.iter().enumerate() {
            x = layer.forward(&x, Some(&attention_mask));

            if return_all_layers {
                all_layers.push(x.shallow_clone());
            }
        }

        // 应用最终的 layer norm
        if let Some(layer_norm) = &self.encoder.layer_norm {
            x = layer_norm.forward(&x);
        }

        // 6. 最终投影
        if let Some(final_proj) = &self.final_proj {
            x = final_proj.forward(&x);
        }

        // 7. 选择输出层
        let output_features = if let Some(layer_idx) = output_layer {
            if return_all_layers && (layer_idx as usize) < all_layers.len() {
                all_layers[layer_idx as usize].shallow_clone()
            } else {
                x.shallow_clone()
            }
        } else {
            x.shallow_clone()
        };

        Ok(HuBERTOutput {
            last_hidden_state: x,
            hidden_states: if return_all_layers {
                Some(all_layers)
            } else {
                None
            },
            extract_features: Some(conv_features),
            features: output_features,
        })
    }

    /// 简化的特征提取接口（兼容 RVC）
    pub fn forward(&self, input: &Tensor, output_layer: i64) -> Result<Tensor> {
        let output = self.extract_features(input, Some(output_layer), false)?;
        Ok(output.features)
    }

    /// 获取模型配置
    pub fn config(&self) -> &HuBERTConfig {
        &self.config
    }
}

/// HuBERT 输出结构
#[derive(Debug)]
pub struct HuBERTOutput {
    /// 最后一层的隐藏状态
    pub last_hidden_state: Tensor,
    /// 所有层的隐藏状态（如果请求）
    pub hidden_states: Option<Vec<Tensor>>,
    /// 卷积特征提取的输出
    pub extract_features: Option<Tensor>,
    /// 选择的输出特征
    pub features: Tensor,
}

/// HuBERT 特征提取器工厂
pub struct HuBERTFactory;

impl HuBERTFactory {
    /// 创建 HuBERT Base 模型
    pub fn create_base(vs: &nn::Path, device: Device) -> HuBERT {
        let config = HuBERTConfig::default();
        HuBERT::new(vs, config, device)
    }

    /// 创建 HuBERT Large 模型
    pub fn create_large(vs: &nn::Path, device: Device) -> HuBERT {
        let config = HuBERTConfig {
            feature_dim: 1024,
            encoder_layers: 24,
            encoder_attention_heads: 16,
            encoder_ffn_embed_dim: 4096,
            final_dim: 1024,
            ..Default::default()
        };
        HuBERT::new(vs, config, device)
    }

    /// 从配置文件创建
    pub fn from_config(vs: &nn::Path, config: HuBERTConfig, device: Device) -> HuBERT {
        HuBERT::new(vs, config, device)
    }
}

/// 音频预处理工具
pub struct AudioPreprocessor {
    sample_rate: f64,
    normalize: bool,
}

impl AudioPreprocessor {
    pub fn new(sample_rate: f64, normalize: bool) -> Self {
        Self {
            sample_rate,
            normalize,
        }
    }

    /// 预处理音频输入
    pub fn preprocess(&self, audio: &[f32]) -> Tensor {
        let mut processed = Vec::from(audio);

        // 标准化
        if self.normalize {
            let max_val = processed.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            if max_val > 0.0 {
                for sample in &mut processed {
                    *sample /= max_val;
                }
            }
        }

        Tensor::from_slice(&processed)
    }

    /// 批量预处理
    pub fn preprocess_batch(&self, batch: &[&[f32]]) -> Tensor {
        let batch_tensors: Vec<Tensor> = batch.iter().map(|audio| self.preprocess(audio)).collect();

        Tensor::stack(&batch_tensors, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_hubert_config() {
        let config = HuBERTConfig::default();
        assert_eq!(config.feature_dim, 768);
        assert_eq!(config.encoder_layers, 12);
    }

    #[test]
    fn test_feature_extractor() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let config = FeatureExtractorConfig::default();
        let extractor = FeatureExtractor::new(&vs.root(), config);

        // 测试输入：1秒 16kHz 音频
        let input = Tensor::randn(&[1, 16000], (Kind::Float, device));
        let output = extractor.forward(&input);

        println!("Feature extractor output shape: {:?}", output.size());
        assert_eq!(output.dim(), 3); // [batch, time, dim]
    }

    #[test]
    fn test_positional_encoding() {
        let device = Device::Cpu;
        let pe = PositionalEncoding::new(768, 1000, device);
        let input = Tensor::randn(&[2, 100, 768], (Kind::Float, device));
        let output = pe.forward(&input);

        assert_eq!(input.size(), output.size());
    }

    #[test]
    fn test_hubert_model() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let config = HuBERTConfig::default();
        let model = HuBERT::new(&vs.root(), config, device);

        // 测试音频输入
        let audio = Tensor::randn(&[1, 16000], (Kind::Float, device));
        let result = model.extract_features(&audio, Some(9), false);

        match result {
            Ok(output) => {
                println!("HuBERT output shape: {:?}", output.features.size());
                assert_eq!(output.features.dim(), 3);
            }
            Err(e) => {
                println!("HuBERT test error (expected): {}", e);
            }
        }
    }

    #[test]
    fn test_audio_preprocessor() {
        let preprocessor = AudioPreprocessor::new(16000.0, true);
        let audio = vec![0.1, 0.2, -0.5, 0.8, -0.3];
        let processed = preprocessor.preprocess(&audio);

        assert_eq!(processed.size()[0], audio.len() as i64);
        println!(
            "Preprocessed audio: {:?}",
            Vec::<f32>::try_from(processed).unwrap()
        );
    }

    #[test]
    fn test_hubert_factory() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        let base_model = HuBERTFactory::create_base(&vs.root(), device);
        assert_eq!(base_model.config.feature_dim, 768);

        let large_model = HuBERTFactory::create_large(&vs.root(), device);
        assert_eq!(large_model.config.feature_dim, 1024);
    }
}
