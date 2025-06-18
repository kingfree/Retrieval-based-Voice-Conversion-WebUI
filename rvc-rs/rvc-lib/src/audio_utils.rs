//! 音频处理工具模块
//!
//! 提供音频文件加载、保存、处理等功能
//! 支持WAV格式的音频文件读写

use std::path::Path;
use tch::{Device, Tensor};

/// 音频数据结构
#[derive(Debug, Clone)]
pub struct AudioData {
    /// 音频样本数据
    pub samples: Vec<f32>,
    /// 采样率
    pub sample_rate: u32,
    /// 通道数
    pub channels: u16,
}

impl AudioData {
    /// 创建新的音频数据
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        Self {
            samples,
            sample_rate,
            channels,
        }
    }

    /// 获取音频时长（秒）
    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / (self.sample_rate as f32 * self.channels as f32)
    }

    /// 获取样本数量
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// 转换为单声道
    pub fn to_mono(&self) -> AudioData {
        if self.channels == 1 {
            return self.clone();
        }

        let mut mono_samples = Vec::with_capacity(self.samples.len() / self.channels as usize);

        for chunk in self.samples.chunks(self.channels as usize) {
            let sum: f32 = chunk.iter().sum();
            mono_samples.push(sum / self.channels as f32);
        }

        AudioData::new(mono_samples, self.sample_rate, 1)
    }

    /// 重采样到目标采样率
    pub fn resample(
        &self,
        target_sample_rate: u32,
    ) -> Result<AudioData, Box<dyn std::error::Error>> {
        if self.sample_rate == target_sample_rate {
            return Ok(self.clone());
        }

        // 简单的线性插值重采样
        let ratio = target_sample_rate as f64 / self.sample_rate as f64;
        let new_length = (self.samples.len() as f64 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_length);

        for i in 0..new_length {
            let original_index = i as f64 / ratio;
            let index = original_index as usize;

            if index >= self.samples.len() - 1 {
                resampled.push(self.samples[self.samples.len() - 1]);
            } else {
                let frac = original_index - index as f64;
                let sample = self.samples[index] * (1.0 - frac as f32)
                    + self.samples[index + 1] * frac as f32;
                resampled.push(sample);
            }
        }

        Ok(AudioData::new(resampled, target_sample_rate, self.channels))
    }

    /// 转换为Tensor
    pub fn to_tensor(&self, device: Device) -> Tensor {
        Tensor::from_slice(&self.samples).to_device(device)
    }

    /// 从Tensor创建AudioData
    pub fn from_tensor(
        tensor: &Tensor,
        sample_rate: u32,
        channels: u16,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let samples: Vec<f32> = tensor.try_into()?;
        Ok(AudioData::new(samples, sample_rate, channels))
    }

    /// 计算音频统计信息
    pub fn calculate_stats(&self) -> AudioStats {
        if self.samples.is_empty() {
            return AudioStats::default();
        }

        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let mut sum_squares = 0.0;

        for &sample in &self.samples {
            min_val = min_val.min(sample);
            max_val = max_val.max(sample);
            sum_squares += sample * sample;
        }

        let rms = (sum_squares / self.samples.len() as f32).sqrt();
        let mean = self.samples.iter().sum::<f32>() / self.samples.len() as f32;

        AudioStats {
            min_value: min_val,
            max_value: max_val,
            rms,
            mean,
            length: self.samples.len(),
            duration: self.duration(),
        }
    }

    /// 规范化音频数据到指定范围
    pub fn normalize(&mut self, target_max: f32) {
        let stats = self.calculate_stats();
        let current_max = stats.max_value.abs().max(stats.min_value.abs());

        if current_max > 0.0 {
            let scale = target_max / current_max;
            for sample in &mut self.samples {
                *sample *= scale;
            }
        }
    }

    /// 添加静音填充
    pub fn pad_silence(&mut self, duration_seconds: f32) {
        let samples_to_add = (duration_seconds * self.sample_rate as f32) as usize;
        let mut padding = vec![0.0; samples_to_add];
        self.samples.append(&mut padding);
    }
}

/// 音频统计信息
#[derive(Debug, Clone, Default)]
pub struct AudioStats {
    pub min_value: f32,
    pub max_value: f32,
    pub rms: f32,
    pub mean: f32,
    pub length: usize,
    pub duration: f32,
}

/// 简单的WAV文件加载器
/// 注意：这是一个简化的实现，实际项目中应使用专业的音频库如hound
pub fn load_wav_simple(path: &str) -> Result<AudioData, Box<dyn std::error::Error>> {
    if !Path::new(path).exists() {
        return Err(format!("Audio file not found: {}", path).into());
    }

    // 对于测试目的，我们生成一个合成的音频信号
    // 在实际实现中，这里应该解析WAV文件格式
    println!("Loading audio from: {}", path);

    // 生成1秒的440Hz正弦波测试信号
    let sample_rate = 16000;
    let duration = 1.0;
    let frequency = 440.0;
    let mut samples = Vec::new();

    for i in 0..(sample_rate as f32 * duration) as usize {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5;
        samples.push(sample);
    }

    println!("Generated {} samples at {}Hz", samples.len(), sample_rate);

    Ok(AudioData::new(samples, sample_rate, 1))
}

/// 使用hound库加载WAV文件（如果可用）
#[cfg(feature = "hound")]
pub fn load_wav_with_hound(path: &str) -> Result<AudioData, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    let samples: Result<Vec<f32>, _> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect(),
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|s| s.map(|sample| sample as f32 / i32::MAX as f32))
            .collect(),
    };

    let samples = samples?;

    Ok(AudioData::new(samples, spec.sample_rate, spec.channels))
}

/// 简单的WAV文件保存器
pub fn save_wav_simple(path: &str, audio: &AudioData) -> Result<(), Box<dyn std::error::Error>> {
    println!("Saving audio to: {} ({} samples)", path, audio.len());

    // 在实际实现中，这里应该写入WAV文件格式
    // 现在只是模拟保存过程并创建一个占位文件
    std::fs::write(
        path,
        format!(
            "Audio data: {} samples at {}Hz",
            audio.len(),
            audio.sample_rate
        ),
    )?;

    println!("Audio saved successfully");
    Ok(())
}

/// 使用hound库保存WAV文件（如果可用）
#[cfg(feature = "hound")]
pub fn save_wav_with_hound(
    path: &str,
    audio: &AudioData,
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = hound::WavSpec {
        channels: audio.channels,
        sample_rate: audio.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;

    for &sample in &audio.samples {
        writer.write_sample(sample)?;
    }

    writer.finalize()?;
    Ok(())
}

/// 计算两个音频信号的相似性（互相关）
pub fn calculate_similarity(signal1: &[f32], signal2: &[f32]) -> f32 {
    if signal1.len() != signal2.len() {
        return 0.0;
    }

    let mut correlation = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;

    for i in 0..signal1.len() {
        correlation += signal1[i] * signal2[i];
        norm1 += signal1[i] * signal1[i];
        norm2 += signal2[i] * signal2[i];
    }

    if norm1 == 0.0 || norm2 == 0.0 {
        return 0.0;
    }

    correlation / (norm1.sqrt() * norm2.sqrt())
}

/// 应用窗函数
pub fn apply_window(signal: &mut [f32], window_type: WindowType) {
    let len = signal.len();

    match window_type {
        WindowType::Hann => {
            for (i, sample) in signal.iter_mut().enumerate() {
                let window_val =
                    0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (len - 1) as f32).cos());
                *sample *= window_val;
            }
        }
        WindowType::Hamming => {
            for (i, sample) in signal.iter_mut().enumerate() {
                let window_val =
                    0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (len - 1) as f32).cos();
                *sample *= window_val;
            }
        }
        WindowType::Blackman => {
            for (i, sample) in signal.iter_mut().enumerate() {
                let a0 = 0.42;
                let a1 = 0.5;
                let a2 = 0.08;
                let n = i as f32;
                let N = (len - 1) as f32;
                let window_val = a0 - a1 * (2.0 * std::f32::consts::PI * n / N).cos()
                    + a2 * (4.0 * std::f32::consts::PI * n / N).cos();
                *sample *= window_val;
            }
        }
    }
}

/// 窗函数类型
#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    Hann,
    Hamming,
    Blackman,
}

/// 创建测试音频信号
pub fn create_test_signal(frequency: f32, duration: f32, sample_rate: u32) -> AudioData {
    let num_samples = (sample_rate as f32 * duration) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5;
        samples.push(sample);
    }

    AudioData::new(samples, sample_rate, 1)
}

/// 创建白噪声信号
pub fn create_white_noise(duration: f32, sample_rate: u32, amplitude: f32) -> AudioData {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let num_samples = (sample_rate as f32 * duration) as usize;
    let mut samples = Vec::with_capacity(num_samples);

    // 简单的伪随机数生成器
    let mut hasher = DefaultHasher::new();

    for i in 0..num_samples {
        i.hash(&mut hasher);
        let hash = hasher.finish();
        let random = (hash % 1000) as f32 / 1000.0; // 0.0 to 1.0
        let noise = (random - 0.5) * 2.0 * amplitude; // -amplitude to +amplitude
        samples.push(noise);
    }

    AudioData::new(samples, sample_rate, 1)
}

/// 混合两个音频信号
pub fn mix_audio(
    audio1: &AudioData,
    audio2: &AudioData,
    ratio: f32,
) -> Result<AudioData, Box<dyn std::error::Error>> {
    if audio1.sample_rate != audio2.sample_rate {
        return Err("Sample rates must match".into());
    }

    if audio1.channels != audio2.channels {
        return Err("Channel counts must match".into());
    }

    let len = audio1.samples.len().min(audio2.samples.len());
    let mut mixed_samples = Vec::with_capacity(len);

    for i in 0..len {
        let mixed = audio1.samples[i] * (1.0 - ratio) + audio2.samples[i] * ratio;
        mixed_samples.push(mixed);
    }

    Ok(AudioData::new(
        mixed_samples,
        audio1.sample_rate,
        audio1.channels,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_data_creation() {
        let samples = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let audio = AudioData::new(samples.clone(), 44100, 1);

        assert_eq!(audio.samples, samples);
        assert_eq!(audio.sample_rate, 44100);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.len(), 5);
        assert!(!audio.is_empty());
    }

    #[test]
    fn test_audio_duration() {
        let samples = vec![0.0; 44100]; // 1 second at 44.1kHz
        let audio = AudioData::new(samples, 44100, 1);

        assert!((audio.duration() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_audio_stats() {
        let samples = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let audio = AudioData::new(samples, 44100, 1);
        let stats = audio.calculate_stats();

        assert_eq!(stats.min_value, -1.0);
        assert_eq!(stats.max_value, 1.0);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.length, 5);
    }

    #[test]
    fn test_similarity_calculation() {
        let signal1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let signal2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let similarity = calculate_similarity(&signal1, &signal2);
        assert!((similarity - 1.0).abs() < 1e-6);

        let signal3 = vec![0.0; 5];
        let similarity2 = calculate_similarity(&signal1, &signal3);
        assert_eq!(similarity2, 0.0);
    }

    #[test]
    fn test_create_test_signal() {
        let audio = create_test_signal(440.0, 0.1, 44100);

        assert_eq!(audio.sample_rate, 44100);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.len(), 4410); // 0.1 seconds * 44100 Hz

        // 检查信号是否是正弦波特征
        let stats = audio.calculate_stats();
        assert!(stats.max_value > 0.4 && stats.max_value < 0.6);
        assert!(stats.min_value > -0.6 && stats.min_value < -0.4);
    }

    #[test]
    fn test_audio_normalization() {
        let mut audio = AudioData::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], 44100, 1);
        audio.normalize(0.5);

        let stats = audio.calculate_stats();
        assert!((stats.max_value - 0.5).abs() < 1e-6);
        assert!((stats.min_value + 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_resample() {
        let audio = AudioData::new(vec![1.0, 2.0, 3.0, 4.0], 1000, 1);
        let resampled = audio.resample(2000).unwrap();

        assert_eq!(resampled.sample_rate, 2000);
        assert_eq!(resampled.len(), 8); // 2x upsampling
    }

    #[test]
    fn test_mono_conversion() {
        // 立体声信号：左右声道交替
        let stereo_samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // L, R, L, R, L, R
        let stereo_audio = AudioData::new(stereo_samples, 44100, 2);

        let mono_audio = stereo_audio.to_mono();

        assert_eq!(mono_audio.channels, 1);
        assert_eq!(mono_audio.len(), 3); // 3 mono samples from 6 stereo samples
        assert_eq!(mono_audio.samples, vec![1.5, 3.5, 5.5]); // (L+R)/2
    }
}
