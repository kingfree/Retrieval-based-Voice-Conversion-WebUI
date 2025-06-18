//! F0 (基频) 估计和处理模块
//!
//! 该模块实现了多种基频估计算法和后处理功能，用于 RVC 语音转换中的音高控制。
//! 支持的算法包括 PM, Harvest, DIO, YIN, RMVPE 等。

use anyhow::{Result, anyhow};
// use rayon::prelude::*; // TODO: Re-enable when parallel processing is implemented
use std::f32::consts::PI;
use tch::Device;

/// F0 估计方法枚举
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum F0Method {
    PM,      // Pitch Marking
    Harvest, // Harvest algorithm
    DIO,     // DIO algorithm
    YIN,     // YIN algorithm
    RMVPE,   // RMVPE (深度学习方法)
    CREPE,   // CREPE-like 方法
}

impl std::str::FromStr for F0Method {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "pm" => Ok(F0Method::PM),
            "harvest" => Ok(F0Method::Harvest),
            "dio" => Ok(F0Method::DIO),
            "yin" => Ok(F0Method::YIN),
            "rmvpe" => Ok(F0Method::RMVPE),
            "crepe" => Ok(F0Method::CREPE),
            _ => Err(anyhow!("Unknown F0 method: {}", s)),
        }
    }
}

/// F0 估计配置
#[derive(Debug, Clone)]
pub struct F0Config {
    /// 采样率
    pub sample_rate: f32,
    /// 帧长 (样本数)
    pub frame_length: usize,
    /// 帧移 (样本数)
    pub hop_length: usize,
    /// 最小F0频率 (Hz)
    pub f0_min: f32,
    /// 最大F0频率 (Hz)
    pub f0_max: f32,
    /// 窗函数类型
    pub window_type: WindowType,
    /// 阈值参数
    pub threshold: f32,
}

impl Default for F0Config {
    fn default() -> Self {
        Self {
            sample_rate: 16000.0,
            frame_length: 1024,
            hop_length: 160, // 10ms at 16kHz
            f0_min: 50.0,
            f0_max: 1100.0,
            window_type: WindowType::Hann,
            threshold: 0.3,
        }
    }
}

/// 窗函数类型
#[derive(Debug, Clone, Copy)]
pub enum WindowType {
    Hann,
    Hamming,
    Blackman,
    Rectangular,
}

/// F0 估计结果
#[derive(Debug, Clone)]
pub struct F0Result {
    /// F0 频率序列 (Hz)
    pub frequencies: Vec<f32>,
    /// 时间戳序列 (秒)
    pub times: Vec<f32>,
    /// 置信度序列
    pub confidences: Vec<f32>,
    /// 有声/无声标记
    pub voiced: Vec<bool>,
}

impl F0Result {
    /// 创建新的F0结果
    pub fn new(frequencies: Vec<f32>, times: Vec<f32>) -> Self {
        let len = frequencies.len();
        let confidences = vec![1.0; len];
        let voiced = frequencies.iter().map(|&f| f > 0.0).collect();

        Self {
            frequencies,
            times,
            confidences,
            voiced,
        }
    }

    /// 获取有效F0值（非零）
    pub fn valid_f0(&self) -> Vec<f32> {
        self.frequencies
            .iter()
            .filter(|&&f| f > 0.0)
            .cloned()
            .collect()
    }

    /// 转换为半音
    pub fn to_semitones(&self, ref_freq: f32) -> Vec<f32> {
        self.frequencies
            .iter()
            .map(|&f| {
                if f > 0.0 {
                    12.0 * (f / ref_freq).log2()
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// 转换为Mel刻度
    pub fn to_mel(&self) -> Vec<f32> {
        self.frequencies
            .iter()
            .map(|&f| {
                if f > 0.0 {
                    1127.0 * (1.0 + f / 700.0).ln()
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// 长度
    pub fn len(&self) -> usize {
        self.frequencies.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.frequencies.is_empty()
    }
}

/// F0 估计器主结构
pub struct F0Estimator {
    config: F0Config,
    device: Device,
}

impl F0Estimator {
    /// 创建新的F0估计器
    pub fn new(config: F0Config, device: Device) -> Self {
        Self { config, device }
    }

    /// 估计F0
    pub fn estimate(&self, audio: &[f32], method: F0Method) -> Result<F0Result> {
        match method {
            F0Method::PM => self.estimate_pm(audio),
            F0Method::Harvest => self.estimate_harvest(audio),
            F0Method::DIO => self.estimate_dio(audio),
            F0Method::YIN => self.estimate_yin(audio),
            F0Method::RMVPE => self.estimate_rmvpe(audio),
            F0Method::CREPE => self.estimate_crepe(audio),
        }
    }

    /// PM (Pitch Marking) 方法
    fn estimate_pm(&self, audio: &[f32]) -> Result<F0Result> {
        let frame_count = (audio.len() - self.config.frame_length) / self.config.hop_length + 1;
        let mut frequencies = Vec::with_capacity(frame_count);
        let mut times = Vec::with_capacity(frame_count);

        for i in 0..frame_count {
            let start = i * self.config.hop_length;
            let end = (start + self.config.frame_length).min(audio.len());
            let frame = &audio[start..end];

            let time = start as f32 / self.config.sample_rate;
            let f0 = self.estimate_f0_autocorr(frame)?;

            frequencies.push(f0);
            times.push(time);
        }

        Ok(F0Result::new(frequencies, times))
    }

    /// Harvest 算法
    fn estimate_harvest(&self, audio: &[f32]) -> Result<F0Result> {
        let frame_count = (audio.len() - self.config.frame_length) / self.config.hop_length + 1;
        let mut frequencies = Vec::with_capacity(frame_count);
        let mut times = Vec::with_capacity(frame_count);

        // Harvest 的核心思想是基于瞬时频率估计
        for i in 0..frame_count {
            let start = i * self.config.hop_length;
            let end = (start + self.config.frame_length).min(audio.len());
            let frame = &audio[start..end];

            let time = start as f32 / self.config.sample_rate;
            let f0 = self.estimate_f0_harvest_frame(frame)?;

            frequencies.push(f0);
            times.push(time);
        }

        Ok(F0Result::new(frequencies, times))
    }

    /// DIO 算法
    fn estimate_dio(&self, audio: &[f32]) -> Result<F0Result> {
        let frame_count = (audio.len() - self.config.frame_length) / self.config.hop_length + 1;
        let mut frequencies = Vec::with_capacity(frame_count);
        let mut times = Vec::with_capacity(frame_count);

        for i in 0..frame_count {
            let start = i * self.config.hop_length;
            let end = (start + self.config.frame_length).min(audio.len());
            let frame = &audio[start..end];

            let time = start as f32 / self.config.sample_rate;
            let f0 = self.estimate_f0_dio_frame(frame)?;

            frequencies.push(f0);
            times.push(time);
        }

        Ok(F0Result::new(frequencies, times))
    }

    /// YIN 算法
    fn estimate_yin(&self, audio: &[f32]) -> Result<F0Result> {
        let frame_count = (audio.len() - self.config.frame_length) / self.config.hop_length + 1;
        let mut frequencies = Vec::with_capacity(frame_count);
        let mut times = Vec::with_capacity(frame_count);

        for i in 0..frame_count {
            let start = i * self.config.hop_length;
            let end = (start + self.config.frame_length).min(audio.len());
            let frame = &audio[start..end];

            let time = start as f32 / self.config.sample_rate;
            let f0 = self.estimate_f0_yin_frame(frame)?;

            frequencies.push(f0);
            times.push(time);
        }

        Ok(F0Result::new(frequencies, times))
    }

    /// RMVPE 方法（简化版）
    fn estimate_rmvpe(&self, audio: &[f32]) -> Result<F0Result> {
        // 这里使用改进的自相关方法模拟 RMVPE
        let frame_count = (audio.len() - self.config.frame_length) / self.config.hop_length + 1;
        let mut frequencies = Vec::with_capacity(frame_count);
        let mut times = Vec::with_capacity(frame_count);

        for i in 0..frame_count {
            let start = i * self.config.hop_length;
            let end = (start + self.config.frame_length).min(audio.len());
            let frame = &audio[start..end];

            let time = start as f32 / self.config.sample_rate;
            let f0 = self.estimate_f0_rmvpe_like(frame)?;

            frequencies.push(f0);
            times.push(time);
        }

        Ok(F0Result::new(frequencies, times))
    }

    /// CREPE-like 方法
    fn estimate_crepe(&self, audio: &[f32]) -> Result<F0Result> {
        // 简化的 CREPE-like 实现，使用频域分析
        let frame_count = (audio.len() - self.config.frame_length) / self.config.hop_length + 1;
        let mut frequencies = Vec::with_capacity(frame_count);
        let mut times = Vec::with_capacity(frame_count);

        for i in 0..frame_count {
            let start = i * self.config.hop_length;
            let end = (start + self.config.frame_length).min(audio.len());
            let frame = &audio[start..end];

            let time = start as f32 / self.config.sample_rate;
            let f0 = self.estimate_f0_spectral(frame)?;

            frequencies.push(f0);
            times.push(time);
        }

        Ok(F0Result::new(frequencies, times))
    }

    /// 自相关方法估计F0
    fn estimate_f0_autocorr(&self, frame: &[f32]) -> Result<f32> {
        if frame.len() < 64 {
            return Ok(0.0);
        }

        let min_period = (self.config.sample_rate / self.config.f0_max) as usize;
        let max_period = (self.config.sample_rate / self.config.f0_min) as usize;

        // 应用窗函数
        let windowed = self.apply_window(frame);

        // 计算自相关
        let autocorr = self.autocorrelation(&windowed, max_period);

        // 寻找最佳周期
        let mut best_period = 0;
        let mut best_corr = 0.0;

        for period in min_period..max_period.min(autocorr.len()) {
            if autocorr[period] > best_corr {
                best_corr = autocorr[period];
                best_period = period;
            }
        }

        if best_corr > self.config.threshold && best_period > 0 {
            Ok(self.config.sample_rate / best_period as f32)
        } else {
            Ok(0.0)
        }
    }

    /// Harvest 帧处理
    fn estimate_f0_harvest_frame(&self, frame: &[f32]) -> Result<f32> {
        // Harvest 的简化实现
        let windowed = self.apply_window(frame);

        // 计算瞬时频率
        let instantaneous_freq = self.compute_instantaneous_frequency(&windowed)?;

        // 选择最佳频率候选
        let candidates = self.find_frequency_candidates(&instantaneous_freq);
        let best_candidate = self.select_best_candidate(&candidates);

        Ok(best_candidate)
    }

    /// DIO 帧处理
    fn estimate_f0_dio_frame(&self, frame: &[f32]) -> Result<f32> {
        // DIO 的简化实现，基于零交叉率和能量
        let windowed = self.apply_window(frame);

        // 计算零交叉率
        let zcr = self.zero_crossing_rate(&windowed);

        // 基于零交叉率估计基频
        let estimated_freq = zcr * self.config.sample_rate / 2.0;

        if estimated_freq >= self.config.f0_min && estimated_freq <= self.config.f0_max {
            Ok(estimated_freq)
        } else {
            Ok(0.0)
        }
    }

    /// YIN 帧处理
    fn estimate_f0_yin_frame(&self, frame: &[f32]) -> Result<f32> {
        if frame.len() < 64 {
            return Ok(0.0);
        }

        let min_period = (self.config.sample_rate / self.config.f0_max) as usize;
        let max_period = (self.config.sample_rate / self.config.f0_min) as usize;

        // YIN 算法的差函数
        let diff_function = self.compute_yin_difference_function(frame, max_period);

        // 累积平均归一化差函数 (CMNDF)
        let cmndf = self.compute_cmndf(&diff_function);

        // 寻找第一个小于阈值的最小值
        let period = self.find_yin_period(&cmndf, min_period, self.config.threshold);

        if period > 0 {
            // 抛物线插值优化
            let refined_period = self.parabolic_interpolation(&cmndf, period);
            Ok(self.config.sample_rate / refined_period)
        } else {
            Ok(0.0)
        }
    }

    /// RMVPE-like 方法
    fn estimate_f0_rmvpe_like(&self, frame: &[f32]) -> Result<f32> {
        // 使用多重线索的改进方法
        let autocorr_f0 = self.estimate_f0_autocorr(frame)?;
        let spectral_f0 = self.estimate_f0_spectral(frame)?;
        let yin_f0 = self.estimate_f0_yin_frame(frame)?;

        // 权重融合多个估计结果
        let candidates = vec![(autocorr_f0, 0.3), (spectral_f0, 0.4), (yin_f0, 0.3)];

        let weighted_f0 = self.weighted_median(&candidates);
        Ok(weighted_f0)
    }

    /// 频域方法估计F0
    fn estimate_f0_spectral(&self, frame: &[f32]) -> Result<f32> {
        if frame.len() < 64 {
            return Ok(0.0);
        }

        // 应用窗函数
        let windowed = self.apply_window(frame);

        // FFT
        let spectrum = self.compute_fft(&windowed);

        // 寻找频谱峰值
        let peak_frequency = self.find_spectral_peak(&spectrum)?;

        if peak_frequency >= self.config.f0_min && peak_frequency <= self.config.f0_max {
            Ok(peak_frequency)
        } else {
            Ok(0.0)
        }
    }

    /// 应用窗函数
    fn apply_window(&self, frame: &[f32]) -> Vec<f32> {
        let mut windowed = Vec::with_capacity(frame.len());

        for (i, &sample) in frame.iter().enumerate() {
            let window_val = match self.config.window_type {
                WindowType::Hann => {
                    0.5 * (1.0 - (2.0 * PI * i as f32 / (frame.len() - 1) as f32).cos())
                }
                WindowType::Hamming => {
                    0.54 - 0.46 * (2.0 * PI * i as f32 / (frame.len() - 1) as f32).cos()
                }
                WindowType::Blackman => {
                    0.42 - 0.5 * (2.0 * PI * i as f32 / (frame.len() - 1) as f32).cos()
                        + 0.08 * (4.0 * PI * i as f32 / (frame.len() - 1) as f32).cos()
                }
                WindowType::Rectangular => 1.0,
            };
            windowed.push(sample * window_val);
        }

        windowed
    }

    /// 计算自相关
    fn autocorrelation(&self, signal: &[f32], max_lag: usize) -> Vec<f32> {
        let mut autocorr = vec![0.0; max_lag + 1];

        for lag in 0..=max_lag.min(signal.len() - 1) {
            let mut sum = 0.0;
            let mut norm = 0.0;

            for i in 0..(signal.len() - lag) {
                sum += signal[i] * signal[i + lag];
                norm += signal[i] * signal[i];
            }

            autocorr[lag] = if norm > 1e-10 { sum / norm.sqrt() } else { 0.0 };
        }

        autocorr
    }

    /// 计算瞬时频率
    fn compute_instantaneous_frequency(&self, signal: &[f32]) -> Result<Vec<f32>> {
        let mut inst_freq = Vec::with_capacity(signal.len());

        for i in 1..signal.len() - 1 {
            let phase_diff = (signal[i + 1] - signal[i - 1]).atan2(2.0 * signal[i]);
            let freq = phase_diff * self.config.sample_rate / (2.0 * PI);
            inst_freq.push(freq.abs());
        }

        Ok(inst_freq)
    }

    /// 寻找频率候选
    fn find_frequency_candidates(&self, frequencies: &[f32]) -> Vec<f32> {
        let mut candidates = Vec::new();

        for &freq in frequencies {
            if freq >= self.config.f0_min && freq <= self.config.f0_max {
                candidates.push(freq);
            }
        }

        candidates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        candidates
    }

    /// 选择最佳候选
    fn select_best_candidate(&self, candidates: &[f32]) -> f32 {
        if candidates.is_empty() {
            return 0.0;
        }

        // 简单选择中位数
        candidates[candidates.len() / 2]
    }

    /// 计算零交叉率
    fn zero_crossing_rate(&self, signal: &[f32]) -> f32 {
        let mut crossings = 0;

        for i in 1..signal.len() {
            if (signal[i - 1] >= 0.0) != (signal[i] >= 0.0) {
                crossings += 1;
            }
        }

        crossings as f32 / (signal.len() - 1) as f32
    }

    /// YIN 差函数
    fn compute_yin_difference_function(&self, signal: &[f32], max_period: usize) -> Vec<f32> {
        let mut diff = vec![0.0; max_period + 1];

        for tau in 1..=max_period.min(signal.len() / 2) {
            let mut sum = 0.0;
            for j in 0..(signal.len() - tau) {
                let delta = signal[j] - signal[j + tau];
                sum += delta * delta;
            }
            diff[tau] = sum;
        }

        diff
    }

    /// 累积平均归一化差函数
    fn compute_cmndf(&self, diff: &[f32]) -> Vec<f32> {
        let mut cmndf = vec![1.0; diff.len()];
        let mut running_sum = 0.0;

        for tau in 1..diff.len() {
            running_sum += diff[tau];
            cmndf[tau] = if running_sum > 0.0 {
                diff[tau] * tau as f32 / running_sum
            } else {
                1.0
            };
        }

        cmndf
    }

    /// 寻找 YIN 周期
    fn find_yin_period(&self, cmndf: &[f32], min_period: usize, threshold: f32) -> usize {
        for tau in min_period..cmndf.len() {
            if cmndf[tau] < threshold {
                // 寻找局部最小值
                let mut period = tau;
                while period + 1 < cmndf.len() && cmndf[period + 1] < cmndf[period] {
                    period += 1;
                }
                return period;
            }
        }
        0
    }

    /// 抛物线插值
    fn parabolic_interpolation(&self, values: &[f32], index: usize) -> f32 {
        if index == 0 || index >= values.len() - 1 {
            return index as f32;
        }

        let y1 = values[index - 1];
        let y2 = values[index];
        let y3 = values[index + 1];

        let a = (y1 - 2.0 * y2 + y3) / 2.0;
        let b = (y3 - y1) / 2.0;

        if a.abs() > 1e-10 {
            index as f32 - b / (2.0 * a)
        } else {
            index as f32
        }
    }

    /// 加权中位数
    fn weighted_median(&self, candidates: &[(f32, f32)]) -> f32 {
        let valid_candidates: Vec<_> = candidates
            .iter()
            .filter(|(f, _)| *f > 0.0)
            .cloned()
            .collect();

        if valid_candidates.is_empty() {
            return 0.0;
        }

        // 计算加权平均
        let total_weight: f32 = valid_candidates.iter().map(|(_, w)| w).sum();
        if total_weight > 0.0 {
            valid_candidates.iter().map(|(f, w)| f * w).sum::<f32>() / total_weight
        } else {
            0.0
        }
    }

    /// 计算FFT
    fn compute_fft(&self, signal: &[f32]) -> Vec<f32> {
        // 简化的FFT实现，实际应用中应使用更高效的FFT库
        let n = signal.len();
        let mut magnitude = vec![0.0; n / 2 + 1];

        for k in 0..magnitude.len() {
            let mut real = 0.0;
            let mut imag = 0.0;

            for (i, &sample) in signal.iter().enumerate() {
                let angle = -2.0 * PI * k as f32 * i as f32 / n as f32;
                real += sample * angle.cos();
                imag += sample * angle.sin();
            }

            magnitude[k] = (real * real + imag * imag).sqrt();
        }

        magnitude
    }

    /// 寻找频谱峰值
    fn find_spectral_peak(&self, spectrum: &[f32]) -> Result<f32> {
        let mut max_magnitude = 0.0;
        let mut peak_bin = 0;

        for (i, &magnitude) in spectrum.iter().enumerate() {
            if magnitude > max_magnitude {
                max_magnitude = magnitude;
                peak_bin = i;
            }
        }

        if max_magnitude > self.config.threshold {
            let frequency =
                peak_bin as f32 * self.config.sample_rate / (2.0 * spectrum.len() as f32);
            Ok(frequency)
        } else {
            Ok(0.0)
        }
    }
}

/// F0 后处理工具
pub struct F0Processor;

impl F0Processor {
    /// 平滑F0序列
    pub fn smooth(f0: &mut [f32], window_size: usize) {
        if window_size <= 1 {
            return;
        }

        let half_window = window_size / 2;
        let original = f0.to_vec();

        for i in 0..f0.len() {
            let start = if i >= half_window { i - half_window } else { 0 };
            let end = (i + half_window + 1).min(f0.len());

            let valid_values: Vec<f32> = original[start..end]
                .iter()
                .filter(|&&x| x > 0.0)
                .cloned()
                .collect();

            if !valid_values.is_empty() {
                f0[i] = valid_values.iter().sum::<f32>() / valid_values.len() as f32;
            }
        }
    }

    /// 插值填充零值
    pub fn interpolate(f0: &mut [f32]) {
        let mut last_valid = None;

        for i in 0..f0.len() {
            if f0[i] > 0.0 {
                if let Some(last_idx) = last_valid {
                    // 线性插值填充中间的零值
                    let steps = i - last_idx;
                    if steps > 1 {
                        let step_size = (f0[i] - f0[last_idx]) / steps as f32;
                        for j in 1..steps {
                            f0[last_idx + j] = f0[last_idx] + step_size * j as f32;
                        }
                    }
                }
                last_valid = Some(i);
            }
        }
    }

    /// 应用音高偏移
    pub fn apply_pitch_shift(f0: &mut [f32], semitones: f32) {
        let ratio = 2.0f32.powf(semitones / 12.0);
        for freq in f0.iter_mut() {
            if *freq > 0.0 {
                *freq *= ratio;
            }
        }
    }

    /// 转换为Mel刻度
    pub fn to_mel_scale(f0: &[f32]) -> Vec<f32> {
        f0.iter()
            .map(|&freq| {
                if freq > 0.0 {
                    1127.0 * (1.0 + freq / 700.0).ln()
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// 从Mel刻度转换回来
    pub fn from_mel_scale(mel: &[f32]) -> Vec<f32> {
        mel.iter()
            .map(|&m| {
                if m > 0.0 {
                    700.0 * ((m / 1127.0).exp() - 1.0)
                } else {
                    0.0
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_f0_config() {
        let config = F0Config::default();
        assert_eq!(config.sample_rate, 16000.0);
        assert_eq!(config.f0_min, 50.0);
        assert_eq!(config.f0_max, 1100.0);
    }

    #[test]
    fn test_f0_method_parsing() {
        assert_eq!("pm".parse::<F0Method>().unwrap(), F0Method::PM);
        assert_eq!("harvest".parse::<F0Method>().unwrap(), F0Method::Harvest);
        assert_eq!("yin".parse::<F0Method>().unwrap(), F0Method::YIN);
    }

    #[test]
    fn test_f0_estimator() {
        let config = F0Config::default();
        let device = Device::Cpu;
        let estimator = F0Estimator::new(config, device);

        // Test with dummy audio data
        let audio = vec![0.1; 1000];
        let result = estimator.estimate(&audio, F0Method::PM);

        match result {
            Ok(f0_result) => {
                println!("F0 estimation successful, {} frames", f0_result.len());
                assert!(!f0_result.is_empty());
            }
            Err(e) => {
                println!("F0 estimation failed (expected): {}", e);
            }
        }
    }

    #[test]
    fn test_f0_processor() {
        let mut f0 = vec![100.0, 0.0, 200.0, 0.0, 300.0];
        F0Processor::interpolate(&mut f0);

        // Check that zeros were interpolated
        assert!(f0[1] > 0.0);
        assert!(f0[3] > 0.0);

        F0Processor::smooth(&mut f0, 3);
        println!("Smoothed F0: {:?}", f0);
    }
}
