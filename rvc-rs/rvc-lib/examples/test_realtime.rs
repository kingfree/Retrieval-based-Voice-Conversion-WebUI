use rvc_lib::{GUIConfig, RVC, AudioCallbackConfig};
use std::thread;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("RVC Rust实时推理功能测试");
    println!("========================");

    // 创建测试配置
    let config = create_test_config();

    // 显示配置信息
    print_config(&config);

    // 测试基础功能
    test_basic_functionality(&config)?;

    // 测试音频处理流程
    test_audio_processing(&config)?;

    // 测试实时流处理
    test_realtime_streaming(&config)?;

    println!("\n✅ 所有测试完成！");
    Ok(())
}

fn create_test_config() -> GUIConfig {
    GUIConfig {
        pth_path: "assets/pretrained/f0G40k.pth".to_string(),
        index_path: "logs/added_index.index".to_string(),
        sg_hostapi: "MME".to_string(),
        sg_wasapi_exclusive: false,
        sg_input_device: "default".to_string(),
        sg_output_device: "default".to_string(),
        sr_type: "sr_model".to_string(),
        threshold: -60.0,
        pitch: 0.0,
        formant: 0.0,
        rms_mix_rate: 0.0,
        index_rate: 0.5,
        block_time: 0.25,
        crossfade_length: 0.05,
        extra_time: 2.5,
        n_cpu: 4,
        use_jit: false,
        use_pv: false,
        f0method: "rmvpe".to_string(),
        i_noise_reduce: false,
        o_noise_reduce: false,
    }
}

fn print_config(config: &GUIConfig) {
    println!("配置信息:");
    println!("  模型路径: {}", config.pth_path);
    println!("  索引路径: {}", config.index_path);
    println!("  音高调整: {}", config.pitch);
    println!("  共振峰: {}", config.formant);
    println!("  索引率: {}", config.index_rate);
    println!("  块时间: {}s", config.block_time);
    println!("  CPU核心: {}", config.n_cpu);
    println!("  F0方法: {}", config.f0method);
    println!();
}

fn test_basic_functionality(config: &GUIConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 测试基础功能...");

    // 创建RVC实例
    let mut rvc = RVC::new(config);

    // 测试配置更新
    println!("  - 测试参数更新");
    rvc.change_key(12.0); // 升高一个八度
    rvc.change_formant(2.0); // 改变共振峰
    rvc.change_index_rate(0.8); // 调整索引率

    // 获取模型信息
    let model_info = rvc.get_model_info();
    println!("  - 模型状态:");
    println!("    HuBERT已加载: {}", model_info.hubert_loaded);
    println!("    生成器已加载: {}", model_info.model_loaded);
    println!("    索引已加载: {}", model_info.index_loaded);
    println!("    目标采样率: {}", model_info.target_sr);
    println!("    F0条件: {}", model_info.f0_conditioned);
    println!("    版本: {}", model_info.version);
    println!("    设备: {}", model_info.device);

    // 检查就绪状态
    println!("  - RVC就绪状态: {}", rvc.is_ready());

    // 清除缓存测试
    println!("  - 清除缓存");
    rvc.clear_cache();

    println!("✅ 基础功能测试完成\n");
    Ok(())
}

fn test_audio_processing(config: &GUIConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎵 测试音频处理流程...");

    let mut rvc = RVC::new(config);

    // 创建测试音频数据 (1秒的正弦波，16kHz)
    let sample_rate = 16000;
    let duration = 1.0; // 1秒
    let frequency = 440.0; // A4音符

    let mut test_audio = Vec::new();
    for i in 0..(sample_rate as f32 * duration) as usize {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * std::f32::consts::PI * frequency * t).sin() * 0.5;
        test_audio.push(sample);
    }

    println!("  - 创建测试音频: {}Hz正弦波，{}秒", frequency, duration);
    println!("  - 音频长度: {} 样本", test_audio.len());

    // 测试简单推理
    if rvc.is_ready() {
        println!("  - 执行音频推理...");
        match rvc.infer_simple(&test_audio) {
            Ok(output) => {
                println!("  - 推理成功! 输出长度: {} 样本", output.len());

                // 检查输出的基本统计信息
                let max_val = output.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let min_val = output.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let mean = output.iter().sum::<f32>() / output.len() as f32;

                println!("  - 输出统计: 最大值={:.4}, 最小值={:.4}, 均值={:.4}",
                         max_val, min_val, mean);
            }
            Err(e) => {
                println!("  ⚠️ 推理失败: {}", e);
                println!("  - 这可能是因为模型文件不存在或路径错误");
            }
        }
    } else {
        println!("  ⚠️ RVC未就绪，跳过音频推理测试");
    }

    // 测试F0提取
    println!("  - 测试F0提取功能");
    test_f0_extraction(&mut rvc, &test_audio)?;

    // 测试完整推理流程
    if rvc.is_ready() {
        println!("  - 测试完整推理流程");
        match rvc.infer(&test_audio, 4000, 1600, 2400, "rmvpe") {
            Ok(output) => {
                println!("  - 完整推理成功! 输出长度: {} 样本", output.len());
            }
            Err(e) => {
                println!("  ⚠️ 完整推理失败: {}", e);
            }
        }
    }

    println!("✅ 音频处理测试完成\n");
    Ok(())
}

fn test_f0_extraction(rvc: &mut RVC, audio: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
    let methods = ["pm", "harvest", "rmvpe"];

    for method in &methods {
        println!("    - 测试{}方法", method.to_uppercase());
        let (pitch, pitchf) = rvc.get_f0(audio, 0.0, method);

        println!("      F0提取完成: pitch长度={}, pitchf长度={}",
                 pitch.len(), pitchf.len());

        // 计算基础统计
        if !pitchf.is_empty() {
            let non_zero_pitch: Vec<f32> = pitchf.iter()
                .filter(|&&p| p > 0.0)
                .cloned()
                .collect();

            if !non_zero_pitch.is_empty() {
                let mean_pitch = non_zero_pitch.iter().sum::<f32>() / non_zero_pitch.len() as f32;
                println!("      平均音高: {:.2}Hz", mean_pitch);
            } else {
                println!("      未检测到音高信息");
            }
        }
    }

    Ok(())
}

fn test_realtime_streaming(config: &GUIConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎙️ 测试实时流处理...");

    let mut rvc = RVC::new(config);

    // 测试流初始化
    println!("  - 初始化音频流...");
    match rvc.start_stream(44100, 512) {
        Ok(()) => {
            println!("  - 音频流启动成功");
            println!("  - 流状态: {}", rvc.is_streaming());

            // 获取流信息
            if let Some(info) = rvc.get_stream_info() {
                println!("  - 流信息: {}", info);
            }

            // 测试音频回调创建
            test_audio_callback(&mut rvc)?;

            // 模拟流处理
            simulate_stream_processing(&mut rvc)?;

            // 停止流
            println!("  - 停止音频流...");
            match rvc.stop_stream() {
                Ok(()) => println!("  - 音频流已停止"),
                Err(e) => println!("  ⚠️ 停止流失败: {}", e),
            }
        }
        Err(e) => {
            println!("  ⚠️ 无法启动音频流: {}", e);
            println!("  - 这可能是因为模型未加载或音频设备不可用");
        }
    }

    println!("✅ 实时流处理测试完成\n");
    Ok(())
}

fn test_audio_callback(rvc: &mut RVC) -> Result<(), Box<dyn std::error::Error>> {
    println!("  - 测试音频回调创建...");

    let callback_config = AudioCallbackConfig {
        sample_rate: 44100,
        block_size: 512,
        enable_crossfade: true,
        crossfade_samples: 64,
    };

    match rvc.create_audio_callback(callback_config) {
        Ok(_callback) => {
            println!("  - 音频回调创建成功");
            // 注意：这里我们不能直接测试回调，因为它需要音频上下文
        }
        Err(e) => {
            println!("  ⚠️ 音频回调创建失败: {}", e);
        }
    }

    Ok(())
}

fn simulate_stream_processing(rvc: &mut RVC) -> Result<(), Box<dyn std::error::Error>> {
    println!("  - 模拟流处理...");

    // 创建一些测试块
    let block_size = 512;
    let test_blocks = 5;

    for i in 0..test_blocks {
        // 创建测试音频块
        let mut test_block = vec![0.0f32; block_size];
        let freq = 440.0 + (i as f32 * 100.0); // 变化的频率

        for j in 0..block_size {
            let t = (i * block_size + j) as f32 / 44100.0;
            test_block[j] = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.3;
        }

        // 处理音频块 - 注意这里返回的是处理后的数据，不是空的()
        match rvc.process_stream_chunk(&test_block) {
            Ok(processed_audio) => {
                if i == 0 {
                    println!("    第一个块处理成功，输出长度: {}", processed_audio.len());
                }
            }
            Err(e) => {
                println!("    ⚠️ 块{}处理失败: {}", i + 1, e);
            }
        }

        // 短暂延迟模拟实时处理
        thread::sleep(Duration::from_millis(10));
    }

    println!("  - 流处理模拟完成 ({} 块)", test_blocks);
    Ok(())
}

// 辅助函数：创建测试音频信号
fn create_test_signal(frequency: f32, duration: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration * sample_rate as f32) as usize;
    let mut signal = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * std::f32::consts::PI * frequency * t).sin();
        signal.push(sample);
    }

    signal
}

// 辅助函数：计算音频信号的RMS
fn calculate_rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }

    let sum_squares: f32 = signal.iter().map(|&x| x * x).sum();
    (sum_squares / signal.len() as f32).sqrt()
}

// 辅助函数：应用简单的增益
fn apply_gain(signal: &mut [f32], gain: f32) {
    for sample in signal.iter_mut() {
        *sample *= gain;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = create_test_config();
        assert!(!config.pth_path.is_empty());
        assert_eq!(config.n_cpu, 4);
        assert_eq!(config.pitch, 0.0);
    }

    #[test]
    fn test_signal_generation() {
        let signal = create_test_signal(440.0, 1.0, 16000);
        assert_eq!(signal.len(), 16000);

        let rms = calculate_rms(&signal);
        assert!(rms > 0.0);
        assert!(rms < 1.0);
    }

    #[test]
    fn test_gain_application() {
        let mut signal = vec![0.5, -0.5, 0.8, -0.8];
        apply_gain(&mut signal, 2.0);

        assert_eq!(signal, vec![1.0, -1.0, 1.6, -1.6]);
    }

    #[test]
    fn test_rvc_basic_creation() {
        let config = create_test_config();
        let rvc = RVC::new(&config);

        // 基本创建应该成功
        let model_info = rvc.get_model_info();
        assert!(!model_info.device.is_empty());
    }
}
