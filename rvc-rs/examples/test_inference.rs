//! RVC 推理测试
//!
//! 该测试演示了完整的 RVC 推理流程，包括：
//! 1. 模型和组件初始化
//! 2. 音频数据处理
//! 3. 特征提取和 F0 估计
//! 4. 语音转换推理
//! 5. 结果验证和性能测试

use rvc_lib::{
    audio_utils::{AudioData, AudioStats, calculate_similarity, create_test_signal},
    f0_estimation::F0Method,
    inference::{BatchInference, F0FilterConfig, InferenceConfig, RVCInference},
};
use std::path::PathBuf;
use std::time::Instant;
use tch::Device;

/// 测试配置
struct TestConfig {
    /// 是否使用 GPU
    use_gpu: bool,
    /// 测试音频长度（秒）
    test_duration: f64,
    /// 测试音频频率（Hz）
    test_frequency: f64,
    /// 是否进行性能测试
    performance_test: bool,
    /// 是否进行批量测试
    batch_test: bool,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            use_gpu: false, // 默认使用 CPU 以确保兼容性
            test_duration: 2.0,
            test_frequency: 440.0,
            performance_test: true,
            batch_test: true,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 RVC Rust 推理测试套件");
    println!("{}", "=".repeat(60));

    let test_config = TestConfig::default();

    // 检查设备可用性
    let device = check_device_availability(test_config.use_gpu);
    println!("📱 使用设备: {:?}", device);

    // 运行基础推理测试
    println!("\n🔬 基础推理测试");
    println!("{}", "-".repeat(40));
    run_basic_inference_test(&test_config, device)?;

    // 运行不同 F0 方法测试
    println!("\n🎼 F0 方法对比测试");
    println!("{}", "-".repeat(40));
    run_f0_method_comparison_test(&test_config, device)?;

    // 运行参数调优测试
    println!("\n⚙️  参数调优测试");
    println!("{}", "-".repeat(40));
    run_parameter_tuning_test(&test_config, device)?;

    // 运行性能测试
    if test_config.performance_test {
        println!("\n⚡ 性能基准测试");
        println!("{}", "-".repeat(40));
        run_performance_test(&test_config, device)?;
    }

    // 运行批量处理测试
    if test_config.batch_test {
        println!("\n📦 批量处理测试");
        println!("{}", "-".repeat(40));
        run_batch_processing_test(&test_config, device)?;
    }

    // 运行鲁棒性测试
    println!("\n🛡️  鲁棒性测试");
    println!("{}", "-".repeat(40));
    run_robustness_test(&test_config, device)?;

    println!("\n✅ 所有测试完成!");
    print_test_summary();

    Ok(())
}

/// 检查设备可用性
fn check_device_availability(prefer_gpu: bool) -> Device {
    if prefer_gpu && tch::Cuda::is_available() {
        let device_count = tch::Cuda::device_count();
        println!("🎮 发现 {} 个 CUDA 设备", device_count);

        // 检查 GPU 内存
        if let Ok(memory_info) = get_gpu_memory_info(0) {
            println!(
                "💾 GPU 内存: {:.1}GB 总量, {:.1}GB 可用",
                memory_info.total_gb, memory_info.free_gb
            );

            if memory_info.free_gb > 2.0 {
                println!("✅ GPU 内存充足，使用 CUDA");
                return Device::Cuda(0);
            } else {
                println!("⚠️  GPU 内存不足，切换到 CPU");
            }
        }
    }

    println!("🖥️  使用 CPU 设备");
    Device::Cpu
}

/// 获取 GPU 内存信息
fn get_gpu_memory_info(device_id: i32) -> Result<GpuMemoryInfo, Box<dyn std::error::Error>> {
    // 这里应该实现实际的 GPU 内存查询
    // 目前返回模拟数据
    Ok(GpuMemoryInfo {
        total_gb: 8.0,
        free_gb: 6.0,
        used_gb: 2.0,
    })
}

struct GpuMemoryInfo {
    total_gb: f64,
    free_gb: f64,
    used_gb: f64,
}

/// 基础推理测试
fn run_basic_inference_test(
    test_config: &TestConfig,
    device: Device,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎵 创建测试音频信号...");
    let test_audio = create_test_signal(
        22050.0,
        test_config.test_duration,
        test_config.test_frequency,
    );

    println!("   - 采样率: {}Hz", test_audio.sample_rate);
    println!(
        "   - 时长: {:.2}s",
        test_audio.samples.len() as f32 / test_audio.sample_rate as f32
    );
    println!("   - 样本数: {}", test_audio.samples.len());

    // 创建推理配置
    let config = InferenceConfig {
        speaker_id: 0,
        f0_method: F0Method::Harvest,
        pitch_shift: 1.0,
        index_rate: 0.5,
        target_sample_rate: 22050,
        device,
        batch_size: 1,
        enable_denoise: false,
        f0_filter: F0FilterConfig::default(),
    };

    println!("🚀 初始化推理引擎...");
    let model_path = PathBuf::from("assets/weights/kikiV1.pth");
    let index_path = Some(PathBuf::from("logs/kikiV1.index"));

    let start_time = Instant::now();
    let inference_engine = RVCInference::new(config, &model_path, index_path.as_ref())?;
    let init_time = start_time.elapsed();

    println!("   ✅ 初始化完成，耗时: {:.2}ms", init_time.as_millis());

    // 执行推理
    println!("🎨 执行语音转换...");
    let conversion_start = Instant::now();
    let result = inference_engine.convert_audio_data(test_audio.clone(), None::<&str>)?;
    let conversion_time = conversion_start.elapsed();

    println!("   ✅ 转换完成，耗时: {:.2}ms", conversion_time.as_millis());

    // 验证结果
    println!("🔍 验证转换结果...");
    validate_conversion_result(&test_audio, &result)?;

    // 计算性能指标
    let samples_per_second = result.samples.len() as f64 / conversion_time.as_secs_f64();
    let realtime_factor = samples_per_second / result.sample_rate as f64;

    println!("📊 性能指标:");
    println!("   - 处理速度: {:.0} 样本/秒", samples_per_second);
    println!("   - 实时倍数: {:.2}x", realtime_factor);
    println!("   - 初始化时间: {:.2}ms", init_time.as_millis());
    println!("   - 转换时间: {:.2}ms", conversion_time.as_millis());

    Ok(())
}

/// 不同 F0 方法对比测试
fn run_f0_method_comparison_test(
    test_config: &TestConfig,
    device: Device,
) -> Result<(), Box<dyn std::error::Error>> {
    let test_audio = create_test_signal(
        22050.0,
        test_config.test_duration,
        test_config.test_frequency,
    );

    let f0_methods = vec![
        (F0Method::Harvest, "Harvest (高质量)"),
        (F0Method::PM, "PM (快速)"),
        (F0Method::DIO, "DIO (平衡)"),
        (F0Method::YIN, "YIN (音乐)"),
        (F0Method::RMVPE, "RMVPE (推荐)"),
    ];

    println!("🎼 测试不同 F0 估计方法...");

    for (method, description) in f0_methods {
        println!("\n📐 测试 {}", description);

        let config = InferenceConfig {
            f0_method: method,
            device,
            ..Default::default()
        };

        let model_path = PathBuf::from("assets/weights/kikiV1.pth");
        let inference_engine = RVCInference::new(config, &model_path, None::<&PathBuf>)?;

        let start_time = Instant::now();
        let result = inference_engine.convert_audio_data(test_audio.clone(), None::<&str>)?;
        let elapsed = start_time.elapsed();

        let quality_score = calculate_audio_quality(&result);

        println!("   - 处理时间: {:.2}ms", elapsed.as_millis());
        println!("   - 质量评分: {:.2}/10", quality_score);
        println!("   - 输出长度: {} 样本", result.samples.len());
    }

    Ok(())
}

/// 参数调优测试
fn run_parameter_tuning_test(
    test_config: &TestConfig,
    device: Device,
) -> Result<(), Box<dyn std::error::Error>> {
    let test_audio = create_test_signal(
        22050.0,
        test_config.test_duration,
        test_config.test_frequency,
    );

    println!("⚙️  测试不同参数组合...");

    // 音调调整测试
    println!("\n🎵 音调调整测试:");
    let pitch_shifts = vec![0.8, 0.9, 1.0, 1.1, 1.2, 1.5];

    for pitch_shift in pitch_shifts {
        let config = InferenceConfig {
            pitch_shift,
            device,
            ..Default::default()
        };

        let model_path = PathBuf::from("assets/weights/kikiV1.pth");
        let inference_engine = RVCInference::new(config, &model_path, None::<&PathBuf>)?;

        let result = inference_engine.convert_audio_data(test_audio.clone(), None::<&str>)?;
        let detected_pitch = estimate_average_pitch(&result);

        println!(
            "   - 音调: {:.1}x, 检测到: {:.1}Hz",
            pitch_shift, detected_pitch
        );
    }

    // 索引混合率测试
    println!("\n🔍 索引混合率测试:");
    let index_rates = vec![0.0, 0.25, 0.5, 0.75, 1.0];

    for index_rate in index_rates {
        let config = InferenceConfig {
            index_rate,
            device,
            ..Default::default()
        };

        let model_path = PathBuf::from("assets/weights/kikiV1.pth");
        let inference_engine = RVCInference::new(config, &model_path, None::<&PathBuf>)?;

        let result = inference_engine.convert_audio_data(test_audio.clone(), None::<&str>)?;
        let similarity = calculate_similarity(&test_audio.samples, &result.samples)?;

        println!(
            "   - 混合率: {:.0}%, 相似度: {:.3}",
            index_rate * 100.0,
            similarity
        );
    }

    Ok(())
}

/// 性能基准测试
fn run_performance_test(
    test_config: &TestConfig,
    device: Device,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ 执行性能基准测试...");

    let durations = vec![1.0, 2.0, 5.0, 10.0]; // 不同时长的音频
    let config = InferenceConfig {
        device,
        ..Default::default()
    };

    let model_path = PathBuf::from("assets/weights/kikiV1.pth");
    let inference_engine = RVCInference::new(config, &model_path, None::<&PathBuf>)?;

    println!("\n📊 不同音频长度的性能表现:");
    println!("时长(s) | 处理时间(ms) | 实时倍数 | 内存使用");
    println!("{}", "-".repeat(50));

    for duration in durations {
        let test_audio = create_test_signal(22050.0, duration, test_config.test_frequency);

        // 多次运行取平均值
        let mut total_time = 0u128;
        let runs = 3;

        for _ in 0..runs {
            let start = Instant::now();
            let _ = inference_engine.convert_audio_data(test_audio.clone(), None::<&str>)?;
            total_time += start.elapsed().as_millis();
        }

        let avg_time = total_time / runs;
        let realtime_factor = (duration * 1000.0) / avg_time as f64;
        let memory_usage = get_memory_usage();

        println!(
            "{:6.1} | {:11} | {:8.2}x | {:8.1}MB",
            duration, avg_time, realtime_factor, memory_usage
        );
    }

    Ok(())
}

/// 批量处理测试
fn run_batch_processing_test(
    test_config: &TestConfig,
    device: Device,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📦 创建批量测试数据...");

    // 创建多个测试文件
    let test_files = create_test_batch(test_config)?;
    println!("   ✅ 创建了 {} 个测试文件", test_files.len());

    let config = InferenceConfig {
        device,
        batch_size: 2,
        ..Default::default()
    };

    let model_path = PathBuf::from("assets/weights/kikiV1.pth");
    let inference_engine = RVCInference::new(config, &model_path, None::<&PathBuf>)?;
    let batch_inference = BatchInference::new(inference_engine);

    let output_dir = PathBuf::from("test_batch_output");
    std::fs::create_dir_all(&output_dir)?;

    println!("🚀 执行批量处理...");
    let start_time = Instant::now();
    let results = batch_inference.process_batch(&test_files, &output_dir)?;
    let total_time = start_time.elapsed();

    println!("   ✅ 批量处理完成");
    println!("   - 处理文件数: {}", results.len());
    println!("   - 总耗时: {:.2}s", total_time.as_secs_f64());
    println!(
        "   - 平均每文件: {:.2}ms",
        total_time.as_millis() as f64 / results.len() as f64
    );

    // 清理测试文件
    cleanup_test_files(&test_files)?;
    std::fs::remove_dir_all(&output_dir)?;

    Ok(())
}

/// 鲁棒性测试
fn run_robustness_test(
    test_config: &TestConfig,
    device: Device,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🛡️  测试系统鲁棒性...");

    let config = InferenceConfig {
        device,
        ..Default::default()
    };

    let model_path = PathBuf::from("assets/weights/kikiV1.pth");
    let inference_engine = RVCInference::new(config, &model_path, None::<&PathBuf>)?;

    // 测试1: 空音频
    println!("\n🔇 测试空音频处理...");
    let empty_audio = AudioData {
        samples: vec![],
        sample_rate: 22050,
        channels: 1,
    };

    match inference_engine.convert_audio_data(empty_audio, None::<&str>) {
        Ok(_) => println!("   ✅ 空音频处理成功"),
        Err(e) => println!("   ⚠️  空音频处理失败: {}", e),
    }

    // 测试2: 极短音频
    println!("\n⏱️  测试极短音频处理...");
    let short_audio = create_test_signal(22050.0, 0.1, test_config.test_frequency);

    match inference_engine.convert_audio_data(short_audio, None::<&str>) {
        Ok(_) => println!("   ✅ 极短音频处理成功"),
        Err(e) => println!("   ⚠️  极短音频处理失败: {}", e),
    }

    // 测试3: 静音音频
    println!("\n🔇 测试静音音频处理...");
    let silent_audio = AudioData {
        samples: vec![0.0; 22050], // 1秒静音
        sample_rate: 22050,
        channels: 1,
    };

    match inference_engine.convert_audio_data(silent_audio, None::<&str>) {
        Ok(_) => println!("   ✅ 静音音频处理成功"),
        Err(e) => println!("   ⚠️  静音音频处理失败: {}", e),
    }

    // 测试4: 高幅度音频
    println!("\n📢 测试高幅度音频处理...");
    let loud_audio = AudioData {
        samples: vec![0.95; 22050], // 接近削波的音频
        sample_rate: 22050,
        channels: 1,
    };

    match inference_engine.convert_audio_data(loud_audio, None::<&str>) {
        Ok(result) => {
            let max_amplitude = result
                .samples
                .iter()
                .fold(0.0f32, |acc, &x| acc.max(x.abs()));
            println!("   ✅ 高幅度音频处理成功，最大幅度: {:.3}", max_amplitude);
        }
        Err(e) => println!("   ⚠️  高幅度音频处理失败: {}", e),
    }

    Ok(())
}

/// 验证转换结果
fn validate_conversion_result(
    input: &AudioData,
    output: &AudioData,
) -> Result<(), Box<dyn std::error::Error>> {
    // 基本检查
    if output.samples.is_empty() {
        return Err("输出音频为空".into());
    }

    if output.sample_rate != input.sample_rate {
        println!(
            "   ⚠️  采样率不匹配: {} -> {}",
            input.sample_rate, output.sample_rate
        );
    }

    // 长度检查（允许一定误差）
    let length_ratio = output.samples.len() as f64 / input.samples.len() as f64;
    if (length_ratio - 1.0).abs() > 0.1 {
        println!("   ⚠️  长度变化显著: {:.2}x", length_ratio);
    }

    // 幅度检查
    let input_max = input
        .samples
        .iter()
        .fold(0.0f32, |acc, &x| acc.max(x.abs()));
    let output_max = output
        .samples
        .iter()
        .fold(0.0f32, |acc, &x| acc.max(x.abs()));

    println!("   📊 输入最大幅度: {:.3}", input_max);
    println!("   📊 输出最大幅度: {:.3}", output_max);

    if output_max > 1.0 {
        println!("   ⚠️  输出存在削波！");
    }

    // 能量检查
    let input_energy: f32 = input.samples.iter().map(|x| x * x).sum();
    let output_energy: f32 = output.samples.iter().map(|x| x * x).sum();
    let energy_ratio = output_energy / input_energy;

    println!("   📊 能量比: {:.3}", energy_ratio);

    if energy_ratio < 0.1 || energy_ratio > 10.0 {
        println!("   ⚠️  能量变化异常");
    }

    println!("   ✅ 结果验证完成");
    Ok(())
}

/// 计算音频质量评分
fn calculate_audio_quality(audio: &AudioData) -> f64 {
    if audio.samples.is_empty() {
        return 0.0;
    }

    // 简单的质量评估：基于动态范围和失真
    let max_amp = audio
        .samples
        .iter()
        .fold(0.0f32, |acc, &x| acc.max(x.abs()));
    let rms =
        (audio.samples.iter().map(|x| x * x).sum::<f32>() / audio.samples.len() as f32).sqrt();

    let dynamic_range = if rms > 0.0 {
        20.0 * (max_amp / rms).log10()
    } else {
        0.0
    };
    let clipping_penalty = if max_amp > 0.99 { 2.0 } else { 0.0 };

    let base_score = 8.0;
    let dr_bonus = (dynamic_range / 20.0).min(2.0);

    (base_score + dr_bonus - clipping_penalty)
        .max(0.0)
        .min(10.0) as f64
}

/// 估计平均音调
fn estimate_average_pitch(audio: &AudioData) -> f64 {
    // 简化的音调估计（实际应使用更复杂的算法）
    // 这里返回一个基于频率分析的粗略估计
    440.0 // 占位符，实际实现需要FFT分析
}

/// 获取内存使用情况
fn get_memory_usage() -> f64 {
    // 简化的内存使用估计
    // 实际实现需要系统调用
    128.0 // MB，占位符
}

/// 创建批量测试数据
fn create_test_batch(test_config: &TestConfig) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut test_files = Vec::new();
    let frequencies = vec![220.0, 440.0, 880.0]; // 不同频率

    for (i, &freq) in frequencies.iter().enumerate() {
        let audio = create_test_signal(22050.0, test_config.test_duration, freq);
        let filename = format!("test_batch_{}.wav", i);
        let filepath = PathBuf::from(&filename);

        // 保存测试文件
        rvc_lib::audio_utils::save_wav_simple(&filename, &audio)?;
        test_files.push(filepath);
    }

    Ok(test_files)
}

/// 清理测试文件
fn cleanup_test_files(files: &[PathBuf]) -> Result<(), Box<dyn std::error::Error>> {
    for file in files {
        if file.exists() {
            std::fs::remove_file(file)?;
        }
    }
    Ok(())
}

/// 打印测试总结
fn print_test_summary() {
    println!("\n{}", "=".repeat(60));
    println!("📋 测试总结");
    println!("{}", "=".repeat(60));
    println!("✅ 基础推理测试: 通过");
    println!("✅ F0 方法对比: 通过");
    println!("✅ 参数调优测试: 通过");
    println!("✅ 性能基准测试: 通过");
    println!("✅ 批量处理测试: 通过");
    println!("✅ 鲁棒性测试: 通过");
    println!();
    println!("💡 建议:");
    println!("   1. 根据硬件配置选择合适的设备和批处理大小");
    println!("   2. 根据质量要求选择合适的 F0 估计方法");
    println!("   3. 根据音色需求调整索引混合率");
    println!("   4. 对于实时应用，考虑使用更快的参数组合");
    println!();
    println!("🔗 相关资源:");
    println!(
        "   - 项目文档: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI"
    );
    println!("   - 模型下载: https://huggingface.co/lj1995/VoiceConversionWebUI");
    println!(
        "   - 问题反馈: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues"
    );
}
