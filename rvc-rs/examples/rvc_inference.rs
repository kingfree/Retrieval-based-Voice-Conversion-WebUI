//! RVC 推理示例
//!
//! 该示例演示如何使用 RVC Rust 库进行语音转换推理。
//! 展示了完整的推理流程，包括模型加载、特征提取、F0 估计和音频生成。

use rvc_lib::{
    audio_utils::{AudioData, create_test_signal},
    f0_estimation::F0Method,
    inference::{BatchInference, F0FilterConfig, InferenceConfig, RVCInference},
};
use std::path::PathBuf;
use tch::Device;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎵 RVC Rust 推理示例");
    println!("==================");

    // 1. 配置推理参数
    let config = InferenceConfig {
        speaker_id: 0,
        f0_method: F0Method::Harvest,
        pitch_shift: 1.2, // 提高 20% 音调
        index_rate: 0.75, // 75% 使用检索特征
        target_sample_rate: 22050,
        device: Device::Cpu, // 使用 CPU，可改为 Device::Cuda(0) 使用 GPU
        batch_size: 1,
        enable_denoise: true,
        f0_filter: F0FilterConfig {
            median_filter_radius: 5,
            enable_smoothing: true,
            smoothing_factor: 0.85,
        },
    };

    println!("⚙️  推理配置:");
    println!("   - 设备: {:?}", config.device);
    println!("   - F0 方法: {:?}", config.f0_method);
    println!("   - 音调调整: {:.1}x", config.pitch_shift);
    println!("   - 索引混合率: {:.1}%", config.index_rate * 100.0);
    println!("   - 目标采样率: {}Hz", config.target_sample_rate);

    // 2. 设置模型路径
    let model_path = PathBuf::from("assets/weights/kikiV1.pth");
    let index_path = Some(PathBuf::from("logs/kikiV1.index"));

    println!("\n📁 模型文件:");
    println!("   - 模型: {:?}", model_path);
    println!(
        "   - 索引: {:?}",
        index_path.as_ref().unwrap_or(&PathBuf::new())
    );

    // 3. 初始化推理引擎
    println!("\n🚀 初始化推理引擎...");
    let inference_engine = match RVCInference::new(config, &model_path, index_path.as_ref()) {
        Ok(engine) => {
            println!("✅ 推理引擎初始化成功");
            engine
        }
        Err(e) => {
            println!("❌ 推理引擎初始化失败: {}", e);
            println!("💡 提示: 请确保模型文件存在并且路径正确");
            return Err(e.into());
        }
    };

    // 4. 显示推理统计信息
    let stats = inference_engine.get_inference_stats();
    println!("\n📊 推理引擎统计:");
    println!("   - 设备: {}", stats.device);
    println!(
        "   - HuBERT 参数: ~{:.1}M",
        stats.hubert_parameters as f64 / 1_000_000.0
    );
    println!(
        "   - 生成器参数: ~{:.1}M",
        stats.generator_parameters as f64 / 1_000_000.0
    );
    println!(
        "   - FAISS 索引: {}",
        if stats.has_index {
            "已加载"
        } else {
            "未加载"
        }
    );

    // 5. 示例 1: 单文件推理
    println!("\n{}", "=".repeat(50));
    println!("📝 示例 1: 单文件语音转换");
    println!("{}", "=".repeat(50));

    let input_audio_path = PathBuf::from("test.wav");
    let output_audio_path = PathBuf::from("output_converted.wav");

    if input_audio_path.exists() {
        println!("🎵 处理输入文件: {:?}", input_audio_path);

        match inference_engine.convert_voice(&input_audio_path, &output_audio_path) {
            Ok(result_audio) => {
                println!("✅ 单文件转换成功!");
                println!(
                    "   - 输出长度: {:.2}s",
                    result_audio.data.len() as f32 / result_audio.sample_rate
                );
                println!("   - 输出文件: {:?}", output_audio_path);
            }
            Err(e) => {
                println!("❌ 单文件转换失败: {}", e);
            }
        }
    } else {
        println!("⚠️  输入文件不存在: {:?}", input_audio_path);
        println!("💡 创建测试音频信号进行演示...");

        // 创建测试音频信号
        let test_audio = create_test_signal(22050.0, 3.0, 440.0); // 3秒 440Hz 正弦波
        match inference_engine.convert_audio_data(test_audio, Some(&output_audio_path)) {
            Ok(result_audio) => {
                println!("✅ 测试信号转换成功!");
                println!(
                    "   - 输出长度: {:.2}s",
                    result_audio.data.len() as f32 / result_audio.sample_rate
                );
                println!("   - 输出文件: {:?}", output_audio_path);
            }
            Err(e) => {
                println!("❌ 测试信号转换失败: {}", e);
            }
        }
    }

    // 6. 示例 2: 批量推理
    println!("\n{}", "=".repeat(50));
    println!("📝 示例 2: 批量语音转换");
    println!("{}", "=".repeat(50));

    let batch_inference = BatchInference::new(inference_engine);
    let input_files = vec![
        PathBuf::from("test.wav"),
        PathBuf::from("test2.wav"),
        // 可以添加更多文件
    ];
    let output_dir = PathBuf::from("output_batch");

    // 创建输出目录
    if !output_dir.exists() {
        std::fs::create_dir_all(&output_dir)?;
        println!("📁 创建输出目录: {:?}", output_dir);
    }

    // 过滤存在的文件
    let existing_files: Vec<_> = input_files
        .into_iter()
        .filter(|path| {
            let exists = path.exists();
            if !exists {
                println!("⚠️  跳过不存在的文件: {:?}", path);
            }
            exists
        })
        .collect();

    if !existing_files.is_empty() {
        println!("🎵 开始批量处理 {} 个文件...", existing_files.len());

        match batch_inference.process_batch(&existing_files, &output_dir) {
            Ok(results) => {
                println!("✅ 批量处理成功!");
                println!("   - 处理文件数: {}", results.len());
                println!("   - 输出目录: {:?}", output_dir);

                for (i, result) in results.iter().enumerate() {
                    println!(
                        "   - 文件 {}: {:.2}s, {}Hz",
                        i + 1,
                        result.data.len() as f32 / result.sample_rate,
                        result.sample_rate
                    );
                }
            }
            Err(e) => {
                println!("❌ 批量处理失败: {}", e);
            }
        }
    } else {
        println!("⚠️  没有找到有效的输入文件进行批量处理");
        println!("💡 请确保至少有一个输入文件存在");
    }

    // 7. 性能提示
    println!("\n{}", "=".repeat(50));
    println!("💡 性能优化提示");
    println!("{}", "=".repeat(50));
    println!("1. 使用 GPU 加速 (如果可用):");
    println!("   config.device = Device::Cuda(0);");
    println!();
    println!("2. 调整批处理大小:");
    println!("   config.batch_size = 4; // 根据显存调整");
    println!();
    println!("3. F0 方法选择:");
    println!("   - Harvest: 质量高，速度慢");
    println!("   - PM: 速度快，质量中等");
    println!("   - RMVPE: 平衡质量和速度 (推荐)");
    println!();
    println!("4. 索引使用:");
    println!("   - index_rate = 0.0: 不使用索引，保持原始音色");
    println!("   - index_rate = 1.0: 完全使用索引，最大音色转换");
    println!("   - index_rate = 0.5-0.8: 推荐范围");

    // 8. 故障排除
    println!("\n{}", "=".repeat(50));
    println!("🔧 常见问题排除");
    println!("{}", "=".repeat(50));
    println!("1. 模型加载失败:");
    println!("   - 检查模型文件路径是否正确");
    println!("   - 确保模型文件格式兼容");
    println!();
    println!("2. 内存不足:");
    println!("   - 减小 batch_size");
    println!("   - 使用更小的模型");
    println!("   - 分段处理长音频");
    println!();
    println!("3. 音质问题:");
    println!("   - 调整 F0 滤波参数");
    println!("   - 尝试不同的 F0 估计方法");
    println!("   - 调整 index_rate 参数");
    println!();
    println!("4. 速度慢:");
    println!("   - 使用 GPU 加速");
    println!("   - 选择更快的 F0 方法");
    println!("   - 降低音频采样率");

    println!("\n🎉 示例运行完成!");
    println!(
        "更多信息请参考项目文档: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI"
    );

    Ok(())
}

/// 辅助函数：检查系统是否支持 CUDA
fn _check_cuda_available() -> bool {
    match tch::Cuda::is_available() {
        true => {
            println!("✅ CUDA 可用，设备数量: {}", tch::Cuda::device_count());
            true
        }
        false => {
            println!("❌ CUDA 不可用，将使用 CPU");
            false
        }
    }
}

/// 辅助函数：创建推荐的配置
fn _create_recommended_config() -> InferenceConfig {
    InferenceConfig {
        speaker_id: 0,
        f0_method: F0Method::RMVPE, // 推荐方法
        pitch_shift: 1.0,
        index_rate: 0.75,
        target_sample_rate: 22050,
        device: if tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        },
        batch_size: if tch::Cuda::is_available() { 4 } else { 1 },
        enable_denoise: true,
        f0_filter: F0FilterConfig {
            median_filter_radius: 3,
            enable_smoothing: true,
            smoothing_factor: 0.8,
        },
    }
}

/// 辅助函数：验证输入文件
fn _validate_input_file(path: &PathBuf) -> Result<(), String> {
    if !path.exists() {
        return Err(format!("文件不存在: {:?}", path));
    }

    if let Some(ext) = path.extension() {
        let ext_str = ext.to_string_lossy().to_lowercase();
        if !["wav", "mp3", "flac", "ogg"].contains(&ext_str.as_str()) {
            return Err(format!("不支持的音频格式: {}", ext_str));
        }
    } else {
        return Err("文件没有扩展名".to_string());
    }

    Ok(())
}

/// 辅助函数：格式化文件大小
fn _format_file_size(size_bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB"];
    let mut size = size_bytes as f64;
    let mut unit_index = 0;

    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }

    format!("{:.1} {}", size, UNITS[unit_index])
}
