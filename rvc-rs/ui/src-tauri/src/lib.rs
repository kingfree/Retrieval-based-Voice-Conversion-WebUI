use chrono;
use log::{error, info, warn};
use rvc_lib::{set_devices, start_vc_with_callback, update_devices, DeviceInfo, GUIConfig, GUI};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, State};

// Global flag to indicate if real VC is running
static VC_IS_ACTIVE: AtomicBool = AtomicBool::new(false);

// Simple state without storing the VC instance directly
#[derive(Default)]
pub struct VcState {
    pub is_running: Arc<Mutex<bool>>,
    pub audio_streaming: Arc<Mutex<bool>>,
}

#[derive(Serialize, Clone, Debug)]
struct AudioMetrics {
    inference_time: f32,
    total_processing_time: f32,
    timestamp: String,
    input_level: f32,
    output_level: f32,
}

#[derive(Serialize, Clone)]
struct AudioData {
    input_data: Vec<f32>,
    output_data: Vec<f32>,
    sample_rate: u32,
    timestamp: String,
}

// Helper function to resolve paths relative to project root
fn resolve_path(path: &str) -> PathBuf {
    if path.is_empty() {
        return PathBuf::new();
    }

    let path_buf = PathBuf::from(path);

    // If it's already an absolute path, return as-is
    if path_buf.is_absolute() {
        return path_buf;
    }

    // Try to resolve relative to the project root
    // First, get the current executable directory and work backwards to find project root
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            // In development, the executable is in target/debug or target/release
            // In production, it might be in the app bundle
            let mut current = exe_dir;

            // Look for project markers (assets, configs directories)
            for _ in 0..5 {
                // Limit search depth
                let assets_path = current.join("assets");
                let configs_path = current.join("configs");

                if assets_path.exists() && configs_path.exists() {
                    return current.join(path);
                }

                if let Some(parent) = current.parent() {
                    current = parent;
                } else {
                    break;
                }
            }
        }
    }

    // Fallback: try relative to current working directory
    if let Ok(cwd) = std::env::current_dir() {
        let resolved = cwd.join(path);
        if resolved.exists() {
            return resolved;
        }
    }

    // Last resort: return the original path
    path_buf
}

#[tauri::command]
async fn event_handler() -> Result<String, String> {
    Ok("Event handler ready".to_string())
}

#[tauri::command]
async fn frontend_event(event: String, data: serde_json::Value) -> Result<(), String> {
    info!("Frontend event: {} with data: {:?}", event, data);
    Ok(())
}

#[tauri::command]
async fn get_init_config() -> Result<GUIConfig, String> {
    info!("📋 Loading initial configuration...");
    match GUI::load() {
        Ok(config) => {
            info!("✅ Configuration loaded successfully");
            Ok(config)
        }
        Err(e) => {
            let error = format!("❌ Failed to load configuration: {}", e);
            error!("{}", error);
            Err(error)
        }
    }
}

#[tauri::command]
async fn save_config(config: GUIConfig) -> Result<(), String> {
    info!("💾 Saving configuration...");
    match GUI::save(&config) {
        Ok(()) => {
            info!("✅ Configuration saved successfully");
            Ok(())
        }
        Err(e) => {
            let error = format!("❌ Failed to save configuration: {}", e);
            error!("{}", error);
            Err(error)
        }
    }
}

#[tauri::command]
async fn get_device_info(hostapi: Option<String>) -> Result<DeviceInfo, String> {
    info!("🎵 Enumerating audio devices for hostapi: {:?}", hostapi);
    match update_devices(hostapi.as_deref()) {
        Ok(device_info) => {
            info!(
                "✅ Found {} host APIs, {} input devices, {} output devices",
                device_info.hostapis.len(),
                device_info.input_devices.len(),
                device_info.output_devices.len()
            );
            Ok(device_info)
        }
        Err(e) => {
            let error = format!("❌ Failed to enumerate devices: {}", e);
            error!("{}", error);
            Err(error)
        }
    }
}

#[tauri::command]
async fn set_audio_devices(
    hostapi: String,
    inputDevice: String,
    outputDevice: String,
) -> Result<u32, String> {
    info!(
        "🔧 Setting audio devices - Host: {}, Input: {}, Output: {}",
        hostapi, inputDevice, outputDevice
    );
    match set_devices(&hostapi, &inputDevice, &outputDevice) {
        Ok(sample_rate) => {
            info!(
                "✅ Audio devices set successfully, sample rate: {} Hz",
                sample_rate
            );
            Ok(sample_rate)
        }
        Err(e) => {
            let error = format!("❌ Failed to set audio devices: {}", e);
            error!("{}", error);
            Err(error)
        }
    }
}

#[tauri::command]
async fn start_voice_conversion(
    app: AppHandle,
    state: State<'_, VcState>,
    pth: String,
    index: String,
    hostapi: String,
    inputDevice: String,
    outputDevice: String,
    wasapiExclusive: bool,
    srType: String,
    threshold: f32,
    pitch: f32,
    formant: f32,
    indexRate: f32,
    rmsMixRate: f32,
    f0method: String,
    blockTime: f32,
    crossfadeLength: f32,
    nCpu: u32,
    extraTime: f32,
    iNoiseReduce: bool,
    oNoiseReduce: bool,
    usePv: bool,
) -> Result<(), String> {
    info!("🚀 Starting voice conversion with parameters:");
    info!("  Model: {}", pth);
    info!("  Index: {}", index);
    info!("  F0 Method: {}", f0method);
    info!("  Pitch: {}", pitch);
    info!("  Index Rate: {}", indexRate);

    // Check if already running
    let mut is_running = match state.inner().is_running.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            warn!("Mutex was poisoned, recovering...");
            poisoned.into_inner()
        }
    };
    if *is_running {
        let error = "Voice conversion is already running".to_string();
        warn!("{}", error);
        let _ = app.emit(
            "rvc_error",
            serde_json::json!({
                "error": error.clone(),
                "type": "already_running",
                "timestamp": chrono::Utc::now().to_rfc3339()
            }),
        );
        return Err(error);
    }

    // Resolve and validate model file exists
    let resolved_pth = resolve_path(&pth);
    if !resolved_pth.exists() {
        let error = format!(
            "❌ Model file not found: {} (resolved to: {})",
            pth,
            resolved_pth.display()
        );
        error!("{}", error);
        let _ = app.emit(
            "rvc_error",
            serde_json::json!({
                "error": error.clone(),
                "type": "file_not_found",
                "file": pth,
                "resolved_path": resolved_pth.to_string_lossy(),
                "timestamp": chrono::Utc::now().to_rfc3339()
            }),
        );
        return Err(error);
    }

    // Resolve and validate index file exists (if provided)
    let resolved_index = if !index.is_empty() {
        let resolved = resolve_path(&index);
        if !resolved.exists() {
            let error = format!(
                "❌ Index file not found: {} (resolved to: {})",
                index,
                resolved.display()
            );
            error!("{}", error);
            let _ = app.emit(
                "rvc_error",
                serde_json::json!({
                    "error": error.clone(),
                    "type": "file_not_found",
                    "file": index,
                    "resolved_path": resolved.to_string_lossy(),
                    "timestamp": chrono::Utc::now().to_rfc3339()
                }),
            );
            return Err(error);
        }
        resolved.to_string_lossy().to_string()
    } else {
        String::new()
    };

    // Set audio devices first
    if let Err(e) = set_devices(&hostapi, &inputDevice, &outputDevice) {
        let error = format!("❌ Failed to set audio devices: {}", e);
        error!("{}", error);
        let _ = app.emit(
            "rvc_error",
            serde_json::json!({
                "error": error.clone(),
                "type": "device_error",
                "timestamp": chrono::Utc::now().to_rfc3339()
            }),
        );
        return Err(error);
    }

    // Update and save configuration to rvc-lib with resolved paths
    let config = GUIConfig {
        pth_path: resolved_pth.to_string_lossy().to_string(),
        index_path: resolved_index,
        sg_hostapi: hostapi,
        sg_input_device: inputDevice,
        sg_output_device: outputDevice,
        sg_wasapi_exclusive: wasapiExclusive,
        sr_type: srType,
        threshold,
        pitch,
        formant,
        index_rate: indexRate,
        rms_mix_rate: rmsMixRate,
        f0method,
        block_time: blockTime,
        crossfade_length: crossfadeLength,
        n_cpu: nCpu,
        extra_time: extraTime,
        i_noise_reduce: iNoiseReduce,
        o_noise_reduce: oNoiseReduce,
        use_pv: usePv,
        ..Default::default()
    };

    // Save configuration using rvc-lib
    if let Err(e) = GUI::save(&config) {
        let error = format!("❌ Failed to save configuration: {}", e);
        error!("{}", error);
        return Err(error);
    }

    // Create audio callback for real-time metrics and data streaming
    let app_handle = app.clone();
    let audio_callback = Arc::new(
        move |input_data: &[f32], output_data: &[f32], inference_time: f32, total_time: f32| {
            println!("🎤 Audio callback triggered! Input: {} samples, Output: {} samples, Inference: {:.2}ms, Total: {:.2}ms",
                input_data.len(), output_data.len(), inference_time, total_time);

            // Debug: Print first few samples of input data
            if !input_data.is_empty() {
                let preview_count = std::cmp::min(5, input_data.len());
                println!(
                    "🔍 Input data preview (first {} samples): {:?}",
                    preview_count,
                    &input_data[..preview_count]
                );
            }

            // Debug: Print first few samples of output data
            if !output_data.is_empty() {
                let preview_count = std::cmp::min(5, output_data.len());
                println!(
                    "🔍 Output data preview (first {} samples): {:?}",
                    preview_count,
                    &output_data[..preview_count]
                );
            }

            // Calculate audio levels
            let input_level = if !input_data.is_empty() {
                (input_data.iter().map(|x| x * x).sum::<f32>() / input_data.len() as f32).sqrt()
            } else {
                0.0
            };
            let output_level = if !output_data.is_empty() {
                (output_data.iter().map(|x| x * x).sum::<f32>() / output_data.len() as f32).sqrt()
            } else {
                0.0
            };

            println!(
                "📊 Audio levels - Input: {:.4}, Output: {:.4}",
                input_level, output_level
            );

            let timestamp = chrono::Utc::now().to_rfc3339();

            // Send audio metrics
            let metrics = AudioMetrics {
                inference_time,
                total_processing_time: total_time,
                timestamp: timestamp.clone(),
                input_level,
                output_level,
            };

            println!("📈 Emitting audio_metrics: {:?}", metrics);
            match app_handle.emit("audio_metrics", &metrics) {
                Ok(_) => println!("✅ Successfully emitted audio_metrics event"),
                Err(e) => println!("❌ Failed to emit audio_metrics: {}", e),
            }

            // Send audio data for visualization (limit data size for performance)
            const MAX_SAMPLES: usize = 1024; // Limit to prevent UI lag
            let input_samples: Vec<f32> = input_data.iter().take(MAX_SAMPLES).copied().collect();
            let output_samples: Vec<f32> = output_data.iter().take(MAX_SAMPLES).copied().collect();

            let audio_data = AudioData {
                input_data: input_samples.clone(),
                output_data: output_samples.clone(),
                sample_rate: 22050, // Default sample rate, could be made configurable
                timestamp,
            };

            println!(
                "🎵 Emitting audio_data: input_len={}, output_len={}, sample_rate={}",
                input_samples.len(),
                output_samples.len(),
                audio_data.sample_rate
            );
            match app_handle.emit("audio_data", &audio_data) {
                Ok(_) => println!("✅ Successfully emitted audio_data event"),
                Err(e) => println!("❌ Failed to emit audio_data: {}", e),
            }
        },
    );

    // Start voice conversion using rvc-lib
    match start_vc_with_callback(Some(audio_callback)) {
        Ok(_vc_instance) => {
            // The VC instance is dropped here, but the streams remain active
            // This is the current limitation of the architecture
            *is_running = true;
            match state.inner().audio_streaming.lock() {
                Ok(mut guard) => *guard = true,
                Err(poisoned) => {
                    warn!("Audio streaming mutex was poisoned, recovering...");
                    *poisoned.into_inner() = true;
                }
            }
            VC_IS_ACTIVE.store(true, Ordering::SeqCst);

            info!("✅ Voice conversion started successfully");

            // Emit success event to frontend
            let _ = app.emit(
                "rvc_status",
                serde_json::json!({
                    "status": "started",
                    "message": "Voice conversion started successfully",
                    "timestamp": chrono::Utc::now().to_rfc3339()
                }),
            );

            Ok(())
        }
        Err(e) => {
            let error = format!("❌ Failed to start voice conversion: {}", e);
            error!("{}", error);

            // Emit error event to frontend
            let _ = app.emit(
                "rvc_error",
                serde_json::json!({
                    "error": error.clone(),
                    "timestamp": chrono::Utc::now().to_rfc3339()
                }),
            );

            Err(error)
        }
    }
}

#[tauri::command]
async fn stop_voice_conversion(app: AppHandle, state: State<'_, VcState>) -> Result<(), String> {
    info!("🛑 Stopping voice conversion...");

    let mut is_running = match state.inner().is_running.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            warn!("Mutex was poisoned during stop, recovering...");
            poisoned.into_inner()
        }
    };

    if !*is_running {
        let message = "Voice conversion is not running".to_string();
        warn!("{}", message);
        return Ok(());
    }

    // Set flags to indicate VC should stop
    VC_IS_ACTIVE.store(false, Ordering::SeqCst);
    *is_running = false;

    match state.inner().audio_streaming.lock() {
        Ok(mut guard) => *guard = false,
        Err(poisoned) => {
            warn!("Audio streaming mutex was poisoned during stop, recovering...");
            *poisoned.into_inner() = false;
        }
    }

    info!("✅ Voice conversion stopped successfully");

    // Emit stop event to frontend
    let _ = app.emit(
        "rvc_status",
        serde_json::json!({
            "status": "stopped",
            "message": "Voice conversion stopped successfully",
            "timestamp": chrono::Utc::now().to_rfc3339()
        }),
    );

    Ok(())
}

#[tauri::command]
async fn get_vc_status(state: State<'_, VcState>) -> Result<serde_json::Value, String> {
    let is_running = match state.inner().is_running.lock() {
        Ok(guard) => *guard,
        Err(poisoned) => {
            warn!("Mutex was poisoned during status check, recovering...");
            *poisoned.into_inner()
        }
    };

    let audio_streaming = match state.inner().audio_streaming.lock() {
        Ok(guard) => *guard,
        Err(poisoned) => {
            warn!("Audio streaming mutex was poisoned during status check, recovering...");
            *poisoned.into_inner()
        }
    };

    Ok(serde_json::json!({
        "is_running": is_running,
        "audio_streaming": audio_streaming,
        "vc_active": VC_IS_ACTIVE.load(Ordering::SeqCst),
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

#[tauri::command]
async fn clear_audio_buffers() -> Result<(), String> {
    info!("🧹 Clearing audio buffers...");
    // Note: The actual buffer clearing would be handled by rvc-lib if such functionality exists
    // For now, this is a placeholder
    Ok(())
}

#[tauri::command]
async fn update_rms_mix_rate(rate: f32) -> Result<(), String> {
    info!("🎵 Updating RMS mix rate to: {}", rate);
    // Note: This would need access to the active VC instance
    // For now, this updates the configuration
    let mut config = GUI::load().map_err(|e| e.to_string())?;
    config.rms_mix_rate = rate.clamp(0.0, 1.0);
    GUI::save(&config).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
async fn update_noise_reduce(input: bool, output: bool) -> Result<(), String> {
    info!(
        "🔇 Updating noise reduction - Input: {}, Output: {}",
        input, output
    );
    let mut config = GUI::load().map_err(|e| e.to_string())?;
    config.i_noise_reduce = input;
    config.o_noise_reduce = output;
    GUI::save(&config).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
async fn update_use_pv(use_pv: bool) -> Result<(), String> {
    info!("🔄 Updating phase vocoder usage to: {}", use_pv);
    let mut config = GUI::load().map_err(|e| e.to_string())?;
    config.use_pv = use_pv;
    GUI::save(&config).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
async fn update_threshold(threshold: f32) -> Result<(), String> {
    info!("📊 Updating audio threshold to: {} dB", threshold);
    let mut config = GUI::load().map_err(|e| e.to_string())?;
    config.threshold = threshold;
    GUI::save(&config).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
async fn update_f0_method(method: String) -> Result<(), String> {
    info!("🎼 Updating F0 method to: {}", method);
    let mut config = GUI::load().map_err(|e| e.to_string())?;
    config.f0method = method;
    GUI::save(&config).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_log::Builder::new().build())
        .plugin(tauri_plugin_dialog::init())
        .manage(VcState::default())
        .invoke_handler(tauri::generate_handler![
            event_handler,
            frontend_event,
            get_init_config,
            save_config,
            get_device_info,
            set_audio_devices,
            start_voice_conversion,
            stop_voice_conversion,
            get_vc_status,
            clear_audio_buffers,
            update_rms_mix_rate,
            update_noise_reduce,
            update_use_pv,
            update_threshold,
            update_f0_method
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
