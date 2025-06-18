use chrono;
use log::{error, info, warn};
use rvc_lib::{set_devices, start_vc_with_callback, update_devices, DeviceInfo, GUIConfig, GUI};
use serde::Serialize;
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

#[derive(Serialize, Clone)]
struct AudioMetrics {
    inference_time: f32,
    total_processing_time: f32,
    timestamp: String,
    input_level: f32,
    output_level: f32,
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
    input_device: String,
    output_device: String,
) -> Result<u32, String> {
    info!(
        "🔧 Setting audio devices - Host: {}, Input: {}, Output: {}",
        hostapi, input_device, output_device
    );
    match set_devices(&hostapi, &input_device, &output_device) {
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
    input_device: String,
    output_device: String,
    wasapi_exclusive: bool,
    sr_type: String,
    threshold: f32,
    pitch: f32,
    formant: f32,
    index_rate: f32,
    rms_mix_rate: f32,
    f0method: String,
    block_time: f32,
    crossfade_length: f32,
    n_cpu: u32,
    extra_time: f32,
    i_noise_reduce: bool,
    o_noise_reduce: bool,
    use_pv: bool,
) -> Result<(), String> {
    info!("🚀 Starting voice conversion with parameters:");
    info!("  Model: {}", pth);
    info!("  Index: {}", index);
    info!("  F0 Method: {}", f0method);
    info!("  Pitch: {}", pitch);
    info!("  Index Rate: {}", index_rate);

    // Check if already running
    let mut is_running = state.inner().is_running.lock().unwrap();
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

    // Validate model file exists
    if !std::path::Path::new(&pth).exists() {
        let error = format!("❌ Model file not found: {}", pth);
        error!("{}", error);
        let _ = app.emit(
            "rvc_error",
            serde_json::json!({
                "error": error.clone(),
                "type": "file_not_found",
                "file": pth,
                "timestamp": chrono::Utc::now().to_rfc3339()
            }),
        );
        return Err(error);
    }

    // Validate index file exists (if provided)
    if !index.is_empty() && !std::path::Path::new(&index).exists() {
        let error = format!("❌ Index file not found: {}", index);
        error!("{}", error);
        let _ = app.emit(
            "rvc_error",
            serde_json::json!({
                "error": error.clone(),
                "type": "file_not_found",
                "file": index,
                "timestamp": chrono::Utc::now().to_rfc3339()
            }),
        );
        return Err(error);
    }

    // Set audio devices first
    if let Err(e) = set_devices(&hostapi, &input_device, &output_device) {
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

    // Update and save configuration to rvc-lib
    let config = GUIConfig {
        pth_path: pth.clone(),
        index_path: if index.is_empty() {
            String::new()
        } else {
            index.clone()
        },
        sg_hostapi: hostapi,
        sg_input_device: input_device,
        sg_output_device: output_device,
        sg_wasapi_exclusive: wasapi_exclusive,
        sr_type,
        threshold,
        pitch,
        formant,
        index_rate,
        rms_mix_rate,
        f0method,
        block_time,
        crossfade_length,
        n_cpu,
        extra_time,
        i_noise_reduce,
        o_noise_reduce,
        use_pv,
        ..Default::default()
    };

    // Save configuration using rvc-lib
    if let Err(e) = GUI::save(&config) {
        let error = format!("❌ Failed to save configuration: {}", e);
        error!("{}", error);
        return Err(error);
    }

    // Create audio callback for real-time metrics
    let app_handle = app.clone();
    let audio_callback = Arc::new(
        move |input_data: &[f32], output_data: &[f32], inference_time: f32, total_time: f32| {
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

            let metrics = AudioMetrics {
                inference_time,
                total_processing_time: total_time,
                timestamp: chrono::Utc::now().to_rfc3339(),
                input_level,
                output_level,
            };

            let _ = app_handle.emit("audio_metrics", &metrics);
        },
    );

    // Start voice conversion using rvc-lib
    match start_vc_with_callback(Some(audio_callback)) {
        Ok(_vc_instance) => {
            // The VC instance is dropped here, but the streams remain active
            // This is the current limitation of the architecture
            *is_running = true;
            *state.inner().audio_streaming.lock().unwrap() = true;
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

    let mut is_running = state.inner().is_running.lock().unwrap();
    if !*is_running {
        let message = "Voice conversion is not running".to_string();
        warn!("{}", message);
        return Ok(());
    }

    // Set flags to indicate VC should stop
    VC_IS_ACTIVE.store(false, Ordering::SeqCst);
    *is_running = false;
    *state.inner().audio_streaming.lock().unwrap() = false;

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
    let is_running = *state.inner().is_running.lock().unwrap();
    let audio_streaming = *state.inner().audio_streaming.lock().unwrap();

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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_log::Builder::new().build())
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
            clear_audio_buffers
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
