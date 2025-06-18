use chrono;
use log::{error, info, warn};
use rvc_lib::{start_vc, start_vc_with_callback, AudioDataCallback, DeviceInfo, GUIConfig, GUI};
use serde::Serialize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, State};

// Global flag to indicate if real VC is running
static VC_IS_ACTIVE: AtomicBool = AtomicBool::new(false);

#[derive(Serialize, Clone)]
struct AudioData {
    samples: Vec<f32>,
    sample_rate: u32,
    timestamp: u64,
}

#[derive(Serialize, Clone)]
struct AudioStats {
    input_sample_rate: u32,
    output_sample_rate: u32,
    buffer_size: usize,
    processed_samples: u64,
    dropped_frames: u32,
    latency: f32,
}

#[derive(Serialize, Clone)]
struct PerformanceMetrics {
    inference_time_ms: f32,
    algorithm_latency_ms: f32,
    timestamp: u64,
    buffer_latency_ms: f32,
}

// Global state for voice conversion
/// 存储配置、运行状态，不存储VC句柄（VC不是Send/Sync）
struct VcState {
    config: Arc<Mutex<GUIConfig>>,
    is_running: Arc<Mutex<bool>>,
    delay_time: Arc<Mutex<f32>>,
    function_mode: Arc<Mutex<String>>,
    audio_streaming: Arc<Mutex<bool>>,
    input_buffer: Arc<Mutex<Vec<f32>>>,
    output_buffer: Arc<Mutex<Vec<f32>>>,
    stats: Arc<Mutex<AudioStats>>,
}

impl Default for VcState {
    fn default() -> Self {
        Self {
            config: Arc::new(Mutex::new(GUIConfig::default())),
            is_running: Arc::new(Mutex::new(false)),
            delay_time: Arc::new(Mutex::new(0.0)),
            function_mode: Arc::new(Mutex::new("vc".into())),
            audio_streaming: Arc::new(Mutex::new(false)),
            input_buffer: Arc::new(Mutex::new(Vec::new())),
            output_buffer: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(Mutex::new(AudioStats {
                input_sample_rate: 44100,
                output_sample_rate: 44100,
                buffer_size: 2048,
                processed_samples: 0,
                dropped_frames: 0,
                latency: 0.0,
            })),
        }
    }
}

/// Comprehensive event handler that mimics Python's event_handler functionality
#[tauri::command]
async fn event_handler(
    app: AppHandle,
    state: State<'_, VcState>,
    event: String,
    value: Option<String>,
) -> Result<(), String> {
    info!("📨 Event received: {} = {:?}", event, value);

    let mut config = state.inner().config.lock().unwrap();
    let mut is_running = state.inner().is_running.lock().unwrap();

    match event.as_str() {
        // Device management events
        "reload_devices" | "sg_hostapi" => {
            if let Some(hostapi) = value {
                config.sg_hostapi = hostapi.clone();
                let device_info =
                    rvc_lib::update_devices(Some(&hostapi)).map_err(|e| e.to_string())?;

                // Update devices if current selections are no longer available
                if !device_info.input_devices.contains(&config.sg_input_device)
                    && !device_info.input_devices.is_empty()
                {
                    config.sg_input_device = device_info.input_devices[0].clone();
                }

                if !device_info
                    .output_devices
                    .contains(&config.sg_output_device)
                    && !device_info.output_devices.is_empty()
                {
                    config.sg_output_device = device_info.output_devices[0].clone();
                }

                // Emit updated device lists to frontend
                app.emit("devices_updated", &device_info).unwrap();
            }
        }

        // Voice conversion control
        "start_vc" => {
            if !*is_running {
                info!("Starting voice conversion...");

                // Save current configuration
                GUI::save(&config).map_err(|e| e.to_string())?;

                // Start audio data streaming
                *state.inner().audio_streaming.lock().unwrap() = true;

                // Create audio data callback for real-time visualization
                let app_for_callback = app.clone();
                let input_buffer_callback = state.inner().input_buffer.clone();
                let output_buffer_callback = state.inner().output_buffer.clone();
                let stats_callback = state.inner().stats.clone();

                let audio_callback: AudioDataCallback = Arc::new(
                    move |input_data: &[f32],
                          output_data: &[f32],
                          inference_time_ms: f32,
                          algorithm_latency_ms: f32| {
                        let timestamp = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_millis() as u64;

                        // Update input buffer
                        {
                            let mut input_buf = input_buffer_callback.lock().unwrap();
                            input_buf.extend_from_slice(input_data);
                            if input_buf.len() > 8192 {
                                let excess = input_buf.len() - 8192;
                                input_buf.drain(0..excess);
                            }
                        }

                        // Update output buffer
                        {
                            let mut output_buf = output_buffer_callback.lock().unwrap();
                            output_buf.extend_from_slice(output_data);
                            if output_buf.len() > 8192 {
                                let excess = output_buf.len() - 8192;
                                output_buf.drain(0..excess);
                            }
                        }

                        // Calculate buffer latency
                        let buffer_latency_ms = (input_data.len() as f32 / 44100.0) * 1000.0;

                        // Update stats
                        {
                            let mut stats_guard = stats_callback.lock().unwrap();
                            stats_guard.processed_samples += input_data.len() as u64;
                            stats_guard.latency = algorithm_latency_ms;
                        }

                        // Emit real-time audio data to frontend
                        let input_audio_data = AudioData {
                            samples: input_data.to_vec(),
                            sample_rate: 44100,
                            timestamp,
                        };

                        let output_audio_data = AudioData {
                            samples: output_data.to_vec(),
                            sample_rate: 44100,
                            timestamp,
                        };

                        let current_stats = stats_callback.lock().unwrap().clone();

                        // Create performance metrics
                        let performance_metrics = PerformanceMetrics {
                            inference_time_ms,
                            algorithm_latency_ms,
                            buffer_latency_ms,
                            timestamp,
                        };

                        // Emit events (ignore errors as they're not critical)
                        let _ = app_for_callback.emit("input_audio_data", &input_audio_data);
                        let _ = app_for_callback.emit("output_audio_data", &output_audio_data);
                        let _ = app_for_callback.emit("audio_stats", &current_stats);
                        let _ = app_for_callback.emit("performance_metrics", &performance_metrics);
                    },
                );

                // Start voice conversion with callback
                match start_vc_with_callback(Some(audio_callback)) {
                    Ok(_vc_handle) => {
                        // Set global flag that VC is active
                        VC_IS_ACTIVE.store(true, Ordering::SeqCst);

                        // Note: _vc_handle stays in scope and will be dropped when this function exits
                        // For now, we rely on the callback mechanism to provide audio data
                        // The VC will remain active until the process ends or until we implement
                        // a proper cleanup mechanism

                        *is_running = true;

                        // Calculate initial delay time like Python implementation
                        let delay = config.block_time
                            + config.crossfade_length
                            + 0.01
                            + if config.i_noise_reduce {
                                config.crossfade_length.min(0.04)
                            } else {
                                0.0
                            };
                        *state.inner().delay_time.lock().unwrap() = delay;
                        app.emit("delay_time", &((delay * 1000.0).round() as u32))
                            .unwrap();

                        app.emit("vc_started", ()).unwrap();
                        app.emit("audio_stream_started", ()).unwrap();
                        info!("Voice conversion started successfully");
                    }
                    Err(e) => {
                        *state.inner().audio_streaming.lock().unwrap() = false;
                        return Err(format!("Failed to start voice conversion: {}", e));
                    }
                }
            }
        }

        "stop_vc" => {
            if *is_running {
                info!("Stopping voice conversion...");

                // Signal that VC should stop
                VC_IS_ACTIVE.store(false, Ordering::SeqCst);

                // Stop audio streaming
                *state.inner().audio_streaming.lock().unwrap() = false;

                // Clear buffers
                state.inner().input_buffer.lock().unwrap().clear();
                state.inner().output_buffer.lock().unwrap().clear();

                // Update state immediately
                *is_running = false;
                app.emit("vc_stopped", ()).unwrap();
                app.emit("audio_stream_stopped", ()).unwrap();

                info!("Voice conversion stopped");

                // Note: The VC handle will continue to exist until the application restarts
                // This is a limitation of the current architecture where CPAL streams
                // are not easily stoppable once started
            }
        }

        // Hot parameter updates (these can be changed while VC is running)
        "threshold" => {
            if let Some(val) = value {
                if let Ok(threshold) = val.parse::<f32>() {
                    config.threshold = threshold;
                }
            }
        }

        "pitch" => {
            if let Some(val) = value {
                if let Ok(pitch) = val.parse::<f32>() {
                    config.pitch = pitch;
                    // 实时参数变更暂时无法直接作用于 VC 实例
                }
            }
        }

        "formant" => {
            if let Some(val) = value {
                if let Ok(formant) = val.parse::<f32>() {
                    config.formant = formant;
                    // 实时参数变更暂时无法直接作用于 VC 实例
                }
            }
        }

        "index_rate" => {
            if let Some(val) = value {
                if let Ok(rate) = val.parse::<f32>() {
                    config.index_rate = rate;
                    // 实时参数变更暂时无法直接作用于 VC 实例
                }
            }
        }

        "rms_mix_rate" => {
            if let Some(val) = value {
                if let Ok(rate) = val.parse::<f32>() {
                    config.rms_mix_rate = rate;
                }
            }
        }

        // F0 method selection
        "pm" | "harvest" | "crepe" | "rmvpe" | "fcpe" => {
            config.f0method = event.clone();
        }

        // Noise reduction toggles
        "I_noise_reduce" => {
            if let Some(val) = value {
                let flag = val.parse().unwrap_or(false);
                let delta = if flag {
                    config.crossfade_length.min(0.04)
                } else {
                    -config.crossfade_length.min(0.04)
                };
                config.i_noise_reduce = flag;
                if *is_running {
                    let mut d = state.inner().delay_time.lock().unwrap();
                    *d += delta;
                    app.emit("delay_time", &((*d * 1000.0).round() as u32))
                        .unwrap();
                }
            }
        }

        "O_noise_reduce" => {
            if let Some(val) = value {
                config.o_noise_reduce = val.parse().unwrap_or(false);
            }
        }

        "use_pv" => {
            if let Some(val) = value {
                config.use_pv = val.parse().unwrap_or(false);
            }
        }

        // Function mode
        "function_mode" => {
            if let Some(mode) = value {
                *state.inner().function_mode.lock().unwrap() = mode.clone();
                info!("Function mode changed to: {}", mode);
            }
        }

        // Device selection
        "sg_input_device" => {
            if let Some(device) = value {
                config.sg_input_device = device;
            }
        }

        "sg_output_device" => {
            if let Some(device) = value {
                config.sg_output_device = device;
            }
        }

        "sg_wasapi_exclusive" => {
            if let Some(flag) = value {
                config.sg_wasapi_exclusive = flag.parse().unwrap_or(false);
            }
        }

        "pth_path" => {
            if let Some(path) = value {
                config.pth_path = path;
            }
        }

        "index_path" => {
            if let Some(path) = value {
                config.index_path = path;
            }
        }

        // Parameters that require restart (stop current VC)
        "block_time" | "crossfade_length" | "extra_time" | "n_cpu" | "sr_type" => {
            if *is_running {
                // Stop current VC session as these parameters require restart
                // Note: VC handle is dropped automatically when it goes out of scope
                *is_running = false;
                app.emit("vc_stopped", ()).unwrap();
                info!(
                    "Voice conversion stopped due to parameter change: {}",
                    event
                );
            }

            // Update the parameter
            match event.as_str() {
                "block_time" => {
                    if let Some(val) = value {
                        if let Ok(time) = val.parse::<f32>() {
                            config.block_time = time;
                        }
                    }
                }
                "crossfade_length" => {
                    if let Some(val) = value {
                        if let Ok(length) = val.parse::<f32>() {
                            config.crossfade_length = length;
                        }
                    }
                }
                "extra_time" => {
                    if let Some(val) = value {
                        if let Ok(time) = val.parse::<f32>() {
                            config.extra_time = time;
                        }
                    }
                }
                "n_cpu" => {
                    if let Some(val) = value {
                        if let Ok(cpu) = val.parse::<u32>() {
                            config.n_cpu = cpu;
                        }
                    }
                }
                "sr_type" => {
                    if let Some(sr_type) = value {
                        config.sr_type = sr_type;
                    }
                }
                _ => {}
            }
        }

        _ => {
            info!("Unhandled event: {}", event);
        }
    }

    Ok(())
}

/// Print events coming from the front-end (legacy compatibility).
#[tauri::command]
fn frontend_event(event: String, value: Option<String>) {
    match value {
        Some(v) => info!("frontend event: {} = {}", event, v),
        None => info!("frontend event: {}", event),
    }
}

#[tauri::command]
fn get_init_config() -> Result<GUIConfig, String> {
    GUI::load().map_err(|e| e.to_string())
}

#[tauri::command]
fn set_values(values: GUIConfig) -> Result<(), String> {
    GUI::save(&values).map_err(|e| e.to_string())
}

#[tauri::command]
fn save_config(config: serde_json::Value) -> Result<(), String> {
    use std::fs;
    use std::path::Path;

    // Create config directory if it doesn't exist
    let config_dir = Path::new("config");
    if !config_dir.exists() {
        fs::create_dir_all(config_dir)
            .map_err(|e| format!("Failed to create config directory: {}", e))?;
    }

    // Save configuration to file
    let config_path = config_dir.join("frontend_config.json");
    let config_str = serde_json::to_string_pretty(&config)
        .map_err(|e| format!("Failed to serialize config: {}", e))?;

    fs::write(&config_path, config_str)
        .map_err(|e| format!("Failed to write config file: {}", e))?;

    info!("💾 Configuration saved to {:?}", config_path);
    Ok(())
}

#[tauri::command]
fn load_config() -> Result<serde_json::Value, String> {
    use std::fs;
    use std::path::Path;

    let config_path = Path::new("config/frontend_config.json");

    if !config_path.exists() {
        info!("📋 No saved config found, using defaults");
        // Return default configuration
        return Ok(serde_json::json!({
            "pth": "assets/weights/kikiV1.pth",
            "index": "logs/kikiV1.index",
            "hostapi": "",
            "wasapiExclusive": false,
            "inputDevice": "",
            "outputDevice": "",
            "srType": "sr_model",
            "threshold": -60,
            "pitch": 0,
            "formant": 0.0,
            "indexRate": 0.75,
            "rmsMixRate": 0.25,
            "f0method": "fcpe",
            "blockTime": 0.25,
            "crossfadeLength": 0.05,
            "nCpu": 2,
            "extraTime": 2.5,
            "iNoiseReduce": true,
            "oNoiseReduce": true,
            "usePv": false,
            "functionMode": "vc"
        }));
    }

    let config_str = fs::read_to_string(&config_path)
        .map_err(|e| format!("Failed to read config file: {}", e))?;

    let config: serde_json::Value = serde_json::from_str(&config_str)
        .map_err(|e| format!("Failed to parse config file: {}", e))?;

    info!("✅ Configuration loaded from {:?}", config_path);
    Ok(config)
}

#[tauri::command]
async fn start_vc(
    app: AppHandle,
    state: State<'_, VcState>,
    pth: String,
    index: String,
    hostapi: String,
    input_device: String,
    output_device: String,
    wasapi_exclusive: bool,
    sr_type: String,
    threshold: f64,
    pitch: f64,
    formant: f64,
    index_rate: f64,
    rms_mix_rate: f64,
    f0method: String,
    block_time: f64,
    crossfade_length: f64,
    n_cpu: i32,
    extra_time: f64,
    i_noise_reduce: bool,
    o_noise_reduce: bool,
    use_pv: bool,
    function_mode: String,
) -> Result<(), String> {
    info!("🚀 Starting voice conversion with parameters:");
    info!("  Model: {}", pth);
    info!("  Index: {}", index);
    info!("  F0 Method: {}", f0method);
    info!("  Pitch: {}", pitch);
    info!("  Index Rate: {}", index_rate);

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

    // Update configuration
    {
        let mut config = state.inner().config.lock().unwrap();
        config.pth_path = pth.clone();
        config.index_path = if index.is_empty() {
            String::new()
        } else {
            index.clone()
        };
        config.sg_hostapi = hostapi;
        config.sg_input_device = input_device;
        config.sg_output_device = output_device;
        config.sg_wasapi_exclusive = wasapi_exclusive;
        config.sr_type = sr_type;
        config.threshold = threshold;
        config.pitch = pitch;
        config.formant = formant;
        config.index_rate = index_rate as f32;
        config.rms_mix_rate = rms_mix_rate as f32;
        config.f0method = f0method;
        config.block_time = block_time as f32;
        config.crossfade_length = crossfade_length as f32;
        config.n_cpu = n_cpu as u32;
        config.extra_time = extra_time as f32;
        config.i_noise_reduce = i_noise_reduce;
        config.o_noise_reduce = o_noise_reduce;
        config.use_pv = use_pv;
        // Note: function_mode is not a field in GUIConfig, handle separately if needed
    }

    // Attempt to start voice conversion
    let config = state.inner().config.lock().unwrap().clone();

    // Log detailed configuration for debugging
    info!("🔧 Starting with configuration:");
    info!("  - Model path: {}", config.pth_path);
    info!("  - Index path: {}", config.index_path);
    info!("  - F0 method: {}", config.f0method);
    info!("  - Pitch: {}", config.pitch);
    info!("  - Index rate: {}", config.index_rate);
    info!("  - Block time: {}", config.block_time);
    info!("  - Crossfade length: {}", config.crossfade_length);

    match start_vc(move |gui_config| {
        info!("🔄 Configuring RVC with parameters...");
        *gui_config = config.clone();
        info!("✅ RVC configuration applied");
    }) {
        Ok(_) => {
            *is_running = true;
            info!("✅ Voice conversion started successfully");

            // Start audio streaming
            *state.inner().audio_streaming.lock().unwrap() = true;
            info!("🎵 Audio streaming enabled");

            // Emit success event to frontend
            let _ = app.emit(
                "rvc_status",
                serde_json::json!({
                    "status": "started",
                    "message": "Voice conversion started successfully"
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
async fn stop_vc(app: AppHandle, state: State<'_, VcState>) -> Result<(), String> {
    info!("🛑 Stopping voice conversion...");

    let mut is_running = state.inner().is_running.lock().unwrap();
    if !*is_running {
        let message = "Voice conversion is not running";
        warn!("{}", message);
        return Ok(()); // Don't return error, just log warning
    }

    // Stop audio streaming first
    *state.inner().audio_streaming.lock().unwrap() = false;

    // Stop voice conversion by setting VC_IS_ACTIVE to false
    VC_IS_ACTIVE.store(false, std::sync::atomic::Ordering::Relaxed);
    *is_running = false;
    info!("✅ Voice conversion stopped successfully");

    // Emit stop event to frontend
    let _ = app.emit(
        "rvc_status",
        serde_json::json!({
            "status": "stopped",
            "message": "Voice conversion stopped successfully"
        }),
    );

    Ok(())
}

#[tauri::command]
fn update_devices(hostapi: Option<String>) -> Result<DeviceInfo, String> {
    rvc_lib::update_devices(hostapi.as_deref()).map_err(|e| e.to_string())
}

#[tauri::command]
fn set_devices(
    hostapi: String,
    input_device: String,
    output_device: String,
) -> Result<u32, String> {
    rvc_lib::set_devices(&hostapi, &input_device, &output_device).map_err(|e| e.to_string())
}

#[tauri::command]
async fn get_audio_stream_status(state: State<'_, VcState>) -> Result<bool, String> {
    Ok(*state.inner().audio_streaming.lock().unwrap())
}

#[tauri::command]
async fn clear_audio_buffers(state: State<'_, VcState>) -> Result<(), String> {
    state.inner().input_buffer.lock().unwrap().clear();
    state.inner().output_buffer.lock().unwrap().clear();
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(VcState::default())
        .invoke_handler(tauri::generate_handler![
            event_handler,
            frontend_event,
            get_init_config,
            set_values,
            save_config,
            load_config,
            start_vc,
            stop_vc,
            update_devices,
            set_devices,
            get_audio_stream_status,
            clear_audio_buffers
        ])
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
