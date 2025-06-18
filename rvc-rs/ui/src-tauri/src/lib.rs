use log::info;
use rvc_lib::{start_vc, DeviceInfo, GUIConfig, GUI};
use serde::Serialize;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, State};

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

// Global state for voice conversion
/// 只存储配置和运行状态，不再存储 VC 句柄（VC 不是 Send/Sync，不能放在 State）
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
    info!("Event: {} = {:?}", event, value);

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
                start_audio_streaming(app.clone(), state.clone());

                // Start voice conversion（VC 句柄不再存储于 State）
                match start_vc() {
                    Ok(_vc_handle) => {
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
                // Stop audio streaming
                *state.inner().audio_streaming.lock().unwrap() = false;
                // Clear buffers
                state.inner().input_buffer.lock().unwrap().clear();
                state.inner().output_buffer.lock().unwrap().clear();

                // VC 句柄不再存储于 State，无需手动 drop
                *is_running = false;
                app.emit("vc_stopped", ()).unwrap();
                app.emit("audio_stream_stopped", ()).unwrap();
                info!("Voice conversion stopped");
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
fn update_devices(hostapi: Option<String>) -> Result<DeviceInfo, String> {
    rvc_lib::update_devices(hostapi.as_deref())
}

#[tauri::command]
fn set_devices(
    hostapi: String,
    input_device: String,
    output_device: String,
) -> Result<u32, String> {
    rvc_lib::set_devices(&hostapi, &input_device, &output_device)
}

/// Start audio data streaming for waveform visualization
fn start_audio_streaming(app: AppHandle, state: State<'_, VcState>) {
    let app_clone = app.clone();
    let audio_streaming = state.inner().audio_streaming.clone();
    let input_buffer = state.inner().input_buffer.clone();
    let output_buffer = state.inner().output_buffer.clone();
    let stats = state.inner().stats.clone();

    // Spawn background thread for audio data simulation/capture
    thread::spawn(move || {
        let mut frame_count = 0u64;
        let mut last_emit = Instant::now();
        let emit_interval = Duration::from_millis(50); // 20 FPS for smooth waveform updates

        while *audio_streaming.lock().unwrap() {
            thread::sleep(Duration::from_millis(10)); // 100 Hz processing rate

            // Simulate audio data capture (in real implementation, this would come from audio callback)
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;

            // Generate simulated audio data for demonstration
            let sample_count = 1024;
            let mut input_samples = Vec::with_capacity(sample_count);
            let mut output_samples = Vec::with_capacity(sample_count);

            for i in 0..sample_count {
                let t = (frame_count * sample_count as u64 + i as u64) as f32 / 44100.0;

                // Simulate input audio with some noise and signal
                let input_signal = 0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
                    + 0.05 * ((t * 1000.0) % 1.0 - 0.5); // Add some noise

                // Simulate output audio (processed version)
                let output_signal = 0.4 * (2.0 * std::f32::consts::PI * 523.25 * t).sin() // C note
                    + 0.15 * (2.0 * std::f32::consts::PI * 659.25 * t).sin(); // E note

                input_samples.push(input_signal);
                output_samples.push(output_signal);
            }

            // Update buffers
            {
                let mut input_buf = input_buffer.lock().unwrap();
                input_buf.extend_from_slice(&input_samples);
                if input_buf.len() > 8192 {
                    let excess = input_buf.len() - 8192;
                    input_buf.drain(0..excess);
                }
            }

            {
                let mut output_buf = output_buffer.lock().unwrap();
                output_buf.extend_from_slice(&output_samples);
                if output_buf.len() > 8192 {
                    let excess = output_buf.len() - 8192;
                    output_buf.drain(0..excess);
                }
            }

            // Update stats
            {
                let mut stats_guard = stats.lock().unwrap();
                stats_guard.processed_samples += sample_count as u64;
                stats_guard.latency = 25.0 + (frame_count % 10) as f32 * 2.0; // Simulate latency variation
            }

            frame_count += 1;

            // Emit audio data to frontend at regular intervals
            if last_emit.elapsed() >= emit_interval {
                let input_data = {
                    let buf = input_buffer.lock().unwrap();
                    AudioData {
                        samples: buf.clone(),
                        sample_rate: 44100,
                        timestamp,
                    }
                };

                let output_data = {
                    let buf = output_buffer.lock().unwrap();
                    AudioData {
                        samples: buf.clone(),
                        sample_rate: 44100,
                        timestamp,
                    }
                };

                let current_stats = stats.lock().unwrap().clone();

                // Emit events to frontend
                if let Err(e) = app_clone.emit("input_audio_data", &input_data) {
                    eprintln!("Failed to emit input audio data: {}", e);
                }

                if let Err(e) = app_clone.emit("output_audio_data", &output_data) {
                    eprintln!("Failed to emit output audio data: {}", e);
                }

                if let Err(e) = app_clone.emit("audio_stats", &current_stats) {
                    eprintln!("Failed to emit audio stats: {}", e);
                }

                // Emit inference time simulation
                let infer_time = 15.0 + (frame_count % 20) as f32 * 5.0; // Simulate 15-35ms inference time
                if let Err(e) = app_clone.emit("infer_time", &(infer_time as u32)) {
                    eprintln!("Failed to emit inference time: {}", e);
                }

                last_emit = Instant::now();
            }
        }

        info!("Audio streaming thread terminated");
    });
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
