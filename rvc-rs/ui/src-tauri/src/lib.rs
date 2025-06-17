use log::info;
use rvc_lib::{start_vc, DeviceInfo, GUIConfig, GUI};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, State};

// Global state for voice conversion
struct VcState {
    config: Arc<Mutex<GUIConfig>>,
    is_running: Arc<Mutex<bool>>,
}

impl Default for VcState {
    fn default() -> Self {
        Self {
            config: Arc::new(Mutex::new(GUIConfig::default())),
            is_running: Arc::new(Mutex::new(false)),
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

                // Start voice conversion
                match start_vc() {
                    Ok(_vc_handle) => {
                        // Note: We don't store the VC handle in state due to thread safety issues
                        // The VC handle manages its own lifecycle
                        *is_running = true;
                        app.emit("vc_started", ()).unwrap();
                        info!("Voice conversion started successfully");
                    }
                    Err(e) => {
                        return Err(format!("Failed to start voice conversion: {}", e));
                    }
                }
            }
        }

        "stop_vc" => {
            if *is_running {
                info!("Stopping voice conversion...");
                // Note: VC handle is dropped automatically when it goes out of scope
                *is_running = false;
                app.emit("vc_stopped", ()).unwrap();
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
                    // TODO: Update RVC instance pitch in real-time
                }
            }
        }

        "formant" => {
            if let Some(val) = value {
                if let Ok(formant) = val.parse::<f32>() {
                    config.formant = formant;
                    // TODO: Update RVC instance formant in real-time
                }
            }
        }

        "index_rate" => {
            if let Some(val) = value {
                if let Ok(rate) = val.parse::<f32>() {
                    config.index_rate = rate;
                    // TODO: Update RVC instance index rate in real-time
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
                config.i_noise_reduce = val.parse().unwrap_or(false);
                // TODO: Update delay time calculation
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
                // TODO: Implement function mode switching (vc/im)
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
            set_devices
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
