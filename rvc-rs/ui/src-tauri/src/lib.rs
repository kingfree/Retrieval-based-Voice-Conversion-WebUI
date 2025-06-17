use log::info;
use rvc_lib::{DeviceInfo, GUIConfig, GUI};

/// Print events coming from the front-end.
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
        .invoke_handler(tauri::generate_handler![
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
