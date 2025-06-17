use log::info;
use rvc_lib::{GUI, GUIConfig};
use cpal::traits::{DeviceTrait, HostTrait};
use serde::Serialize;

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

#[derive(Serialize)]
struct DeviceInfo {
  hostapis: Vec<String>,
  input_devices: Vec<String>,
  output_devices: Vec<String>,
}

#[tauri::command]
fn update_devices(hostapi: Option<String>) -> Result<DeviceInfo, String> {
  let host_ids = cpal::available_hosts();
  let host_names: Vec<String> = host_ids.iter().map(|id| id.name().to_string()).collect();

  let host = if let Some(name) = hostapi {
    if let Some(id) = host_ids.iter().find(|id| id.name() == name).copied() {
      cpal::host_from_id(id).map_err(|e| e.to_string())?
    } else {
      cpal::default_host()
    }
  } else {
    cpal::default_host()
  };

  let input_devices = host
    .input_devices()
    .map_err(|e| e.to_string())?
    .filter_map(|d| d.name().ok())
    .collect();
  let output_devices = host
    .output_devices()
    .map_err(|e| e.to_string())?
    .filter_map(|d| d.name().ok())
    .collect();

  Ok(DeviceInfo { hostapis: host_names, input_devices, output_devices })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .invoke_handler(tauri::generate_handler![frontend_event, get_init_config, set_values, update_devices])
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
