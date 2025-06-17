use log::info;
use rvc_lib::{GUI, GUIConfig};

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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .invoke_handler(tauri::generate_handler![frontend_event, get_init_config])
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
