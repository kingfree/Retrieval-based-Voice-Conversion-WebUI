use tauri::{Manager};
use rvc_lib::greet;
use tokio::task;

#[tauri::command]
async fn frontend_event(event: String, value: Option<String>) {
    task::spawn(async move {
        println!("event: {} value: {:?}", event, value);
    });
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![frontend_event])
        .setup(|app| {
            let _window = app.get_window("main").unwrap();
            println!("{}", greet("Tauri"));
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
