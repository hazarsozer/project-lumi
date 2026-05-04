use tauri::Manager;
use tauri_plugin_shell::ShellExt;

/// Read the IPC bearer token written by the Brain on startup.
/// Returns the trimmed token string, or an error message if the file is missing.
#[tauri::command]
fn read_ipc_token() -> Result<String, String> {
    let home = std::env::var("HOME").map_err(|e| e.to_string())?;
    let token_path = std::path::Path::new(&home).join(".lumi").join("ipc_token");
    std::fs::read_to_string(&token_path)
        .map(|s| s.trim().to_string())
        .map_err(|e| format!("read_ipc_token: {e}"))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![read_ipc_token])
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            // A second launch was attempted — bring the overlay to the front.
            if let Some(win) = app.get_webview_window("overlay") {
                let _ = win.show();
                let _ = win.set_focus();
            }
        }))
        .setup(|app| {
            // Spawn the Brain sidecar. The Child handle is stored in app state
            // so Tauri keeps it alive for the full lifetime of the app and kills
            // it automatically when the app exits.
            //
            // The sidecar binary must exist at:
            //   app/src-tauri/binaries/lumi-brain-x86_64-unknown-linux-gnu/
            // Build it with: bash scripts/build_brain.sh
            let shell = app.shell();
            let sidecar_cmd = shell
                .sidecar("lumi-brain")
                .expect("lumi-brain sidecar not found — run scripts/build_brain.sh first");
            let (_rx, child) = sidecar_cmd
                .spawn()
                .expect("failed to start lumi-brain sidecar");
            app.manage(child);

            // Hide the overlay on close rather than exiting, so it stays
            // resident in the background and can be re-shown via tray/hotkey.
            if let Some(window) = app.get_webview_window("overlay") {
                let win = window.clone();
                window.on_window_event(move |event| {
                    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                        api.prevent_close();
                        win.hide().expect("failed to hide window on close");
                    }
                });
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running Lumi");
}
