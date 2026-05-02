use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            // A second launch was attempted — bring the overlay to the front.
            if let Some(win) = app.get_webview_window("overlay") {
                let _ = win.show();
                let _ = win.set_focus();
            }
        }))
        .setup(|app| {
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
