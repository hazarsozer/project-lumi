/**
 * tauriCompat.ts — thin wrappers around Tauri APIs that degrade gracefully
 * to no-ops when running in a plain browser (VITE_MOCK_WS=true / dev:web).
 *
 * Detection: Tauri injects `__TAURI_INTERNALS__` on `window` at runtime.
 * When that key is absent we are in a plain browser and all Tauri calls
 * must be skipped to avoid import errors.
 */

const isTauri = (): boolean => "__TAURI_INTERNALS__" in window;

/**
 * Emit a Tauri app event. No-op in browser mode.
 */
export const tauriEmit = async (
  event: string,
  payload?: unknown,
): Promise<void> => {
  if (!isTauri()) return;
  const { emit } = await import("@tauri-apps/api/event");
  await emit(event, payload);
};

/**
 * Listen for a Tauri app event.
 * Returns a cleanup function (unlisten). In browser mode returns a no-op cleanup.
 */
export const tauriListen = async <T>(
  event: string,
  handler: (payload: T) => void,
): Promise<() => void> => {
  if (!isTauri()) return () => {};
  const { listen } = await import("@tauri-apps/api/event");
  const unlisten = await listen<T>(event, (e) => handler(e.payload));
  return unlisten;
};

/**
 * Get all Tauri webview windows.
 * Returns an empty array in browser mode.
 */
export const tauriGetAllWindows = async (): Promise<
  Array<{ label: string; isVisible: () => Promise<boolean>; hide: () => Promise<void>; show: () => Promise<void>; setFocus: () => Promise<void> }>
> => {
  if (!isTauri()) return [];
  const { getAllWebviewWindows } = await import("@tauri-apps/api/webviewWindow");
  return getAllWebviewWindows();
};
