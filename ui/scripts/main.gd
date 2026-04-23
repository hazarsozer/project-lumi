extends Node

@onready var client: LumiClient = $LumiClient
@onready var avatar: AvatarController = $AvatarController
@onready var settings_panel = $SettingsPanel
@onready var chat_panel = $ChatPanel
@onready var status_dot: ColorRect = $CompactOverlay/ButtonTray/TrayLayout/StatusDot
@onready var state_label: Label = $CompactOverlay/ButtonTray/TrayLayout/StateLabel
@onready var settings_btn: Button = $CompactOverlay/ButtonTray/TrayLayout/SettingsBtn
@onready var chat_btn: Button = $CompactOverlay/ButtonTray/TrayLayout/ChatBtn
@onready var sprite: AnimatedSprite2D = $AvatarController/AnimatedSprite2D
@onready var glow_ring: Panel = $CompactOverlay/GlowRing

var _glow_style: StyleBoxFlat = null
var _drag_active: bool = false
var _drag_offset: Vector2i = Vector2i.ZERO

# Accent colors per state (matches design_tokens.json).
const STATE_COLORS: Dictionary = {
	"idle":       Color(0.0,    0.5961, 0.8196, 1),
	"listening":  Color(0.0,    0.6627, 0.3137, 1),
	"processing": Color(0.9216, 0.5373, 0.0,    1),
	"speaking":   Color(0.8941, 0.9255, 0.9490, 1),
}
const OPACITY_IDLE:   float = 0.42
const OPACITY_ACTIVE: float = 1.0


func _ready() -> void:
	client.message_received.connect(_on_message)
	client.connected_to_brain.connect(_on_connected)
	client.disconnected_from_brain.connect(_on_disconnected)
	client.config_schema_received.connect(_on_config_schema_received)
	client.config_update_result_received.connect(_on_config_update_result_received)
	settings_panel.apply_requested.connect(_on_settings_apply_requested)
	settings_btn.pressed.connect(_on_settings_pressed)
	chat_btn.pressed.connect(_on_chat_pressed)
	chat_panel.message_submitted.connect(_on_chat_message_submitted)
	_glow_style = StyleBoxFlat.new()
	_glow_style.bg_color = Color(0, 0, 0, 0)
	_glow_style.border_width_left = 2
	_glow_style.border_width_top = 2
	_glow_style.border_width_right = 2
	_glow_style.border_width_bottom = 2
	_glow_style.set_corner_radius_all(12)
	glow_ring.add_theme_stylebox_override("panel", _glow_style)
	_apply_state_ui("idle")


func _input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_cancel"):
		client.send_interrupt()
	if event is InputEventKey and event.pressed and not event.echo:
		if event.ctrl_pressed and event.keycode == KEY_COMMA:
			_on_settings_pressed()
	if event is InputEventMouseMotion and _drag_active:
		DisplayServer.window_set_position(Vector2i(DisplayServer.mouse_get_position()) + _drag_offset)
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT and not event.pressed:
		_drag_active = false


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
		_drag_active = true
		_drag_offset = DisplayServer.window_get_position() - Vector2i(DisplayServer.mouse_get_position())


func _on_connected() -> void:
	state_label.text = "Connected"
	state_label.add_theme_color_override("font_color", Color(0.0, 0.6627, 0.3137, 1))


func _on_disconnected() -> void:
	_apply_state_ui("idle")
	state_label.text = "Disconnected"
	state_label.add_theme_color_override("font_color", Color(0.427, 0.486, 0.529, 1))
	avatar.on_state_change("idle")


func _on_settings_pressed() -> void:
	if client._conn_state == "connected":
		client.request_config_schema()
	else:
		_place_panel_beside_overlay(settings_panel)
		settings_panel.open(_offline_demo_schema(), _offline_demo_values())


func _on_chat_pressed() -> void:
	if chat_panel.visible:
		chat_panel.hide()
	else:
		_place_panel_beside_overlay(chat_panel)
		chat_panel.show()


func _on_chat_message_submitted(text: String) -> void:
	client.send_event("text_input", {"text": text})


func _on_config_schema_received(fields: Dictionary, current_values: Dictionary) -> void:
	_place_panel_beside_overlay(settings_panel)
	settings_panel.open(fields, current_values)


func _on_config_update_result_received(applied_live: Array, pending_restart: Array, errors: Dictionary) -> void:
	settings_panel.handle_update_result(applied_live, pending_restart, errors)
	if not errors.is_empty():
		push_warning("main.gd: config_update errors: %s" % str(errors))


func _on_settings_apply_requested(changes: Dictionary, persist: bool) -> void:
	client.send_config_update(changes, persist)


func _on_message(event_name: String, payload: Dictionary) -> void:
	match event_name:
		"state_change":
			if payload.has("state"):
				var s: String = payload["state"]
				avatar.on_state_change(s)
				_apply_state_ui(s)
			else:
				push_warning("main.gd: state_change missing 'state' key")
		"tts_viseme":
			if payload.has("viseme") and payload.has("duration_ms"):
				avatar.on_viseme(payload["viseme"], payload["duration_ms"])
			else:
				push_warning("main.gd: tts_viseme missing required keys")
		"tts_start":
			chat_panel.begin_assistant_turn()
		"tts_stop":
			avatar.on_tts_stop()
		"transcript":
			var text: String = payload.get("text", "")
			if not text.is_empty():
				chat_panel.add_user_message(text)
		"llm_token":
			if payload.has("token"):
				chat_panel.add_token(payload["token"])
			else:
				push_warning("main.gd: llm_token missing 'token' key")
		"rag_retrieval":
			if payload.has("top_doc_paths"):
				chat_panel.add_citations(payload["top_doc_paths"])
			else:
				push_warning("main.gd: rag_retrieval missing 'top_doc_paths' key")
		"error":
			push_warning("Brain error [%s]: %s" % [payload.get("code", "?"), payload.get("message", "?")])
		"config_schema", "config_update_result":
			pass
		_:
			push_warning("main.gd: unhandled event '%s'" % event_name)


func _apply_state_ui(state: String) -> void:
	var color: Color = STATE_COLORS.get(state, STATE_COLORS["idle"])
	status_dot.color = color
	var is_active: bool = state != "idle"
	sprite.modulate.a = OPACITY_ACTIVE if is_active else OPACITY_IDLE
	var label_text: String = state.capitalize()
	state_label.text = label_text
	state_label.add_theme_color_override(
		"font_color",
		color if is_active else Color(0.427, 0.486, 0.529, 1)
	)
	if _glow_style != null:
		_glow_style.border_color = color if is_active else Color(0.110, 0.165, 0.212, 0.5)


# ---------------------------------------------------------------------------
# Panel placement — position floating window to the right of the overlay
# (or left if right side would be off-screen)
# ---------------------------------------------------------------------------

func _place_panel_beside_overlay(panel: Window) -> void:
	var overlay_pos := DisplayServer.window_get_position()
	var screen_size := DisplayServer.screen_get_size()
	var panel_w := int(panel.size.x)
	var target_x := overlay_pos.x + 170
	if target_x + panel_w > screen_size.x:
		target_x = overlay_pos.x - panel_w - 10
	panel.position = Vector2i(target_x, max(0, overlay_pos.y))


# ---------------------------------------------------------------------------
# Offline demo schema (shown when Brain is not connected)
# ---------------------------------------------------------------------------

func _offline_demo_schema() -> Dictionary:
	return {
		"edition": {"label": "Performance Edition", "control": "select",
			"options": ["light", "standard", "pro"], "restart_required": true,
			"help": "Hardware tier for LLM GPU offload."},
		"log_level": {"label": "Log Level", "control": "select",
			"options": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
			"restart_required": false, "help": "Root logger verbosity."},
		"json_logs": {"label": "JSON Log Format", "control": "toggle",
			"restart_required": false, "help": "Structured JSON output for log aggregators."},
		"audio.sensitivity": {"label": "Wake-Word Sensitivity", "control": "slider",
			"min": 0.0, "max": 1.0, "step": 0.01, "restart_required": false,
			"help": "Detection threshold [0–1]. Lower = more sensitive."},
		"audio.vad_threshold": {"label": "VAD Threshold", "control": "slider",
			"min": 0.0, "max": 1.0, "step": 0.01, "restart_required": false,
			"help": "Voice activity detection threshold."},
		"audio.silence_timeout_s": {"label": "Silence Timeout (s)", "control": "slider",
			"min": 0.1, "max": 10.0, "step": 0.1, "restart_required": false,
			"help": "Seconds of silence before recording stops."},
		"audio.wake_word_model_path": {"label": "Wake-Word Model Path", "control": "path",
			"restart_required": true, "help": "Path to the hey-Lumi ONNX model."},
		"scribe.model_size": {"label": "Whisper Model Size", "control": "select",
			"options": ["tiny", "tiny.en", "base", "small", "medium", "large-v3"],
			"restart_required": true, "help": "STT model variant."},
		"scribe.compute_type": {"label": "Compute Type", "control": "select",
			"options": ["int8", "float16", "float32"], "restart_required": true,
			"help": "int8 is fastest on CPU."},
		"llm.model_path": {"label": "LLM Model Path", "control": "path",
			"restart_required": true, "help": "Path to the GGUF model file."},
		"llm.temperature": {"label": "Temperature", "control": "slider",
			"min": 0.0, "max": 2.0, "step": 0.01, "restart_required": false,
			"help": "Sampling temperature. Lower = more deterministic."},
		"llm.max_tokens": {"label": "Max Output Tokens", "control": "slider",
			"min": 64, "max": 4096, "step": 64, "restart_required": false,
			"help": "Maximum tokens per response."},
		"tts.enabled": {"label": "Enable TTS", "control": "toggle",
			"restart_required": false, "help": "Disable for silent/headless mode."},
		"tts.voice": {"label": "TTS Voice", "control": "text",
			"restart_required": false, "help": "Kokoro voice identifier."},
		"tools.enabled": {"label": "Enable OS Tools", "control": "toggle",
			"restart_required": false, "help": "Allow Lumi to run OS action tools."},
		"rag.enabled": {"label": "Enable RAG", "control": "toggle",
			"restart_required": false, "help": "Search your personal documents before responding."},
		"rag.retrieval_top_k": {"label": "Retrieval Top-K", "control": "slider",
			"min": 1, "max": 20, "step": 1, "restart_required": false,
			"help": "Candidates retrieved per mode before RRF re-ranking."},
		"ipc.enabled": {"label": "Enable IPC Server", "control": "toggle",
			"restart_required": true, "help": "Connects the Godot frontend to the Brain."},
		"ipc.port": {"label": "IPC Port", "control": "number",
			"min": 1024, "max": 65535, "restart_required": true,
			"help": "Port for the ZMQ socket."},
		"persona.system_prompt": {"label": "System Prompt", "control": "text",
			"restart_required": false, "help": "Override Lumi's default persona. Leave empty for default."},
	}


func _offline_demo_values() -> Dictionary:
	return {
		"edition": "standard",
		"log_level": "INFO",
		"json_logs": false,
		"audio.sensitivity": 0.8,
		"audio.vad_threshold": 0.5,
		"audio.silence_timeout_s": 1.5,
		"audio.wake_word_model_path": "models/hey_lumi.onnx",
		"scribe.model_size": "tiny.en",
		"scribe.compute_type": "int8",
		"llm.model_path": "models/llm/phi-3.5-mini.gguf",
		"llm.temperature": 0.7,
		"llm.max_tokens": 512,
		"tts.enabled": true,
		"tts.voice": "af_heart",
		"tools.enabled": true,
		"rag.enabled": false,
		"rag.retrieval_top_k": 8,
		"ipc.enabled": false,
		"ipc.port": 5555,
		"persona.system_prompt": "",
	}
