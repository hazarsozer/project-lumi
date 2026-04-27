class_name SettingsPanel
extends Window

signal settings_closed
signal apply_requested(changes: Dictionary, persist: bool)

var _schema: Dictionary = {}
var _current_values: Dictionary = {}
var _dirty_values: Dictionary = {}
var _row_map: Dictionary = {}

@onready var _panel: PanelContainer = $PanelContainer
@onready var _tab_container: TabContainer = $PanelContainer/VBox/TabContainer
@onready var _apply_btn: Button = $PanelContainer/VBox/Footer/ApplyBtn
@onready var _save_btn: Button = $PanelContainer/VBox/Footer/SaveBtn
@onready var _cancel_btn: Button = $PanelContainer/VBox/Footer/CancelBtn
@onready var _restart_bar: Label = $PanelContainer/VBox/RestartBar
@onready var _close_btn: Button = $PanelContainer/VBox/Header/CloseBtn


func _ready() -> void:
	hide()
	close_requested.connect(_on_cancel)
	_apply_btn.pressed.connect(_on_apply)
	_save_btn.pressed.connect(_on_save)
	_cancel_btn.pressed.connect(_on_cancel)
	_close_btn.pressed.connect(_on_cancel)


func open(fields: Dictionary, current_values: Dictionary) -> void:
	_schema = fields
	_current_values = current_values
	_dirty_values = {}
	_row_map = {}
	_restart_bar.hide()
	_populate_tabs()
	show()


const TAB_FIELDS: Dictionary = {
	"General": ["edition", "log_level", "json_logs"],
	"Voice Input": [
		"audio.sample_rate", "audio.chunk_size", "audio.sensitivity",
		"audio.vad_threshold", "audio.silence_timeout_s",
		"audio.recording_timeout_s", "audio.wake_word_model_path"
	],
	"Transcription": [
		"scribe.model_size", "scribe.beam_size", "scribe.compute_type",
		"scribe.model_path", "scribe.initial_prompt"
	],
	"LLM": [
		"llm.model_path", "llm.n_gpu_layers", "llm.context_length",
		"llm.max_tokens", "llm.temperature", "llm.vram_budget_gb",
		"llm.memory_dir"
	],
	"TTS": ["tts.enabled", "tts.voice", "tts.model_path", "tts.voices_path"],
	"Knowledge Base": [
		"rag.enabled", "rag.db_path", "rag.embedding_model",
		"rag.chunk_size", "rag.chunk_overlap", "rag.retrieval_top_k",
		"rag.context_char_budget", "rag.min_score",
		"rag.corpus_dir", "rag.retrieval_timeout_s"
	],
	"Advanced": [
		"ipc.enabled", "ipc.address", "ipc.port",
		"tools.enabled", "tools.allowed_tools", "tools.execution_timeout_s",
		"vision.enabled", "vision.model_path", "vision.capture_method",
		"vision.max_resolution", "persona.system_prompt"
	],
}

const TAB_ORDER: Array = [
	"General", "Voice Input", "Transcription", "LLM",
	"TTS", "Knowledge Base", "Advanced"
]


func _populate_tabs() -> void:
	for child in _tab_container.get_children():
		child.queue_free()

	var row_scene: PackedScene = preload("res://scenes/setting_row.tscn")

	for tab_name in TAB_ORDER:
		var keys: Array = TAB_FIELDS[tab_name]
		var scroll := ScrollContainer.new()
		scroll.name = tab_name
		scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
		var vbox := VBoxContainer.new()
		vbox.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		scroll.add_child(vbox)
		_tab_container.add_child(scroll)

		for key in keys:
			if not _schema.has(key):
				continue
			var meta: Dictionary = _schema[key]
			var current_val: Variant = _current_values.get(key)
			var row: SettingRow = row_scene.instantiate()
			vbox.add_child(row)
			row.setup(key, meta, current_val)
			row.value_changed.connect(_on_row_value_changed)
			_row_map[key] = row


func handle_update_result(
		applied_live: Array,
		pending_restart: Array,
		errors: Dictionary
) -> void:
	for row in _row_map.values():
		row.clear_error()

	for key in errors.keys():
		if _row_map.has(key):
			_row_map[key].show_error(errors[key])
		else:
			push_warning("SettingsPanel: error for unknown key '%s': %s" % [key, errors[key]])

	if pending_restart.size() > 0:
		_restart_bar.text = "Some changes require a restart to take effect."
		_restart_bar.show()


func _on_row_value_changed(key: String, value: Variant) -> void:
	_dirty_values[key] = value

	var needs_restart := false
	for k in _dirty_values.keys():
		if _schema.get(k, {}).get("restart_required", false):
			needs_restart = true
			break

	if needs_restart:
		_restart_bar.text = "Some changes require a restart to take effect."
		_restart_bar.show()
	else:
		_restart_bar.hide()


func _on_apply() -> void:
	emit_signal("apply_requested", _dirty_values.duplicate(), false)
	_dirty_values.clear()
	_restart_bar.hide()


func _on_save() -> void:
	emit_signal("apply_requested", _dirty_values.duplicate(), true)
	_dirty_values.clear()
	_restart_bar.hide()


func _on_cancel() -> void:
	hide()
	_dirty_values.clear()
	_restart_bar.hide()
	emit_signal("settings_closed")
