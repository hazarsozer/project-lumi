class_name RagToggle
extends CanvasLayer

## Small always-visible status pill in the top-left corner that shows whether
## RAG retrieval is enabled or disabled.
##
## Keyboard shortcut: Ctrl+R toggles the state.
##
## Persistence: the current state is saved to and loaded from
##   user://user_prefs.cfg  under section [rag], key "enabled".
##
## The signal rag_state_changed(enabled: bool) is emitted whenever the state
## changes.  main.gd connects this signal to LumiClient.send_event so the
## Brain receives a "set_config" wire frame.

signal rag_state_changed(enabled: bool)

const PREFS_PATH: String = "user://user_prefs.cfg"
const PREFS_SECTION: String = "rag"
const PREFS_KEY: String = "enabled"

const COLOR_ON: Color = Color(0.18, 0.72, 0.36, 0.92)   # green
const COLOR_OFF: Color = Color(0.45, 0.45, 0.45, 0.80)  # grey

@onready var _pill_bg: ColorRect = $Control/PillBG
@onready var _status_label: Label = $Control/PillBG/StatusLabel

var _rag_enabled: bool = true


func _ready() -> void:
	_rag_enabled = _load_pref()
	_refresh_pill()


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_R and event.ctrl_pressed:
			_toggle()
			get_viewport().set_input_as_handled()


## Returns the current RAG enabled state.
func is_enabled() -> bool:
	return _rag_enabled


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

func _toggle() -> void:
	_rag_enabled = not _rag_enabled
	_save_pref(_rag_enabled)
	_refresh_pill()
	emit_signal("rag_state_changed", _rag_enabled)


func _refresh_pill() -> void:
	if _pill_bg == null or _status_label == null:
		return
	_pill_bg.color = COLOR_ON if _rag_enabled else COLOR_OFF
	_status_label.text = "RAG ON" if _rag_enabled else "RAG OFF"


func _load_pref() -> bool:
	var cfg := ConfigFile.new()
	var err := cfg.load(PREFS_PATH)
	if err != OK:
		# File does not exist yet; default to enabled.
		return true
	return cfg.get_value(PREFS_SECTION, PREFS_KEY, true)


func _save_pref(value: bool) -> void:
	var cfg := ConfigFile.new()
	# Load existing prefs so we don't clobber other sections.
	cfg.load(PREFS_PATH)
	cfg.set_value(PREFS_SECTION, PREFS_KEY, value)
	var err := cfg.save(PREFS_PATH)
	if err != OK:
		push_warning("RagToggle: failed to save prefs (err=%d)" % err)
