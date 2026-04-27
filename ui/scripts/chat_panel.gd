class_name ChatPanel
extends Window

signal message_submitted(text: String)

# Reference to the in-progress assistant label during token streaming.
var _streaming_label: RichTextLabel = null

@onready var _messages_vbox: VBoxContainer = $PanelContainer/VBox/MessagesScroll/MessagesVBox
@onready var _messages_scroll: ScrollContainer = $PanelContainer/VBox/MessagesScroll
@onready var _input_field: LineEdit = $PanelContainer/VBox/InputRow/InputField
@onready var _send_btn: Button = $PanelContainer/VBox/InputRow/SendBtn
@onready var _clear_btn: Button = $PanelContainer/VBox/Header/ClearBtn


func _ready() -> void:
	hide()
	close_requested.connect(hide)
	_send_btn.pressed.connect(_on_send)
	_clear_btn.pressed.connect(clear_history)
	_input_field.text_submitted.connect(_on_text_submitted)


# Called when a transcript arrives — shows what the user said.
func add_user_message(text: String) -> void:
	_streaming_label = null
	_add_bubble(text, "user")


# Called for each streaming token from the LLM.
func add_token(token: String) -> void:
	if _streaming_label == null:
		_streaming_label = _create_bubble("assistant")
		_messages_vbox.add_child(_streaming_label)
	_streaming_label.append_text(token)
	_scroll_to_bottom()


# Called when TTS starts a new turn — caps the previous streaming label.
func begin_assistant_turn() -> void:
	_streaming_label = null


# Called when RAG sources arrive for the most recent response.
func add_citations(top_doc_paths: Array) -> void:
	if top_doc_paths.is_empty():
		return
	var hbox := HBoxContainer.new()
	hbox.add_theme_constant_override("separation", 4)
	var prefix := Label.new()
	prefix.text = "Sources:"
	prefix.add_theme_font_size_override("font_size", 10)
	prefix.add_theme_color_override("font_color", Color(0.427, 0.486, 0.529, 1))
	hbox.add_child(prefix)
	for path in top_doc_paths:
		var btn := Button.new()
		var short_name: String = path.get_file() if "/" in path else path
		btn.text = short_name
		btn.tooltip_text = path
		btn.add_theme_font_size_override("font_size", 10)
		hbox.add_child(btn)
	_messages_vbox.add_child(hbox)
	_scroll_to_bottom()


func clear_history() -> void:
	_streaming_label = null
	for child in _messages_vbox.get_children():
		child.queue_free()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

func _add_bubble(text: String, role: String) -> void:
	var label := _create_bubble(role)
	label.text = text
	_messages_vbox.add_child(label)
	_scroll_to_bottom()


func _create_bubble(role: String) -> RichTextLabel:
	var label := RichTextLabel.new()
	label.bbcode_enabled = false
	label.fit_content = true
	label.scroll_active = false
	label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	label.set_meta("role", role)
	if role == "user":
		label.add_theme_color_override("default_color", Color(0.859, 0.886, 0.906, 1))
	else:
		label.add_theme_color_override("default_color", Color(0.427, 0.486, 0.529, 1))
	label.add_theme_font_size_override("normal_font_size", 13)
	return label


func _scroll_to_bottom() -> void:
	await get_tree().process_frame
	_messages_scroll.scroll_vertical = int(_messages_scroll.get_v_scroll_bar().max_value)


func _on_send() -> void:
	var text := _input_field.text.strip_edges()
	if text.is_empty():
		return
	_input_field.text = ""
	add_user_message(text)
	emit_signal("message_submitted", text)


func _on_text_submitted(_text: String) -> void:
	_on_send()
