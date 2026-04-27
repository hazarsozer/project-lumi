class_name SettingRow
extends HBoxContainer

signal value_changed(field_key: String, new_value: Variant)

const C_TEXT_PRIMARY   := Color(0.859, 0.886, 0.906, 1)
const C_TEXT_SECONDARY := Color(0.427, 0.486, 0.529, 1)
const C_ACCENT         := Color(0.0,   0.600, 0.824, 1)
const C_BORDER         := Color(0.110, 0.165, 0.212, 1)
const C_SURFACE_TOP    := Color(0.063, 0.118, 0.165, 1)

var field_key: String = ""
var _control_type: String = ""
var _control_node: Control = null
var _label_node: Label = null
var _restart_badge: Label = null
var _slider_value_label: Label = null
var _error_label: Label = null


func setup(key: String, meta: Dictionary, current_value: Variant) -> void:
	field_key = key
	_control_type = meta.get("control", "text")

	# --- Label ---
	_label_node = Label.new()
	_label_node.text = meta.get("label", key)
	_label_node.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	add_child(_label_node)

	# --- Control ---
	_control_node = _build_control(meta, current_value)
	if _control_node != null:
		add_child(_control_node)

	# --- Slider value display label (created by _build_control for sliders) ---
	if _slider_value_label != null:
		add_child(_slider_value_label)

	# --- Restart badge ---
	if meta.get("restart_required", false):
		_restart_badge = Label.new()
		_restart_badge.text = "[↻]"
		_restart_badge.tooltip_text = "Restart required for this change to take effect."
		add_child(_restart_badge)

	_apply_row_style()


func show_error(message: String) -> void:
	if _error_label == null:
		_error_label = Label.new()
		_error_label.add_theme_color_override("font_color", Color(1, 0.2, 0.2, 1))
		_error_label.custom_minimum_size = Vector2(0, 16)
		add_child(_error_label)
	_error_label.text = message
	_error_label.show()


func clear_error() -> void:
	if _error_label != null:
		_error_label.hide()


func get_value() -> Variant:
	if _control_node == null:
		return null
	match _control_type:
		"slider":
			return (_control_node as HSlider).value
		"toggle":
			return (_control_node as CheckBox).button_pressed
		"select":
			var opt := _control_node as OptionButton
			return opt.get_item_text(opt.selected)
		"text", "path":
			return (_control_node as LineEdit).text
		"number":
			return (_control_node as SpinBox).value
		"multiselect":
			# Returns an Array of the option strings whose CheckBox is checked.
			var result: Array = []
			for child in _control_node.get_children():
				var cb := child as CheckBox
				if cb != null and cb.button_pressed:
					result.append(cb.text)
			return result
	return null


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

func _apply_row_style() -> void:
	add_theme_constant_override("separation", 8)
	alignment = BoxContainer.ALIGNMENT_CENTER

	if _label_node != null:
		_label_node.add_theme_color_override("font_color", C_TEXT_SECONDARY)
		_label_node.add_theme_font_size_override("font_size", 12)

	if _slider_value_label != null:
		_slider_value_label.add_theme_color_override("font_color", C_TEXT_SECONDARY)
		_slider_value_label.add_theme_font_size_override("font_size", 11)

	if _restart_badge != null:
		_restart_badge.add_theme_color_override("font_color", C_ACCENT)
		_restart_badge.add_theme_font_size_override("font_size", 10)


func _build_control(meta: Dictionary, current_value: Variant) -> Control:
	match _control_type:
		"slider":
			return _build_slider(meta, current_value)
		"toggle":
			return _build_toggle(current_value)
		"select":
			return _build_select(meta, current_value)
		"text", "path":
			return _build_line_edit(current_value)
		"number":
			return _build_spinbox(meta, current_value)
		"multiselect":
			return _build_multiselect(meta, current_value)
	# Unknown control type — fall back to read-only label.
	push_warning("SettingRow: unknown control type '%s' for key '%s'; using label fallback." % [_control_type, field_key])
	var lbl := Label.new()
	lbl.text = str(current_value)
	return lbl


func _build_slider(meta: Dictionary, current_value: Variant) -> HSlider:
	var slider := HSlider.new()
	slider.min_value = meta.get("min", 0.0)
	slider.max_value = meta.get("max", 1.0)
	slider.step = meta.get("step", 0.01)
	slider.custom_minimum_size = Vector2(160, 0)
	if current_value != null:
		slider.value = float(current_value)

	var track_style := StyleBoxFlat.new()
	track_style.bg_color = C_BORDER
	track_style.set_corner_radius_all(3)
	track_style.content_margin_top = 3.0
	track_style.content_margin_bottom = 3.0

	var active_style := StyleBoxFlat.new()
	active_style.bg_color = C_ACCENT
	active_style.set_corner_radius_all(3)
	active_style.content_margin_top = 3.0
	active_style.content_margin_bottom = 3.0

	slider.add_theme_stylebox_override("slider", track_style)
	slider.add_theme_stylebox_override("grabber_area", active_style)
	slider.add_theme_stylebox_override("grabber_area_highlight", active_style)

	# Companion label that mirrors the numeric value.
	_slider_value_label = Label.new()
	_slider_value_label.custom_minimum_size = Vector2(40, 0)
	_slider_value_label.text = "%.2f" % slider.value

	slider.value_changed.connect(func(v: float) -> void:
		_slider_value_label.text = "%.2f" % v
		emit_signal("value_changed", field_key, v)
	)
	return slider


func _build_toggle(current_value: Variant) -> CheckBox:
	var cb := CheckBox.new()
	if current_value != null:
		cb.button_pressed = bool(current_value)
	cb.toggled.connect(func(pressed: bool) -> void:
		emit_signal("value_changed", field_key, pressed)
	)
	return cb


func _build_select(meta: Dictionary, current_value: Variant) -> OptionButton:
	var opt := OptionButton.new()
	var options: Array = meta.get("options", [])
	var selected_idx := 0
	for i in options.size():
		opt.add_item(str(options[i]))
		if str(options[i]) == str(current_value):
			selected_idx = i
	if options.size() > 0:
		opt.selected = selected_idx
	opt.item_selected.connect(func(idx: int) -> void:
		emit_signal("value_changed", field_key, opt.get_item_text(idx))
	)
	return opt


func _build_line_edit(current_value: Variant) -> LineEdit:
	var le := LineEdit.new()
	le.custom_minimum_size = Vector2(200, 0)
	if current_value != null:
		le.text = str(current_value)
	le.text_changed.connect(func(new_text: String) -> void:
		emit_signal("value_changed", field_key, new_text)
	)
	return le


func _build_spinbox(meta: Dictionary, current_value: Variant) -> SpinBox:
	var sb := SpinBox.new()
	if meta.has("min"):
		sb.min_value = float(meta["min"])
	else:
		sb.min_value = -1e9
	if meta.has("max"):
		sb.max_value = float(meta["max"])
	else:
		sb.max_value = 1e9
	if meta.has("step"):
		sb.step = float(meta["step"])
	if current_value != null:
		sb.value = float(current_value)
	sb.value_changed.connect(func(v: float) -> void:
		emit_signal("value_changed", field_key, v)
	)
	return sb


func _build_multiselect(meta: Dictionary, current_value: Variant) -> HBoxContainer:
	var container := HBoxContainer.new()
	var options: Array = meta.get("options", [])
	# current_value for multiselect is expected to be an Array of selected strings.
	var selected: Array = []
	if current_value is Array:
		selected = current_value
	for opt_str in options:
		var cb := CheckBox.new()
		cb.text = str(opt_str)
		cb.button_pressed = selected.has(str(opt_str))
		cb.toggled.connect(func(_pressed: bool) -> void:
			# Re-collect all checked options and emit.
			var result: Array = []
			for child in container.get_children():
				var item := child as CheckBox
				if item != null and item.button_pressed:
					result.append(item.text)
			emit_signal("value_changed", field_key, result)
		)
		container.add_child(cb)
	return container
