class_name AvatarController
extends Node

@export var sprite: AnimatedSprite2D

var _current_state: String = "idle"
var _viseme_timer: float = 0.0


func _process(delta: float) -> void:
	if _viseme_timer > 0.0:
		_viseme_timer -= delta
		if _viseme_timer <= 0.0:
			_set_mouth_closed()


func on_state_change(state: String) -> void:
	_current_state = state
	match state:
		"idle":
			_play_animation("idle_breathe")
		"listening":
			_play_animation("listening_pulse")
		"processing":
			_play_animation("processing_spin")
		"speaking":
			_play_animation("speaking_lipsync")
		_:
			push_warning("AvatarController: unknown state '%s'" % state)


func on_viseme(viseme: String, duration_ms: int) -> void:
	# Phase 5: simple open/close driven by any viseme signal.
	# Phase 6+ will map viseme strings to specific mouth shapes.
	_set_mouth_open()
	_viseme_timer = duration_ms / 1000.0


func on_tts_stop() -> void:
	_set_mouth_closed()
	_viseme_timer = 0.0


# --- Private helpers ---

func _play_animation(anim_name: String) -> void:
	if sprite == null:
		push_warning("AvatarController: sprite not assigned")
		return
	if sprite.sprite_frames and sprite.sprite_frames.has_animation(anim_name):
		sprite.play(anim_name)
	else:
		push_warning("AvatarController: animation '%s' not found in SpriteFrames" % anim_name)


func _set_mouth_open() -> void:
	if sprite == null:
		return
	if sprite.sprite_frames and sprite.sprite_frames.has_animation("speaking_open"):
		sprite.play("speaking_open")


func _set_mouth_closed() -> void:
	if sprite == null:
		return
	if _current_state == "speaking":
		_play_animation("speaking_lipsync")
