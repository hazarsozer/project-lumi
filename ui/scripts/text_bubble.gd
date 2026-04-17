class_name TextBubble
extends Control

## Displays streaming LLM tokens in a text bubble.
##
## Lifecycle:
##   - add_token() is called once per "llm_token" event during LLM inference.
##   - clear() is called when "tts_start" fires; the Brain is about to speak
##     the full response, so the accumulated text is no longer needed.
##
## The node starts hidden. It becomes visible on the first add_token() call
## and hides itself again on clear().

# Emitted after clear() completes — useful for any parent that wants to
# animate the bubble out before starting TTS visuals.
signal bubble_cleared

# Wire this to the RichTextLabel child in the Godot editor (or via the scene
# file). The export makes it inspectable in the editor.
@export var label: RichTextLabel

var _current_text: String = ""
var _visible_bubble: bool = false


func add_token(token: String) -> void:
	_current_text += token
	if label:
		label.text = _current_text
	if not _visible_bubble:
		_visible_bubble = true
		show()


func clear() -> void:
	_current_text = ""
	if label:
		label.text = ""
	_visible_bubble = false
	hide()
	bubble_cleared.emit()
