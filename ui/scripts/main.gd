extends Node

@onready var client: LumiClient = $LumiClient
@onready var avatar: AvatarController = $AvatarController
# TextBubble must be added as a child node named "TextBubble" in the main
# scene (ui/scenes/main.tscn) via the Godot editor before this reference
# will resolve. Use the text_bubble.tscn scene as the source.
@onready var text_bubble: TextBubble = $TextBubble
# CitationPanel is instanced from citation_panel.tscn and wired in main.tscn.
@onready var citation_panel: CitationPanel = $CitationPanel
# RagToggle is instanced from rag_toggle.tscn and wired in main.tscn.
@onready var rag_toggle: RagToggle = $RagToggle


func _ready() -> void:
	client.message_received.connect(_on_message)
	client.connected_to_brain.connect(_on_connected)
	client.disconnected_from_brain.connect(_on_disconnected)
	rag_toggle.rag_state_changed.connect(_on_rag_state_changed)


func _input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_cancel"):  # Escape key
		client.send_interrupt()


func _on_connected() -> void:
	print("main.gd: connected to Brain")
	# Sync the Brain with the persisted RAG toggle state on (re)connect.
	_send_rag_config(rag_toggle.is_enabled())


func _on_disconnected() -> void:
	print("main.gd: disconnected from Brain")
	avatar.on_state_change("idle")


func _on_rag_state_changed(enabled: bool) -> void:
	_send_rag_config(enabled)


func _send_rag_config(enabled: bool) -> void:
	client.send_event("set_config", {"key": "rag_enabled", "value": enabled})


func _on_message(event_name: String, payload: Dictionary) -> void:
	match event_name:
		"state_change":
			if payload.has("state"):
				avatar.on_state_change(payload["state"])
			else:
				push_warning("main.gd: state_change missing 'state' key")
		"tts_viseme":
			if payload.has("viseme") and payload.has("duration_ms"):
				avatar.on_viseme(payload["viseme"], payload["duration_ms"])
			else:
				push_warning("main.gd: tts_viseme missing required keys")
		"tts_start":
			# LLM finished streaming — clear the bubble before TTS begins.
			text_bubble.clear()
		"tts_stop":
			avatar.on_tts_stop()
		"transcript":
			push_warning("main.gd: transcript received: %s" % payload.get("text", ""))
		"llm_token":
			if payload.has("token"):
				text_bubble.add_token(payload["token"])
			else:
				push_warning("main.gd: llm_token missing 'token' key")
		"rag_retrieval":
			if payload.has("top_doc_paths"):
				citation_panel.show_citations(payload)
			else:
				push_warning("main.gd: rag_retrieval missing 'top_doc_paths' key")
		"error":
			push_warning("Brain error [%s]: %s" % [payload.get("code", "?"), payload.get("message", "?")])
		_:
			push_warning("main.gd: unhandled event '%s'" % event_name)
