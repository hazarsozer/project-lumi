class_name CitationPanel
extends CanvasLayer

## Displays RAG retrieval results (source documents) in an overlay panel.
##
## Lifecycle:
##   - show_citations(data) is called with the "rag_retrieval" wire payload.
##   - The panel auto-shows on show_citations() and hides on the Hide button.
##
## Expected payload keys (from zmq_server.py on_rag_retrieval):
##   "query"         — String: the user query that triggered retrieval
##   "hit_count"     — int: number of documents retrieved
##   "latency_ms"    — int: retrieval latency in milliseconds
##   "top_doc_paths" — Array[String]: document paths, one per citation row

@onready var _citations_list: VBoxContainer = $PanelContainer/VBoxContainer/CitationsList
@onready var _hide_button: Button = $PanelContainer/VBoxContainer/HeaderRow/HideButton


func _ready() -> void:
	hide()
	_hide_button.pressed.connect(_on_hide_pressed)


## Display the panel and populate it with the rag_retrieval payload.
func show_citations(data: Dictionary) -> void:
	if not data.has("top_doc_paths"):
		push_warning("CitationPanel: payload missing 'top_doc_paths'; skipping display")
		return

	_clear_list()

	var paths: Array = data["top_doc_paths"]
	var hit_count: int = data.get("hit_count", paths.size())
	var latency_ms: int = data.get("latency_ms", 0)

	for i in range(paths.size()):
		var path: String = str(paths[i])
		var doc_name: String = path.get_file() if path != "" else "(unknown)"
		var rank: int = i + 1

		var row := Label.new()
		row.text = "%d.  %s" % [rank, doc_name]
		row.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		_citations_list.add_child(row)

	var footer := Label.new()
	footer.text = "%d source(s)  —  %d ms" % [hit_count, latency_ms]
	footer.add_theme_color_override("font_color", Color(0.6, 0.6, 0.6))
	_citations_list.add_child(footer)

	show()


func _clear_list() -> void:
	for child in _citations_list.get_children():
		child.queue_free()


func _on_hide_pressed() -> void:
	hide()
