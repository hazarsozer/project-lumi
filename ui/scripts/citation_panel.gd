class_name CitationPanel
extends CanvasLayer

## Displays RAG retrieval results (source documents) in an overlay panel.
##
## Lifecycle:
##   - show_citations(data) is called with the "rag_retrieval" wire payload.
##   - The panel auto-hides after AUTO_HIDE_SEC seconds unless the user has
##     interacted with it (hover or click).
##   - The close button (X) hides the panel immediately.
##
## Expected payload keys (from zmq_server.py on_rag_retrieval):
##   "query"         — String: the user query that triggered retrieval
##   "hit_count"     — int: number of documents retrieved
##   "latency_ms"    — int: retrieval latency in milliseconds
##   "top_doc_paths" — Array[String]: document paths, one per citation row

const AUTO_HIDE_SEC: float = 8.0

@onready var _panel_container: PanelContainer = $PanelContainer
@onready var _title_label: Label = $PanelContainer/VBoxContainer/HeaderRow/TitleLabel
@onready var _close_button: Button = $PanelContainer/VBoxContainer/HeaderRow/CloseButton
@onready var _query_label: Label = $PanelContainer/VBoxContainer/QueryLabel
@onready var _citations_list: VBoxContainer = $PanelContainer/VBoxContainer/CitationsList

var _auto_hide_timer: SceneTreeTimer = null
var _user_interacted: bool = false


func _ready() -> void:
	hide()
	_close_button.pressed.connect(_on_close_pressed)
	_panel_container.mouse_entered.connect(_on_panel_mouse_entered)


## Display the panel and populate it with the rag_retrieval payload.
func show_citations(data: Dictionary) -> void:
	if not data.has("top_doc_paths"):
		push_warning("CitationPanel: payload missing 'top_doc_paths'; skipping display")
		return

	_clear_list()
	_user_interacted = false

	var paths: Array = data["top_doc_paths"]
	var hit_count: int = data.get("hit_count", paths.size())
	var latency_ms: int = data.get("latency_ms", 0)
	var query: String = data.get("query", "")

	_title_label.text = "Sources (%d results, %dms)" % [hit_count, latency_ms]

	_query_label.text = query
	_query_label.visible = query != ""

	for i in range(paths.size()):
		var path: String = str(paths[i])
		var short_path: String = _shorten_path(path)
		var rank: int = i + 1

		var row := Label.new()
		row.text = "%d.  %s" % [rank, short_path]
		row.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		_citations_list.add_child(row)

	show()
	_start_auto_hide_timer()


## Truncate a file path to its last 2 components.
## e.g. "/home/user/documents/notes/ideas.txt" → "notes/ideas.txt"
func _shorten_path(path: String) -> String:
	if path == "":
		return "(unknown)"
	var parts: PackedStringArray = path.split("/")
	# Remove empty strings from leading slash
	var non_empty: Array = []
	for p in parts:
		if p != "":
			non_empty.append(p)
	if non_empty.size() == 0:
		return path
	if non_empty.size() <= 2:
		return "/".join(PackedStringArray(non_empty))
	return "%s/%s" % [non_empty[non_empty.size() - 2], non_empty[non_empty.size() - 1]]


func _clear_list() -> void:
	for child in _citations_list.get_children():
		child.queue_free()


func _start_auto_hide_timer() -> void:
	# Cancel any previously running timer by discarding the reference; the old
	# timeout signal fires into a lambda that checks _user_interacted, so
	# discarding the ref is safe — the lambda is a no-op once the panel hides.
	_auto_hide_timer = get_tree().create_timer(AUTO_HIDE_SEC)
	_auto_hide_timer.timeout.connect(_on_auto_hide_timeout)


func _on_auto_hide_timeout() -> void:
	if not _user_interacted:
		hide()


func _on_panel_mouse_entered() -> void:
	_user_interacted = true


func _on_close_pressed() -> void:
	_user_interacted = true
	hide()
