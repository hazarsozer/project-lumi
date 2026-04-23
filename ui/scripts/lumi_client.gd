class_name LumiClient
extends Node

signal message_received(event_name: String, payload: Dictionary)
signal connected_to_brain
signal disconnected_from_brain
signal config_schema_received(fields: Dictionary, current_values: Dictionary)
signal config_update_result_received(applied_live: Array, pending_restart: Array, errors: Dictionary)

const HOST: String = "127.0.0.1"
const PORT: int = 5555
const RECONNECT_DELAY_SEC: float = 2.0

var _stream: StreamPeerTCP
var _buffer: PackedByteArray
var _reconnect_timer: float = 0.0
# Three states: "disconnected", "connecting", "connected"
var _conn_state: String = "disconnected"


func _ready() -> void:
	_stream = StreamPeerTCP.new()
	_buffer = PackedByteArray()
	_begin_connect()


func _process(delta: float) -> void:
	match _conn_state:
		"disconnected":
			_reconnect_timer -= delta
			if _reconnect_timer <= 0.0:
				_begin_connect()
			return
		"connecting":
			_stream.poll()
			var status := _stream.get_status()
			match status:
				StreamPeerTCP.STATUS_CONNECTED:
					_conn_state = "connected"
					emit_signal("connected_to_brain")
					print("LumiClient: connected to Brain at %s:%d" % [HOST, PORT])
				StreamPeerTCP.STATUS_NONE, StreamPeerTCP.STATUS_ERROR:
					_on_disconnected()
			return
		"connected":
			_stream.poll()
			var status := _stream.get_status()
			if status == StreamPeerTCP.STATUS_NONE or status == StreamPeerTCP.STATUS_ERROR:
				_on_disconnected()
				return
			# Read all available bytes into buffer
			var available := _stream.get_available_bytes()
			if available > 0:
				var chunk := _stream.get_data(available)
				if chunk[0] == OK:
					_buffer.append_array(chunk[1])
				_drain_buffer()


func send_event(event_name: String, payload: Dictionary) -> void:
	if _conn_state != "connected":
		push_warning("LumiClient: cannot send '%s', not connected" % event_name)
		return
	var frame := IPCProtocol.encode(event_name, payload)
	var err := _stream.put_data(frame)
	if err != OK:
		push_warning("LumiClient: put_data failed with error %d" % err)


func send_interrupt() -> void:
	send_event("interrupt", {})


func send_user_text(text: String) -> void:
	send_event("user_text", {"text": text})


func request_config_schema() -> void:
	send_event("config_schema_request", {})


func send_config_update(changes: Dictionary, persist: bool) -> void:
	send_event("config_update", {"changes": changes, "persist": persist})


func _begin_connect() -> void:
	_buffer.clear()
	var err := _stream.connect_to_host(HOST, PORT)
	if err != OK:
		push_warning("LumiClient: connect_to_host failed (err=%d); retrying in %.1fs" % [err, RECONNECT_DELAY_SEC])
		_conn_state = "disconnected"
		_reconnect_timer = RECONNECT_DELAY_SEC
		return
	# connect_to_host is non-blocking; STATUS_CONNECTED is confirmed in _process
	_conn_state = "connecting"


func _on_disconnected() -> void:
	_conn_state = "disconnected"
	_stream.disconnect_from_host()
	_reconnect_timer = RECONNECT_DELAY_SEC
	emit_signal("disconnected_from_brain")
	print("LumiClient: disconnected; retrying in %.1fs" % RECONNECT_DELAY_SEC)


func _drain_buffer() -> void:
	# decode_frames returns { "messages": Array[Dictionary], "remainder": PackedByteArray }
	# because GDScript passes PackedByteArray by value; we must write the remainder back.
	var result := IPCProtocol.decode_frames(_buffer)
	_buffer = result["remainder"]
	var messages: Array = result["messages"]
	for msg in messages:
		# Handle IPC handshake: Brain sends {"type":"hello",...}, we reply with hello_ack.
		if msg.get("type", "") == "hello":
			_handle_hello(msg)
			continue
		if msg.has("event") and msg.has("payload"):
			var ev: String = msg["event"]
			var pl: Dictionary = msg["payload"]
			if ev == "config_schema":
				emit_signal("config_schema_received",
					pl.get("fields", {}),
					pl.get("current_values", {}))
			elif ev == "config_update_result":
				emit_signal("config_update_result_received",
					pl.get("applied_live", []),
					pl.get("pending_restart", []),
					pl.get("errors", {}))
			else:
				emit_signal("message_received", ev, pl)
		else:
			push_warning("LumiClient: dropping malformed message (missing 'event' or 'payload')")


func _handle_hello(hello: Dictionary) -> void:
	var remote_version: String = hello.get("version", "")
	var status: String = "ok"
	if remote_version != IPCProtocol.VERSION:
		status = "version_mismatch"
		push_warning(
			"LumiClient: IPC version mismatch — Brain=%s, Godot=%s; continuing." \
			% [remote_version, IPCProtocol.VERSION]
		)
	var ack := {
		"type": "hello_ack",
		"version": IPCProtocol.VERSION,
		"status": status,
	}
	var ack_json: String = JSON.stringify(ack)
	var ack_bytes: PackedByteArray = ack_json.to_utf8_buffer()
	var length: int = ack_bytes.size()
	var frame := PackedByteArray()
	frame.append((length >> 24) & 0xFF)
	frame.append((length >> 16) & 0xFF)
	frame.append((length >> 8) & 0xFF)
	frame.append(length & 0xFF)
	frame.append_array(ack_bytes)
	var err := _stream.put_data(frame)
	if err != OK:
		push_warning("LumiClient: failed to send hello_ack (err=%d)" % err)
