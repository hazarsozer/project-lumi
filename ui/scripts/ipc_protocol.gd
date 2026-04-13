class_name IPCProtocol

const VERSION: String = "1.0"


## Encode an event into a length-prefixed frame (4-byte big-endian + UTF-8 JSON).
static func encode(event_name: String, payload: Dictionary) -> PackedByteArray:
	var msg := {
		"event": event_name,
		"payload": payload,
		"timestamp": Time.get_unix_time_from_system(),
		"version": VERSION
	}
	var json_str: String = JSON.stringify(msg)
	var json_bytes: PackedByteArray = json_str.to_utf8_buffer()
	var length: int = json_bytes.size()
	var frame := PackedByteArray()
	# 4-byte big-endian length prefix
	frame.append((length >> 24) & 0xFF)
	frame.append((length >> 16) & 0xFF)
	frame.append((length >> 8) & 0xFF)
	frame.append(length & 0xFF)
	frame.append_array(json_bytes)
	return frame


## Decode all complete frames from a buffer.
##
## Returns a Dictionary with two keys:
##   "messages"  — Array[Dictionary] of fully parsed frames
##   "remainder" — PackedByteArray of unconsumed bytes (incomplete trailing frame)
##
## GDScript passes PackedByteArray by value, so the caller must write back
## the "remainder" to keep the accumulated buffer correct.
static func decode_frames(buffer: PackedByteArray) -> Dictionary:
	var messages: Array[Dictionary] = []
	while buffer.size() >= 4:
		var length: int = (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3]
		if buffer.size() < 4 + length:
			break  # incomplete frame; wait for more data
		var json_bytes := buffer.slice(4, 4 + length)
		buffer = buffer.slice(4 + length)
		var json_str: String = json_bytes.get_string_from_utf8()
		var parsed = JSON.parse_string(json_str)
		if parsed is Dictionary:
			messages.append(parsed)
		else:
			push_warning("IPCProtocol: failed to parse JSON frame")
	return {"messages": messages, "remainder": buffer}
