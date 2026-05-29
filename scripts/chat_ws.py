"""WebSocket REPL/batch client for Lumi Brain — bypasses the Tauri frontend.

Reads the bearer token from ~/.lumi/ipc_token, completes the handshake,
sends user_text events, and streams llm_token deltas back to stdout.

Usage:
    # interactive REPL
    uv run python scripts/chat_ws.py

    # batch mode (one prompt per line on stdin)
    echo "Who are you?" | uv run python scripts/chat_ws.py --batch

    # batch from inline args
    uv run python scripts/chat_ws.py --prompt "Who are you?" --prompt "Send an email"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import websockets


WS_URL = "ws://127.0.0.1:5556"
TOKEN_PATH = Path("~/.lumi/ipc_token").expanduser()


def read_token() -> str | None:
    if not TOKEN_PATH.exists():
        return None
    return TOKEN_PATH.read_text().strip() or None


async def chat(prompts: list[str] | None) -> None:
    token = read_token()
    if token is None:
        print(f"WARN: no token at {TOKEN_PATH}; running unauthenticated", file=sys.stderr)

    async with websockets.connect(WS_URL) as ws:
        # 1. Wait for Brain's hello
        hello_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
        hello = json.loads(hello_raw)
        if hello.get("type") != "hello":
            print(f"ERROR: expected hello, got {hello}", file=sys.stderr)
            return

        # 2. Send hello_ack immediately (token if present)
        ack: dict[str, object] = {"type": "hello_ack", "version": "1.0", "status": "ok"}
        if token is not None:
            ack["token"] = token
        await ws.send(json.dumps(ack))

        # Reader task: prints state_change, llm_token deltas, error, transcript
        assistant_buf: list[str] = []
        done = asyncio.Event()

        dump_path = Path("/tmp/lumi-ws-frames.jsonl")
        dump_path.write_text("")

        async def reader() -> None:
            async for raw in ws:
                # dump everything before parsing
                with dump_path.open("a") as fh:
                    fh.write(raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace"))
                    fh.write("\n")
                msg = json.loads(raw)
                event = msg.get("event")
                payload = msg.get("payload", {})
                if event == "llm_token":
                    delta = payload.get("delta", "")
                    assistant_buf.append(delta)
                    print(delta, end="", flush=True)
                elif event == "tts_start":
                    text = payload.get("text", "")
                    if text and not assistant_buf:
                        assistant_buf.append(text)
                        print(text, flush=True)
                elif event == "tts_stop":
                    done.set()
                elif event == "state_change":
                    state = payload.get("state")
                    if state == "idle":
                        done.set()
                elif event == "error":
                    print(f"\n[brain error] {payload}", file=sys.stderr)
                    done.set()

        reader_task = asyncio.create_task(reader())

        async def ask(prompt: str) -> str:
            assistant_buf.clear()
            done.clear()
            print(f"\n> {prompt}\n", flush=True)
            await ws.send(json.dumps({
                "event": "user_text",
                "payload": {"text": prompt},
                "timestamp": time.time(),
                "version": "1.0",
            }))
            try:
                await asyncio.wait_for(done.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                print("\n[timeout waiting for response]", file=sys.stderr)
            return "".join(assistant_buf).strip()

        try:
            if prompts:
                for p in prompts:
                    await ask(p)
            else:
                # interactive
                print("\nLumi REPL — type a prompt (Ctrl-D / 'quit' to exit)")
                loop = asyncio.get_event_loop()
                while True:
                    try:
                        line = await loop.run_in_executor(None, sys.stdin.readline)
                    except (EOFError, KeyboardInterrupt):
                        break
                    if not line:
                        break
                    text = line.strip()
                    if text.lower() in {"quit", "exit", "/quit"}:
                        break
                    if not text:
                        continue
                    await ask(text)
        finally:
            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", action="append", default=[], help="inline prompt (repeatable)")
    ap.add_argument("--batch", action="store_true", help="read prompts from stdin (one per line)")
    args = ap.parse_args()

    prompts: list[str] | None = None
    if args.prompt:
        prompts = args.prompt
    elif args.batch:
        prompts = [ln.strip() for ln in sys.stdin if ln.strip()]

    asyncio.run(chat(prompts))


if __name__ == "__main__":
    main()
