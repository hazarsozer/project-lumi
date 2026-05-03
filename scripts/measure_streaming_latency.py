"""Measure perceived TTS streaming latency for a multi-sentence LLM response.

Connects to a running Lumi Brain over WebSocket, sends a user_text that will
produce a multi-sentence response, and records:
  - Time from user_text sent → first llm_token received
  - Time from user_text sent → first tts_start received  (key perceived latency)
  - Time from user_text sent → tts_stop received         (total speech duration)

Usage
-----
    # Start the Brain first:
    uv run python -m src.main

    # Then in another terminal:
    uv run python scripts/measure_streaming_latency.py
    uv run python scripts/measure_streaming_latency.py --host 127.0.0.1 --port 5556 --repeat 3

Expected results post C4 streaming fix
---------------------------------------
  First token latency:  50-200 ms   (model warm start)
  First TTS latency:    400-900 ms  (first sentence boundary hit)
  Total speech:         varies with response length

Spec target: 5 s → 800 ms perceived latency for first TTS start.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import websockets.asyncio.client as ws_client

_QUERY = "Tell me two interesting facts about the moon."
_HELLO_TIMEOUT = 3.0
_RESULT_TIMEOUT = 30.0


async def _run_single(host: str, port: int) -> dict[str, float]:
    uri = f"ws://{host}:{port}"
    timings: dict[str, float] = {}

    async with ws_client.connect(uri) as ws:
        # Handshake
        raw = await asyncio.wait_for(ws.recv(), timeout=_HELLO_TIMEOUT)
        msg = json.loads(raw)
        assert msg.get("type") == "hello", f"Expected hello, got {msg!r}"
        await ws.send(json.dumps({"type": "hello_ack", "version": "1.0", "status": "ok"}))

        # Send query
        t0 = time.perf_counter()
        envelope = {
            "event": "user_text",
            "payload": {"text": _QUERY},
            "timestamp": time.time(),
            "version": "1.0",
        }
        await ws.send(json.dumps(envelope))

        # Collect frames until tts_stop or timeout
        deadline = time.perf_counter() + _RESULT_TIMEOUT
        while time.perf_counter() < deadline:
            remaining = deadline - time.perf_counter()
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=min(1.0, remaining))
            except asyncio.TimeoutError:
                continue

            frame = json.loads(raw)
            event = frame.get("event", "")
            elapsed = (time.perf_counter() - t0) * 1000  # ms

            if event == "llm_token" and "first_token_ms" not in timings:
                timings["first_token_ms"] = elapsed
            elif event == "tts_start" and "first_tts_ms" not in timings:
                timings["first_tts_ms"] = elapsed
                print(f"  First TTS start: {elapsed:.0f} ms")
            elif event == "tts_stop":
                timings["total_ms"] = elapsed
                break

    return timings


async def _main(host: str, port: int, repeat: int) -> None:
    all_first_tts: list[float] = []
    all_first_token: list[float] = []

    for i in range(repeat):
        print(f"\nRun {i + 1}/{repeat}: '{_QUERY[:60]}...'")
        try:
            t = await _run_single(host, port)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue

        if "first_token_ms" in t:
            all_first_token.append(t["first_token_ms"])
            print(f"  First token:   {t['first_token_ms']:.0f} ms")
        if "first_tts_ms" in t:
            all_first_tts.append(t["first_tts_ms"])
        if "total_ms" in t:
            print(f"  Total speech:  {t['total_ms']:.0f} ms")

    print("\n─── Summary ───────────────────────────────────")
    if all_first_token:
        print(f"First token latency:  mean={statistics.mean(all_first_token):.0f} ms")
    if all_first_tts:
        mean_tts = statistics.mean(all_first_tts)
        print(f"First TTS latency:    mean={mean_tts:.0f} ms")
        spec_target = 800
        status = "✅ PASS" if mean_tts <= spec_target else "❌ FAIL"
        print(f"Spec target ≤{spec_target} ms:  {status} (mean={mean_tts:.0f} ms)")
    print("────────────────────────────────────────────────")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Lumi streaming TTS latency.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--repeat", type=int, default=1, help="Number of runs to average")
    args = parser.parse_args()

    print(f"Connecting to ws://{args.host}:{args.port}")
    print("Make sure the Brain is running: uv run python -m src.main\n")
    asyncio.run(_main(args.host, args.port, args.repeat))


if __name__ == "__main__":
    main()
