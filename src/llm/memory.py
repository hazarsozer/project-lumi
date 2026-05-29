"""Conversation memory with JSON persistence for Project Lumi."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)


# Threshold at which rotation is triggered. Promote to LumiConfig.llm.memory_max_turns
# when user-configurable memory size becomes a requested feature.
MAX_TURNS: int = 40

# Turns to keep verbatim after rotation. The older turns are summarized into a
# single system message. Must be < MAX_TURNS.
RETAIN_RECENT: int = 20


class ConversationMemory:
    """Stores conversation turns with optional JSON persistence and LLM-based rotation.

    Thread safety
    -------------
    An ``RLock`` guards the mutable ``_history`` list.  All reads and writes
    go through that lock.

    The LLM summariser and the disk write are dispatched to a single-worker
    ``ThreadPoolExecutor`` (daemon thread) so that ``add_turn()`` returns
    promptly without blocking on either I/O or LLM inference.

    The summariser is called **without** holding the memory lock, so there is
    no risk of a deadlock with the model's ``_VRAM_LOCK``.  The lock is
    re-acquired only for the final history splice and save, which are fast
    operations.

    Rotation strategy (triggered when len(history) > MAX_TURNS):
    - If a summariser callable is registered: the oldest (len - RETAIN_RECENT)
      turns are passed to the summariser, which returns a summary string. They
      are replaced by a single system message: {"role": "system", "content":
      "Summary of earlier conversation: <text>"}. The newest RETAIN_RECENT
      turns are kept verbatim.
    - If no summariser is registered: simple truncation to RETAIN_RECENT (no
      LLM call, no crash — graceful degradation).
    - If the summariser raises: fall back to truncation and log a warning.
    """

    def __init__(
        self,
        memory_dir: str,
        summariser: Callable[[list[dict[str, str]]], str] | None = None,
        max_age_s: float = 0.0,
        _now_fn: Callable[[], float] | None = None,
    ) -> None:
        """Initialise ConversationMemory.

        Args:
            memory_dir: Directory for the JSON persistence file.
            summariser: Optional callable that summarises a list of turns into a
                        single string.  Called off-thread during rotation.
            max_age_s: Maximum age (in seconds) for persisted conversation
                       entries.  Entries whose ``ts`` field is older than
                       ``now - max_age_s`` are purged on load and save.
                       0.0 disables age-based retention entirely.
            _now_fn: Callable returning the current time as a float (UNIX
                     epoch).  Defaults to ``time.time``.  Override in tests to
                     inject a deterministic clock without wall-clock sleeps.
        """
        expanded = Path(memory_dir).expanduser()
        expanded.mkdir(parents=True, exist_ok=True)
        self._history: list[dict[str, str]] = []
        self._file: Path = expanded / "conversation.json"
        self._summariser = summariser
        self._max_age_s: float = max_age_s
        self._now_fn: Callable[[], float] = _now_fn if _now_fn is not None else time.time
        self._lock: threading.RLock = threading.RLock()
        # Guards against concurrent rotations. True while a background rotation
        # task is queued or running. add_turn checks this under the lock so that
        # at most one rotation is in-flight at any time. Any turns appended
        # while the flag is set are preserved by the writeback splice (see
        # _rotate_background) rather than being discarded.
        self._rotating: bool = False
        # Single worker serialises background rotations; daemon=True so the
        # thread doesn't prevent interpreter shutdown.
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="memory-bg"
        )

    def set_summariser(
        self,
        callback: Callable[[list[dict[str, str]]], str] | None,
    ) -> None:
        """Inject or replace the summariser callable after construction."""
        with self._lock:
            self._summariser = callback

    def add_turn(self, role: str, content: str) -> None:
        """Append a turn and schedule background rotation when history exceeds MAX_TURNS.

        Returns promptly — the LLM summariser and disk write run off-thread.

        At most one rotation is in-flight at any time.  Turns appended while a
        rotation is already running are kept: the writeback splices using the
        live ``_history[k:]`` tail rather than a stale snapshot, so no turn is
        ever silently discarded.

        Each entry is stamped with a ``ts`` field (UNIX epoch float) so that
        age-based retention can later purge stale entries.  The ``ts`` field is
        an implementation detail and is not returned by ``get_history()`` —
        callers only see ``role`` and ``content``.
        """
        with self._lock:
            self._history.append({"role": role, "content": content, "ts": self._now_fn()})
            # Only dispatch a rotation if one is not already pending or running.
            # Turns added while _rotating is True will be preserved by the
            # in-flight rotation's splice-based writeback.
            needs_rotation = (
                len(self._history) > MAX_TURNS and not self._rotating
            )
            if needs_rotation:
                self._rotating = True
                summariser = self._summariser

        # Lock is now released.  Dispatch rotation work to background thread.
        if needs_rotation:
            self._executor.submit(self._rotate_background, summariser)

    def _rotate_background(
        self,
        summariser: Callable[[list[dict[str, str]]], str] | None,
    ) -> None:
        """Run LLM summarisation and disk write on a background thread.

        Design guarantees (all three must hold):
          (a) No turn appended during summarisation is lost.
          (b) Concurrent rotations cannot drop or double-process turns —
              ``_rotating`` ensures at most one rotation is in-flight; extra
              add_turns accumulate in ``_history`` and are swept up by the
              next rotation after this one clears the flag.
          (c) The summariser is called with the memory lock NOT held, so
              there is no risk of deadlock with ``_VRAM_LOCK``.

        Algorithm:
          1. Under lock: snapshot ``oldest = _history[:k]`` (turns to summarise)
             where ``k = max(0, len(_history) - RETAIN_RECENT)``.  Save ``k``.
          2. Release lock; call summariser with ``oldest`` snapshot — may be slow.
          3. Under lock: splice ``_history = [summary_entry] + _history[k:]``.
             Using the LIVE ``_history[k:]`` (not a stale recent snapshot) means
             any turns appended during step 2 are automatically preserved.
             Clear ``_rotating`` so the next ``add_turn`` can dispatch again.
        """
        # Step 1: snapshot the prefix to summarise.
        with self._lock:
            k = max(0, len(self._history) - RETAIN_RECENT)
            oldest = list(self._history[:k])

        # Step 2: call summariser outside the lock (may acquire _VRAM_LOCK).
        summary_entry: dict[str, str] | None = None
        if summariser is not None:
            try:
                summary_text = summariser(oldest)
                summary_entry = {
                    "role": "system",
                    "content": f"Summary of earlier conversation: {summary_text}",
                }
            except Exception:
                logger.warning(
                    "Memory summariser raised — falling back to truncation",
                    exc_info=True,
                )

        # Step 3: splice under lock, preserving any turns added since step 1.
        with self._lock:
            if summary_entry is not None:
                # Keep the live tail _history[k:] — includes any interim turns.
                self._history = [summary_entry] + self._history[k:]
            else:
                # Fallback (no summariser or summariser raised): discard oldest
                # using the live history so interim turns are still preserved.
                self._history = self._history[k:]
            # Clear flag so future add_turns can schedule the next rotation.
            self._rotating = False
            self._save_locked()

    def get_history(self) -> list[dict[str, str]]:
        """Return the conversation history with only ``role`` and ``content`` keys.

        The internal ``ts`` (timestamp) field is an implementation detail for
        retention bookkeeping and is stripped from the returned entries so that
        callers always receive the ``{role, content}`` shape they expect.
        """
        with self._lock:
            return [
                {"role": e["role"], "content": e["content"]}
                for e in self._history
            ]

    def purge_old_entries(self) -> int:
        """Remove entries older than ``max_age_s`` seconds (thread-safe).

        Uses ``self._now_fn()`` as the reference clock so that tests can inject
        a deterministic timestamp without relying on wall-clock sleeps.

        Returns the number of entries removed.  No-op (returns 0) when
        ``max_age_s`` is 0.0 (retention disabled).

        Must be called with the lock NOT held — acquires it internally.
        """
        if self._max_age_s <= 0.0:
            return 0
        with self._lock:
            return self._purge_locked()

    def _purge_locked(self) -> int:
        """Inner purge — must be called with ``_lock`` held."""
        if self._max_age_s <= 0.0:
            return 0
        cutoff = self._now_fn() - self._max_age_s
        before = len(self._history)
        # Entries without a 'ts' field (e.g. summary entries written before
        # this feature shipped) are preserved — they have no age information.
        self._history = [
            e for e in self._history
            if not isinstance(e.get("ts"), (int, float)) or e["ts"] >= cutoff
        ]
        removed = before - len(self._history)
        if removed:
            logger.debug(
                "Purged %d conversation entries older than %.0f s",
                removed,
                self._max_age_s,
            )
        return removed

    def prune(self, max_turns: int) -> None:
        """Keep only the last *max_turns* entries, discarding the oldest."""
        with self._lock:
            self._history = self._history[-max_turns:]

    def clear(self) -> None:
        """Wipe in-memory history and delete the persistence file if it exists."""
        with self._lock:
            self._history = []
            if self._file.exists():
                try:
                    self._file.unlink()
                except OSError:
                    logger.warning("Failed to delete persistence file: %s", self._file)

    def save(self) -> None:
        """Write current history to the JSON persistence file (thread-safe)."""
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        """Purge stale entries, then write ``_history`` to disk.

        Must be called with ``_lock`` held.  The purge is inline (no lock
        acquisition needed) because we already hold the lock.
        """
        self._purge_locked()
        try:
            with self._file.open("w", encoding="utf-8") as fh:
                json.dump(self._history, fh, indent=2, ensure_ascii=False)
        except OSError:
            logger.error("Failed to save conversation history to %s", self._file)

    def flush(self) -> None:
        """Block until all pending background rotation/save tasks have completed.

        Intended for tests and for ordered shutdown.  After this returns,
        ``get_history()`` reflects the result of every previously submitted
        ``add_turn()`` call.
        """
        # Submitting a no-op to the executor and waiting for it ensures all
        # previously submitted tasks have been processed (FIFO single-worker).
        future = self._executor.submit(lambda: None)
        future.result()

    def load(self) -> None:
        """Load history from the JSON file and rotate if needed.

        Malformed entries (not a dict, or missing/non-string role/content) are
        skipped with a warning rather than crashing the caller.
        """
        if not self._file.exists():
            return
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load %s (%s) — starting fresh", self._file, exc)
            with self._lock:
                self._history = []
            return

        if not isinstance(data, list):
            logger.warning("Unexpected format in %s — starting fresh", self._file)
            with self._lock:
                self._history = []
            return

        valid: list[dict[str, str]] = []
        skipped = 0
        for entry in data:
            if (
                isinstance(entry, dict)
                and isinstance(entry.get("role"), str)
                and isinstance(entry.get("content"), str)
            ):
                valid.append(entry)
            else:
                skipped += 1
                logger.warning(
                    "Skipping malformed history entry in %s: %r", self._file, entry
                )

        if skipped:
            logger.warning(
                "Loaded %d valid entries from %s; skipped %d malformed entries",
                len(valid),
                self._file,
                skipped,
            )

        with self._lock:
            self._history = valid
            # Purge age-expired entries before rotation so stale turns don't
            # count against the MAX_TURNS threshold and are not summarised.
            self._purge_locked()
            if len(self._history) > MAX_TURNS:
                # Inline synchronous rotation on load — no background needed here
                # because load() itself is called from a single-threaded startup
                # context before the inference threads are running.
                self._rotate_inline()

    def _rotate_inline(self) -> None:
        """Synchronous rotation used during ``load()``.

        Must be called with ``_lock`` held.
        """
        recent = self._history[-RETAIN_RECENT:]
        oldest = self._history[:-RETAIN_RECENT]
        # Summariser snapshot under lock (set_summariser cannot race here since
        # load() holds the lock).
        summariser = self._summariser

        if summariser is not None:
            try:
                # Temporarily release lock while calling the summariser so we
                # don't hold it across a potentially slow LLM call.
                self._lock.release()
                try:
                    summary_text = summariser(oldest)
                finally:
                    self._lock.acquire()
                summary_entry: dict[str, str] = {
                    "role": "system",
                    "content": f"Summary of earlier conversation: {summary_text}",
                }
                self._history = [summary_entry] + recent
            except Exception:
                logger.warning(
                    "Memory summariser raised during load — falling back to truncation",
                    exc_info=True,
                )
                self._history = recent
        else:
            self._history = recent
