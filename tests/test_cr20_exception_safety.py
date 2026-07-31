"""
CR-20 / Issue #21 — Exception safety & resource cleanup.

Tests for:
1. utils.py bare except → KeyboardInterrupt / SystemExit propagate
2. Orchestrator RAG init failure is logged (caplog)
3. Orchestrator.__init__ partial failure cleans up acquired resources
4. IPCTransport releases socket on exception during start()
5. Hygiene: ears.py __future__ import; utils.py already fixed in same pass

All tests are written to be RED (failing) against the pre-fix code and
GREEN after the fixes land.
"""

from __future__ import annotations

import logging
import socket
import threading
from unittest.mock import MagicMock, patch

import pytest

from src.audio.speaker import SpeakerThread


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------


def _make_base_config(
    *,
    rag_enabled: bool = False,
    ipc_enabled: bool = False,
    ptt_enabled: bool = False,
):
    """Return a MagicMock config that satisfies Orchestrator.__init__ requirements."""
    cfg = MagicMock()
    cfg.rag.enabled = rag_enabled
    cfg.tools.enabled = False
    cfg.vision.enabled = False
    cfg.ipc.enabled = ipc_enabled
    cfg.audio.ptt_enabled = ptt_enabled
    cfg.audio.send_visemes = False
    cfg.llm.memory_dir = "/tmp/lumi_cr20_test_memory"
    cfg.llm.max_tokens = 5
    cfg.llm.temperature = 0.7
    cfg.llm.context_length = 512
    cfg.llm.inference_timeout_s = 30.0
    cfg.llm.tools_in_system_prompt = False
    cfg.observability.heartbeat_interval_s = 0.0
    return cfg


def _mock_speaker() -> MagicMock:
    sp = MagicMock(spec=SpeakerThread)
    sp.start = MagicMock()
    sp.stop = MagicMock()
    return sp


# ---------------------------------------------------------------------------
# Fix 1 — utils.py bare except
# ---------------------------------------------------------------------------


class TestUtilsBareExcept:
    """utils.py bare `except:` must be narrowed so signal exceptions propagate."""

    @pytest.mark.unit
    def test_keyboard_interrupt_propagates_from_play_ready_sound(self) -> None:
        """KeyboardInterrupt raised inside speaker.enqueue() must NOT be swallowed."""
        sp = MagicMock(spec=SpeakerThread)
        sp.enqueue.side_effect = KeyboardInterrupt

        from src.utils import play_ready_sound

        # Pre-fix: bare except catches KeyboardInterrupt → no raise → test PASSES
        # Post-fix: except Exception (or narrower) → KeyboardInterrupt escapes → test PASSES
        # So we assert it raises; this is RED until the fix is in place IF the
        # implementation uses bare except and hides it (the existing code already
        # uses `except Exception` but we double-check the escape hatch works).
        #
        # The test documents the requirement: play_ready_sound must NOT contain
        # a bare `except:` that suppresses BaseException subclasses.
        with pytest.raises(KeyboardInterrupt):
            play_ready_sound(sp)

    @pytest.mark.unit
    def test_system_exit_propagates_from_play_ready_sound(self) -> None:
        """SystemExit raised inside speaker.enqueue() must NOT be swallowed."""
        sp = MagicMock(spec=SpeakerThread)
        sp.enqueue.side_effect = SystemExit(0)

        from src.utils import play_ready_sound

        with pytest.raises(SystemExit):
            play_ready_sound(sp)

    @pytest.mark.unit
    def test_regular_exception_is_still_swallowed(self) -> None:
        """Regular Exception from speaker.enqueue() is caught and logged (unchanged)."""
        sp = MagicMock(spec=SpeakerThread)
        sp.enqueue.side_effect = RuntimeError("audio device disappeared")

        from src.utils import play_ready_sound

        # Must NOT raise — regular exceptions are intentionally swallowed here.
        play_ready_sound(sp)


# ---------------------------------------------------------------------------
# Fix 2 — Orchestrator RAG init logged
# ---------------------------------------------------------------------------


class TestOrchestratorRAGInitLogged:
    """RAG init failure must produce a visible log record instead of being silent."""

    @pytest.mark.unit
    def test_rag_init_failure_logs_warning_or_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """When RAG is enabled but DocumentStore raises, the error must be logged."""
        config = _make_base_config(rag_enabled=True)
        speaker = _mock_speaker()

        boom = RuntimeError("database is corrupted")

        with (
            patch("src.core.orchestrator.ConversationMemory") as mock_mem,
            patch("src.core.orchestrator.ModelLoader"),
            patch("src.core.orchestrator.PromptEngine"),
            patch("src.core.orchestrator.ReasoningRouter"),
            patch("src.core.orchestrator.ReflexRouter"),
            patch("src.core.orchestrator.ToolRegistry"),
            patch("src.core.orchestrator.ToolExecutor"),
            patch("src.core.orchestrator.LLMInferenceDispatcher"),
            # Make DocumentStore raise on construction
            patch("src.rag.store.DocumentStore", side_effect=boom),
            patch("src.rag.retriever.RAGRetriever"),
            caplog.at_level(logging.WARNING, logger="src.core.orchestrator"),
        ):
            mock_mem.return_value.load = MagicMock()

            from src.core.orchestrator import Orchestrator

            orch = Orchestrator(config, speaker=speaker)

        # Must log at WARNING or ERROR level with context about the failure
        rag_log_records = [
            r for r in caplog.records
            if "rag" in r.message.lower() or "rag" in r.name.lower()
        ]
        assert rag_log_records, (
            "Expected at least one log record mentioning RAG failure; got none. "
            f"All records: {[r.message for r in caplog.records]}"
        )
        assert any(r.levelno >= logging.WARNING for r in rag_log_records), (
            "RAG failure log record must be WARNING or higher"
        )

    @pytest.mark.unit
    def test_rag_init_failure_disables_rag_runtime(self) -> None:
        """When RAG init fails, _rag_runtime_enabled must be set to False."""
        config = _make_base_config(rag_enabled=True)
        speaker = _mock_speaker()

        with (
            patch("src.core.orchestrator.ConversationMemory") as mock_mem,
            patch("src.core.orchestrator.ModelLoader"),
            patch("src.core.orchestrator.PromptEngine"),
            patch("src.core.orchestrator.ReasoningRouter"),
            patch("src.core.orchestrator.ReflexRouter"),
            patch("src.core.orchestrator.ToolRegistry"),
            patch("src.core.orchestrator.ToolExecutor"),
            patch("src.core.orchestrator.LLMInferenceDispatcher"),
            patch("src.rag.store.DocumentStore", side_effect=RuntimeError("db broken")),
            patch("src.rag.retriever.RAGRetriever"),
        ):
            mock_mem.return_value.load = MagicMock()

            from src.core.orchestrator import Orchestrator

            orch = Orchestrator(config, speaker=speaker)

        assert orch._rag_runtime_enabled is False


# ---------------------------------------------------------------------------
# Fix 3 — Orchestrator.__init__ exception safety / resource cleanup
# ---------------------------------------------------------------------------


class TestOrchestratorInitExceptionSafety:
    """Partial __init__ failure must not leave acquired resources dangling.

    Injection-point rationale (orchestrator.__init__ resource-acquisition order):
      1. ConversationMemory.load()      ← no thread/socket resource
      2. ReasoningRouter / ReflexRouter ← no thread/socket resource
      3. ToolRegistry / ToolExecutor    ← no thread/socket resource
      4. LLMInferenceDispatcher         ← no thread yet (lazy)
      5. SpeakerThread.start()          ← THREAD acquired here
      6. PTTListener (if ptt_enabled)   ← optional thread
      7. register_handler() (no-fail)
      8. EventBridge (if ipc_enabled)   ← start() spawns a thread/socket

    Tests force failures at step 8 (after step 5 already ran) so the speaker
    thread is "live" when the failure occurs and must be stopped on cleanup.
    """

    @pytest.mark.unit
    def test_speaker_stopped_when_init_fails_after_speaker_start(self) -> None:
        """If __init__ fails after self._speaker.start(), speaker.stop() is called.

        EventBridge.start() is the injection point because it is the first
        fallible step that runs AFTER the speaker thread has been started.
        With ipc_enabled=True the orchestrator creates and starts an EventBridge;
        making EventBridge.__init__ raise simulates a late-init failure.
        """
        config = _make_base_config(ipc_enabled=True)
        speaker = _mock_speaker()

        with (
            patch("src.core.orchestrator.ConversationMemory") as mock_mem,
            patch("src.core.orchestrator.ModelLoader"),
            patch("src.core.orchestrator.PromptEngine"),
            patch("src.core.orchestrator.ReasoningRouter"),
            patch("src.core.orchestrator.ReflexRouter"),
            patch("src.core.orchestrator.ToolRegistry"),
            patch("src.core.orchestrator.ToolExecutor"),
            patch("src.core.orchestrator.LLMInferenceDispatcher"),
            # EventBridge is created AFTER speaker.start() — force it to fail.
            patch(
                "src.core.orchestrator.EventBridge",
                side_effect=RuntimeError("bridge init failed"),
            ),
        ):
            mock_mem.return_value.load = MagicMock()

            from src.core.orchestrator import Orchestrator

            with pytest.raises(RuntimeError, match="bridge init failed"):
                Orchestrator(config, speaker=speaker)

        # The speaker was started before EventBridge — it must be stopped on cleanup.
        speaker.stop.assert_called_once()

    @pytest.mark.unit
    def test_speaker_not_stopped_when_init_fails_before_speaker_start(self) -> None:
        """If __init__ fails before speaker.start(), speaker.stop() is NOT called.

        This is the inverse guard: we must not call stop() on resources that
        were never started.  ToolRegistry is instantiated BEFORE speaker.start(),
        so making it raise means the speaker was never started.
        """
        config = _make_base_config()
        speaker = _mock_speaker()

        with (
            patch("src.core.orchestrator.ConversationMemory") as mock_mem,
            patch("src.core.orchestrator.ModelLoader"),
            patch("src.core.orchestrator.PromptEngine"),
            patch("src.core.orchestrator.ReasoningRouter"),
            patch("src.core.orchestrator.ReflexRouter"),
            # ToolRegistry is created BEFORE speaker.start() — fail here.
            patch(
                "src.core.orchestrator.ToolRegistry",
                side_effect=RuntimeError("registry init boom"),
            ),
            patch("src.core.orchestrator.ToolExecutor"),
            patch("src.core.orchestrator.LLMInferenceDispatcher"),
        ):
            mock_mem.return_value.load = MagicMock()

            from src.core.orchestrator import Orchestrator

            with pytest.raises(RuntimeError, match="registry init boom"):
                Orchestrator(config, speaker=speaker)

        # Speaker was never started — stop() must NOT have been called.
        speaker.stop.assert_not_called()

    @pytest.mark.unit
    def test_event_bridge_stopped_when_init_fails_after_bridge_start(self) -> None:
        """If __init__ fails after EventBridge.start(), EventBridge.stop() is called.

        Injection point: the `on_tts_stop` attribute on the EventBridge mock
        raises AttributeError when accessed inside the `register_handler()` call
        that immediately follows `_bridge_started = True`.  This simulates a
        late-init failure while the bridge thread is already live.
        """
        config = _make_base_config(ipc_enabled=True)
        speaker = _mock_speaker()

        mock_bridge = MagicMock()
        mock_bridge.start = MagicMock()
        mock_bridge.stop = MagicMock()

        # Make the first attribute access after EventBridge.start() raise.
        # `on_tts_stop` is accessed on line:
        #   self.register_handler(SpeechCompletedEvent, self._event_bridge.on_tts_stop)
        # which is the first handler registered from the auto-created bridge path.
        type(mock_bridge).on_tts_stop = property(
            fget=lambda self: (_ for _ in ()).throw(RuntimeError("tts_stop wiring boom"))
        )

        with (
            patch("src.core.orchestrator.ConversationMemory") as mock_mem,
            patch("src.core.orchestrator.ModelLoader"),
            patch("src.core.orchestrator.PromptEngine"),
            patch("src.core.orchestrator.ReasoningRouter"),
            patch("src.core.orchestrator.ReflexRouter"),
            patch("src.core.orchestrator.ToolRegistry"),
            patch("src.core.orchestrator.ToolExecutor"),
            patch("src.core.orchestrator.LLMInferenceDispatcher"),
            patch("src.core.orchestrator.EventBridge", return_value=mock_bridge),
        ):
            mock_mem.return_value.load = MagicMock()

            from src.core.orchestrator import Orchestrator

            with pytest.raises(RuntimeError, match="tts_stop wiring boom"):
                Orchestrator(config, speaker=speaker)

        # EventBridge was started; it must be stopped on cleanup.
        mock_bridge.stop.assert_called_once()
        # Speaker was started before EventBridge — must also be stopped.
        speaker.stop.assert_called_once()


# ---------------------------------------------------------------------------
# Fix 4 — IPCTransport socket released on exception
# ---------------------------------------------------------------------------


class TestIPCTransportSocketCleanup:
    """IPCTransport.start() must close the server socket on ALL exception paths."""

    @pytest.mark.unit
    def test_socket_released_when_bind_raises(self) -> None:
        """If socket.bind() raises, the created socket is closed (not leaked)."""
        from src.core.ipc_transport import IPCTransport

        transport = IPCTransport("127.0.0.1", 0)

        created_socks: list[MagicMock] = []

        def _make_socket(family, kind):
            # Don't use spec=socket.socket here because socket.socket has already
            # been replaced by the patch — spec-ing a Mock raises InvalidSpecError.
            s = MagicMock()
            s.bind.side_effect = OSError("address already in use")
            created_socks.append(s)
            return s

        with patch("src.core.ipc_transport.socket.socket", side_effect=_make_socket):
            with pytest.raises(OSError, match="address already in use"):
                transport.start()

        assert created_socks, "Expected socket.socket() to be called"
        # All created sockets must have been closed.
        for sock in created_socks:
            sock.close.assert_called()

    @pytest.mark.unit
    def test_socket_released_when_listen_raises(self) -> None:
        """If socket.listen() raises after bind(), the socket is still closed."""
        from src.core.ipc_transport import IPCTransport

        transport = IPCTransport("127.0.0.1", 0)

        created_socks: list[MagicMock] = []

        def _make_socket(family, kind):
            s = MagicMock()
            s.bind.return_value = None
            s.listen.side_effect = OSError("listen failed")
            created_socks.append(s)
            return s

        with patch("src.core.ipc_transport.socket.socket", side_effect=_make_socket):
            with pytest.raises(OSError, match="listen failed"):
                transport.start()

        assert created_socks
        for sock in created_socks:
            sock.close.assert_called()

    @pytest.mark.unit
    def test_server_sock_is_none_after_failed_start(self) -> None:
        """After a failed start(), _server_sock must be None (no dangling reference)."""
        from src.core.ipc_transport import IPCTransport

        transport = IPCTransport("127.0.0.1", 0)

        def _make_socket(family, kind):
            s = MagicMock()
            s.bind.side_effect = OSError("port in use")
            return s

        with patch("src.core.ipc_transport.socket.socket", side_effect=_make_socket):
            with pytest.raises(OSError):
                transport.start()

        assert transport._server_sock is None, (
            "_server_sock must be None after a failed start() to prevent dangling reference"
        )


# ---------------------------------------------------------------------------
# Fix 5 — Hygiene: ears.py __future__ annotations
# ---------------------------------------------------------------------------


class TestEarsModuleHygiene:
    """ears.py must have `from __future__ import annotations` for consistency."""

    @pytest.mark.unit
    def test_ears_module_has_future_annotations(self) -> None:
        """ears.py should import annotations from __future__ for style consistency."""
        import ast
        import inspect
        import src.audio.ears as ears_module

        source = inspect.getsource(ears_module)
        tree = ast.parse(source)

        has_future_annotations = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "__future__":
                    for alias in node.names:
                        if alias.name == "annotations":
                            has_future_annotations = True
                            break

        assert has_future_annotations, (
            "ears.py is missing `from __future__ import annotations`; "
            "all source modules should be consistent."
        )
