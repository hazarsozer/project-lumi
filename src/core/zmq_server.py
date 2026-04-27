"""Compatibility shim — import EventBridge from src.core.event_bridge instead."""

from src.core.event_bridge import EventBridge as ZMQServer  # noqa: F401

__all__ = ["ZMQServer"]
