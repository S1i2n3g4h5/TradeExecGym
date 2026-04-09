"""Compatibility shim for legacy ``tasks.factory`` imports."""

from server.tasks import get_task

__all__ = ["get_task"]
