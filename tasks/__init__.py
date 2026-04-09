"""Compatibility shim for legacy imports.

The project's task implementation lives in ``server.tasks``, but several
local validators still import ``tasks`` or ``tasks.factory``.
"""

from server.tasks import *  # noqa: F401,F403

