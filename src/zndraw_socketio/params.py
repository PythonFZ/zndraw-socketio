"""Dependency injection parameters for zndraw-socketio.

Provides a lightweight Depends compatible with FastAPI's Depends.
When FastAPI is installed, its Depends is used directly instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class _DependsBase:
    """Internal sentinel class for dependency declarations.

    Use the :func:`Depends` function instead of instantiating directly.
    """

    dependency: Optional[Callable[..., Any]] = None
    use_cache: bool = True


def Depends(  # noqa: N802
    dependency: Optional[Callable[..., Any]] = None,
    *,
    use_cache: bool = True,
) -> Any:
    """Declare a dependency for a Socket.IO event handler.

    Returns ``Any`` so that ``param: str = Depends(get_value)`` passes
    type-checking — the same convention FastAPI uses.

    When FastAPI is installed, use ``fastapi.Depends`` instead — it is
    automatically recognised by the wrapper.

    Example::

        from zndraw_socketio.params import Depends

        async def get_redis():
            return Redis()

        @tsio.on(MyEvent)
        async def handler(sid: str, data: MyEvent, redis: Redis = Depends(get_redis)):
            ...
    """
    return _DependsBase(dependency=dependency, use_cache=use_cache)
