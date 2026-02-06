"""Typed wrapper for python-socketio with Pydantic validation.

This module provides a thin wrapper around socketio instances that adds:
- Typed emit/call methods with automatic event name derivation
- Handler registration with Pydantic validation from type hints
- Support for union and discriminated union response types
- FastAPI integration via Depends()
- Exception handlers for Socket.IO events

Note on Union Types (PEP 747):
    The response_model parameter uses TypeForm[T] from PEP 747 for type inference.
    See: https://peps.python.org/pep-0747/
"""

from __future__ import annotations

import asyncio
import inspect
import re
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import (
    Annotated,
    Any,
    Callable,
    Type,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from pydantic import BaseModel, TypeAdapter, validate_call
from pydantic_core import to_jsonable_python
from socketio import (
    AsyncClient,
    AsyncServer,
    AsyncSimpleClient,
    Client,
    Server,
    SimpleClient,
)
from typing_extensions import TypeForm

try:
    from fastapi.params import Depends as _DependsClass
except ImportError:
    from zndraw_socketio.params import _DependsBase as _DependsClass

try:
    from fastapi import Request
except ImportError:
    Request = None  # type: ignore[assignment, misc]

T = TypeVar("T")


# =============================================================================
# Event Context
# =============================================================================


@dataclass
class EventContext:
    """Context for Socket.IO event handler exceptions.

    Provides full context about the event that raised the exception,
    similar to FastAPI's Request object for exception handlers.

    Attributes:
        sid: Session ID of the client that triggered the event.
        event: Event name that raised the exception.
        namespace: Namespace the event was sent to (e.g., "/" or "/chat").
        data: Original data sent by the client.
        sio: Wrapper instance for emitting events, accessing rooms, etc.
    """

    sid: str
    event: str
    namespace: str
    data: Any
    sio: "AsyncServerWrapper"


@dataclass
class SioRequest:
    """Minimal Request-compatible shim for Socket.IO dependency injection.

    Provides ``request.app`` access so that FastAPI-style dependencies
    like ``def get_db(request: Request)`` can be reused in Socket.IO
    handlers. Unsupported Request attributes (url, headers, etc.)
    raise ``AttributeError`` naturally.
    """

    app: Any


# =============================================================================
# Event Name Helper
# =============================================================================


def get_event_name(model: Type[BaseModel] | BaseModel) -> str:
    """Get event name from a Pydantic model class or instance.

    Checks for an `event_name` class attribute first, then falls back to
    converting the class name from PascalCase to snake_case.

    Args:
        model: A Pydantic model class or instance.

    Returns:
        The event name string.

    Examples:
        >>> class Ping(BaseModel):
        ...     message: str
        >>> get_event_name(Ping)
        'ping'

        >>> from typing import ClassVar
        >>> class CustomEvent(BaseModel):
        ...     event_name: ClassVar[str] = "my_custom_event"
        ...     data: str
        >>> get_event_name(CustomEvent)
        'my_custom_event'
    """
    cls = model if isinstance(model, type) else type(model)

    if hasattr(cls, "event_name"):
        return cls.event_name  # type: ignore[return-value]

    # Convert PascalCase to snake_case
    name = cls.__name__
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


# =============================================================================
# Shared Helper Functions
# =============================================================================


def _resolve_emit_args(event: str | BaseModel, data: Any = None) -> tuple[str, Any]:
    """Resolve event name and payload from emit arguments.

    Args:
        event: Either a string event name or a BaseModel instance.
        data: Optional data payload (used when event is a string).

    Returns:
        Tuple of (event_name, serialized_payload).

    Raises:
        TypeError: If event is a BaseModel and data is also provided.
    """
    if isinstance(event, BaseModel):
        if data is not None:
            raise TypeError(
                "Cannot provide both a BaseModel instance as event and a data argument. "
                "Use emit(MyModel(...)) or emit('event_name', data=...), not both."
            )
        return get_event_name(event), to_jsonable_python(event)
    return event, to_jsonable_python(data) if isinstance(data, BaseModel) else data


def _validate_response(response: Any, response_model: TypeForm[T] | None) -> T | Any:
    """Validate response against response_model if provided.

    Args:
        response: The raw response from socketio.
        response_model: Optional type form (single type, union, or Annotated).

    Returns:
        Validated response if response_model is provided, otherwise raw response.
    """
    if response_model is not None:
        return TypeAdapter(response_model).validate_python(response)
    return response


def _extract_dependencies(handler: Callable) -> dict[str, Callable]:
    """Extract Depends() declarations from a handler's type hints and defaults.

    Checks both ``Annotated[T, Depends(...)]`` metadata and plain default values.

    Args:
        handler: The event handler function.

    Returns:
        Dict mapping parameter name to the dependency callable.
    """
    deps: dict[str, Callable] = {}

    # Check Annotated metadata: param: Annotated[Redis, Depends(get_redis)]
    try:
        hints = get_type_hints(handler, include_extras=True)
    except Exception:
        hints = {}
    for name, hint in hints.items():
        if get_origin(hint) is Annotated:
            for metadata in get_args(hint)[1:]:
                if (
                    isinstance(metadata, _DependsClass)
                    and metadata.dependency is not None
                ):
                    deps[name] = metadata.dependency
                    break

    # Check default values: param: Redis = Depends(get_redis)
    sig = inspect.signature(handler)
    for name, param in sig.parameters.items():
        if name not in deps and isinstance(param.default, _DependsClass):
            if param.default.dependency is not None:
                deps[name] = param.default.dependency

    return deps


async def _resolve_dependencies(
    deps: dict[str, Callable],
    *,
    app: Any = None,
    stack: AsyncExitStack,
) -> dict[str, Any]:
    """Resolve dependency callables into their values.

    Supports sync/async callables and sync/async generators.
    Generator dependencies are wrapped as context managers and registered
    with the ``AsyncExitStack`` so teardown runs automatically when the
    stack exits (following FastAPI's pattern).

    If a dependency's signature includes a parameter typed as
    ``Request``, a ``SioRequest`` shim is injected automatically.

    Args:
        deps: Dict mapping parameter name to dependency callable.
        app: Optional FastAPI app instance for Request injection.
        stack: AsyncExitStack for managing generator dependency lifecycle.

    Returns:
        Dict mapping parameter name to resolved value.
    """
    resolved: dict[str, Any] = {}

    for name, dep_fn in deps.items():
        # Build kwargs: inject SioRequest for Request-typed params
        kwargs: dict[str, Any] = {}
        if Request is not None:
            for pname, param in inspect.signature(dep_fn).parameters.items():
                if param.annotation is Request:
                    kwargs[pname] = SioRequest(app=app)

        if asyncio.iscoroutinefunction(dep_fn):
            resolved[name] = await dep_fn(**kwargs)
        elif inspect.isasyncgenfunction(dep_fn):
            cm = asynccontextmanager(dep_fn)(**kwargs)
            resolved[name] = await stack.enter_async_context(cm)
        elif inspect.isgeneratorfunction(dep_fn):
            cm = contextmanager(dep_fn)(**kwargs)
            resolved[name] = stack.enter_context(cm)
        else:
            resolved[name] = dep_fn(**kwargs)

    return resolved


def _create_async_handler_wrapper(
    handler: Callable, *, app_getter: Callable[[], Any] | None = None
) -> Callable:
    """Wrap async handler with Pydantic validation, DI, and serialization.

    Uses pydantic's validate_call to validate input arguments and return value
    based on the function's type annotations. Resolves any Depends()
    dependencies before calling the handler. Generator dependencies are
    managed via AsyncExitStack for automatic cleanup.

    Args:
        handler: The async event handler function.
        app_getter: Optional callable that returns the FastAPI app at event
            time. Defers app resolution so that handlers registered before
            the app exists still work.

    Returns:
        Wrapped async handler with validation and dependency injection.
    """
    deps = _extract_dependencies(handler)

    if deps:

        @wraps(handler)
        async def _dep_handler(*args: Any, **kwargs: Any) -> Any:
            async with AsyncExitStack() as stack:
                app = app_getter() if app_getter is not None else None
                resolved = await _resolve_dependencies(deps, app=app, stack=stack)
                kwargs.update(resolved)
                return await handler(*args, **kwargs)

        sig = inspect.signature(handler)
        _dep_handler.__signature__ = sig.replace(
            parameters=[p for n, p in sig.parameters.items() if n not in deps]
        )
        _dep_handler.__annotations__ = {
            k: v for k, v in handler.__annotations__.items() if k not in deps
        }
        validated = validate_call(validate_return=True)(_dep_handler)
    else:
        validated = validate_call(validate_return=True)(handler)

    @wraps(handler)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = await validated(*args, **kwargs)
        if isinstance(result, BaseModel):
            return to_jsonable_python(result)
        return result

    return wrapper


def _create_sync_handler_wrapper(handler: Callable) -> Callable:
    """Wrap sync handler with Pydantic validation and serialization.

    Uses pydantic's validate_call to validate input arguments and return value
    based on the function's type annotations.

    Args:
        handler: The sync event handler function.

    Returns:
        Wrapped sync handler with validation.
    """
    validated = validate_call(validate_return=True)(handler)

    @wraps(handler)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = validated(*args, **kwargs)
        if isinstance(result, BaseModel):
            return to_jsonable_python(result)
        return result

    return wrapper


# =============================================================================
# Async Client Wrapper
# =============================================================================


class AsyncClientWrapper:
    """Typed wrapper for socketio.AsyncClient.

    Provides typed emit, call, and on methods while passing through all other
    attributes to the underlying AsyncClient instance.

    Example:
        >>> import socketio
        >>> from zndraw_socketio import wrap
        >>> sio = wrap(socketio.AsyncClient())
        >>> await sio.emit(Ping(message="hello"))
    """

    def __init__(self, sio: AsyncClient) -> None:
        """Initialize wrapper with an AsyncClient instance.

        Args:
            sio: The socketio AsyncClient to wrap.
        """
        self._sio = sio
        self._app: Any = None

    @property
    def app(self) -> Any:
        return self._app

    @app.setter
    def app(self, value: Any) -> None:
        self._app = value

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying socketio instance."""
        return getattr(self._sio, name)

    # emit overloads
    @overload
    async def emit(
        self,
        event: BaseModel,
        **kwargs: Any,
    ) -> None: ...

    @overload
    async def emit(
        self,
        event: str,
        data: Any = None,
        **kwargs: Any,
    ) -> None: ...

    async def emit(
        self,
        event: str | BaseModel,
        data: Any = None,
        **kwargs: Any,
    ) -> None:
        """Emit an event to the server.

        Args:
            event: Either a string event name or a BaseModel instance.
                If BaseModel, event name is derived from the class name.
            data: Optional data payload (used when event is a string).
            **kwargs: Additional arguments passed to socketio's emit.
        """
        event_name, payload = _resolve_emit_args(event, data)
        await self._sio.emit(event_name, payload, **kwargs)

    @overload
    async def call(
        self,
        event: BaseModel,
        *,
        response_model: TypeForm[T],
        **kwargs: Any,
    ) -> T: ...

    @overload
    async def call(
        self,
        event: BaseModel,
        **kwargs: Any,
    ) -> Any: ...

    @overload
    async def call(
        self,
        event: str,
        data: Any = None,
        *,
        response_model: TypeForm[T],
        **kwargs: Any,
    ) -> T: ...

    @overload
    async def call(
        self,
        event: str,
        data: Any = None,
        **kwargs: Any,
    ) -> Any: ...

    async def call(
        self,
        event: str | BaseModel,
        data: Any = None,
        *,
        response_model: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Emit an event and wait for a response.

        Args:
            event: Either a string event name or a BaseModel instance.
            data: Optional data payload (used when event is a string).
            response_model: Optional type to validate response against (PEP 747 TypeForm).
            **kwargs: Additional arguments passed to socketio's call.

        Returns:
            Validated response if response_model is provided, otherwise raw response.
        """
        event_name, payload = _resolve_emit_args(event, data)
        response = await self._sio.call(event_name, payload, **kwargs)
        return _validate_response(response, response_model)

    def on(
        self,
        event: str | Type[BaseModel],
        handler: Callable | None = None,
        **kwargs: Any,
    ) -> Callable[[Callable], Callable] | Callable:
        """Register an event handler.

        Can be used as a decorator or called directly with a handler.

        Args:
            event: Either a string event name or a BaseModel class.
                If BaseModel class, event name is derived from the class name.
            handler: Optional handler function (if not using as decorator).
            **kwargs: Additional arguments passed to socketio's on (e.g., namespace).

        Returns:
            Decorator that registers the handler with validation, or the handler
            if called directly.

        Example:
            >>> @tsio.on(Ping)
            ... async def handle_ping(data: Ping) -> Pong:
            ...     return Pong(reply=data.message)

            >>> @tsio.on(Ping, namespace='/chat')
            ... async def handle_ping(data: Ping) -> Pong:
            ...     return Pong(reply=data.message)
        """
        if isinstance(event, type) and issubclass(event, BaseModel):
            event_name = get_event_name(event)
        else:
            event_name = event

        def decorator(handler: Callable) -> Callable:
            wrapped = _create_async_handler_wrapper(
                handler, app_getter=lambda: self._app
            )
            self._sio.on(event_name, wrapped, **kwargs)
            return handler

        if handler is not None:
            return decorator(handler)
        return decorator

    def event(self, handler: Callable | None = None, **kwargs: Any) -> Callable:
        """Register an event handler using the function name as the event name.

        This decorator uses the function name directly as the event name and
        wraps the handler with Pydantic validation based on type annotations.

        Args:
            handler: The event handler function.
            **kwargs: Additional arguments passed to socketio's on (e.g., namespace).

        Returns:
            The original handler (unmodified).

        Example:
            >>> @tsio.event
            ... async def ping(data: Ping) -> Pong:
            ...     return Pong(reply=data.message)

            >>> @tsio.event(namespace='/chat')
            ... async def ping(data: Ping) -> Pong:
            ...     return Pong(reply=data.message)
        """

        def decorator(handler: Callable) -> Callable:
            event_name = handler.__name__
            wrapped = _create_async_handler_wrapper(
                handler, app_getter=lambda: self._app
            )
            self._sio.on(event_name, wrapped, **kwargs)
            return handler

        if handler is not None:
            return decorator(handler)
        return decorator


# =============================================================================
# Async Server Wrapper
# =============================================================================


class AsyncServerWrapper:
    """Typed wrapper for socketio.AsyncServer.

    Provides typed emit, call, and on methods while passing through all other
    attributes to the underlying AsyncServer instance.

    Also provides FastAPI integration via Depends() support and exception
    handlers for Socket.IO events.

    Example:
        >>> import socketio
        >>> from fastapi import FastAPI, Depends
        >>> from zndraw_socketio import wrap
        >>> from typing import Annotated
        >>>
        >>> app = FastAPI()
        >>> tsio = wrap(socketio.AsyncServer(async_mode='asgi'))
        >>>
        >>> # Create type alias for dependency injection
        >>> SioServer = Annotated[AsyncServerWrapper, Depends(tsio)]
        >>>
        >>> @app.post("/notify")
        ... async def notify(server: SioServer):
        ...     await server.emit("notification", {"msg": "hello"})
        ...     return {"status": "sent"}
        >>>
        >>> @tsio.exception_handler(ValueError)
        ... async def handle_error(ctx: EventContext, exc: ValueError):
        ...     return {"error": str(exc)}
        >>>
        >>> # Create combined ASGI app
        >>> combined_app = socketio.ASGIApp(tsio, app)
    """

    def __init__(self, sio: AsyncServer) -> None:
        """Initialize wrapper with an AsyncServer instance.

        Args:
            sio: The socketio AsyncServer to wrap.
        """
        self._sio = sio
        self._app: Any = None
        # {namespace: {ExceptionType: handler_fn}}
        # namespace=None means global handler
        self._exception_handlers: dict[str | None, dict[type[Exception], Callable]] = {}

    @property
    def app(self) -> Any:
        return self._app

    @app.setter
    def app(self, value: Any) -> None:
        self._app = value

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying socketio instance."""
        return getattr(self._sio, name)

    def __call__(self) -> AsyncServerWrapper:
        """Return the wrapper instance for FastAPI Depends().

        This makes the wrapper callable, allowing it to be used directly
        with FastAPI's dependency injection. Returning the wrapper (rather
        than the raw server) ensures that routes have access to typed
        emit() and call() methods.

        Returns:
            The AsyncServerWrapper instance.

        Example:
            >>> from typing import Annotated
            >>> from fastapi import Depends
            >>>
            >>> SioServer = Annotated[AsyncServerWrapper, Depends(tsio)]
            >>>
            >>> @app.get("/emit")
            ... async def emit(sio: SioServer):
            ...     await sio.emit(MyModel(data="hello"))
        """
        return self

    def exception_handler(
        self,
        exc_type: type[Exception],
        namespace: str | None = None,
    ) -> Callable[[Callable], Callable]:
        """Register an exception handler for Socket.IO events.

        Similar to FastAPI's `@app.exception_handler()` decorator, this allows
        catching exceptions from event handlers and returning structured responses.

        Args:
            exc_type: The exception type to handle.
            namespace: Optional namespace to scope the handler to.
                       If None, the handler applies globally to all namespaces.

        Returns:
            Decorator that registers the async handler function.

        Example:
            >>> @tsio.exception_handler(ValidationError)
            ... async def handle_validation(ctx: EventContext, exc: ValidationError):
            ...     return {"error": "validation_error", "details": exc.errors()}
            >>>
            >>> @tsio.exception_handler(ValueError, namespace="/chat")
            ... async def handle_chat_error(ctx: EventContext, exc: ValueError):
            ...     return {"error": "chat_error", "message": str(exc)}
        """

        def decorator(handler: Callable) -> Callable:
            if namespace not in self._exception_handlers:
                self._exception_handlers[namespace] = {}
            self._exception_handlers[namespace][exc_type] = handler
            return handler

        return decorator

    def _find_exception_handler(
        self,
        exc: Exception,
        namespace: str,
    ) -> Callable | None:
        """Find the most specific exception handler for the given exception.

        Resolution order (most specific first):
        1. Namespace-specific handler for exact exception type
        2. Namespace-specific handler for parent exception type (MRO order)
        3. Global handler for exact exception type
        4. Global handler for parent exception type (MRO order)

        Args:
            exc: The exception that was raised.
            namespace: The namespace where the event was triggered.

        Returns:
            The handler function if found, None otherwise.
        """
        exc_type = type(exc)

        # Check namespace-specific handlers first, then global (None)
        for ns in (namespace, None):
            handlers = self._exception_handlers.get(ns, {})

            # Walk MRO to find matching handler
            for cls in exc_type.__mro__:
                if cls in handlers:
                    return handlers[cls]

        return None

    # emit overloads
    @overload
    async def emit(
        self,
        event: BaseModel,
        **kwargs: Any,
    ) -> None: ...

    @overload
    async def emit(
        self,
        event: str,
        data: Any = None,
        **kwargs: Any,
    ) -> None: ...

    async def emit(
        self,
        event: str | BaseModel,
        data: Any = None,
        **kwargs: Any,
    ) -> None:
        """Emit an event to connected clients.

        Args:
            event: Either a string event name or a BaseModel instance.
            data: Optional data payload (used when event is a string).
            **kwargs: Additional arguments passed to socketio's emit (to, room, etc).
        """
        event_name, payload = _resolve_emit_args(event, data)
        await self._sio.emit(event_name, payload, **kwargs)

    # call overloads (see PEP 747 note in AsyncClientWrapper)
    @overload
    async def call(
        self,
        event: BaseModel,
        *,
        response_model: TypeForm[T],
        **kwargs: Any,
    ) -> T: ...

    @overload
    async def call(
        self,
        event: BaseModel,
        **kwargs: Any,
    ) -> Any: ...

    @overload
    async def call(
        self,
        event: str,
        data: Any = None,
        *,
        response_model: TypeForm[T],
        **kwargs: Any,
    ) -> T: ...

    @overload
    async def call(
        self,
        event: str,
        data: Any = None,
        **kwargs: Any,
    ) -> Any: ...

    async def call(
        self,
        event: str | BaseModel,
        data: Any = None,
        *,
        response_model: Any = None,
        **kwargs: Any,
    ) -> T | Any:
        """Emit an event and wait for a response from a client.

        Args:
            event: Either a string event name or a BaseModel instance.
            data: Optional data payload (used when event is a string).
            response_model: Optional Pydantic model type to validate response.
            **kwargs: Additional arguments passed to socketio's call (to, sid, etc).

        Returns:
            Validated response if response_model is provided, otherwise raw response.
        """
        event_name, payload = _resolve_emit_args(event, data)
        response = await self._sio.call(event_name, payload, **kwargs)
        return _validate_response(response, response_model)

    def on(
        self,
        event: str | Type[BaseModel],
        handler: Callable | None = None,
        **kwargs: Any,
    ) -> Callable[[Callable], Callable] | Callable:
        """Register an event handler with exception handling support.

        Wraps handlers with Pydantic validation and exception handling that
        routes to registered exception handlers.

        Args:
            event: Either a string event name or a BaseModel class.
            handler: Optional handler function (if not using as decorator).
            **kwargs: Additional arguments passed to socketio's on (e.g., namespace).

        Returns:
            Decorator that registers the handler with validation and exception handling.
        """
        if isinstance(event, type) and issubclass(event, BaseModel):
            event_name = get_event_name(event)
        else:
            event_name = event

        namespace = kwargs.get("namespace", "/")

        def decorator(handler: Callable) -> Callable:
            # Wrap with validation
            validated_wrapped = _create_async_handler_wrapper(
                handler, app_getter=lambda: self._app
            )

            # Add exception handling wrapper
            @wraps(validated_wrapped)
            async def exc_wrapped(sid: str, *args: Any, **kw: Any) -> Any:
                data = args[0] if args else kw.get("data")
                try:
                    return await validated_wrapped(sid, *args, **kw)
                except Exception as exc:
                    exc_handler = self._find_exception_handler(exc, namespace)
                    if exc_handler is None:
                        raise
                    ctx = EventContext(
                        sid=sid,
                        event=event_name,
                        namespace=namespace,
                        data=data,
                        sio=self,
                    )
                    result = await exc_handler(ctx, exc)
                    if isinstance(result, BaseModel):
                        return to_jsonable_python(result)
                    return result

            self._sio.on(event_name, exc_wrapped, **kwargs)
            return handler

        if handler is not None:
            return decorator(handler)
        return decorator

    def event(self, handler: Callable | None = None, **kwargs: Any) -> Callable:
        """Register an event handler using the function name as the event name.

        Args:
            handler: The event handler function.
            **kwargs: Additional arguments passed to socketio's on (e.g., namespace).

        Returns:
            The original handler (unmodified).
        """
        namespace = kwargs.get("namespace", "/")

        def decorator(handler: Callable) -> Callable:
            event_name = handler.__name__
            # Wrap with validation
            validated_wrapped = _create_async_handler_wrapper(
                handler, app_getter=lambda: self._app
            )

            # Add exception handling wrapper
            @wraps(validated_wrapped)
            async def exc_wrapped(sid: str, *args: Any, **kw: Any) -> Any:
                data = args[0] if args else kw.get("data")
                try:
                    return await validated_wrapped(sid, *args, **kw)
                except Exception as exc:
                    exc_handler = self._find_exception_handler(exc, namespace)
                    if exc_handler is None:
                        raise
                    ctx = EventContext(
                        sid=sid,
                        event=event_name,
                        namespace=namespace,
                        data=data,
                        sio=self,
                    )
                    result = await exc_handler(ctx, exc)
                    if isinstance(result, BaseModel):
                        return to_jsonable_python(result)
                    return result

            self._sio.on(event_name, exc_wrapped, **kwargs)
            return handler

        if handler is not None:
            return decorator(handler)
        return decorator


# =============================================================================
# Sync Client Wrapper
# =============================================================================


class SyncClientWrapper:
    """Typed wrapper for socketio.Client.

    Provides typed emit, call, and on methods while passing through all other
    attributes to the underlying Client instance.
    """

    def __init__(self, sio: Client) -> None:
        """Initialize wrapper with a Client instance.

        Args:
            sio: The socketio Client to wrap.
        """
        self._sio = sio

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying socketio instance."""
        return getattr(self._sio, name)

    # emit overloads
    @overload
    def emit(
        self,
        event: BaseModel,
        **kwargs: Any,
    ) -> None: ...

    @overload
    def emit(
        self,
        event: str,
        data: Any = None,
        **kwargs: Any,
    ) -> None: ...

    def emit(
        self,
        event: str | BaseModel,
        data: Any = None,
        **kwargs: Any,
    ) -> None:
        """Emit an event to the server.

        Args:
            event: Either a string event name or a BaseModel instance.
            data: Optional data payload (used when event is a string).
            **kwargs: Additional arguments passed to socketio's emit.
        """
        event_name, payload = _resolve_emit_args(event, data)
        self._sio.emit(event_name, payload, **kwargs)

    @overload
    def call(
        self,
        event: BaseModel,
        *,
        response_model: TypeForm[T],
        **kwargs: Any,
    ) -> T: ...

    @overload
    def call(
        self,
        event: BaseModel,
        **kwargs: Any,
    ) -> Any: ...

    @overload
    def call(
        self,
        event: str,
        data: Any = None,
        *,
        response_model: TypeForm[T],
        **kwargs: Any,
    ) -> T: ...

    @overload
    def call(
        self,
        event: str,
        data: Any = None,
        **kwargs: Any,
    ) -> Any: ...

    def call(
        self,
        event: str | BaseModel,
        data: Any = None,
        *,
        response_model: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Emit an event and wait for a response.

        Args:
            event: Either a string event name or a BaseModel instance.
            data: Optional data payload (used when event is a string).
            response_model: Optional type to validate response against (PEP 747 TypeForm).
            **kwargs: Additional arguments passed to socketio's call.

        Returns:
            Validated response if response_model is provided, otherwise raw response.
        """
        event_name, payload = _resolve_emit_args(event, data)
        response = self._sio.call(event_name, payload, **kwargs)
        return _validate_response(response, response_model)

    def on(
        self,
        event: str | Type[BaseModel],
        handler: Callable | None = None,
        **kwargs: Any,
    ) -> Callable[[Callable], Callable] | Callable:
        """Register an event handler.

        Args:
            event: Either a string event name or a BaseModel class.
            handler: Optional handler function (if not using as decorator).
            **kwargs: Additional arguments passed to socketio's on (e.g., namespace).

        Returns:
            Decorator that registers the handler with validation.
        """
        if isinstance(event, type) and issubclass(event, BaseModel):
            event_name = get_event_name(event)
        else:
            event_name = event

        def decorator(handler: Callable) -> Callable:
            wrapped = _create_sync_handler_wrapper(handler)
            self._sio.on(event_name, wrapped, **kwargs)
            return handler

        if handler is not None:
            return decorator(handler)
        return decorator

    def event(self, handler: Callable | None = None, **kwargs: Any) -> Callable:
        """Register an event handler using the function name as the event name.

        Args:
            handler: The event handler function.
            **kwargs: Additional arguments passed to socketio's on (e.g., namespace).

        Returns:
            The original handler (unmodified).
        """

        def decorator(handler: Callable) -> Callable:
            event_name = handler.__name__
            wrapped = _create_sync_handler_wrapper(handler)
            self._sio.on(event_name, wrapped, **kwargs)
            return handler

        if handler is not None:
            return decorator(handler)
        return decorator


# =============================================================================
# Async Simple Client Wrapper
# =============================================================================


class AsyncSimpleClientWrapper:
    """Typed wrapper for socketio.AsyncSimpleClient.

    Provides typed emit, call, and receive methods while passing through all
    other attributes to the underlying AsyncSimpleClient instance.

    The SimpleClient API uses receive() instead of event handlers, making it
    ideal for testing and simple scripts.

    Example:
        >>> import socketio
        >>> from zndraw_socketio import wrap
        >>> tsio = wrap(socketio.AsyncSimpleClient())
        >>> await tsio.connect('http://localhost:5000')
        >>> await tsio.emit(Ping(message="hello"))
        >>> event_name, data = await tsio.receive(response_model=Pong)
    """

    def __init__(self, sio: AsyncSimpleClient) -> None:
        """Initialize wrapper with an AsyncSimpleClient instance.

        Args:
            sio: The socketio AsyncSimpleClient to wrap.
        """
        self._sio = sio

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying socketio instance."""
        return getattr(self._sio, name)

    # emit overloads
    @overload
    async def emit(
        self,
        event: BaseModel,
    ) -> None: ...

    @overload
    async def emit(
        self,
        event: str,
        data: Any = None,
    ) -> None: ...

    async def emit(
        self,
        event: str | BaseModel,
        data: Any = None,
    ) -> None:
        """Emit an event to the server.

        Args:
            event: Either a string event name or a BaseModel instance.
                If BaseModel, event name is derived from the class name.
            data: Optional data payload (used when event is a string).
        """
        event_name, payload = _resolve_emit_args(event, data)
        await self._sio.emit(event_name, payload)

    # call overloads
    @overload
    async def call(
        self,
        event: BaseModel,
        *,
        response_model: TypeForm[T],
        timeout: int = 60,
    ) -> T: ...

    @overload
    async def call(
        self,
        event: BaseModel,
        *,
        timeout: int = 60,
    ) -> Any: ...

    @overload
    async def call(
        self,
        event: str,
        data: Any = None,
        *,
        response_model: TypeForm[T],
        timeout: int = 60,
    ) -> T: ...

    @overload
    async def call(
        self,
        event: str,
        data: Any = None,
        *,
        timeout: int = 60,
    ) -> Any: ...

    async def call(
        self,
        event: str | BaseModel,
        data: Any = None,
        *,
        response_model: Any = None,
        timeout: int = 60,
    ) -> Any:
        """Emit an event and wait for a response.

        Args:
            event: Either a string event name or a BaseModel instance.
            data: Optional data payload (used when event is a string).
            response_model: Optional type to validate response against (PEP 747 TypeForm).
            timeout: Timeout in seconds (default 60).

        Returns:
            Validated response if response_model is provided, otherwise raw response.
        """
        event_name, payload = _resolve_emit_args(event, data)
        response = await self._sio.call(event_name, payload, timeout=timeout)
        return _validate_response(response, response_model)

    # receive overloads
    @overload
    async def receive(
        self,
        *,
        response_model: TypeForm[T],
        timeout: float | None = None,
    ) -> tuple[str, T]: ...

    @overload
    async def receive(
        self,
        *,
        timeout: float | None = None,
    ) -> tuple[str, Any]: ...

    async def receive(
        self,
        *,
        response_model: Any = None,
        timeout: float | None = None,
    ) -> tuple[str, Any]:
        """Wait for an event from the server.

        Args:
            response_model: Optional type to validate the event data against.
            timeout: Timeout in seconds (None for no timeout).

        Returns:
            Tuple of (event_name, validated_data). If response_model is provided,
            the data is validated against it.

        Raises:
            TimeoutError: If timeout is reached before receiving an event.
        """
        result = await self._sio.receive(timeout=timeout)
        event_name = result[0]
        # SimpleClient receive returns [event_name, *args]
        event_data = result[1] if len(result) > 1 else None
        validated_data = _validate_response(event_data, response_model)
        return event_name, validated_data


# =============================================================================
# Simple Client Wrapper
# =============================================================================


class SimpleClientWrapper:
    """Typed wrapper for socketio.SimpleClient.

    Provides typed emit, call, and receive methods while passing through all
    other attributes to the underlying SimpleClient instance.

    The SimpleClient API uses receive() instead of event handlers, making it
    ideal for testing and simple scripts.

    Example:
        >>> import socketio
        >>> from zndraw_socketio import wrap
        >>> tsio = wrap(socketio.SimpleClient())
        >>> tsio.connect('http://localhost:5000')
        >>> tsio.emit(Ping(message="hello"))
        >>> event_name, data = tsio.receive(response_model=Pong)
    """

    def __init__(self, sio: SimpleClient) -> None:
        """Initialize wrapper with a SimpleClient instance.

        Args:
            sio: The socketio SimpleClient to wrap.
        """
        self._sio = sio

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying socketio instance."""
        return getattr(self._sio, name)

    @overload
    def emit(
        self,
        event: BaseModel,
    ) -> None: ...

    @overload
    def emit(
        self,
        event: str,
        data: Any = None,
    ) -> None: ...

    def emit(
        self,
        event: str | BaseModel,
        data: Any = None,
    ) -> None:
        """Emit an event to the server.

        Args:
            event: Either a string event name or a BaseModel instance.
                If BaseModel, event name is derived from the class name.
            data: Optional data payload (used when event is a string).
        """
        event_name, payload = _resolve_emit_args(event, data)
        self._sio.emit(event_name, payload)

    @overload
    def call(
        self,
        event: BaseModel,
        *,
        response_model: TypeForm[T],
        timeout: int = 60,
    ) -> T: ...

    @overload
    def call(
        self,
        event: BaseModel,
        *,
        timeout: int = 60,
    ) -> Any: ...

    @overload
    def call(
        self,
        event: str,
        data: Any = None,
        *,
        response_model: TypeForm[T],
        timeout: int = 60,
    ) -> T: ...

    @overload
    def call(
        self,
        event: str,
        data: Any = None,
        *,
        timeout: int = 60,
    ) -> Any: ...

    def call(
        self,
        event: str | BaseModel,
        data: Any = None,
        *,
        response_model: Any = None,
        timeout: int = 60,
    ) -> Any:
        """Emit an event and wait for a response.

        Args:
            event: Either a string event name or a BaseModel instance.
            data: Optional data payload (used when event is a string).
            response_model: Optional type to validate response against (PEP 747 TypeForm).
            timeout: Timeout in seconds (default 60).

        Returns:
            Validated response if response_model is provided, otherwise raw response.
        """
        event_name, payload = _resolve_emit_args(event, data)
        response = self._sio.call(event_name, payload, timeout=timeout)
        return _validate_response(response, response_model)

    @overload
    def receive(
        self,
        *,
        response_model: TypeForm[T],
        timeout: float | None = None,
    ) -> tuple[str, T]: ...

    @overload
    def receive(
        self,
        *,
        timeout: float | None = None,
    ) -> tuple[str, Any]: ...

    def receive(
        self,
        *,
        response_model: Any = None,
        timeout: float | None = None,
    ) -> tuple[str, Any]:
        """Wait for an event from the server.

        Args:
            response_model: Optional type to validate the event data against.
            timeout: Timeout in seconds (None for no timeout).

        Returns:
            Tuple of (event_name, validated_data). If response_model is provided,
            the data is validated against it.

        Raises:
            TimeoutError: If timeout is reached before receiving an event.
        """
        result = self._sio.receive(timeout=timeout)
        event_name = result[0]
        # SimpleClient receive returns [event_name, *args]
        event_data = result[1] if len(result) > 1 else None
        validated_data = _validate_response(event_data, response_model)
        return event_name, validated_data


# =============================================================================
# Sync Server Wrapper
# =============================================================================


class SyncServerWrapper:
    """Typed wrapper for socketio.Server.

    Provides typed emit, call, and on methods while passing through all other
    attributes to the underlying Server instance.
    """

    def __init__(self, sio: Server) -> None:
        """Initialize wrapper with a Server instance.

        Args:
            sio: The socketio Server to wrap.
        """
        self._sio = sio

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying socketio instance."""
        return getattr(self._sio, name)

    # emit overloads
    @overload
    def emit(
        self,
        event: BaseModel,
        **kwargs: Any,
    ) -> None: ...

    @overload
    def emit(
        self,
        event: str,
        data: Any = None,
        **kwargs: Any,
    ) -> None: ...

    def emit(
        self,
        event: str | BaseModel,
        data: Any = None,
        **kwargs: Any,
    ) -> None:
        """Emit an event to connected clients.

        Args:
            event: Either a string event name or a BaseModel instance.
            data: Optional data payload (used when event is a string).
            **kwargs: Additional arguments passed to socketio's emit.
        """
        event_name, payload = _resolve_emit_args(event, data)
        self._sio.emit(event_name, payload, **kwargs)

    # call overloads (see PEP 747 note in AsyncClientWrapper)
    @overload
    def call(
        self,
        event: BaseModel,
        *,
        response_model: TypeForm[T],
        **kwargs: Any,
    ) -> T: ...

    @overload
    def call(
        self,
        event: BaseModel,
        **kwargs: Any,
    ) -> Any: ...

    @overload
    def call(
        self,
        event: str,
        data: Any = None,
        *,
        response_model: TypeForm[T],
        **kwargs: Any,
    ) -> T: ...

    @overload
    def call(
        self,
        event: str,
        data: Any = None,
        **kwargs: Any,
    ) -> Any: ...

    def call(
        self,
        event: str | BaseModel,
        data: Any = None,
        *,
        response_model: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Emit an event and wait for a response from a client.

        Args:
            event: Either a string event name or a BaseModel instance.
            data: Optional data payload (used when event is a string).
            response_model: Optional type to validate response against (PEP 747 TypeForm).
            **kwargs: Additional arguments passed to socketio's call.

        Returns:
            Validated response if response_model is provided, otherwise raw response.
        """
        event_name, payload = _resolve_emit_args(event, data)
        response = self._sio.call(event_name, payload, **kwargs)
        return _validate_response(response, response_model)

    def on(
        self,
        event: str | Type[BaseModel],
        handler: Callable | None = None,
        **kwargs: Any,
    ) -> Callable[[Callable], Callable] | Callable:
        """Register an event handler.

        Args:
            event: Either a string event name or a BaseModel class.
            handler: Optional handler function (if not using as decorator).
            **kwargs: Additional arguments passed to socketio's on (e.g., namespace).

        Returns:
            Decorator that registers the handler with validation.
        """
        if isinstance(event, type) and issubclass(event, BaseModel):
            event_name = get_event_name(event)
        else:
            event_name = event

        def decorator(handler: Callable) -> Callable:
            wrapped = _create_sync_handler_wrapper(handler)
            self._sio.on(event_name, wrapped, **kwargs)
            return handler

        if handler is not None:
            return decorator(handler)
        return decorator

    def event(self, handler: Callable | None = None, **kwargs: Any) -> Callable:
        """Register an event handler using the function name as the event name.

        Args:
            handler: The event handler function.
            **kwargs: Additional arguments passed to socketio's on (e.g., namespace).

        Returns:
            The original handler (unmodified).
        """

        def decorator(handler: Callable) -> Callable:
            event_name = handler.__name__
            wrapped = _create_sync_handler_wrapper(handler)
            self._sio.on(event_name, wrapped, **kwargs)
            return handler

        if handler is not None:
            return decorator(handler)
        return decorator


# =============================================================================
# Factory Function
# =============================================================================


@overload
def wrap(sio: AsyncSimpleClient) -> AsyncSimpleClientWrapper: ...


@overload
def wrap(sio: SimpleClient) -> SimpleClientWrapper: ...


@overload
def wrap(sio: AsyncClient) -> AsyncClientWrapper: ...


@overload
def wrap(sio: AsyncServer) -> AsyncServerWrapper: ...


@overload
def wrap(sio: Client) -> SyncClientWrapper: ...


@overload
def wrap(sio: Server) -> SyncServerWrapper: ...


def wrap(
    sio: AsyncSimpleClient | SimpleClient | AsyncClient | AsyncServer | Client | Server,
) -> (
    AsyncSimpleClientWrapper
    | SimpleClientWrapper
    | AsyncClientWrapper
    | AsyncServerWrapper
    | SyncClientWrapper
    | SyncServerWrapper
):
    """Wrap a socketio instance with typed emit, call, and on methods.

    This is the main entry point for the wrapper API. It auto-detects the
    type of socketio instance and returns the appropriate wrapper.

    To enable FastAPI-style ``Request`` injection in dependencies, set the
    ``.app`` property on the returned wrapper (supported by
    ``AsyncServerWrapper`` and ``AsyncClientWrapper``)::

        tsio = wrap(socketio.AsyncServer(async_mode='asgi'))
        tsio.app = app  # can be set later, e.g. inside a lifespan

    Args:
        sio: A socketio Client, AsyncClient, Server, AsyncServer,
            SimpleClient, or AsyncSimpleClient instance.

    Returns:
        The appropriate wrapper class for the given socketio instance.

    Raises:
        TypeError: If sio is not a recognized socketio instance type.

    Example:
        >>> import socketio
        >>> from zndraw_socketio import wrap
        >>>
        >>> # Wrap standard client
        >>> tsio = wrap(socketio.AsyncClient())
        >>> await tsio.emit(Ping(message="hello"))
        >>>
        >>> # Wrap simple client
        >>> tsio = wrap(socketio.SimpleClient())
        >>> tsio.emit(Ping(message="hello"))
        >>> event_name, data = tsio.receive(response_model=Pong)
    """
    # Check SimpleClient types first (they are subclasses in some sense)
    if isinstance(sio, AsyncSimpleClient):
        return AsyncSimpleClientWrapper(sio)
    elif isinstance(sio, SimpleClient):
        return SimpleClientWrapper(sio)
    elif isinstance(sio, AsyncClient):
        return AsyncClientWrapper(sio)
    elif isinstance(sio, AsyncServer):
        return AsyncServerWrapper(sio)
    elif isinstance(sio, Client):
        return SyncClientWrapper(sio)
    elif isinstance(sio, Server):
        return SyncServerWrapper(sio)
    raise TypeError(f"Expected socketio instance, got {type(sio)}")
