"""Integration tests for FastAPI + Socket.IO exception handlers."""

import socketio
from pydantic import BaseModel

from zndraw_socketio import EventContext, wrap


class ErrorResponse(BaseModel):
    """RFC 9457 Problem Details compliant error response."""

    type: str = "about:blank"
    title: str
    status: int | None = None
    detail: str | None = None


class ErrorTriggerEvent(BaseModel):
    """Event that optionally triggers an error."""

    error: bool


class OkayResponse(BaseModel):
    """Successful response."""

    message: str


class ValueErrorResponse(BaseModel):
    """Response for ValueError exceptions."""

    error_type: str = "value_error"
    message: str


class TestExceptionHandlerRegistration:
    """Tests for exception handler registration on AsyncServerWrapper."""

    def test_register_global_handler(self):
        """Global handlers are registered with namespace=None."""
        sio = socketio.AsyncServer(async_mode="asgi")
        tsio = wrap(sio)

        @tsio.exception_handler(ValueError)
        async def handle_value_error(ctx: EventContext, exc: ValueError):
            return {"error": "value_error"}

        assert None in tsio._exception_handlers
        assert ValueError in tsio._exception_handlers[None]

    def test_register_namespace_handler(self):
        """Namespace-specific handlers are registered correctly."""
        sio = socketio.AsyncServer(async_mode="asgi")
        tsio = wrap(sio)

        @tsio.exception_handler(ValueError, namespace="/chat")
        async def handle_value_error(ctx: EventContext, exc: ValueError):
            return {"error": "value_error"}

        assert "/chat" in tsio._exception_handlers
        assert ValueError in tsio._exception_handlers["/chat"]

    def test_register_multiple_handlers(self):
        """Multiple handlers for different exception types."""
        sio = socketio.AsyncServer(async_mode="asgi")
        tsio = wrap(sio)

        @tsio.exception_handler(ValueError)
        async def handle_value_error(ctx: EventContext, exc: ValueError):
            return {"error": "value_error"}

        @tsio.exception_handler(TypeError)
        async def handle_type_error(ctx: EventContext, exc: TypeError):
            return {"error": "type_error"}

        assert ValueError in tsio._exception_handlers[None]
        assert TypeError in tsio._exception_handlers[None]


class TestFindExceptionHandler:
    """Tests for exception handler lookup."""

    def test_find_exact_type_match(self):
        """Finds handler for exact exception type."""
        sio = socketio.AsyncServer(async_mode="asgi")
        tsio = wrap(sio)

        @tsio.exception_handler(ValueError)
        async def handle_value_error(ctx: EventContext, exc: ValueError):
            return {"error": "value_error"}

        handler = tsio._find_exception_handler(ValueError("test"), "/")
        assert handler is not None

    def test_find_parent_type_match(self):
        """Finds handler for parent exception type via MRO."""
        sio = socketio.AsyncServer(async_mode="asgi")
        tsio = wrap(sio)

        @tsio.exception_handler(Exception)
        async def handle_any(ctx: EventContext, exc: Exception):
            return {"error": "any"}

        # ValueError is a subclass of Exception
        handler = tsio._find_exception_handler(ValueError("test"), "/")
        assert handler is not None

    def test_namespace_specific_takes_precedence(self):
        """Namespace-specific handler takes precedence over global."""
        sio = socketio.AsyncServer(async_mode="asgi")
        tsio = wrap(sio)

        @tsio.exception_handler(ValueError)
        async def global_handler(ctx: EventContext, exc: ValueError):
            return {"error": "global"}

        @tsio.exception_handler(ValueError, namespace="/chat")
        async def chat_handler(ctx: EventContext, exc: ValueError):
            return {"error": "chat"}

        handler = tsio._find_exception_handler(ValueError("test"), "/chat")
        assert handler is chat_handler

        handler = tsio._find_exception_handler(ValueError("test"), "/")
        assert handler is global_handler

    def test_no_handler_returns_none(self):
        """Returns None when no handler matches."""
        sio = socketio.AsyncServer(async_mode="asgi")
        tsio = wrap(sio)

        handler = tsio._find_exception_handler(ValueError("test"), "/")
        assert handler is None

    def test_exact_type_before_parent_type(self):
        """Exact type match takes precedence over parent type."""
        sio = socketio.AsyncServer(async_mode="asgi")
        tsio = wrap(sio)

        @tsio.exception_handler(Exception)
        async def handle_any(ctx: EventContext, exc: Exception):
            return {"error": "any"}

        @tsio.exception_handler(ValueError)
        async def handle_value_error(ctx: EventContext, exc: ValueError):
            return {"error": "value_error"}

        handler = tsio._find_exception_handler(ValueError("test"), "/")
        assert handler is handle_value_error
