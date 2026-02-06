"""Tests for Request auto-injection and generator dependencies."""

from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Annotated, AsyncIterator

import pytest
import socketio
from pydantic import BaseModel

from zndraw_socketio import wrap
from zndraw_socketio.wrapper import (
    Request,
    SioRequest,
    _resolve_dependencies,
)

try:
    from fastapi import Depends
except ImportError:
    from zndraw_socketio.params import Depends


@dataclass
class FakeState:
    db_url: str = "sqlite://"


@dataclass
class FakeApp:
    state: FakeState = field(default_factory=FakeState)


def test_sio_request_exposes_app():
    """SioRequest.app returns the app instance."""
    app = FakeApp()
    req = SioRequest(app=app)
    assert req.app is app


def test_sio_request_app_state():
    """SioRequest.app.state is accessible."""
    app = FakeApp(state=FakeState(db_url="postgres://localhost/mydb"))
    req = SioRequest(app=app)
    assert req.app.state.db_url == "postgres://localhost/mydb"


def test_sio_request_no_url():
    """Unsupported attrs raise AttributeError."""
    req = SioRequest(app=FakeApp())
    with pytest.raises(AttributeError):
        _ = req.url  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# wrap() app kwarg tests
# ---------------------------------------------------------------------------


def test_wrap_app_property():
    """wrap(sio) then .app = app stores app on the wrapper."""
    app = FakeApp()
    tsio = wrap(socketio.AsyncServer(async_mode="asgi"))
    tsio.app = app
    assert tsio.app is app


def test_wrap_without_app():
    """wrap(sio) without setting .app leaves it as None."""
    tsio = wrap(socketio.AsyncServer(async_mode="asgi"))
    assert tsio.app is None


def test_wrap_app_property_on_client():
    """.app property works on AsyncClientWrapper too."""
    app = FakeApp()
    tsio = wrap(socketio.AsyncClient())
    tsio.app = app
    assert tsio.app is app


# ---------------------------------------------------------------------------
# _resolve_dependencies tests (Request injection & generator support)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_injects_request():
    """Dependencies with Request param get SioRequest injected."""
    app = FakeApp(state=FakeState(db_url="postgres://"))

    def get_db_url(request: Request) -> str:
        return request.app.state.db_url

    deps = {"db_url": get_db_url}
    async with AsyncExitStack() as stack:
        resolved = await _resolve_dependencies(deps, app=app, stack=stack)
    assert resolved["db_url"] == "postgres://"


@pytest.mark.asyncio
async def test_resolve_async_generator():
    """Async generator deps yield value and cleanup runs on stack exit."""
    opened = False
    closed = False

    async def get_resource() -> AsyncIterator[str]:
        nonlocal opened, closed
        opened = True
        yield "resource_value"
        closed = True

    deps = {"res": get_resource}
    async with AsyncExitStack() as stack:
        resolved = await _resolve_dependencies(deps, app=None, stack=stack)
        assert resolved["res"] == "resource_value"
        assert opened is True
        assert closed is False  # not yet cleaned up — stack still open
    assert closed is True  # stack exited, cleanup ran


@pytest.mark.asyncio
async def test_resolve_sync_generator():
    """Sync generator deps yield value and cleanup runs on stack exit."""
    from typing import Iterator

    closed = False

    def get_resource() -> Iterator[str]:
        nonlocal closed
        yield "sync_value"
        closed = True

    deps = {"res": get_resource}
    async with AsyncExitStack() as stack:
        resolved = await _resolve_dependencies(deps, app=None, stack=stack)
        assert resolved["res"] == "sync_value"
        assert closed is False
    assert closed is True


@pytest.mark.asyncio
async def test_resolve_generator_with_request():
    """Generator dep that also takes Request param."""
    app = FakeApp(state=FakeState(db_url="sqlite://"))

    async def get_session(request: Request) -> AsyncIterator[str]:
        url = request.app.state.db_url
        yield f"session:{url}"

    deps = {"session": get_session}
    async with AsyncExitStack() as stack:
        resolved = await _resolve_dependencies(deps, app=app, stack=stack)
        assert resolved["session"] == "session:sqlite://"


class EchoRequest(BaseModel):
    msg: str


class EchoResponse(BaseModel):
    reply: str


# ---------------------------------------------------------------------------
# Deferred app tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deferred_app_set_after_handler_registration(server_factory):
    """Handler registered before .app is set still gets the app at event time."""
    from fastapi import FastAPI

    tsio = wrap(socketio.AsyncServer(async_mode="asgi"))

    def get_prefix(request: Request) -> str:
        return request.app.state.prefix

    @tsio.on(EchoRequest)
    async def handle(
        sid: str, data: EchoRequest, prefix: Annotated[str, Depends(get_prefix)]
    ) -> EchoResponse:
        return EchoResponse(reply=f"{prefix}, {data.msg}")

    # Set app AFTER handler registration
    app = FastAPI()
    app.state.prefix = "Deferred"
    tsio.app = app

    url = await server_factory(socketio.ASGIApp(tsio, app))
    client = wrap(socketio.AsyncSimpleClient())
    await client.connect(url)

    resp = await client.call(EchoRequest(msg="world"), response_model=EchoResponse)
    assert resp.reply == "Deferred, world"
    await client.disconnect()


@pytest.mark.asyncio
async def test_request_app_none_when_not_set():
    """When .app is not set, dependency gets SioRequest(app=None)."""
    from contextlib import AsyncExitStack

    def get_app_value(request: Request) -> object:
        return request.app

    deps = {"val": get_app_value}
    async with AsyncExitStack() as stack:
        resolved = await _resolve_dependencies(deps, app=None, stack=stack)
    assert resolved["val"] is None


# ---------------------------------------------------------------------------
# Integration tests: @tsio.on() with Request injection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_decorator_injects_request(server_factory):
    """@tsio.on handler receives Request-injected dependency."""
    from fastapi import FastAPI

    app = FastAPI()
    app.state.prefix = "Hello"
    tsio = wrap(socketio.AsyncServer(async_mode="asgi"))
    tsio.app = app

    def get_prefix(request: Request) -> str:
        return request.app.state.prefix

    @tsio.on(EchoRequest)
    async def handle(
        sid: str, data: EchoRequest, prefix: Annotated[str, Depends(get_prefix)]
    ) -> EchoResponse:
        return EchoResponse(reply=f"{prefix}, {data.msg}")

    url = await server_factory(socketio.ASGIApp(tsio, app))
    client = wrap(socketio.AsyncSimpleClient())
    await client.connect(url)

    resp = await client.call(EchoRequest(msg="world"), response_model=EchoResponse)
    assert resp.reply == "Hello, world"
    await client.disconnect()


@pytest.mark.asyncio
async def test_on_decorator_async_generator_dep(server_factory):
    """@tsio.on handler with async generator dependency (yield pattern).

    AsyncExitStack ensures cleanup runs before the handler returns.
    """
    from fastapi import FastAPI

    app = FastAPI()
    app.state.db_url = "test://"
    tsio = wrap(socketio.AsyncServer(async_mode="asgi"))
    tsio.app = app

    cleanup_called = False

    async def get_session(request: Request) -> AsyncIterator[str]:
        nonlocal cleanup_called
        yield f"session:{request.app.state.db_url}"
        cleanup_called = True

    @tsio.on(EchoRequest)
    async def handle(
        sid: str, data: EchoRequest, session: Annotated[str, Depends(get_session)]
    ) -> EchoResponse:
        return EchoResponse(reply=session)

    url = await server_factory(socketio.ASGIApp(tsio, app))
    client = wrap(socketio.AsyncSimpleClient())
    await client.connect(url)

    resp = await client.call(EchoRequest(msg="x"), response_model=EchoResponse)
    assert resp.reply == "session:test://"
    # AsyncExitStack cleanup runs before handler returns — no sleep needed
    assert cleanup_called is True
    await client.disconnect()


@pytest.mark.asyncio
async def test_event_decorator_injects_request(server_factory):
    """@tsio.event handler receives Request-injected dependency."""
    from fastapi import FastAPI

    app = FastAPI()
    app.state.prefix = "Hey"
    tsio = wrap(socketio.AsyncServer(async_mode="asgi"))
    tsio.app = app

    def get_prefix(request: Request) -> str:
        return request.app.state.prefix

    @tsio.event
    async def echo_request(
        sid: str,
        data: EchoRequest,
        prefix: str = Depends(get_prefix),
    ) -> EchoResponse:
        return EchoResponse(reply=f"{prefix}, {data.msg}")

    url = await server_factory(socketio.ASGIApp(tsio, app))
    client = wrap(socketio.AsyncSimpleClient())
    await client.connect(url)

    resp = await client.call(EchoRequest(msg="there"), response_model=EchoResponse)
    assert resp.reply == "Hey, there"
    await client.disconnect()


# ---------------------------------------------------------------------------
# Integration tests: connect/disconnect with DI
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_handler_with_depends(server_factory):
    """connect handler can use Depends() with Request injection."""
    from fastapi import FastAPI

    app = FastAPI()
    app.state.connections = []
    tsio = wrap(socketio.AsyncServer(async_mode="asgi"))
    tsio.app = app

    def get_connections(request: Request) -> list:
        return request.app.state.connections

    @tsio.on("connect")
    async def on_connect(
        sid: str,
        environ: dict,
        auth: dict | None = None,
        connections: list = Depends(get_connections),
    ) -> None:
        connections.append(sid)

    url = await server_factory(socketio.ASGIApp(tsio, app))
    client = wrap(socketio.AsyncSimpleClient())
    await client.connect(url)

    # Give the connect handler a moment to run
    import asyncio

    await asyncio.sleep(0.1)

    assert len(app.state.connections) == 1
    await client.disconnect()


@pytest.mark.asyncio
async def test_disconnect_handler_with_depends(server_factory):
    """disconnect handler can use Depends() with Request injection."""
    from fastapi import FastAPI

    app = FastAPI()
    app.state.disconnections = []
    tsio = wrap(socketio.AsyncServer(async_mode="asgi"))
    tsio.app = app

    def get_disconnections(request: Request) -> list:
        return request.app.state.disconnections

    @tsio.on("disconnect")
    async def on_disconnect(
        sid: str,
        reason: str,
        disconnections: Annotated[list, Depends(get_disconnections)],
    ) -> None:
        disconnections.append(sid)

    url = await server_factory(socketio.ASGIApp(tsio, app))
    client = wrap(socketio.AsyncSimpleClient())
    await client.connect(url)
    await client.disconnect()

    # Give the disconnect handler a moment to run
    import asyncio

    await asyncio.sleep(0.1)

    assert len(app.state.disconnections) == 1
