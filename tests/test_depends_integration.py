"""Integration tests for Depends() dependency injection in Socket.IO handlers."""

from typing import Annotated

import pytest
import socketio
from fastapi import Depends, FastAPI
from pydantic import BaseModel

from zndraw_socketio import wrap

# =============================================================================
# Test Models & Dependencies
# =============================================================================


class GreetRequest(BaseModel):
    name: str


class GreetResponse(BaseModel):
    greeting: str


class CounterResponse(BaseModel):
    count: int


class FakeRedis:
    """Minimal fake Redis for testing."""

    def __init__(self):
        self._store: dict[str, int] = {}

    async def incr(self, key: str) -> int:
        self._store[key] = self._store.get(key, 0) + 1
        return self._store[key]


_fake_redis = FakeRedis()


async def get_redis() -> FakeRedis:
    return _fake_redis


def get_prefix() -> str:
    return "Hello"


RedisDep = Annotated[FakeRedis, Depends(get_redis)]


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.asyncio
async def test_depends_annotated_async(server_factory):
    """Depends via Annotated[T, Depends(fn)] with async dependency."""
    app = FastAPI()
    tsio = wrap(socketio.AsyncServer(async_mode="asgi"))

    @tsio.on(GreetRequest)
    async def handle_greet(
        sid: str, data: GreetRequest, redis: RedisDep
    ) -> GreetResponse:
        count = await redis.incr(f"greet:{data.name}")
        return GreetResponse(greeting=f"Hi {data.name} (#{count})")

    url = await server_factory(socketio.ASGIApp(tsio, app))
    client = wrap(socketio.AsyncSimpleClient())
    await client.connect(url)

    resp = await client.call(GreetRequest(name="Alice"), response_model=GreetResponse)
    assert resp.greeting == "Hi Alice (#1)"

    resp = await client.call(GreetRequest(name="Alice"), response_model=GreetResponse)
    assert resp.greeting == "Hi Alice (#2)"

    await client.disconnect()


@pytest.mark.asyncio
async def test_depends_default_value_sync(server_factory):
    """Depends via default value with sync dependency."""
    app = FastAPI()
    tsio = wrap(socketio.AsyncServer(async_mode="asgi"))

    @tsio.on(GreetRequest)
    async def handle_greet(
        sid: str, data: GreetRequest, prefix: str = Depends(get_prefix)
    ) -> GreetResponse:
        return GreetResponse(greeting=f"{prefix}, {data.name}!")

    url = await server_factory(socketio.ASGIApp(tsio, app))
    client = wrap(socketio.AsyncSimpleClient())
    await client.connect(url)

    resp = await client.call(GreetRequest(name="Bob"), response_model=GreetResponse)
    assert resp.greeting == "Hello, Bob!"

    await client.disconnect()


@pytest.mark.asyncio
async def test_depends_multiple(server_factory):
    """Multiple dependencies in a single handler."""
    app = FastAPI()
    tsio = wrap(socketio.AsyncServer(async_mode="asgi"))

    @tsio.on(GreetRequest)
    async def handle_greet(
        sid: str,
        data: GreetRequest,
        redis: RedisDep,
        prefix: str = Depends(get_prefix),
    ) -> GreetResponse:
        count = await redis.incr(f"multi:{data.name}")
        return GreetResponse(greeting=f"{prefix}, {data.name} (#{count})")

    url = await server_factory(socketio.ASGIApp(tsio, app))
    client = wrap(socketio.AsyncSimpleClient())
    await client.connect(url)

    resp = await client.call(GreetRequest(name="Carol"), response_model=GreetResponse)
    assert resp.greeting == "Hello, Carol (#1)"

    await client.disconnect()


@pytest.mark.asyncio
async def test_depends_with_event_decorator(server_factory):
    """Depends works with @tsio.event decorator too."""
    app = FastAPI()
    tsio = wrap(socketio.AsyncServer(async_mode="asgi"))

    @tsio.event
    async def greet_request(
        sid: str, data: GreetRequest, prefix: str = Depends(get_prefix)
    ) -> GreetResponse:
        return GreetResponse(greeting=f"{prefix}, {data.name}!")

    url = await server_factory(socketio.ASGIApp(tsio, app))
    client = wrap(socketio.AsyncSimpleClient())
    await client.connect(url)

    resp = await client.call(GreetRequest(name="Dave"), response_model=GreetResponse)
    assert resp.greeting == "Hello, Dave!"

    await client.disconnect()


@pytest.mark.asyncio
async def test_event_decorator_no_depends(server_factory):
    """@tsio.event works without Depends — plain handler with Pydantic validation."""
    app = FastAPI()
    tsio = wrap(socketio.AsyncServer(async_mode="asgi"))

    @tsio.event
    async def greet_request(sid: str, data: GreetRequest) -> GreetResponse:
        return GreetResponse(greeting=f"Hi, {data.name}!")

    url = await server_factory(socketio.ASGIApp(tsio, app))
    client = wrap(socketio.AsyncSimpleClient())
    await client.connect(url)

    resp = await client.call(GreetRequest(name="Eve"), response_model=GreetResponse)
    assert resp.greeting == "Hi, Eve!"

    await client.disconnect()
