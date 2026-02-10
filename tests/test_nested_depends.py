"""Tests for nested (sub-)dependency resolution."""

from contextlib import AsyncExitStack

import pytest
import socketio
from fastapi import FastAPI
from pydantic import BaseModel

from zndraw_socketio import wrap
from zndraw_socketio.wrapper import (
    Request,
    SioRequest,
    _resolve_dependencies,
    _resolve_single,
)

try:
    from fastapi import Depends
except ImportError:
    from zndraw_socketio.params import Depends


def dep_a() -> str:
    return "aaa"


def dep_b(a: str = Depends(dep_a)) -> str:
    return a + "-bbb"


@pytest.mark.asyncio
async def test_nested_sync_deps_resolved():
    """dep_b depends on dep_a; both should be resolved."""
    async with AsyncExitStack() as stack:
        resolved = await _resolve_dependencies({"val": dep_b}, stack=stack)
    assert resolved["val"] == "aaa-bbb"


@pytest.mark.asyncio
async def test_nested_async_deps_resolved():
    """Async sub-dependency is awaited before parent."""

    async def async_dep_a() -> str:
        return "async-aaa"

    def dep_with_async_sub(a: str = Depends(async_dep_a)) -> str:
        return a + "-bbb"

    async with AsyncExitStack() as stack:
        resolved = await _resolve_dependencies(
            {"val": dep_with_async_sub}, stack=stack
        )
    assert resolved["val"] == "async-aaa-bbb"


@pytest.mark.asyncio
async def test_three_level_nesting():
    """dep_c -> dep_b -> dep_a, three levels deep."""

    def level_a() -> int:
        return 1

    def level_b(a: int = Depends(level_a)) -> int:
        return a + 10

    def level_c(b: int = Depends(level_b)) -> int:
        return b + 100

    async with AsyncExitStack() as stack:
        resolved = await _resolve_dependencies({"val": level_c}, stack=stack)
    assert resolved["val"] == 111


@pytest.mark.asyncio
async def test_diamond_dependency_pattern():
    """
    Diamond:  Handler -> A(c, d) -> C(d) -> D
                       -> B(c)    -> C(d) -> D
    D and C should each be resolved once (cached).
    """
    d_count = 0
    c_count = 0

    def dep_d() -> int:
        nonlocal d_count
        d_count += 1
        return 1

    def dep_c(d: int = Depends(dep_d)) -> int:
        nonlocal c_count
        c_count += 1
        return d + 10

    def dep_a(c: int = Depends(dep_c), d: int = Depends(dep_d)) -> int:
        return c + d + 100

    def dep_b(c: int = Depends(dep_c)) -> int:
        return c + 200

    async with AsyncExitStack() as stack:
        resolved = await _resolve_dependencies(
            {"a": dep_a, "b": dep_b}, stack=stack
        )

    # D=1, C=11, A=11+1+100=112, B=11+200=211
    assert resolved["a"] == 112
    assert resolved["b"] == 211
    assert d_count == 1, f"dep_d called {d_count} times, expected 1"
    assert c_count == 1, f"dep_c called {c_count} times, expected 1"


@pytest.mark.asyncio
async def test_use_cache_false_resolves_fresh():
    """use_cache=False skips the cache -- sub-dep called again."""
    call_count = 0

    def shared() -> str:
        nonlocal call_count
        call_count += 1
        return f"v{call_count}"

    def consumer_1(s: str = Depends(shared, use_cache=False)) -> str:
        return s

    def consumer_2(s: str = Depends(shared, use_cache=False)) -> str:
        return s

    async with AsyncExitStack() as stack:
        await _resolve_dependencies(
            {"c1": consumer_1, "c2": consumer_2}, stack=stack
        )

    assert call_count == 2


@pytest.mark.asyncio
async def test_nested_sync_generator_dep():
    """Sync generator sub-dependency yields a value and cleans up."""
    cleanup_order: list[str] = []

    def gen_a():
        yield "gen-a"
        cleanup_order.append("a-cleanup")

    def dep_using_gen(a: str = Depends(gen_a)) -> str:
        return a + "-used"

    async with AsyncExitStack() as stack:
        resolved = await _resolve_dependencies(
            {"val": dep_using_gen}, stack=stack
        )
        assert resolved["val"] == "gen-a-used"

    assert cleanup_order == ["a-cleanup"]


@pytest.mark.asyncio
async def test_nested_async_generator_dep():
    """Async generator sub-dependency yields and cleans up."""
    cleanup_order: list[str] = []

    async def async_gen_a():
        yield "async-gen-a"
        cleanup_order.append("async-a-cleanup")

    async def dep_using_async_gen(a: str = Depends(async_gen_a)) -> str:
        return a + "-used"

    async with AsyncExitStack() as stack:
        resolved = await _resolve_dependencies(
            {"val": dep_using_async_gen}, stack=stack
        )
        assert resolved["val"] == "async-gen-a-used"

    assert cleanup_order == ["async-a-cleanup"]


# =============================================================================
# Integration Test
# =============================================================================


class NestReq(BaseModel):
    value: int


class NestResp(BaseModel):
    result: str


@pytest.mark.asyncio
async def test_nested_depends_integration(server_factory):
    """Full server round-trip with diamond deps: offset and scale share base."""

    def get_base() -> int:
        return 10

    def get_offset(base: int = Depends(get_base)) -> int:
        return base + 5

    def get_scale(base: int = Depends(get_base)) -> int:
        return base * 2

    app = FastAPI()
    tsio = wrap(socketio.AsyncServer(async_mode="asgi"))

    @tsio.on(NestReq)
    async def handle(
        sid: str,
        data: NestReq,
        offset: int = Depends(get_offset),
        scale: int = Depends(get_scale),
    ) -> NestResp:
        return NestResp(result=f"{data.value * scale + offset}")

    url = await server_factory(socketio.ASGIApp(tsio, app))
    client = wrap(socketio.AsyncSimpleClient())
    await client.connect(url)

    # base=10, offset=15, scale=20 → 1*20+15=35
    resp = await client.call(NestReq(value=1), response_model=NestResp)
    assert resp.result == "35"

    await client.disconnect()


# =============================================================================
# Issue 1: Top-level use_cache=False respected for handler dependencies
# =============================================================================


@pytest.mark.asyncio
async def test_top_level_use_cache_false_respected():
    """Handler-level Depends(fn, use_cache=False) is passed through to resolver."""
    call_count = 0

    def fresh_dep() -> str:
        nonlocal call_count
        call_count += 1
        return f"v{call_count}"

    async def handler(
        sid: str,
        a: str = Depends(fresh_dep, use_cache=False),
        b: str = Depends(fresh_dep, use_cache=False),
    ) -> str:
        return a + b

    from zndraw_socketio.wrapper import (
        _extract_dependencies,
        _get_use_cache,
    )

    deps = _extract_dependencies(handler)
    use_cache_flags = {name: _get_use_cache(handler, name) for name in deps}

    # Both should be False
    assert use_cache_flags["a"] is False
    assert use_cache_flags["b"] is False

    async with AsyncExitStack() as stack:
        resolved = await _resolve_dependencies(
            deps, stack=stack, use_cache_overrides=use_cache_flags
        )

    # fresh_dep should be called twice (once per param) because use_cache=False
    assert call_count == 2
    assert resolved["a"] == "v1"
    assert resolved["b"] == "v2"


@pytest.mark.asyncio
async def test_top_level_use_cache_true_default():
    """Handler-level Depends(fn) defaults to use_cache=True (cached)."""
    call_count = 0

    def cached_dep() -> str:
        nonlocal call_count
        call_count += 1
        return f"v{call_count}"

    async def handler(
        sid: str,
        a: str = Depends(cached_dep),
        b: str = Depends(cached_dep),
    ) -> str:
        return a + b

    from zndraw_socketio.wrapper import (
        _extract_dependencies,
        _get_use_cache,
    )

    deps = _extract_dependencies(handler)
    use_cache_flags = {name: _get_use_cache(handler, name) for name in deps}

    # Both should be True (default)
    assert use_cache_flags["a"] is True
    assert use_cache_flags["b"] is True

    async with AsyncExitStack() as stack:
        resolved = await _resolve_dependencies(
            deps, stack=stack, use_cache_overrides=use_cache_flags
        )

    # cached_dep should be called only once because use_cache=True
    assert call_count == 1
    assert resolved["a"] == "v1"
    assert resolved["b"] == "v1"


# =============================================================================
# Issue 2: Cycle detection test
# =============================================================================


@pytest.mark.asyncio
async def test_cycle_detection_raises():
    """Circular dependency should raise RuntimeError."""

    def dep_fn() -> str:
        return "x"

    async with AsyncExitStack() as stack:
        _cache: dict[int, object] = {}
        _resolving: set[int] = set()

        # Pre-mark dep_fn as "being resolved" to simulate a cycle
        _resolving.add(id(dep_fn))

        with pytest.raises(RuntimeError, match="Circular dependency"):
            await _resolve_single(
                dep_fn,
                stack=stack,
                _cache=_cache,
                _resolving=_resolving,
            )


# =============================================================================
# Issue 3: Nested SioRequest injection test
# =============================================================================


@pytest.mark.asyncio
async def test_nested_request_injection():
    """Sub-dependency receives SioRequest when it has a Request param."""
    received_request = None

    def sub_dep(request: Request) -> str:
        nonlocal received_request
        received_request = request
        return "from-request"

    def parent_dep(val: str = Depends(sub_dep)) -> str:
        return val + "-parent"

    fake_app = object()
    async with AsyncExitStack() as stack:
        resolved = await _resolve_dependencies(
            {"result": parent_dep}, app=fake_app, stack=stack
        )

    assert resolved["result"] == "from-request-parent"
    assert isinstance(received_request, SioRequest)
    assert received_request.app is fake_app
