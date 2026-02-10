"""Tests for nested (sub-)dependency resolution."""

from contextlib import AsyncExitStack
from typing import Annotated

import pytest
import socketio
from pydantic import BaseModel

from zndraw_socketio import wrap
from zndraw_socketio.wrapper import _extract_dependencies, _resolve_dependencies

try:
    from fastapi import Depends, FastAPI
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
        resolved = await _resolve_dependencies(
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
