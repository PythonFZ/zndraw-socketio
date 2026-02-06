"""Type inference tests using assert_type.

These tests verify that type inference works correctly at runtime.
The assert_type function checks that the inferred type matches expectations.
"""

from typing import Annotated, Any
from unittest.mock import MagicMock

import socketio

# Import test models from conftest
from conftest import Error, Ping, Pong, Success
from pydantic import Discriminator
from typing_extensions import assert_type

from zndraw_socketio import (
    AsyncClientWrapper,
    AsyncServerWrapper,
    AsyncSimpleClientWrapper,
    SimpleClientWrapper,
    SyncClientWrapper,
    SyncServerWrapper,
    wrap,
)

# =============================================================================
# Tests for wrap() return types
# =============================================================================


def test_wrap_async_client_type() -> None:
    """Verify wrap(AsyncClient) returns AsyncClientWrapper."""
    sio = socketio.AsyncClient()
    tsio = wrap(sio)
    assert_type(tsio, AsyncClientWrapper)


def test_wrap_async_server_type() -> None:
    """Verify wrap(AsyncServer) returns AsyncServerWrapper."""
    sio = socketio.AsyncServer()
    tsio = wrap(sio)
    assert_type(tsio, AsyncServerWrapper)


def test_wrap_sync_client_type() -> None:
    """Verify wrap(Client) returns SyncClientWrapper."""
    sio = socketio.Client()
    tsio = wrap(sio)
    assert_type(tsio, SyncClientWrapper)


def test_wrap_sync_server_type() -> None:
    """Verify wrap(Server) returns SyncServerWrapper."""
    sio = socketio.Server()
    tsio = wrap(sio)
    assert_type(tsio, SyncServerWrapper)


def test_wrap_simple_client_type() -> None:
    """Verify wrap(SimpleClient) returns SimpleClientWrapper."""
    sio = socketio.SimpleClient()
    tsio = wrap(sio)
    assert_type(tsio, SimpleClientWrapper)


def test_wrap_async_simple_client_type() -> None:
    """Verify wrap(AsyncSimpleClient) returns AsyncSimpleClientWrapper."""
    sio = socketio.AsyncSimpleClient()
    tsio = wrap(sio)
    assert_type(tsio, AsyncSimpleClientWrapper)


# =============================================================================
# Tests for call() with response_model returns T
# =============================================================================


def test_sync_call_returns_response_model_type() -> None:
    """Verify sync call() with response_model returns the correct type."""
    mock_sio = MagicMock(spec=socketio.Client)
    mock_sio.call.return_value = {"reply": "world"}
    tsio = SyncClientWrapper(mock_sio)

    response = tsio.call(Ping(message="hello"), response_model=Pong)
    assert_type(response, Pong)
    assert isinstance(response, Pong)


def test_call_without_response_model_returns_any() -> None:
    """Verify call() without response_model returns Any."""
    mock_sio = MagicMock(spec=socketio.Client)
    mock_sio.call.return_value = {"data": 123}
    tsio = SyncClientWrapper(mock_sio)

    response = tsio.call("event", {"data": 123})
    assert_type(response, Any)


# =============================================================================
# Tests for call() with union types
# =============================================================================


def test_call_with_union_type_success() -> None:
    """Verify call() with union response_model returns correct variant."""
    mock_sio = MagicMock(spec=socketio.Client)
    mock_sio.call.return_value = {"kind": "success", "data": "result"}
    tsio = SyncClientWrapper(mock_sio)

    response = tsio.call(Ping(message="hello"), response_model=Success | Error)
    assert_type(response, Success | Error)
    assert isinstance(response, Success)


def test_call_with_union_type_error() -> None:
    """Verify call() with union response_model returns error variant."""
    mock_sio = MagicMock(spec=socketio.Client)
    mock_sio.call.return_value = {"kind": "error", "message": "failed"}
    tsio = SyncClientWrapper(mock_sio)

    response = tsio.call(Ping(message="hello"), response_model=Success | Error)
    assert_type(response, Success | Error)
    assert isinstance(response, Error)


# =============================================================================
# Tests for call() with discriminated unions
# =============================================================================


def test_call_with_discriminated_union() -> None:
    """Verify call() with discriminated union returns annotated type."""
    mock_sio = MagicMock(spec=socketio.Client)
    mock_sio.call.return_value = {"kind": "success", "data": "result"}
    tsio = SyncClientWrapper(mock_sio)

    ResponseType = Annotated[Success | Error, Discriminator("kind")]
    response = tsio.call(Ping(message="hello"), response_model=ResponseType)
    assert_type(response, Success | Error)
    assert isinstance(response, Success)


# =============================================================================
# Tests for SimpleClient receive() return type
# =============================================================================


def test_simple_client_receive_returns_tuple() -> None:
    """Verify receive() returns tuple of (str, T)."""
    mock_sio = MagicMock(spec=socketio.SimpleClient)
    mock_sio.receive.return_value = ["pong", {"reply": "world"}]
    tsio = SimpleClientWrapper(mock_sio)

    result = tsio.receive(response_model=Pong)
    assert_type(result, tuple[str, Pong])
    event_name, data = result
    assert event_name == "pong"
    assert isinstance(data, Pong)


def test_simple_client_receive_without_model_returns_any() -> None:
    """Verify receive() without response_model returns tuple of (str, Any)."""
    mock_sio = MagicMock(spec=socketio.SimpleClient)
    mock_sio.receive.return_value = ["event", {"raw": "data"}]
    tsio = SimpleClientWrapper(mock_sio)

    result = tsio.receive()
    assert_type(result, tuple[str, Any])
    event_name, data = result
    assert event_name == "event"
    assert data == {"raw": "data"}
