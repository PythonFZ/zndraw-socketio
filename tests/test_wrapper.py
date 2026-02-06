"""Unit tests for the wrapper module."""

import asyncio
from typing import Annotated

import pytest
import socketio

# Import test models from conftest
from conftest import CustomEvent, Error, PascalCaseModel, Ping, Pong, Success
from pydantic import Discriminator, ValidationError

from zndraw_socketio import (
    AsyncClientWrapper,
    AsyncServerWrapper,
    AsyncSimpleClientWrapper,
    SimpleClientWrapper,
    SyncClientWrapper,
    SyncServerWrapper,
    get_event_name,
    wrap,
)
from zndraw_socketio.wrapper import (
    _create_async_handler_wrapper,
    _create_sync_handler_wrapper,
    _resolve_emit_args,
    _validate_response,
)

# =============================================================================
# Tests for get_event_name()
# =============================================================================


class TestGetEventName:
    """Tests for get_event_name function."""

    @pytest.mark.parametrize(
        "model,expected",
        [
            (Ping, "ping"),
            (PascalCaseModel, "pascal_case_model"),
            (CustomEvent, "my_custom_event"),
        ],
    )
    def test_event_name_from_class(self, model, expected):
        """Test event name derivation from model classes."""
        assert get_event_name(model) == expected

    @pytest.mark.parametrize(
        "instance,expected",
        [
            (Ping(message="hello"), "ping"),
            (CustomEvent(data="test"), "my_custom_event"),
        ],
    )
    def test_event_name_from_instance(self, instance, expected):
        """Test event name derivation from model instances."""
        assert get_event_name(instance) == expected


# =============================================================================
# Tests for _resolve_emit_args()
# =============================================================================


class TestResolveEmitArgs:
    """Tests for _resolve_emit_args helper function."""

    @pytest.mark.parametrize(
        "event,data,expected_name,expected_payload",
        [
            (Ping(message="hello"), None, "ping", {"message": "hello"}),
            ("custom", Ping(message="hello"), "custom", {"message": "hello"}),
            ("custom", {"key": "value"}, "custom", {"key": "value"}),
            ("custom", None, "custom", None),
        ],
    )
    def test_resolve_emit_args(self, event, data, expected_name, expected_payload):
        """Test emit argument resolution."""
        event_name, payload = _resolve_emit_args(event, data)
        assert event_name == expected_name
        assert payload == expected_payload

    def test_model_with_data_raises_error(self):
        """Test that providing both model and data raises TypeError."""
        with pytest.raises(TypeError, match="Cannot provide both"):
            _resolve_emit_args(Ping(message="hello"), data={"extra": "data"})


# =============================================================================
# Tests for _validate_response()
# =============================================================================


class TestValidateResponse:
    """Tests for _validate_response helper function."""

    def test_no_response_model_returns_raw(self):
        """Test that None response_model returns raw response."""
        response = {"reply": "hello"}
        assert _validate_response(response, None) == response

    def test_single_model_validation(self):
        """Test validation with a single model type."""
        result = _validate_response({"reply": "hello"}, Pong)
        assert isinstance(result, Pong)
        assert result.reply == "hello"

    @pytest.mark.parametrize(
        "response,expected_type",
        [
            ({"kind": "success", "data": "test"}, Success),
            ({"kind": "error", "message": "failed"}, Error),
        ],
    )
    def test_union_type_validation(self, response, expected_type):
        """Test validation with union type."""
        result = _validate_response(response, Success | Error)
        assert isinstance(result, expected_type)

    @pytest.mark.parametrize(
        "response,expected_type",
        [
            ({"kind": "success", "data": "test"}, Success),
            ({"kind": "error", "message": "failed"}, Error),
        ],
    )
    def test_discriminated_union_validation(self, response, expected_type):
        """Test validation with discriminated union."""
        ResponseType = Annotated[Success | Error, Discriminator("kind")]
        result = _validate_response(response, ResponseType)
        assert isinstance(result, expected_type)


# =============================================================================
# Tests for wrap() factory
# =============================================================================


class TestWrapFactory:
    """Tests for wrap() factory function."""

    @pytest.mark.parametrize(
        "sio_class,wrapper_class",
        [
            (socketio.AsyncClient, AsyncClientWrapper),
            (socketio.AsyncServer, AsyncServerWrapper),
            (socketio.Client, SyncClientWrapper),
            (socketio.Server, SyncServerWrapper),
            (socketio.SimpleClient, SimpleClientWrapper),
            (socketio.AsyncSimpleClient, AsyncSimpleClientWrapper),
        ],
    )
    def test_wrap_returns_correct_wrapper(self, sio_class, wrapper_class):
        """Test wrap() returns correct wrapper type for each socketio type."""
        sio = sio_class()
        wrapped = wrap(sio)
        assert isinstance(wrapped, wrapper_class)
        assert wrapped._sio is sio

    def test_wrap_invalid_type_raises(self):
        """Test wrap() raises TypeError for invalid types."""
        with pytest.raises(TypeError, match="Expected socketio instance"):
            wrap("not a socketio instance")  # type: ignore


# =============================================================================
# Tests for emit()
# =============================================================================


class TestEmit:
    """Tests for emit() across all wrapper types."""

    def test_sync_client_emit_model(self, sync_client_wrapper, mock_sync_client):
        """Test SyncClientWrapper.emit with model instance."""
        sync_client_wrapper.emit(Ping(message="hello"))
        mock_sync_client.emit.assert_called_once_with("ping", {"message": "hello"})

    def test_async_client_emit_model(self, async_client_wrapper, mock_async_client):
        """Test AsyncClientWrapper.emit with model instance."""
        asyncio.get_event_loop().run_until_complete(
            async_client_wrapper.emit(Ping(message="hello"))
        )
        mock_async_client.emit.assert_called_once_with("ping", {"message": "hello"})

    def test_simple_client_emit_model(self, simple_client_wrapper, mock_simple_client):
        """Test SimpleClientWrapper.emit with model instance."""
        simple_client_wrapper.emit(Ping(message="hello"))
        mock_simple_client.emit.assert_called_once_with("ping", {"message": "hello"})

    def test_async_simple_client_emit_model(
        self, async_simple_client_wrapper, mock_async_simple_client
    ):
        """Test AsyncSimpleClientWrapper.emit with model instance."""
        asyncio.get_event_loop().run_until_complete(
            async_simple_client_wrapper.emit(Ping(message="hello"))
        )
        mock_async_simple_client.emit.assert_called_once_with(
            "ping", {"message": "hello"}
        )

    def test_emit_with_explicit_event_name(self, sync_client_wrapper, mock_sync_client):
        """Test emit with explicit string event name."""
        sync_client_wrapper.emit("custom-event", Ping(message="hello"))
        mock_sync_client.emit.assert_called_once_with(
            "custom-event", {"message": "hello"}
        )

    def test_emit_with_dict_data(self, sync_client_wrapper, mock_sync_client):
        """Test emit with dict data passthrough."""
        sync_client_wrapper.emit("raw-event", {"payload": 123})
        mock_sync_client.emit.assert_called_once_with("raw-event", {"payload": 123})

    def test_emit_model_with_data_raises(self, sync_client_wrapper):
        """Test that emit(Model(...), data=...) raises TypeError."""
        with pytest.raises(TypeError, match="Cannot provide both"):
            sync_client_wrapper.emit(Ping(message="hello"), data={"extra": "data"})

    def test_server_emit_with_kwargs(self, mock_sync_server):
        """Test server emit passes through kwargs like room, to."""
        wrapped = SyncServerWrapper(mock_sync_server)
        wrapped.emit(Ping(message="hello"), room="room1")
        mock_sync_server.emit.assert_called_once_with(
            "ping", {"message": "hello"}, room="room1"
        )


# =============================================================================
# Tests for call()
# =============================================================================


class TestCall:
    """Tests for call() with response_model."""

    def test_sync_call_with_response_model(self, sync_client_wrapper, mock_sync_client):
        """Test sync call() with response_model validates response."""
        result = sync_client_wrapper.call(Ping(message="hello"), response_model=Pong)
        assert isinstance(result, Pong)
        assert result.reply == "world"

    def test_async_call_with_response_model(
        self, async_client_wrapper, mock_async_client
    ):
        """Test async call() with response_model validates response."""
        result = asyncio.get_event_loop().run_until_complete(
            async_client_wrapper.call(Ping(message="hello"), response_model=Pong)
        )
        assert isinstance(result, Pong)
        assert result.reply == "world"
        mock_async_client.call.assert_called_once_with("ping", {"message": "hello"})

    def test_call_without_response_model(self, sync_client_wrapper, mock_sync_client):
        """Test call without response_model returns raw response."""
        mock_sync_client.call.return_value = {"raw": "data"}
        result = sync_client_wrapper.call("event", {"payload": 1})
        assert result == {"raw": "data"}

    def test_call_with_explicit_event_name(self, sync_client_wrapper, mock_sync_client):
        """Test call with explicit string event name."""
        result = sync_client_wrapper.call(
            "custom-ping", Ping(message="hello"), response_model=Pong
        )
        assert isinstance(result, Pong)
        mock_sync_client.call.assert_called_once_with(
            "custom-ping", {"message": "hello"}
        )

    @pytest.mark.parametrize(
        "response,expected_type",
        [
            ({"kind": "success", "data": "result"}, Success),
            ({"kind": "error", "message": "failed"}, Error),
        ],
    )
    def test_call_with_union_response(
        self, sync_client_wrapper, mock_sync_client, response, expected_type
    ):
        """Test call with union response type."""
        mock_sync_client.call.return_value = response
        result = sync_client_wrapper.call(
            Ping(message="test"), response_model=Success | Error
        )
        assert isinstance(result, expected_type)

    def test_simple_client_call_with_timeout(
        self, simple_client_wrapper, mock_simple_client
    ):
        """Test SimpleClient call passes timeout parameter."""
        simple_client_wrapper.call(
            Ping(message="hello"), response_model=Pong, timeout=30
        )
        mock_simple_client.call.assert_called_once_with(
            "ping", {"message": "hello"}, timeout=30
        )

    def test_call_with_dict_data_and_response_model(
        self, sync_client_wrapper, mock_sync_client
    ):
        """Test call() with dict data and response_model validates response."""
        result = sync_client_wrapper.call(
            "ping", {"request": "hello"}, response_model=Pong
        )
        assert isinstance(result, Pong)
        assert result.reply == "world"
        mock_sync_client.call.assert_called_once_with("ping", {"request": "hello"})


# =============================================================================
# Tests for receive() (SimpleClient only)
# =============================================================================


class TestReceive:
    """Tests for SimpleClient receive()."""

    def test_receive_with_response_model(
        self, simple_client_wrapper, mock_simple_client
    ):
        """Test receive() validates data with response_model."""
        event_name, data = simple_client_wrapper.receive(response_model=Pong)
        assert event_name == "pong"
        assert isinstance(data, Pong)
        assert data.reply == "world"
        mock_simple_client.receive.assert_called_once_with(timeout=None)

    def test_receive_with_timeout(self, simple_client_wrapper, mock_simple_client):
        """Test receive() passes timeout parameter."""
        simple_client_wrapper.receive(timeout=5.0)
        mock_simple_client.receive.assert_called_once_with(timeout=5.0)

    def test_receive_without_response_model(
        self, simple_client_wrapper, mock_simple_client
    ):
        """Test receive() without response_model returns raw data."""
        mock_simple_client.receive.return_value = ["event", {"raw": "data"}]
        event_name, data = simple_client_wrapper.receive()
        assert event_name == "event"
        assert data == {"raw": "data"}

    def test_receive_event_only(self, simple_client_wrapper, mock_simple_client):
        """Test receive() with event that has no data."""
        mock_simple_client.receive.return_value = ["heartbeat"]
        event_name, data = simple_client_wrapper.receive()
        assert event_name == "heartbeat"
        assert data is None

    def test_async_receive_with_response_model(
        self, async_simple_client_wrapper, mock_async_simple_client
    ):
        """Test async receive() validates data with response_model."""
        event_name, data = asyncio.get_event_loop().run_until_complete(
            async_simple_client_wrapper.receive(response_model=Pong)
        )
        assert event_name == "pong"
        assert isinstance(data, Pong)


# =============================================================================
# Tests for handler registration
# =============================================================================


class TestHandlerRegistration:
    """Tests for on() and event() decorators."""

    @pytest.mark.parametrize(
        "event,expected_name",
        [
            (Ping, "ping"),
            (CustomEvent, "my_custom_event"),
            ("custom-event", "custom-event"),
        ],
    )
    def test_on_decorator_event_names(
        self, sync_client_wrapper, mock_sync_client, event, expected_name
    ):
        """Test @sio.on() derives correct event name."""

        @sync_client_wrapper.on(event)
        def handler(data):
            pass

        call_args = mock_sync_client.on.call_args
        assert call_args[0][0] == expected_name

    def test_event_decorator_uses_function_name(
        self, sync_client_wrapper, mock_sync_client
    ):
        """Test @sio.event uses function name as event name."""

        @sync_client_wrapper.event
        def my_custom_handler(data: Ping) -> Pong:
            return Pong(reply=data.message)

        call_args = mock_sync_client.on.call_args
        assert call_args[0][0] == "my_custom_handler"

    def test_on_passes_namespace(self, sync_client_wrapper, mock_sync_client):
        """Test @sio.on() passes namespace to underlying sio."""

        @sync_client_wrapper.on(Ping, namespace="/chat")
        def handler(data):
            pass

        call_args = mock_sync_client.on.call_args
        assert call_args[0][0] == "ping"
        assert call_args[1]["namespace"] == "/chat"

    def test_event_passes_namespace(self, sync_client_wrapper, mock_sync_client):
        """Test @sio.event() passes namespace to underlying sio."""

        @sync_client_wrapper.event(namespace="/chat")
        def my_handler(data: Ping) -> Pong:
            return Pong(reply=data.message)

        call_args = mock_sync_client.on.call_args
        assert call_args[0][0] == "my_handler"
        assert call_args[1]["namespace"] == "/chat"

    def test_on_with_direct_handler(self, sync_client_wrapper, mock_sync_client):
        """Test sio.on() called directly with handler (not as decorator)."""

        def handler(data):
            pass

        sync_client_wrapper.on("my_event", handler, namespace="/admin")

        call_args = mock_sync_client.on.call_args
        assert call_args[0][0] == "my_event"
        assert call_args[1]["namespace"] == "/admin"


# =============================================================================
# Tests for return value validation
# =============================================================================


class TestReturnValueValidation:
    """Tests for handler return value validation."""

    def test_sync_handler_validates_and_serializes(self):
        """Test sync handler validates return type and serializes to dict."""

        def handler(data: Ping) -> Pong:
            return Pong(reply=data.message)

        wrapped = _create_sync_handler_wrapper(handler)
        result = wrapped(Ping(message="hello"))
        assert result == {"reply": "hello"}

    def test_async_handler_validates_and_serializes(self):
        """Test async handler validates return type and serializes to dict."""

        async def handler(data: Ping) -> Pong:
            return Pong(reply=data.message)

        wrapped = _create_async_handler_wrapper(handler)
        result = asyncio.get_event_loop().run_until_complete(
            wrapped(Ping(message="hello"))
        )
        assert result == {"reply": "hello"}

    def test_handler_with_invalid_return_raises(self):
        """Test handler with invalid return type raises ValidationError."""

        def handler(data: Ping) -> Pong:
            return "not a Pong"  # type: ignore

        wrapped = _create_sync_handler_wrapper(handler)
        with pytest.raises(ValidationError):
            wrapped(Ping(message="hello"))


# =============================================================================
# Tests for passthrough behavior
# =============================================================================


class TestPassthrough:
    """Tests for __getattr__ passthrough."""

    def test_passthrough_attributes(self, sync_client_wrapper, mock_sync_client):
        """Test that unknown attributes are passed through."""
        mock_sync_client.connected = True
        mock_sync_client.sid = "abc123"
        assert sync_client_wrapper.connected is True
        assert sync_client_wrapper.sid == "abc123"

    def test_passthrough_methods(self, sync_client_wrapper, mock_sync_client):
        """Test that unknown methods are passed through."""
        sync_client_wrapper.disconnect()
        mock_sync_client.disconnect.assert_called_once()

    def test_simple_client_passthrough(self, simple_client_wrapper, mock_simple_client):
        """Test SimpleClient passthrough for connect/disconnect."""
        simple_client_wrapper.connect("http://localhost:5000")
        mock_simple_client.connect.assert_called_once_with("http://localhost:5000")

        simple_client_wrapper.disconnect()
        mock_simple_client.disconnect.assert_called_once()
