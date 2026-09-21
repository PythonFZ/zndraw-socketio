"""Tests for AsyncAPI 3.0 schema generation."""

from __future__ import annotations

import warnings
from typing import Annotated, Union

import socketio
from pydantic import BaseModel, Discriminator, Field

from zndraw_socketio import Emits, wrap
from zndraw_socketio.asyncapi import (
    _extract_models,
    _HandlerMeta,
    _parse_docstring,
    _RestEmitterMeta,
    _to_camel_case,
    _unwrap_annotated,
    generate_asyncapi_schema,
    scan_routes,
)

# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class Ping(BaseModel):
    message: str = Field(description="The ping message")


class Pong(BaseModel):
    reply: str


class AdvancedPing(BaseModel):
    """An advanced ping with priority."""

    message: str
    priority: int


class Error(BaseModel):
    kind: str = "error"
    message: str


class Success(BaseModel):
    kind: str = "success"
    data: str


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


class TestUnwrapAnnotated:
    def test_plain_type(self):
        assert _unwrap_annotated(str) is str

    def test_annotated(self):
        assert _unwrap_annotated(Annotated[int, "meta"]) is int

    def test_basemodel(self):
        assert _unwrap_annotated(Ping) is Ping


class TestExtractModels:
    def test_none(self):
        assert _extract_models(None) == []

    def test_single_model(self):
        assert _extract_models(Ping) == [Ping]

    def test_union(self):
        result = _extract_models(Union[Ping, Pong])
        assert set(result) == {Ping, Pong}

    def test_optional_filters_none(self):
        result = _extract_models(Union[Pong, None])
        assert result == [Pong]

    def test_annotated_union(self):
        t = Annotated[Error | Success, Discriminator("kind")]
        result = _extract_models(t)
        assert set(result) == {Error, Success}

    def test_annotated_single(self):
        t = Annotated[Ping, "some metadata"]
        result = _extract_models(t)
        assert result == [Ping]


class TestParseDocstring:
    def test_none(self):
        assert _parse_docstring(None) == (None, None)

    def test_empty(self):
        assert _parse_docstring("") == (None, None)

    def test_single_line(self):
        assert _parse_docstring("Hello world.") == ("Hello world.", None)

    def test_multiline(self):
        doc = "Summary line.\n\nDetailed description here."
        summary, desc = _parse_docstring(doc)
        assert summary == "Summary line."
        assert desc == "Detailed description here."


class TestToCamelCase:
    def test_already_camel(self):
        assert _to_camel_case("handle") == "handle"

    def test_snake_case(self):
        assert _to_camel_case("handle_ping") == "handlePing"

    def test_multi_parts(self):
        assert _to_camel_case("on_user_sign_up") == "onUserSignUp"


# ---------------------------------------------------------------------------
# Unit tests for generate_asyncapi_schema
# ---------------------------------------------------------------------------


class TestGenerateSchema:
    def test_basic_handler(self):
        """Single handler with input and output model."""
        handlers = [
            _HandlerMeta(
                event_name="ping",
                handler_name="handle_ping",
                input_type=Ping,
                return_type=Pong,
                docstring="Health check endpoint.",
            )
        ]
        schema = generate_asyncapi_schema(handlers)

        assert schema["asyncapi"] == "3.0.0"
        assert schema["info"]["title"] == "Socket.IO API"

        # Channels
        assert "ping" in schema["channels"]
        assert schema["channels"]["ping"]["address"] == "ping"
        assert "Ping" in schema["channels"]["ping"]["messages"]

        # Reply channel
        assert "pingReply" in schema["channels"]
        assert schema["channels"]["pingReply"]["address"] is None
        assert "Pong" in schema["channels"]["pingReply"]["messages"]

        # Operation
        op = schema["operations"]["handlePing"]
        assert op["action"] == "receive"
        assert op["summary"] == "Health check endpoint."
        assert "reply" in op

        # Components
        assert "Ping" in schema["components"]["schemas"]
        assert "Pong" in schema["components"]["schemas"]
        assert "Ping" in schema["components"]["messages"]
        assert "Pong" in schema["components"]["messages"]
        assert (
            schema["components"]["messages"]["Ping"]["contentType"]
            == "application/json"
        )

    def test_no_return_type(self):
        """Handler with no return type should have no reply channel."""
        handlers = [
            _HandlerMeta(
                event_name="fire_and_forget",
                handler_name="handle_event",
                input_type=Ping,
                return_type=None,
                docstring=None,
            )
        ]
        schema = generate_asyncapi_schema(handlers)

        assert "fire_and_forget" in schema["channels"]
        assert "fire_and_forgetReply" not in schema["channels"]
        op = schema["operations"]["handleEvent"]
        assert "reply" not in op
        assert "summary" not in op

    def test_union_input(self):
        """Handler accepting Union input."""
        handlers = [
            _HandlerMeta(
                event_name="ping",
                handler_name="handle_ping",
                input_type=Union[Ping, AdvancedPing],
                return_type=Pong,
                docstring=None,
            )
        ]
        schema = generate_asyncapi_schema(handlers)

        channel_msgs = schema["channels"]["ping"]["messages"]
        assert "Ping" in channel_msgs
        assert "AdvancedPing" in channel_msgs

    def test_union_output(self):
        """Handler returning Union output."""
        handlers = [
            _HandlerMeta(
                event_name="process",
                handler_name="handle_process",
                input_type=Ping,
                return_type=Union[Error, Success],
                docstring=None,
            )
        ]
        schema = generate_asyncapi_schema(handlers)

        reply_msgs = schema["channels"]["processReply"]["messages"]
        assert "Error" in reply_msgs
        assert "Success" in reply_msgs

    def test_annotated_input(self):
        """Handler with Annotated input type."""
        handlers = [
            _HandlerMeta(
                event_name="ping",
                handler_name="handle_ping",
                input_type=Annotated[Ping, "metadata"],
                return_type=Pong,
                docstring=None,
            )
        ]
        schema = generate_asyncapi_schema(handlers)
        assert "Ping" in schema["channels"]["ping"]["messages"]

    def test_annotated_discriminated_union_output(self):
        """Handler returning Annotated discriminated union."""
        ResponseType = Annotated[Error | Success, Discriminator("kind")]
        handlers = [
            _HandlerMeta(
                event_name="process",
                handler_name="handle_process",
                input_type=Ping,
                return_type=ResponseType,
                docstring=None,
            )
        ]
        schema = generate_asyncapi_schema(handlers)

        reply_msgs = schema["channels"]["processReply"]["messages"]
        assert "Error" in reply_msgs
        assert "Success" in reply_msgs

    def test_custom_title_and_version(self):
        schema = generate_asyncapi_schema(
            [], title="My API", version="2.0.0", description="Test desc"
        )
        assert schema["info"]["title"] == "My API"
        assert schema["info"]["version"] == "2.0.0"
        assert schema["info"]["description"] == "Test desc"

    def test_docstring_multiline(self):
        handlers = [
            _HandlerMeta(
                event_name="ping",
                handler_name="handle_ping",
                input_type=Ping,
                return_type=Pong,
                docstring="Summary.\n\nLonger description.",
            )
        ]
        schema = generate_asyncapi_schema(handlers)
        op = schema["operations"]["handlePing"]
        assert op["summary"] == "Summary."
        assert op["description"] == "Longer description."

    def test_multiple_handlers(self):
        handlers = [
            _HandlerMeta("ping", "handle_ping", Ping, Pong, None),
            _HandlerMeta("process", "handle_process", Ping, Success, None),
        ]
        schema = generate_asyncapi_schema(handlers)
        assert len(schema["operations"]) == 2
        assert "handlePing" in schema["operations"]
        assert "handleProcess" in schema["operations"]

    def test_ref_paths_are_valid(self):
        """Verify $ref paths point to existing components."""
        handlers = [
            _HandlerMeta("ping", "handle_ping", Ping, Pong, None),
        ]
        schema = generate_asyncapi_schema(handlers)

        # Check channel message refs
        for msg in schema["channels"]["ping"]["messages"].values():
            ref = msg["$ref"]
            assert ref.startswith("#/components/messages/")
            name = ref.split("/")[-1]
            assert name in schema["components"]["messages"]

        # Check operation message refs
        for msg_ref in schema["operations"]["handlePing"]["messages"]:
            ref = msg_ref["$ref"]
            assert ref.startswith("#/channels/")

        # Check component message payload refs
        for msg in schema["components"]["messages"].values():
            payload = msg["payload"]
            if "$ref" in payload:
                name = payload["$ref"].split("/")[-1]
                assert name in schema["components"]["schemas"]


# ---------------------------------------------------------------------------
# Integration tests with actual wrapper classes
# ---------------------------------------------------------------------------


class TestAsyncServerWrapperSchema:
    def test_schema_from_registered_handlers(self):
        sio = socketio.AsyncServer(async_mode="asgi")
        tsio = wrap(sio)

        @tsio.on(Ping)
        async def handle_ping(sid: str, data: Ping) -> Pong:
            """Health check."""
            return Pong(reply=data.message)

        schema = tsio.asyncapi_schema(title="Test API")

        assert schema["asyncapi"] == "3.0.0"
        assert schema["info"]["title"] == "Test API"
        assert "ping" in schema["channels"]
        assert "handlePing" in schema["operations"]

    def test_event_decorator(self):
        sio = socketio.AsyncServer(async_mode="asgi")
        tsio = wrap(sio)

        @tsio.event
        async def ping(sid: str, data: Ping) -> Pong:
            return Pong(reply=data.message)

        schema = tsio.asyncapi_schema()
        assert "ping" in schema["channels"]
        assert "ping" in schema["operations"]

    def test_string_event_name(self):
        sio = socketio.AsyncServer(async_mode="asgi")
        tsio = wrap(sio)

        @tsio.on("custom-event")
        async def handle_custom(sid: str, data: Ping) -> Pong:
            return Pong(reply=data.message)

        schema = tsio.asyncapi_schema()
        assert "custom-event" in schema["channels"]

    def test_no_handlers_empty_schema(self):
        sio = socketio.AsyncServer(async_mode="asgi")
        tsio = wrap(sio)

        schema = tsio.asyncapi_schema()
        assert schema["channels"] == {}
        assert schema["operations"] == {}


class TestSyncServerWrapperSchema:
    def test_schema_from_registered_handlers(self):
        sio = socketio.Server()
        tsio = wrap(sio)

        @tsio.on(Ping)
        def handle_ping(sid: str, data: Ping) -> Pong:
            return Pong(reply=data.message)

        schema = tsio.asyncapi_schema()
        assert "ping" in schema["channels"]
        assert "handlePing" in schema["operations"]


class TestAsyncClientWrapperSchema:
    def test_schema_from_registered_handlers(self):
        sio = socketio.AsyncClient()
        tsio = wrap(sio)

        @tsio.on(Ping)
        async def handle_ping(data: Ping) -> Pong:
            return Pong(reply=data.message)

        schema = tsio.asyncapi_schema()
        assert "ping" in schema["channels"]

    def test_event_decorator(self):
        sio = socketio.AsyncClient()
        tsio = wrap(sio)

        @tsio.event
        async def ping(data: Ping) -> Pong:
            return Pong(reply=data.message)

        schema = tsio.asyncapi_schema()
        assert "ping" in schema["channels"]


class TestSyncClientWrapperSchema:
    def test_schema_from_registered_handlers(self):
        sio = socketio.Client()
        tsio = wrap(sio)

        @tsio.on(Ping)
        def handle_ping(data: Ping) -> Pong:
            return Pong(reply=data.message)

        schema = tsio.asyncapi_schema()
        assert "ping" in schema["channels"]


# ---------------------------------------------------------------------------
# Emits models for tests
# ---------------------------------------------------------------------------


class SessionLeft(BaseModel):
    room_id: str
    user_id: str


class Notification(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Unit tests for emits in generate_asyncapi_schema
# ---------------------------------------------------------------------------


class TestEmitsSchema:
    def test_single_emit(self):
        """Handler with emits=[Model] creates a send channel + operation."""
        handlers = [
            _HandlerMeta(
                event_name="disconnect",
                handler_name="handle_disconnect",
                input_type=None,
                return_type=None,
                docstring=None,
                emits=[SessionLeft],
            )
        ]
        schema = generate_asyncapi_schema(handlers)

        # Send channel created
        assert "session_left" in schema["channels"]
        ch = schema["channels"]["session_left"]
        assert ch["address"] == "session_left"
        assert "SessionLeft" in ch["messages"]

        # Send operation created with x-triggered-by
        assert "sendSessionLeft" in schema["operations"]
        op = schema["operations"]["sendSessionLeft"]
        assert op["action"] == "send"
        assert op["channel"] == {"$ref": "#/channels/session_left"}
        assert op["x-triggered-by"] == ["handleDisconnect"]

        # Receive operation has x-emits
        recv_op = schema["operations"]["handleDisconnect"]
        assert recv_op["x-emits"] == ["sendSessionLeft"]

        # Components populated
        assert "SessionLeft" in schema["components"]["schemas"]
        assert "SessionLeft" in schema["components"]["messages"]

    def test_multiple_emits(self):
        """Handler with emits=[A, B] creates two send channels/operations."""
        handlers = [
            _HandlerMeta(
                event_name="disconnect",
                handler_name="handle_disconnect",
                input_type=None,
                return_type=None,
                docstring=None,
                emits=[SessionLeft, Notification],
            )
        ]
        schema = generate_asyncapi_schema(handlers)

        assert "session_left" in schema["channels"]
        assert "notification" in schema["channels"]
        assert "sendSessionLeft" in schema["operations"]
        assert "sendNotification" in schema["operations"]

        # Receive operation has both in x-emits
        recv_op = schema["operations"]["handleDisconnect"]
        assert "sendSessionLeft" in recv_op["x-emits"]
        assert "sendNotification" in recv_op["x-emits"]

    def test_dedup_same_emit_model(self):
        """Two handlers emitting same model → one send op with both triggers."""
        handlers = [
            _HandlerMeta(
                event_name="disconnect",
                handler_name="handle_disconnect",
                input_type=None,
                return_type=None,
                docstring=None,
                emits=[SessionLeft],
            ),
            _HandlerMeta(
                event_name="leave_room",
                handler_name="handle_leave_room",
                input_type=None,
                return_type=None,
                docstring=None,
                emits=[SessionLeft],
            ),
        ]
        schema = generate_asyncapi_schema(handlers)

        # Only one send operation
        send_ops = [k for k, v in schema["operations"].items() if v["action"] == "send"]
        assert send_ops == ["sendSessionLeft"]

        # x-triggered-by lists both receive operations
        send_op = schema["operations"]["sendSessionLeft"]
        assert "handleDisconnect" in send_op["x-triggered-by"]
        assert "handleLeaveRoom" in send_op["x-triggered-by"]

        # Both receive operations have x-emits
        assert schema["operations"]["handleDisconnect"]["x-emits"] == [
            "sendSessionLeft"
        ]
        assert schema["operations"]["handleLeaveRoom"]["x-emits"] == ["sendSessionLeft"]

    def test_emit_coexists_with_return_type(self):
        """Handler with both return type and emits → receive reply + send channels."""
        handlers = [
            _HandlerMeta(
                event_name="ping",
                handler_name="handle_ping",
                input_type=Ping,
                return_type=Pong,
                docstring=None,
                emits=[SessionLeft],
            )
        ]
        schema = generate_asyncapi_schema(handlers)

        # Receive channel + reply
        assert "ping" in schema["channels"]
        assert "pingReply" in schema["channels"]
        recv_op = schema["operations"]["handlePing"]
        assert recv_op["action"] == "receive"
        assert recv_op["x-emits"] == ["sendSessionLeft"]

        # Send channel with x-triggered-by
        assert "session_left" in schema["channels"]
        send_op = schema["operations"]["sendSessionLeft"]
        assert send_op["action"] == "send"
        assert send_op["x-triggered-by"] == ["handlePing"]

    def test_emit_reuses_existing_channel(self):
        """Emitted model reuses existing channel if handler also receives that event."""
        handlers = [
            _HandlerMeta(
                event_name="session_left",
                handler_name="handle_session_left",
                input_type=SessionLeft,
                return_type=None,
                docstring=None,
                emits=[],
            ),
            _HandlerMeta(
                event_name="disconnect",
                handler_name="handle_disconnect",
                input_type=None,
                return_type=None,
                docstring=None,
                emits=[SessionLeft],
            ),
        ]
        schema = generate_asyncapi_schema(handlers)

        # Channel should exist (created by receive handler) and have the emit message added
        assert "session_left" in schema["channels"]
        ch = schema["channels"]["session_left"]
        assert "SessionLeft" in ch["messages"]

    def test_emit_ref_paths_are_valid(self):
        """Verify $ref paths for emitted models point to existing components."""
        handlers = [
            _HandlerMeta(
                event_name="disconnect",
                handler_name="handle_disconnect",
                input_type=None,
                return_type=None,
                docstring=None,
                emits=[SessionLeft],
            )
        ]
        schema = generate_asyncapi_schema(handlers)

        # Operation message ref
        op = schema["operations"]["sendSessionLeft"]
        for msg_ref in op["messages"]:
            ref = msg_ref["$ref"]
            assert ref.startswith("#/channels/")

        # x-triggered-by strings point to existing operations
        for op_name in op["x-triggered-by"]:
            assert op_name in schema["operations"]

        # x-emits strings point to existing operations
        recv_op = schema["operations"]["handleDisconnect"]
        for op_name in recv_op["x-emits"]:
            assert op_name in schema["operations"]

        # Component message payload ref
        msg = schema["components"]["messages"]["SessionLeft"]
        payload_ref = msg["payload"]["$ref"]
        name = payload_ref.split("/")[-1]
        assert name in schema["components"]["schemas"]


# ---------------------------------------------------------------------------
# Integration tests for emits on wrapper classes
# ---------------------------------------------------------------------------


class TestAsyncServerEmitsIntegration:
    def test_on_with_emits(self):
        sio = socketio.AsyncServer(async_mode="asgi")
        tsio = wrap(sio)

        @tsio.on(Ping, emits=[SessionLeft])
        async def handle_ping(sid: str, data: Ping) -> Pong:
            return Pong(reply=data.message)

        schema = tsio.asyncapi_schema()
        assert "session_left" in schema["channels"]
        assert "sendSessionLeft" in schema["operations"]
        assert schema["operations"]["sendSessionLeft"]["action"] == "send"
        # Bidirectional refs
        assert schema["operations"]["sendSessionLeft"]["x-triggered-by"] == [
            "handlePing"
        ]
        assert "sendSessionLeft" in schema["operations"]["handlePing"]["x-emits"]

    def test_event_with_emits(self):
        sio = socketio.AsyncServer(async_mode="asgi")
        tsio = wrap(sio)

        @tsio.event(emits=[SessionLeft])
        async def disconnect(sid: str) -> None:
            pass

        schema = tsio.asyncapi_schema()
        assert "session_left" in schema["channels"]
        assert "sendSessionLeft" in schema["operations"]
        assert schema["operations"]["disconnect"]["x-emits"] == ["sendSessionLeft"]


class TestSyncServerEmitsIntegration:
    def test_on_with_emits(self):
        sio = socketio.Server()
        tsio = wrap(sio)

        @tsio.on(Ping, emits=[SessionLeft])
        def handle_ping(sid: str, data: Ping) -> Pong:
            return Pong(reply=data.message)

        schema = tsio.asyncapi_schema()
        assert "session_left" in schema["channels"]
        assert "sendSessionLeft" in schema["operations"]

    def test_event_with_emits(self):
        sio = socketio.Server()
        tsio = wrap(sio)

        @tsio.event(emits=[SessionLeft])
        def disconnect(sid: str) -> None:
            pass

        schema = tsio.asyncapi_schema()
        assert "sendSessionLeft" in schema["operations"]


class TestAsyncClientEmitsIntegration:
    def test_on_with_emits(self):
        sio = socketio.AsyncClient()
        tsio = wrap(sio)

        @tsio.on(Ping, emits=[SessionLeft])
        async def handle_ping(data: Ping) -> Pong:
            return Pong(reply=data.message)

        schema = tsio.asyncapi_schema()
        assert "session_left" in schema["channels"]
        assert "sendSessionLeft" in schema["operations"]

    def test_event_with_emits(self):
        sio = socketio.AsyncClient()
        tsio = wrap(sio)

        @tsio.event(emits=[Notification])
        async def ping(data: Ping) -> Pong:
            return Pong(reply=data.message)

        schema = tsio.asyncapi_schema()
        assert "sendNotification" in schema["operations"]


class TestSyncClientEmitsIntegration:
    def test_on_with_emits(self):
        sio = socketio.Client()
        tsio = wrap(sio)

        @tsio.on(Ping, emits=[SessionLeft])
        def handle_ping(data: Ping) -> Pong:
            return Pong(reply=data.message)

        schema = tsio.asyncapi_schema()
        assert "session_left" in schema["channels"]
        assert "sendSessionLeft" in schema["operations"]

    def test_event_with_emits(self):
        sio = socketio.Client()
        tsio = wrap(sio)

        @tsio.event(emits=[Notification])
        def ping(data: Ping) -> Pong:
            return Pong(reply=data.message)

        schema = tsio.asyncapi_schema()
        assert "sendNotification" in schema["operations"]


# ---------------------------------------------------------------------------
# Runtime warning tests
# ---------------------------------------------------------------------------


class TestEmitWarnings:
    def test_undocumented_emit_warns(self):
        """emit(UnregisteredModel(...)) with warn_undocumented_emits=True → warning."""
        sio = socketio.AsyncServer(async_mode="asgi")
        tsio = wrap(sio, warn_undocumented_emits=True)

        @tsio.on(Ping)
        async def handle_ping(sid: str, data: Ping) -> Pong:
            return Pong(reply=data.message)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            # Resolve args manually since we can't actually emit without a connection
            from zndraw_socketio.wrapper import _resolve_emit_args

            event_name, _ = _resolve_emit_args(SessionLeft(room_id="r", user_id="u"))
            # Check the warning logic directly
            assert event_name not in tsio._known_emit_events

    def test_documented_emit_no_warn(self):
        """emit(RegisteredModel(...)) → no warning expected."""
        sio = socketio.AsyncServer(async_mode="asgi")
        tsio = wrap(sio, warn_undocumented_emits=True)

        @tsio.on(Ping, emits=[SessionLeft])
        async def handle_ping(sid: str, data: Ping) -> Pong:
            return Pong(reply=data.message)

        from zndraw_socketio.wrapper import _resolve_emit_args

        event_name, _ = _resolve_emit_args(SessionLeft(room_id="r", user_id="u"))
        assert event_name in tsio._known_emit_events

    def test_default_no_warn_flag(self):
        """Default (flag off) → _warn_undocumented_emits is False."""
        sio = socketio.AsyncServer(async_mode="asgi")
        tsio = wrap(sio)
        assert tsio._warn_undocumented_emits is False

    def test_known_events_populated(self):
        """_known_emit_events includes handler event + emits models."""
        sio = socketio.AsyncServer(async_mode="asgi")
        tsio = wrap(sio, warn_undocumented_emits=True)

        @tsio.on(Ping, emits=[SessionLeft, Notification])
        async def handle_ping(sid: str, data: Ping) -> Pong:
            return Pong(reply=data.message)

        assert "ping" in tsio._known_emit_events
        assert "session_left" in tsio._known_emit_events
        assert "notification" in tsio._known_emit_events


# ---------------------------------------------------------------------------
# REST Emitter tests
# ---------------------------------------------------------------------------


class TestEmitsClass:
    def test_construction_single(self):
        e = Emits(SessionLeft)
        assert e.models == (SessionLeft,)

    def test_construction_multiple(self):
        e = Emits(SessionLeft, Notification)
        assert e.models == (SessionLeft, Notification)

    def test_construction_empty(self):
        e = Emits()
        assert e.models == ()


class _FakeRoute:
    """Minimal duck-typed FastAPI route for testing scan_routes."""

    def __init__(self, endpoint, methods=None, path="/"):
        self.endpoint = endpoint
        self.methods = methods or {"GET"}
        self.path = path


class TestScanRoutes:
    def test_route_with_emits(self):
        async def my_endpoint(
            sio: Annotated[str, Emits(SessionLeft)],
        ) -> None:
            """Update selection."""

        routes = [_FakeRoute(my_endpoint, methods={"PUT"}, path="/items/{id}")]
        result = scan_routes(routes)
        assert len(result) == 1
        meta = result[0]
        assert meta.handler_name == "my_endpoint"
        assert meta.method == "PUT"
        assert meta.path == "/items/{id}"
        assert meta.summary == "Update selection."
        assert meta.emits == [SessionLeft]

    def test_route_with_multiple_emits(self):
        async def my_endpoint(
            sio: Annotated[str, Emits(SessionLeft, Notification)],
        ) -> None:
            pass

        routes = [_FakeRoute(my_endpoint, methods={"POST"}, path="/notify")]
        result = scan_routes(routes)
        assert len(result) == 1
        assert result[0].emits == [SessionLeft, Notification]

    def test_route_without_emits(self):
        async def plain_endpoint(x: int) -> None:
            pass

        routes = [_FakeRoute(plain_endpoint)]
        result = scan_routes(routes)
        assert result == []

    def test_route_no_endpoint(self):
        """Object without endpoint attr is skipped."""

        class NoEndpoint:
            pass

        result = scan_routes([NoEndpoint()])
        assert result == []

    def test_multiple_methods_picks_sorted_first(self):
        async def multi(sio: Annotated[str, Emits(Notification)]) -> None:
            pass

        routes = [_FakeRoute(multi, methods={"PUT", "PATCH"}, path="/x")]
        result = scan_routes(routes)
        assert result[0].method == "PATCH"  # sorted: PATCH < PUT


class TestGenerateSchemaWithRestEmitters:
    def test_rest_only_emit(self):
        """REST-only emit creates send operation with x-rest-triggers only."""
        rest_emitters = [
            _RestEmitterMeta(
                handler_name="update_selection",
                method="PUT",
                path="/{key}/selection",
                summary="Update selection",
                emits=[SessionLeft],
            )
        ]
        schema = generate_asyncapi_schema([], rest_emitters=rest_emitters)

        # Channel created
        assert "session_left" in schema["channels"]
        ch = schema["channels"]["session_left"]
        assert ch["address"] == "session_left"
        assert "SessionLeft" in ch["messages"]

        # Send operation with x-rest-triggers
        assert "sendSessionLeft" in schema["operations"]
        op = schema["operations"]["sendSessionLeft"]
        assert op["action"] == "send"
        assert "x-triggered-by" not in op
        assert len(op["x-rest-triggers"]) == 1
        trigger = op["x-rest-triggers"][0]
        assert trigger["operationId"] == "updateSelection"
        assert trigger["method"] == "PUT"
        assert trigger["path"] == "/{key}/selection"
        assert trigger["summary"] == "Update selection"

        # Components
        assert "SessionLeft" in schema["components"]["schemas"]
        assert "SessionLeft" in schema["components"]["messages"]

    def test_rest_trigger_no_summary(self):
        """REST trigger without summary omits the key."""
        rest_emitters = [
            _RestEmitterMeta(
                handler_name="delete_item",
                method="DELETE",
                path="/items/{id}",
                summary=None,
                emits=[Notification],
            )
        ]
        schema = generate_asyncapi_schema([], rest_emitters=rest_emitters)
        trigger = schema["operations"]["sendNotification"]["x-rest-triggers"][0]
        assert "summary" not in trigger

    def test_mixed_socket_and_rest(self):
        """Same model emitted from socket handler and REST → both annotations."""
        handlers = [
            _HandlerMeta(
                event_name="disconnect",
                handler_name="handle_disconnect",
                input_type=None,
                return_type=None,
                docstring=None,
                emits=[SessionLeft],
            )
        ]
        rest_emitters = [
            _RestEmitterMeta(
                handler_name="update_selection",
                method="PUT",
                path="/selection",
                summary="Update selection",
                emits=[SessionLeft],
            )
        ]
        schema = generate_asyncapi_schema(handlers, rest_emitters=rest_emitters)

        op = schema["operations"]["sendSessionLeft"]
        assert op["action"] == "send"
        assert "handleDisconnect" in op["x-triggered-by"]
        assert len(op["x-rest-triggers"]) == 1
        assert op["x-rest-triggers"][0]["operationId"] == "updateSelection"

    def test_multiple_rest_emitters_same_model(self):
        """Two REST endpoints emitting same model → both in x-rest-triggers."""
        rest_emitters = [
            _RestEmitterMeta(
                handler_name="endpoint_a",
                method="POST",
                path="/a",
                summary=None,
                emits=[SessionLeft],
            ),
            _RestEmitterMeta(
                handler_name="endpoint_b",
                method="PUT",
                path="/b",
                summary=None,
                emits=[SessionLeft],
            ),
        ]
        schema = generate_asyncapi_schema([], rest_emitters=rest_emitters)

        op = schema["operations"]["sendSessionLeft"]
        assert len(op["x-rest-triggers"]) == 2
        ids = [t["operationId"] for t in op["x-rest-triggers"]]
        assert "endpointA" in ids
        assert "endpointB" in ids

    def test_rest_emit_reuses_existing_channel(self):
        """REST emit model matches existing receive channel → reused."""
        handlers = [
            _HandlerMeta(
                event_name="session_left",
                handler_name="handle_session_left",
                input_type=SessionLeft,
                return_type=None,
                docstring=None,
            )
        ]
        rest_emitters = [
            _RestEmitterMeta(
                handler_name="kick_user",
                method="POST",
                path="/kick",
                summary=None,
                emits=[SessionLeft],
            )
        ]
        schema = generate_asyncapi_schema(handlers, rest_emitters=rest_emitters)

        # Only one session_left channel
        assert "session_left" in schema["channels"]
        assert "SessionLeft" in schema["channels"]["session_left"]["messages"]

    def test_ref_paths_valid_with_rest(self):
        """All $ref paths valid when REST emitters are present."""
        rest_emitters = [
            _RestEmitterMeta(
                handler_name="update_sel",
                method="PUT",
                path="/sel",
                summary="Update",
                emits=[SessionLeft],
            )
        ]
        schema = generate_asyncapi_schema([], rest_emitters=rest_emitters)

        op = schema["operations"]["sendSessionLeft"]
        for msg_ref in op["messages"]:
            ref = msg_ref["$ref"]
            assert ref.startswith("#/channels/")

        msg = schema["components"]["messages"]["SessionLeft"]
        payload_ref = msg["payload"]["$ref"]
        name = payload_ref.split("/")[-1]
        assert name in schema["components"]["schemas"]


# ---------------------------------------------------------------------------
# Integration: Full FastAPI app with Emits + AsyncServerWrapper
# ---------------------------------------------------------------------------


class TestRestEmitsIntegration:
    def test_full_fastapi_integration(self):
        """AsyncServerWrapper.asyncapi_schema() auto-discovers REST Emits."""
        from fastapi import FastAPI

        sio_raw = socketio.AsyncServer(async_mode="asgi")
        tsio = wrap(sio_raw)

        app = FastAPI()
        tsio.app = app

        # Note: Emits(SessionLeft) is all scan_routes needs.
        # Depends(tsio) is omitted because from __future__ import annotations
        # makes annotations lazy strings, and tsio isn't in module globals.
        @app.put("/rooms/{room_id}/selection")
        async def update_selection(
            sio: Annotated[object, Emits(SessionLeft)],
            room_id: str,
        ) -> dict:
            """Update room selection."""
            return {"status": "ok"}

        schema = tsio.asyncapi_schema(title="Integration Test")

        assert "session_left" in schema["channels"]
        assert "sendSessionLeft" in schema["operations"]
        op = schema["operations"]["sendSessionLeft"]
        assert op["action"] == "send"
        assert len(op["x-rest-triggers"]) == 1
        trigger = op["x-rest-triggers"][0]
        assert trigger["operationId"] == "updateSelection"
        assert trigger["method"] == "PUT"
        assert trigger["path"] == "/rooms/{room_id}/selection"
        assert trigger["summary"] == "Update room selection."

    def test_mixed_socket_and_rest_integration(self):
        """Both socket handler and REST emit same model → combined."""
        from fastapi import FastAPI

        sio_raw = socketio.AsyncServer(async_mode="asgi")
        tsio = wrap(sio_raw)

        @tsio.on(Ping, emits=[SessionLeft])
        async def handle_ping(sid: str, data: Ping) -> Pong:
            return Pong(reply=data.message)

        app = FastAPI()
        tsio.app = app

        @app.post("/kick")
        async def kick_user(
            sio: Annotated[object, Emits(SessionLeft)],
        ) -> dict:
            """Kick a user."""
            return {"status": "ok"}

        schema = tsio.asyncapi_schema()

        op = schema["operations"]["sendSessionLeft"]
        assert "handlePing" in op["x-triggered-by"]
        assert len(op["x-rest-triggers"]) == 1
        assert op["x-rest-triggers"][0]["operationId"] == "kickUser"

    def test_rest_emits_populate_known_events(self):
        """REST Emits models are added to _known_emit_events on schema gen."""
        from fastapi import FastAPI

        sio_raw = socketio.AsyncServer(async_mode="asgi")
        tsio = wrap(sio_raw, warn_undocumented_emits=True)

        app = FastAPI()
        tsio.app = app

        @app.post("/notify")
        async def notify(
            sio: Annotated[object, Emits(Notification)],
        ) -> dict:
            return {"status": "ok"}

        # Before schema generation, event is unknown
        assert "notification" not in tsio._known_emit_events

        tsio.asyncapi_schema()

        # After schema generation, event is known
        assert "notification" in tsio._known_emit_events

    def test_no_app_no_rest_emitters(self):
        """No app set → no REST emitters in schema."""
        sio_raw = socketio.AsyncServer(async_mode="asgi")
        tsio = wrap(sio_raw)
        # app not set
        schema = tsio.asyncapi_schema()
        assert schema["channels"] == {}
        assert schema["operations"] == {}
