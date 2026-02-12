"""Tests for AsyncAPI 3.0 schema generation."""

from __future__ import annotations

from typing import Annotated, Union

import socketio
from pydantic import BaseModel, Discriminator, Field

from zndraw_socketio import wrap
from zndraw_socketio.asyncapi import (
    _extract_models,
    _HandlerMeta,
    _parse_docstring,
    _to_camel_case,
    _unwrap_annotated,
    generate_asyncapi_schema,
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
        t = Annotated[Union[Error, Success], Discriminator("kind")]
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
        ResponseType = Annotated[Union[Error, Success], Discriminator("kind")]
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
