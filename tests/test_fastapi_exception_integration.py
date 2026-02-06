from typing import Any

import aiohttp
import pytest
import socketio
from fastapi import Depends, FastAPI
from pydantic import BaseModel

from zndraw_socketio import AsyncServerWrapper, EventContext, wrap


class ErrorResponse(BaseModel):
    error: str
    sid: str
    event: str
    namespace: str
    data: Any
    exc: str


class ErrorTriggerEvent(BaseModel):
    error: bool


class OkayResponse(BaseModel):
    message: str


@pytest.mark.asyncio
async def test_exception_handler_integration_sockets(server_factory):
    app = FastAPI()
    tsio = wrap(socketio.AsyncServer(async_mode="asgi"))

    @tsio.on(ErrorTriggerEvent)
    async def trigger_error(
        sid, data: ErrorTriggerEvent
    ) -> OkayResponse | ErrorResponse:
        if data.error:
            raise ValueError("Intentional error")
        return OkayResponse(message="No error triggered")

    @tsio.exception_handler(ValueError)
    async def handle_value_error(ctx: EventContext, exc: ValueError):
        return ErrorResponse(
            error="value_error",
            sid=ctx.sid,
            event=ctx.event,
            namespace=ctx.namespace,
            data=ctx.data,
            exc=str(exc),
        )

    url = await server_factory(socketio.ASGIApp(tsio, app))
    client = wrap(socketio.AsyncSimpleClient())

    await client.connect(url)
    assert client.connected

    response = await client.call(
        ErrorTriggerEvent(error=True), response_model=ErrorResponse | OkayResponse
    )
    model = ErrorResponse.model_validate(response)
    assert model.error == "value_error"
    assert model.sid == client.sid
    assert model.event == "error_trigger_event"
    assert model.namespace == "/"
    assert model.exc == "Intentional error"
    assert model.data == {"error": True}

    response = await client.call(
        ErrorTriggerEvent(error=False), response_model=ErrorResponse | OkayResponse
    )
    assert OkayResponse.model_validate(response).message == "No error triggered"

    await client.disconnect()


@pytest.mark.asyncio
async def test_exception_handler_integration_rest(server_factory):
    app = FastAPI()
    tsio = wrap(socketio.AsyncServer(async_mode="asgi"))

    @app.get("/trigger_error")
    async def trigger_error(my_tsio: AsyncServerWrapper = Depends(tsio)) -> dict:
        await my_tsio.emit(OkayResponse(message="This is a test"))
        return {"value": "okay"}

    client = wrap(socketio.AsyncSimpleClient())

    socket_app = socketio.ASGIApp(tsio, app)
    url = await server_factory(socket_app)

    await client.connect(url)
    assert client.connected

    async with aiohttp.ClientSession() as session:
        async with session.get(f"{url}/trigger_error") as resp:
            assert resp.status == 200
            data = await resp.json()
            assert data == {"value": "okay"}

    # ensure that the message was received by the client
    event, data = await client.receive(timeout=5)
    assert event == "okay_response"
    assert data["message"] == "This is a test"

    await client.disconnect()
