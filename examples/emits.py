from pprint import pprint

import socketio
from pydantic import BaseModel

from zndraw_socketio import wrap


class Join(BaseModel):
    message: str


class JoinResponse(BaseModel):
    ok: bool


class UserJoined(BaseModel):
    name: str


class Leave(BaseModel):
    message: str


class LeaveResponse(BaseModel):
    ok: bool


class UserLeft(BaseModel):
    name: str


class UserEvent(BaseModel):
    pass


sio = socketio.AsyncServer(async_mode="asgi")
tsio = wrap(sio)


@tsio.on(Join, emits=[UserJoined, UserEvent])
async def handle_join(sid: str, data: Join) -> JoinResponse:
    """Health check."""
    await tsio.emit(UserJoined(name=data.message))
    await tsio.emit(UserEvent())
    return JoinResponse(ok=True)


@tsio.on(Leave, emits=[UserLeft, UserEvent])
async def handle_leave(sid: str, data: Leave) -> LeaveResponse:
    """Health check."""
    await tsio.emit(UserLeft(name=data.message))
    await tsio.emit(UserEvent())
    return LeaveResponse(ok=True)


schema = tsio.asyncapi_schema(title="Test API")
pprint(schema)
