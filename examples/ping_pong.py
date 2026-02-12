from pprint import pprint

import socketio
from pydantic import BaseModel

from zndraw_socketio import wrap


class Ping(BaseModel):
    message: str


class Pong(BaseModel):
    reply: str


sio = socketio.AsyncServer(async_mode="asgi")
tsio = wrap(sio)


@tsio.on(Ping)
async def handle_ping(sid: str, data: Ping) -> Pong:
    """Health check."""
    return Pong(reply=data.message)


schema = tsio.asyncapi_schema(title="Test API")
pprint(schema)
