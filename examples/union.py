from pprint import pprint
from typing import Literal

import socketio
from pydantic import BaseModel

from zndraw_socketio import wrap


class Model(BaseModel):
    name: Literal["A", "B"]


class A(BaseModel):
    type: Literal["A"]
    value: int


class B(BaseModel):
    type: Literal["B"]
    value: str


sio = socketio.AsyncServer(async_mode="asgi")
tsio = wrap(sio)


@tsio.on(Model)
async def handle_ping(sid: str, data: Model) -> A | B:
    """Health check."""
    if data.name == "A":
        return A(type="A", value=42)
    else:
        return B(type="B", value="Hello")


schema = tsio.asyncapi_schema(title="Test API")
pprint(schema)
