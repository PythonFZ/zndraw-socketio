# ZnDraw SocketIO
This package provides an opinionated typed interface to the python-socketio library using pydantic models.

```python
from zndraw_socketio import Wrapper
from pydantic import BaseModel
import socketio

sio = Wrapper(socketio.AsyncClient()) # can be server as well
```

## Emit Pattern
```python
class Ping(BaseModel):
    message: str

# kwargs are passed to socketio's emit method
# emits {"message": "Hello, World!"} to "ping"
await sio.emit(Ping(message="Hello, World!"), **kwargs)
# emits {"message": "Hello, World!"} to "my-ping"
await sio.emit("my-ping", Ping(message="Hello, World!"), **kwargs)
# standard sio behaviour
await sio.emit("event", {"payload": ...})
```

## Call / RPC Pattern
```python
class Pong(BaseModel):
    reply: str

# emits {"message": "Hello, World!"} to "ping" and receives Pong(reply=...) in return
response = await sio.call(Ping(message="Hello, World!"), response_model=Pong)
assert isinstance(response, Pong)
# emits {"message": "Hello, World!"} to "my-ping" and receives Pong(reply=...) in return
response = await sio.call("my-ping", Ping(message="Hello, World!"), response_model=Pong)
assert isinstance(response, Pong)
# standard sio behaviour
response = await sio.call("event", {"payload": ...})
# standard response obj, typically dict
```

## Event Names
By default, the event name is the class name in snake_case. You can customize it by setting the `event_name` attribute.

```python
class CustomEvent(Event):
    ...

assert CustomEvent.event_name == "custom_event"
```

you can override it like this:

```python
from typing import ClassVar

class CustomEvent(Event):
    event_name: ClassVar[str] = "my_custom_event"
```

## Union Return Types
You might want to return `Response|ErrorResponse` from an event handler.

> [!NOTE]
> If your responses share fields, it is recommended to add a discriminator field to avoid ambiguity.


```python
class ProblemDetail(BaseModel):
    """RFC 9457 Problem Details.

    https://www.rfc-editor.org/rfc/rfc9457.html
    """
    kind: Literal["error"] = "error" # The discriminator (nod needed in this example)
    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None

class Response(BaseModel):
    kind: Literal["response"] = "response" # The discriminator (not needed in this example)
    data: str

class ServerRequest(BaseModel):
    query: str


response_model = ... # do we need typing Annotated here?

response = await sio.call(ServerRequest(query="..."), response_model=response_model)
```