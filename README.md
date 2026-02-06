# ZnDraw SocketIO
This package provides an opinionated typed interface to the python-socketio library using pydantic models.

```python
from zndraw_socketio import wrap
from pydantic import BaseModel
import socketio

sio = wrap(socketio.AsyncClient())  # or AsyncServer, Client, Server, etc.
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

## Handler Registration
Handlers are registered with `@sio.on()` or `@sio.event` and get automatic Pydantic validation from type hints.

```python
tsio = wrap(socketio.AsyncServer(async_mode="asgi"))

@tsio.on(Ping)
async def handle_ping(sid: str, data: Ping) -> Pong:
    return Pong(reply=data.message)

# or use the function name as the event name
@tsio.event
async def ping(sid: str, data: Ping) -> Pong:
    return Pong(reply=data.message)
```

## Event Names
By default, the event name is the class name in snake_case. You can customize it by setting the `event_name` attribute.

```python
class CustomEvent(BaseModel):
    ...

get_event_name(CustomEvent) == "custom_event"
```

You can override it like this:

```python
from typing import ClassVar

class CustomEvent(BaseModel):
    event_name: ClassVar[str] = "my_custom_event"
```

## Dependency Injection (Depends)
Handlers support FastAPI-style `Depends()` for dependency injection.

```python
from typing import Annotated
from zndraw_socketio import wrap, Depends

async def get_redis() -> Redis:
    return Redis()

RedisDep = Annotated[Redis, Depends(get_redis)]

@tsio.on(Ping)
async def handle(sid: str, data: Ping, redis: RedisDep) -> Pong:
    await redis.set("last_ping", data.message)
    return Pong(reply=data.message)
```

### FastAPI App Integration
To use dependencies that need `request.app` (e.g. accessing `app.state`), set the `.app` property on the wrapper. The app can be set **after** handler registration — it is resolved at event time, not at registration time.

```python
from fastapi import FastAPI, Request

app = FastAPI()
tsio = wrap(socketio.AsyncServer(async_mode="asgi"))

def get_db(request: Request) -> Database:
    return request.app.state.db

@tsio.on(Ping)
async def handle(sid: str, data: Ping, db: Annotated[Database, Depends(get_db)]) -> Pong:
    ...

# Set app later — e.g. in a lifespan, after handler registration
tsio.app = app
```

## Exception Handlers
Server wrappers support exception handlers similar to FastAPI.

```python
from zndraw_socketio import EventContext

@tsio.exception_handler(ValueError)
async def handle_error(ctx: EventContext, exc: ValueError):
    return {"error": str(exc)}
```

## Union Return Types
You might want to return `Response | ErrorResponse` from an event handler.

> [!NOTE]
> If your responses share fields, it is recommended to add a discriminator field to avoid ambiguity.

```python
class ProblemDetail(BaseModel):
    """RFC 9457 Problem Details."""
    kind: Literal["error"] = "error"
    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None

class Response(BaseModel):
    kind: Literal["response"] = "response"
    data: str

class ServerRequest(BaseModel):
    query: str

response = await sio.call(
    ServerRequest(query="..."),
    response_model=Response | ProblemDetail,
)
```
